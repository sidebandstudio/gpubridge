"""Shared helpers for the GPU validation kit.

Every script in scripts/gpu writes JSON records with the same layout: the kit
version, the environment (hardware, driver, library versions, relevant env
vars), the kind of run, and pass/fail per stage with full error text. The
``run`` block always says whether the run was a split test, so split-test
results can't be mistaken for a real mixed-vendor run.

Records hold only allowlisted environment variables (``recorded_env``), and
everything written goes through ``redact``: no project path, home directory,
user name or URL credentials.
"""

from __future__ import annotations

import contextlib
import faulthandler
import getpass
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import gpubridge

KIT_VERSION = 1
REPO = Path(__file__).resolve().parents[2]

#: The environment variables a record may hold: an allowlist, by exact name or
#: prefix. Nothing else from the environment is recorded.
ENV_PREFIXES = ("GPUBRIDGE_", "NCCL_", "RCCL_", "GLOO_", "TORCH_", "UCX_")
ENV_NAMES = frozenset({
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS", "KIT_TIMEOUT", "STEP_TIMEOUT",
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_NODELIST",
    "SLURM_NNODES", "SLURM_JOB_NUM_NODES", "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID",
    "SLURM_NTASKS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_HET_SIZE",
    # Can put another checkout's gpubridge ahead of the venv's (kit_import).
    "PYTHONPATH",
})
#: Never recorded, even under an allowed prefix: names that may hold a
#: credential or say who ran the job.
SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "USER",
                "HOME", "ACCOUNT")


def recorded_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted environment variables, values redacted."""
    environ = os.environ if environ is None else environ
    return {name: redact(value) for name, value in sorted(environ.items())
            if (name in ENV_NAMES or name.startswith(ENV_PREFIXES))
            and not any(word in name.upper() for word in SECRET_WORDS)}


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # no user name for this uid, in some containers
        return None


def redact(text: str) -> str:
    """``text`` without project paths, the home directory, the user name as a path
    segment, or credentials in URLs; placeholders take their place."""
    end = r"(?=/|$|\s|['\":,])"  # a whole path component, not a prefix of one
    text = re.sub(r"(\w+://)[^/@\s]+@", r"\1***@", text)
    text = re.sub(r"/projects/[^/\s'\":,]+", "$PROJECT_DIR", text)
    project = os.environ.get("PROJECT_DIR", "").rstrip("/")
    if len(project) > 1:
        text = re.sub(re.escape(project) + end, "$PROJECT_DIR", text)
    home = os.path.expanduser("~").rstrip("/")
    if len(home) > 1:
        text = re.sub(re.escape(home) + end, "~", text)
    user = _user()
    if user:
        text = re.sub(f"/{re.escape(user)}{end}", "/$USER", text)
    return text


def scrub(data: Any) -> Any:
    """``data`` with every string in it redacted."""
    if isinstance(data, str):
        return redact(data)
    if isinstance(data, dict):
        return {key: scrub(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [scrub(value) for value in data]
    return data


def default_timeout(fallback: float) -> float:
    """Seconds for gpubridge.init's timeout: KIT_TIMEOUT, which the kit's shell
    scripts set (common.sh), or ``fallback`` when a tool runs on its own."""
    raw = os.environ.get("KIT_TIMEOUT", "").strip()
    return float(raw) if raw else fallback


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as exc:  # recording must never take a run down
        return default if default is not None else f"error: {type(exc).__name__}: {exc}"


def _command(*args: str) -> str | None:
    """A command's stdout, stripped, or None if it is missing or fails."""
    result = _safe(lambda: subprocess.run(args, capture_output=True, text=True, timeout=10))
    if isinstance(result, subprocess.CompletedProcess) and result.returncode == 0:
        return result.stdout.strip()
    return None


def _git_commit() -> str | None:
    return _command("git", "-C", str(REPO), "rev-parse", "--short", "HEAD")


def kit_import() -> dict[str, Any]:
    """Where ``import gpubridge`` came from, and whether that is this checkout's src.

    The venvs install gpubridge editable from one checkout; a job run from another
    checkout (or with PYTHONPATH set) can import a different gpubridge than the
    scripts it runs. ``git_commit`` describes the scripts; this describes the package.
    """
    package = Path(gpubridge.__file__).resolve().parent
    expected = (REPO / "src" / "gpubridge").resolve()
    checkout = _command("git", "-C", str(package), "rev-parse", "--show-toplevel")
    status = (_command("git", "-C", str(package), "status", "--porcelain",
                       "--untracked-files=no") if checkout else None)
    return {
        "gpubridge_file": str(package / "__init__.py"),
        "gpubridge_git_commit": (_command("git", "-C", str(package), "rev-parse", "--short",
                                          "HEAD") if checkout else None),
        "gpubridge_git_dirty": None if status is None else bool(status),
        "gpubridge_in_kit_checkout": package == expected,
        "kit_checkout_src": str(expected),
    }


def triton_check(index: int) -> dict[str, Any]:
    """Whether gpubridge's Triton compression kernels compile and run on GPU ``index``.

    With Triton installed, ``GPUBRIDGE_COMPRESSION_KERNELS=auto`` (the default) uses
    it for GPU tensors, so a Triton that can't compile for this GPU (the MI50's
    gfx906 with pytorch-triton-rocm 3.5.1) fails every compressed collective.
    """
    try:
        import triton
    except ImportError:
        return {"triton": None, "compiles": None, "note": "Triton not installed: torch kernels"}
    except Exception as exc:  # installed but broken: whatever the probe hits, it only records
        return {"triton": "unknown", "compiles": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:  # every kernel a compressed collective launches: quantize and dequantize, both modes
        from gpubridge import _compression_kernels, compression

        device = torch.device("cuda", index)
        x = torch.linspace(-3, 3, 4096, device=device)
        for mode in ("bf16", "fp8"):
            payload = torch.empty(compression.payload_nbytes(mode, x.numel()), dtype=torch.uint8,
                                  device=device)
            _compression_kernels.quantize(mode, x, torch.zeros_like(x), payload)
            _compression_kernels.dequantize_accumulate(mode, payload.view(1, -1),
                                                       torch.zeros_like(x))
        torch.cuda.synchronize(device)
    except Exception as exc:  # a compiler error as Triton reports it, or anything else
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        return {"triton": triton.__version__, "compiles": False,
                "error": f"{type(exc).__name__}: {first}"[:300]}
    return {"triton": triton.__version__, "compiles": True}


def _git_dirty() -> bool | None:
    """Whether tracked files differ from the commit (new results files don't count)."""
    status = _command("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def driver_version() -> str | None:
    """The GPU kernel driver: NVIDIA's from nvidia-smi, AMD's amdgpu module version."""
    if torch.version.cuda:
        out = _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        return out.splitlines()[0].strip() if out else None
    if torch.version.hip:
        # An in-tree amdgpu (no DKMS package) has no module version: rocm-smi then
        # reports the kernel's release, as CSV, which driver_text reduces to the value.
        from validation_tables import driver_text

        return _read("/sys/module/amdgpu/version") or driver_text(_command(
            "rocm-smi", "--showdriverversion", "--csv"))
    return None


def _nccl_version() -> str | None:
    """NCCL's version on CUDA builds, RCCL's on ROCm builds."""
    if not dist.is_nccl_available():
        return None
    version = torch.cuda.nccl.version()
    return ".".join(map(str, version)) if isinstance(version, tuple) else str(version)


#: A CUDA arch-list entry: a cubin (sm_XY) or PTX (compute_XY) for capability X.Y. Entries
#: with a suffix (sm_90a, sm_100f) are for one architecture or family only: never counted.
_CUDA_ARCH = re.compile(r"(sm|compute)_(\d+)")


def gpu_target(props: Any) -> str:
    """A GPU's build target: on ROCm its gcnArchName without the feature flags
    ("gfx906:sramecc+:xnack-" is gfx906); on CUDA sm_XY from its compute capability.
    CUDA builds of PyTorch have a gcnArchName too, holding the product name, so only
    ROCm builds read it."""
    if torch.version.hip:
        return (getattr(props, "gcnArchName", "") or "").split(":")[0] or "unknown"
    return f"sm_{props.major}{props.minor}"


def target_match(target: str, capability: tuple[int, int] | None,
                 arch_list: list[str]) -> str | None:
    """Which rule says a PyTorch build with ``arch_list`` can run a GPU, or None:

    - ``exact``: the list has the GPU's own target (sm_70, gfx906);
    - ``compatible``: a cubin for the same major and an earlier minor (sm_80 for an 8.6
      GPU), which CUDA runs unchanged on later minors of its major (Tegra parts, such
      as sm_72 and sm_87, are the exception: none is a target here);
    - ``ptx``: PTX for the GPU's capability or below (compute_70 for 9.0), which the
      driver compiles when it loads.

    ``capability`` is None on ROCm, where only an exact match counts.
    """
    names = {entry.split(":")[0] for entry in arch_list}
    if target in names:
        return "exact"
    if capability is None:
        return None
    cubins, ptx = [], []
    for entry in names:
        found = _CUDA_ARCH.fullmatch(entry)
        if found and len(found.group(2)) >= 2:
            version = (int(found.group(2)[:-1]), int(found.group(2)[-1]))
            (cubins if found.group(1) == "sm" else ptx).append(version)
    if any(major == capability[0] and minor <= capability[1] for major, minor in cubins):
        return "compatible"
    if any(version <= capability for version in ptx):
        return "ptx"
    return None


def device_info(index: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(index)
    target = gpu_target(props)
    arch_list = _safe(torch.cuda.get_arch_list, [])
    match = target_match(target, None if torch.version.hip else (props.major, props.minor),
                         arch_list)
    return {
        "index": index,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "target": target,
        # False means this PyTorch build has no kernels for this GPU: pick another wheel.
        "target_in_build": match is not None,
        # The rule that matched (exact, compatible or ptx: target_match), or None.
        "target_match": match,
        "total_memory_gb": round(props.total_memory / 2**30, 1),
        "pci_bus_id": getattr(props, "pci_bus_id", None),
        "pci_device_id": getattr(props, "pci_device_id", None),
        "uuid": str(getattr(props, "uuid", "")) or None,
    }


def environment() -> dict[str, Any]:
    """Hardware, versions and allowlisted env vars of this process, redacted."""
    cuda_ok = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda_ok else 0
    return scrub({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "gpubridge": gpubridge.__version__,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        # The imported package, which can differ from the scripts' checkout above.
        "import": _safe(kit_import),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "rocm_version": _read("/opt/rocm/.info/version") if torch.version.hip else None,
        # NCCL on CUDA builds, RCCL on ROCm builds; nccl_version is that library's.
        "collective_library": "rccl" if torch.version.hip else "nccl" if torch.version.cuda
        else None,
        "nccl_version": _safe(_nccl_version),
        "driver_version": _safe(driver_version),
        # Unset means the library picks.
        "interfaces": {name: os.environ.get(name) or "auto"
                       for name in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME")},
        "cuda_available": cuda_ok,
        "device_count": count,
        "arch_list": _safe(torch.cuda.get_arch_list, []) if cuda_ok else [],
        "devices": [_safe(lambda i=i: device_info(i)) for i in range(count)],
        "probe": asdict(gpubridge.probe()),
        "env": recorded_env(),
    })


def run_info(topology: gpubridge.Topology) -> dict[str, Any]:
    """How this run was set up. ``split_test`` and ``real_mixed_vendor`` are never ambiguous."""
    info: dict[str, Any] = {
        "kind": topology.run_kind,
        "split_test": topology.split_test,
        "real_mixed_vendor": topology.run_kind == "gpu-mixed",
        "world_size": topology.world_size,
        "rank": topology.rank,
        "vendor_label": topology.vendor,
        "detected_vendor": topology.detected_vendor,
        "device": str(topology.device),
        "islands": [
            {"vendor": island.vendor, "ranks": list(island.ranks), "leader": island.leader}
            for island in topology.layout.islands
        ],
        # The transport the run used; what it asked for (auto unless it named one) and
        # how init() chose are beside it.
        "bridge": topology.bridge_name,
        "bridge_requested": topology.peers[topology.rank].bridge,
        "bridge_selection": topology.selection,
        # bridge="ucx" on a leader: the path its last collective took (host, direct
        # or staged); scripts refresh it after their collectives. None otherwise.
        "bridge_path": getattr(topology.bridge, "last_path", None),
        # ... and, on the direct path, how the NIC's writes into the GPU were flushed
        # (or why its tensors were staged instead).
        "bridge_flush": getattr(topology.bridge, "flush_method", None),
        "policy": topology.policy_name,
        # GPUBRIDGE_PINNED_STAGING: the default bridge stages through pinned memory.
        "pinned_staging": topology.pinned_staging,
    }
    if topology.split_test:
        info["warning"] = (
            "SPLIT TEST: islands were faked from a single-vendor job with "
            "GPUBRIDGE_SPLIT_TEST. This is NOT a mixed-vendor result."
        )
    return info


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically, so a crash never leaves half a file; every string redacted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(scrub(json.loads(json.dumps(data, default=str))), indent=2))
    tmp.replace(path)


class Record:
    """One JSON record, rewritten after every stage so a crash or hang shows where it stopped."""

    def __init__(self, path: Path, script: str, *, hang_dump_seconds: float = 600) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stacks = open(path.with_suffix(".stacks.txt"), "w")  # noqa: SIM115
        self.hang_dump_seconds = hang_dump_seconds
        self.data: dict[str, Any] = {
            "kit_version": KIT_VERSION,
            "script": script,
            "argv": sys.argv,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": environment(),
            "run": None,
            "stages": [],
            "ok": None,
        }
        self.failed = False
        self.write()

    def write(self) -> None:
        write_json(self.path, self.data)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a stage. Set ``entry["ok"] = False`` for a failed check; exceptions propagate."""
        entry: dict[str, Any] = {"name": name, "ok": None}
        self.data["stages"].append(entry)
        self.write()  # ok=None marks the stage a rank is stuck in
        faulthandler.dump_traceback_later(self.hang_dump_seconds, file=self.stacks)
        start = time.monotonic()
        try:
            yield entry
        except BaseException:
            entry["ok"] = False
            entry["error"] = traceback.format_exc()
            raise
        else:
            if entry["ok"] is None:
                entry["ok"] = True
        finally:
            faulthandler.cancel_dump_traceback_later()
            entry["seconds"] = round(time.monotonic() - start, 4)
            self.failed = self.failed or not entry["ok"]
            self.write()

    def finish(self) -> int:
        """Set the overall result and return a process exit code."""
        self.data["ok"] = not self.failed
        self.data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.write()
        return 0 if self.data["ok"] else 1


def rank_from_env() -> int:
    return int(os.environ.get("RANK", "0"))


def sync(device: torch.device) -> None:
    """Wait for queued GPU work, so timings and checks see finished results."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def p2p_checks(topology: gpubridge.Topology, *, numels=(0, 1, 31, 4099, 1_000_003),
               busy_seconds: float = 0.02) -> list[dict[str, Any]]:
    """Backend P2P bit-exact checks, including producer/consumer streams on GPU.

    Used by check.py --p2p-only and test_gpu.py. CPU smoke tests run this same
    code; only real GPU records can validate the stream and allocator behavior.
    """
    rank, world, device = topology.rank, topology.world_size, topology.device
    if world < 2:
        raise ValueError("P2P checks need at least two ranks")
    src, dst = (rank - 1) % world, (rank + 1) % world
    producer = torch.cuda.Stream(device) if device.type == "cuda" else None
    consumer = torch.cuda.Stream(device) if device.type == "cuda" else None
    if producer is not None:
        producer.wait_stream(torch.cuda.current_stream(device))
    cases = []

    def context(stream):
        return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    def values(peer, numel, dtype):
        return ((torch.arange(numel) * 7 + peer * 3) % 17 - 8).to(dtype)

    for dtype in (torch.int8, torch.int64, torch.float16, torch.bfloat16,
                  torch.float32, torch.float64):
        for numel in numels:
            with context(producer):
                incoming = torch.full((numel,), -99, device=device, dtype=dtype)
                outgoing = torch.full_like(incoming, -99)
                source_device = torch.empty_like(incoming)
                source_device.copy_(values(rank, numel, dtype))
                # Upload pageable host input before the delay: an H2D copy after
                # it would synchronize the producer, hiding a missing stream wait.
                sync(device)
                if producer is not None:
                    queue_gpu_delay(device, busy_seconds)
                outgoing.copy_(source_device)
                works = gpubridge.batch_isend_irecv([
                    dist.P2POp(dist.irecv, incoming, src, tag=42),
                    dist.P2POp(dist.isend, outgoing, dst, tag=42)])
                del outgoing  # the Work/native allocator must keep in-flight storage alive
                # Reuse is stream-local. Allocating on the producer while the
                # send is pending can overwrite an unprotected outgoing block.
                scribble = torch.full_like(incoming, 77)
            with context(consumer):
                for work in works:
                    work.wait()
                observed = incoming.clone().cpu()  # read is queued directly after wait
            del scribble  # keep allocator pressure alive until both Works completed
            ok = torch.equal(observed, values(src, numel, dtype))
            cases.append({"dtype": str(dtype), "numel": numel, "ok": ok})
            if not ok:
                raise AssertionError(f"P2P stream/lifetime check: rank {rank}, {dtype}, n={numel}")
    # Different tags must not block each other. Only cross-island pairs take
    # this test, because native NCCL/RCCL ignore tags.
    for source in range(world):
        for dest in range(source + 1, world):
            if topology.layout.island_of(source) == topology.layout.island_of(dest):
                continue
            if rank == source:
                a = torch.full((31,), 11.0, device=device)
                b = torch.full((31,), 22.0, device=device)
                works = [dist.isend(a, dest, tag=101), dist.isend(b, dest, tag=102)]
                for work in works:
                    work.wait()
            elif rank == dest:
                b = torch.empty(31, device=device)  # float32, as sent
                a = torch.empty_like(b)
                dist.recv(b, source, tag=102)
                dist.recv(a, source, tag=101)
                if not bool((a == 11).all() and (b == 22).all()):
                    raise AssertionError("P2P reverse-tag matching failed")
    dist.barrier()
    return cases


# ---- GPU work without BLAS --------------------------------------------------------
#
# Stream-ordering checks queue GPU work and write their input after it, so that a
# copy that didn't wait would read stale data. That work used to be matmuls, but the
# ROCm wheels' rocBLAS has no kernels for some older GPUs (gfx906: "Cannot read
# .../TensileLibrary.dat"), and gpubridge itself never needs BLAS.

#: Cycles per second of torch.cuda._sleep, per device index; 0 where it can't run.
_SPIN_RATE: dict[int, float] = {}
#: Seconds per elementwise pass over _DELAY_NUMEL float32 values, per device index.
_PASS_SECONDS: dict[int, float] = {}
_DELAY_NUMEL = 1 << 24  # 64 MB: each pass is GPU work, not launch overhead


def queue_gpu_delay(device: torch.device, seconds: float) -> None:
    """Queue about ``seconds`` of GPU work on the current stream; the host doesn't wait.

    Anything queued after it on that stream waits for it. No BLAS: torch's spin kernel
    (``torch.cuda._sleep``), calibrated once per device, or, where that is missing or
    fails, a calibrated chain of elementwise multiplies. Nothing on CPU.
    """
    if device.type != "cuda" or seconds <= 0:
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SPIN_RATE:
        _SPIN_RATE[index] = _spin_rate(index)
    if _SPIN_RATE[index] > 0:
        with torch.cuda.device(index):
            torch.cuda._sleep(max(1, int(_SPIN_RATE[index] * seconds)))
        return
    if index not in _PASS_SECONDS:
        _PASS_SECONDS[index] = _pass_seconds(index)
    x = torch.ones(_DELAY_NUMEL, device=device)
    for _ in range(max(1, round(seconds / _PASS_SECONDS[index]))):
        x.mul_(1.0)


def _timed(index: int, fn: Callable[[], None]) -> float:
    """Seconds ``fn``'s GPU work takes, timed with events on a stream of its own."""
    with torch.cuda.device(index), torch.cuda.stream(torch.cuda.Stream(index)):
        fn()  # warms up: the first launch loads the kernel
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / 1e3


def _spin_rate(index: int) -> float:
    sleep = getattr(torch.cuda, "_sleep", None)
    if sleep is None:
        return 0.0
    cycles = 20_000_000
    try:
        seconds = _timed(index, lambda: sleep(cycles))
    except RuntimeError:
        return 0.0
    # A spin that took no time doesn't spin: fall back to elementwise work. (Real GPU
    # clocks are 10 MHz to a few GHz: 20M cycles take 2 ms to 2 s.)
    return cycles / seconds if 1e-4 < seconds < 10 else 0.0


def _pass_seconds(index: int) -> float:
    x = torch.ones(_DELAY_NUMEL, device=torch.device("cuda", index))
    passes = 10

    def run() -> None:
        for _ in range(passes):
            x.mul_(1.0)

    return max(_timed(index, run) / passes, 1e-6)


def blas_check(index: int = 0, timeout: float = 120) -> dict[str, Any]:
    """Whether a tiny matmul runs on GPU ``index``, and why not. In a child process:
    rocBLAS without kernels for the GPU aborts its process ("rocBLAS error: Cannot read
    .../TensileLibrary.dat", exit -6) rather than raise. gpubridge never needs BLAS;
    the record says so for whoever reads why a matmul elsewhere failed."""
    code = ("import torch; a = torch.ones(4, 4, device='cuda:%d'); "
            "print(float((a @ a).sum()))" % index)
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"blas_available": False, "error": f"a 4x4 matmul ran past {timeout:g} s"}
    if result.returncode == 0 and result.stdout.strip() == "64.0":
        return {"blas_available": True}
    lines = [line for line in (result.stderr + result.stdout).splitlines() if line.strip()]
    # The BLAS error itself, not a warning printed before it (hipBLASLt warns first).
    error = next((line for line in lines if "error" in line.lower() and "blas" in line.lower()),
                 next((line for line in lines if "error" in line.lower()),
                      lines[-1] if lines else "no output"))
    return {"blas_available": False, "exit_code": result.returncode, "error": error[:1000]}
