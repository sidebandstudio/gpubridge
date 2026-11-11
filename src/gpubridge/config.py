"""Runtime configuration: simulation mode, backend and device selection."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Literal

import torch

Vendor = Literal["nvidia", "amd"]
VENDORS: tuple[Vendor, ...] = ("nvidia", "amd")

#: Set to ``nvidia`` or ``amd`` to fake that vendor on CPU (simulation mode).
VENDOR_ENV = "GPUBRIDGE_VENDOR"
#: Set to ``1`` to run on CPU with Gloo islands while still detecting the vendor
#: from the PyTorch build, e.g. to test real CUDA and ROCm builds without GPUs.
CPU_ONLY_ENV = "GPUBRIDGE_CPU_ONLY"
#: TEST ONLY. Set to ``half``, ``alternate`` or ``node`` (``1`` means ``half``) to
#: label some ranks of a single-vendor job as the other vendor, so one cluster
#: can exercise two islands and the bridge. See :mod:`gpubridge.split_test`.
SPLIT_TEST_ENV = "GPUBRIDGE_SPLIT_TEST"
SPLIT_TEST_MODES = ("half", "alternate", "node")
#: Simulation and CPU-only mode only: the machine name this rank reports, so
#: processes on one machine can stand in for several nodes (``node`` split-test
#: mode, the multi-node job scripts tested on CPU). Ignored on GPUs.
SIM_HOSTNAME_ENV = "GPUBRIDGE_SIM_HOSTNAME"
#: Chunk size of the pipelined bridge, in bytes (``4M``, ``65536`` ...). Must be
#: the same on every rank: it decides how many bridge calls a collective makes.
CHUNK_BYTES_ENV = "GPUBRIDGE_CHUNK_BYTES"
#: Path of a thresholds file for the ``auto-tuned`` policy, as written by
#: ``scripts/gpu/bench_all_reduce.py --write-thresholds``.
THRESHOLDS_ENV = "GPUBRIDGE_THRESHOLDS"
#: Set to ``1`` to stage the default bridge's copies (reduce-bridge-broadcast)
#: through one pinned host buffer per leader instead of pageable ``.cpu()``
#: copies. Off by default; must match on every rank. Not validated on GPUs.
PINNED_STAGING_ENV = "GPUBRIDGE_PINNED_STAGING"
#: ``bf16`` or ``fp8`` to compress the bridge step of floating-point SUM and AVG
#: reductions (:mod:`gpubridge.compression`). Off by default; must match on every
#: rank. Changes results: not bit-exact. Not validated on GPUs.
BRIDGE_COMPRESSION_ENV = "GPUBRIDGE_BRIDGE_COMPRESSION"
BRIDGE_COMPRESSION_MODES = ("bf16", "fp8")

#: The collective policy under the torch.distributed backend
#: (``init_process_group(backend="gpubridge")``), which ``gpubridge.init()`` takes as an
#: argument. Default ``auto``; must match on every rank.
POLICY_ENV = "GPUBRIDGE_POLICY"
#: The bridge transport: under the torch.distributed backend, and for
#: ``gpubridge.init()`` when it isn't passed ``bridge=``. Default ``auto`` (``verbs``
#: where every island leader has a usable RDMA device, else ``tcp``); ``gloo`` asks for
#: the Gloo bridge, and with it Gloo groups. Must match on every rank.
BRIDGE_ENV = "GPUBRIDGE_BRIDGE"

#: torch's CPU backend: ``bridge="gloo"``'s groups, the ``flat-gloo`` policy's, and
#: islands of several ranks in CPU simulation. No other job creates a group on it.
CPU_BACKEND = "gloo"
#: Backend of islands on real GPUs. On ROCm builds of PyTorch, "nccl" runs RCCL.
GPU_BACKEND = "nccl"
#: Backend of a one-rank island in CPU simulation on a bridge other than Gloo: a Python
#: group with nothing behind it (:class:`gpubridge.torch_backend.SoloGroup`).
SOLO_BACKEND = "gpubridge_solo"


def _flag(name: str) -> bool:
    """Whether environment variable ``name`` is set to a true value.

    Raises:
        ValueError: if it holds something other than a boolean word.
    """
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name}={os.environ[name]!r} is not a boolean; use 1 or 0")


def is_cpu_only() -> bool:
    """Return True when ``GPUBRIDGE_CPU_ONLY`` is set to a true value.

    Raises:
        ValueError: if the variable holds something other than a boolean word.
    """
    return _flag(CPU_ONLY_ENV)


def pinned_staging_setting() -> bool:
    """Return True when ``GPUBRIDGE_PINNED_STAGING`` is set to a true value.

    Raises:
        ValueError: if the variable holds something other than a boolean word.
    """
    return _flag(PINNED_STAGING_ENV)


def bridge_compression_setting() -> str:
    """``GPUBRIDGE_BRIDGE_COMPRESSION``: ``bf16``, ``fp8``, or empty when off.

    Raises:
        ValueError: for any other value.
    """
    value = os.environ.get(BRIDGE_COMPRESSION_ENV, "").strip().lower()
    if value in ("", "0", "false", "no", "off", "none"):
        return ""
    if value in BRIDGE_COMPRESSION_MODES:
        return value
    raise ValueError(
        f"{BRIDGE_COMPRESSION_ENV}={os.environ[BRIDGE_COMPRESSION_ENV]!r} is not valid; use "
        f"{' or '.join(BRIDGE_COMPRESSION_MODES)}, or leave it unset (off)"
    )


def split_test_mode() -> str | None:
    """Return the ``GPUBRIDGE_SPLIT_TEST`` mode, or None when split-test mode is off.

    This environment variable is the only way to turn split-test mode on.

    Raises:
        ValueError: for a value that is neither off, ``1`` nor a known mode.
    """
    value = os.environ.get(SPLIT_TEST_ENV, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return None
    if value in ("1", "true", "yes", "on"):
        return "half"
    if value in SPLIT_TEST_MODES:
        return value
    raise ValueError(
        f"{SPLIT_TEST_ENV}={os.environ[SPLIT_TEST_ENV]!r} is not valid; use one of "
        f"{', '.join(SPLIT_TEST_MODES)}, 1 (half) or 0 (off)"
    )


_UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}


def parse_size(text: str) -> int:
    """Parse ``"4M"``, ``"64K"``, ``"1G"`` or plain bytes (``"65536"``) into bytes.

    Raises:
        ValueError: for anything else.
    """
    value = text.strip().upper().removesuffix("B")
    try:
        if value and value[-1] in _UNITS:
            return int(float(value[:-1]) * _UNITS[value[-1]])
        return int(value)
    except ValueError:
        raise ValueError(f"{text!r} is not a size; use bytes or a K/M/G suffix, e.g. 4M") from None


def chunk_bytes_setting(default: int) -> int:
    """The pipelined bridge's chunk size: ``GPUBRIDGE_CHUNK_BYTES``, or ``default``.

    Raises:
        ValueError: unless it is a positive multiple of 8 bytes (so a host slot
            can be viewed as any supported dtype).
    """
    raw = os.environ.get(CHUNK_BYTES_ENV, "").strip()
    if not raw:
        return default
    try:
        chunk = parse_size(raw)
    except ValueError as exc:
        raise ValueError(f"{CHUNK_BYTES_ENV}: {exc}") from None
    if chunk < 8 or chunk % 8:
        raise ValueError(f"{CHUNK_BYTES_ENV}={raw!r} must be a positive multiple of 8 bytes")
    return chunk


def is_simulated() -> bool:
    """Return True when gpubridge runs on CPU with Gloo islands instead of on GPUs.

    That is the case when ``GPUBRIDGE_VENDOR`` fakes a vendor (simulation mode)
    or when ``GPUBRIDGE_CPU_ONLY`` is set (CPU-only mode, vendor from the build).
    """
    return bool(os.environ.get(VENDOR_ENV, "").strip()) or is_cpu_only()


def hostname(simulated: bool) -> str:
    """This rank's machine name for discovery: ``GPUBRIDGE_SIM_HOSTNAME`` if
    simulated and set, otherwise ``socket.gethostname()``."""
    if simulated:
        name = os.environ.get(SIM_HOSTNAME_ENV, "").strip()
        if name:
            return name
    return socket.gethostname()


@dataclass(frozen=True)
class Config:
    """How this process communicates, resolved once by :func:`gpubridge.init`."""

    simulated: bool
    """True on CPU, in simulation mode or CPU-only mode: islands are then Gloo groups
    (one-rank islands on a bridge other than Gloo: no backend at all)."""
    island_backend: str
    device: torch.device


def resolve_config() -> Config:
    """Pick the island backend and the device this rank communicates on.

    In simulation and CPU-only mode islands use Gloo and tensors stay on CPU. Otherwise
    islands use the native backend on ``cuda:$LOCAL_RANK``, or on the current
    CUDA device when ``LOCAL_RANK`` is unset. ROCm builds expose AMD GPUs
    through the same ``torch.cuda`` API. In split-test mode only, a local rank
    beyond the visible GPU count wraps around, so ranks can share a GPU.

    The device index is checked here, not left to ``torch.cuda.set_device``:
    discovery turns this error into a problem that fails ``init()`` on every
    rank, whereas a bad ``set_device`` would crash this rank alone and leave
    the others waiting in rendezvous.

    Raises:
        RuntimeError: if not simulating and no GPU is visible to this process, or
            ``LOCAL_RANK`` is not an integer or names a GPU this process can't see.
    """
    if is_simulated():
        return Config(simulated=True, island_backend=CPU_BACKEND, device=torch.device("cpu"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is visible to this process. To develop without GPUs, set "
            f"{VENDOR_ENV}=nvidia or {VENDOR_ENV}=amd to run in CPU simulation mode, "
            f"or {CPU_ONLY_ENV}=1 to run on CPU with the vendor of this PyTorch build."
        )
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        index = torch.cuda.current_device()
    else:
        try:
            index = int(local_rank)
        except ValueError:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank!r} is not an integer. It is normally set by the "
                "launcher (torchrun or srun); don't set it by hand."
            ) from None
    count = torch.cuda.device_count()
    if split_test_mode() is not None and count and index >= count:
        # Test only: lets two islands share GPUs, e.g. two ranks on one GPU.
        index %= count
    if not 0 <= index < count:
        raise RuntimeError(_no_such_gpu(local_rank, index, count))
    return Config(simulated=False, island_backend=GPU_BACKEND, device=torch.device("cuda", index))


def gpu_identity(device: torch.device) -> str | None:
    """The physical GPU behind ``device``, the same in every process that drives it.

    A device index is not: with ``CUDA_VISIBLE_DEVICES`` or Slurm's per-task
    GPU binding, ``cuda:0`` can be a different GPU in every process. The UUID
    and the PCI address are. None for CPU devices. Empty when the GPU reports
    neither (an all-zero value counts as unknown), as on some ROCm builds; init()
    then warns that it can't check that island for shared GPUs.
    """
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:  # noqa: BLE001 - only a check: never stop init() here
        return ""
    parts = []
    uuid = str(getattr(props, "uuid", "") or "")
    if uuid.strip("0-"):
        parts.append(f"uuid {uuid}")
    pci = [getattr(props, f"pci_{name}_id", None) for name in ("domain", "bus", "device")]
    if None not in pci and any(pci):
        parts.append("pci {:04x}:{:02x}:{:02x}".format(*pci))
    return ", ".join(parts)


def _no_such_gpu(local_rank: str | None, index: int, count: int) -> str:
    asked = f"LOCAL_RANK={local_rank}" if local_rank is not None else "The current CUDA device"
    gpus = f"{count} GPU{'s' if count != 1 else ''}"
    return (
        f"{asked} asks for GPU {index}, but this process sees only {gpus}. Each rank on a "
        "node needs its own visible GPU: start no more processes per node than it has GPUs "
        "(torchrun --nproc-per-node), and check what the job makes visible "
        "(CUDA_VISIBLE_DEVICES on NVIDIA, HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES on AMD). "
        "Under Slurm, --gpus-per-task=1 leaves each task a single GPU (index 0), so start "
        "one process per task, or give the task all of the node's GPUs (e.g. --gpus-per-node)."
    )
