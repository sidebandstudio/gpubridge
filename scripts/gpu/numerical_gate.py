"""Small deterministic numerical gate; GPU acceptance requires all three real arms.

Run under four-rank torchrun. --software-only uses CPU/Gloo and labels every
record accordingly. See docs/NUMERICAL_GATE.md for frozen bounds and owner handoff.
No performance measurements are made here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpubridge

ROOT = Path(__file__).resolve().parents[2]
# FP32 roundoff in a tiny dot product plus four-rank reduction, consistent with
# tests/test_train_bench.py's existing float bounds. Freeze before hardware.
RTOL, ATOL = 1e-5, 1e-6
UPDATE_RTOL, UPDATE_ATOL = 1e-4, 2e-8
STEPS, BATCH, SEED = 3, 32, 20261020
SIZES = (17, 65535, 65537, 1048577)  # below/above 256KiB rendezvous and 4MiB chunks
ARMS = ("native", "gloo", "verbs")
CONTROLS = ("none", "sum", "scale", "data")


def fingerprint(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        t = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(t.shape), str(t.dtype)]).encode())
        digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def values(tensors) -> list[float]:
    return torch.cat([t.detach().cpu().reshape(-1).double() for t in tensors]).tolist()


def metric(actual, expected, *, exact=False, update=False) -> dict:
    a = torch.as_tensor(actual, dtype=torch.float64).detach().cpu().reshape(-1)
    b = torch.as_tensor(expected, dtype=torch.float64).detach().cpu().reshape(-1)
    if a.shape != b.shape:
        return {"violations": max(a.numel(), b.numel()), "shape_mismatch": True}
    delta = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    atol, rtol = (UPDATE_ATOL, UPDATE_RTOL) if update else (ATOL, RTOL)
    bad = (~finite) | (delta > (0 if exact else atol + rtol * b.abs()))
    return {"violations": int(bad.sum()), "elements": a.numel(),
            "max_abs": float(delta.max()) if bool(finite.all()) else None,
            "relative_l2": float(torch.linalg.vector_norm(a - b) /
                                 torch.linalg.vector_norm(b).clamp_min(1e-30))
            if bool(finite.all()) else None}


def model() -> nn.Module:
    # CPU construction; no dropout, running buffers, autocast or pretraining.
    torch.manual_seed(SEED)
    m = nn.Sequential(nn.Linear(6, 5, bias=False), nn.Tanh(), nn.Linear(5, 3, bias=False))
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randint(-3, 4, p.shape).float() / 16)
    return m


def batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    # The same global batch on every arm/rank, independent of device RNG.
    generator = torch.Generator().manual_seed(SEED + 100 + step)
    return (torch.randint(-8, 9, (BATCH, 6), generator=generator).float() / 8,
            torch.randint(-8, 9, (BATCH, 3), generator=generator).float() / 8)


def optimizer(parameters, name):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=1 / 32, momentum=0.5, foreach=False)
    return torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.01, eps=1e-8,
                             foreach=False, fused=False)


def optimizer_values(opt):
    # Include momenta, moments and step counters in parameter order.
    return [state[key] for group in opt.param_groups for p in group["params"]
            for state in [opt.state[p]] for key in sorted(state)
            if isinstance(state[key], torch.Tensor)]


def provider_errors(before, after) -> list[str]:
    errors = []
    if after.get("mock") is not False or "libibverbs (/" not in after.get("provider", ""):
        errors.append("actual endpoint must name a non-mock libibverbs provider and path")
    if after.get("state") != "ok" or after.get("failure") or after.get("failed"):
        errors.append("actual endpoint failed")
    if after.get("completion_errors") != 0:
        errors.append("completion errors or missing completion-error evidence")
    for name in ("completions", "bytes_sent", "bytes_received"):
        if after.get(name, 0) <= before.get(name, 0):
            errors.append(f"no positive endpoint {name} delta")
    if (after.get("gpu_path_setting") != "staged"
            or after.get("paths", {}).get("staged", 0) <= before.get("paths", {}).get("staged", 0)):
        errors.append("no actual staged GPU calls")
    if after.get("rndv_bytes") != 256 * 1024 or after.get("chunk_bytes") != 4 * 1024**2:
        errors.append("protocol thresholds differ from the frozen probe sizes")
    return errors


def source_record() -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    package = ROOT / "src/gpubridge"
    paths = [Path(__file__), *sorted(package.rglob("*.py"))]
    imported = [Path(m.__file__).resolve() for name, m in list(sys.modules.items())
                if (name == "gpubridge" or name.startswith("gpubridge."))
                and getattr(m, "__file__", None)]
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(git("status", "--porcelain")),
            "imported_package": str(Path(gpubridge.__file__).resolve()),
            "imports_match": bool(imported) and all(p.is_relative_to(package) for p in imported),
            "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths}}


def run_gate(topology, *, arm, software_only, opt_name="sgd", control="none",
             sizes=SIZES, run_id="software-test") -> dict:
    """All ranks call this, without early exit on a numerical violation."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError("the frozen gate requires four ranks")
    device = topology.device if topology is not None else torch.device(
        "cpu" if software_only else "cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {"schema": "gpubridge.numerical_gate/1", "arm": arm, "rank": rank,
              "world": world, "software_only": software_only, "control": control,
              "run_id": run_id, "optimizer": opt_name, "seed": SEED, "steps": STEPS, "global_batch": BATCH,
              "shares": [BATCH // world] * world, "precision": "float32", "compression": "",
              "probe_sizes": list(sizes), "backend": dist.get_backend(),
              "rtol": RTOL, "atol": ATOL, "device": str(device), "host": socket.gethostname(),
              "update_rtol": UPDATE_RTOL, "update_atol": UPDATE_ATOL,
              "deterministic": True, "tf32": False,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "source": source_record(), "checks": [], "errors": [], "inputs": [],
              "snapshots": [], "created_at": datetime.now(timezone.utc).isoformat()}
    if not software_only and (device.type != "cuda" or report["source"]["dirty"]):
        raise ValueError("hardware gate needs real GPUs and a clean committed source checkout")
    if not report["source"]["imports_match"]:
        raise ValueError("gpubridge imports must resolve to this checkout's src tree")
    if device.type == "cuda":
        from gpubridge.config import gpu_identity

        report["gpu_identity"] = gpu_identity(device)
        report["nccl"] = list(torch.cuda.nccl.version())
    if topology is not None:
        layout = [list(i.ranks) for i in topology.layout.islands]
        report.update(layout=layout, policy=topology.policy_name, run_kind=topology.run_kind,
                      island_backend=dist.get_backend(topology.island_group),
                      transport=topology.bridge_name)
        if layout != [[0, 1], [2, 3]] or topology.bridge_compression:
            raise ValueError("bridge gate requires two equal islands and compression OFF")
    else:
        report.update(layout=[[0, 1, 2, 3]], policy="native", run_kind="native",
                      island_backend=dist.get_backend(), transport="native")
    bridge = topology.bridge if topology is not None else None
    before = bridge.telemetry() if arm == "verbs" and bridge is not None else None
    if arm == "verbs" and rank in (0, 2) and before is None:
        report["errors"].append("expected leader has no actual verbs endpoint")
    if bridge is not None and bridge.name != arm:
        report["errors"].append("actual bridge transport differs from requested arm")

    def check(label, actual, expected, exact=False):
        result = metric(actual, expected, exact=exact, update=label.startswith("update/"))
        report["checks"].append({"name": label, **result})

    # Exact integer-valued probes. Queue AVG calls together before waiting, then
    # verify every element, not a sum or checksum that could hide permutations.
    pending = []
    for n in sizes:
        base = torch.arange(n, dtype=torch.float32)
        expected = world * base + world * (world + 1) / 2
        for average in (False, True):
            tensor = (base + rank + 1).to(device)
            if topology is None:
                work = dist.all_reduce(tensor, async_op=average)
            else:
                work = gpubridge.all_reduce(
                    tensor, op=dist.ReduceOp.AVG if average else dist.ReduceOp.SUM,
                    async_op=average)
            if average:
                pending.append((n, tensor, expected / world, work))
            else:
                check(f"sum/{n}", tensor, expected, exact=True)
    for n, tensor, expected, work in pending:
        work.wait()
        if topology is None:
            tensor.div_(world)
        if control == "sum":  # Deliberate SUM-instead-of-AVG negative control.
            tensor.mul_(world)
        check(f"avg/{n}", tensor, expected, exact=True)

    initial = model()
    initial_hash = fingerprint(initial.named_parameters())
    reference = copy.deepcopy(initial).double()
    candidate = initial.to(device)
    ddp = DDP(candidate, device_ids=[device.index] if device.type == "cuda" else None,
              broadcast_buffers=False, bucket_cap_mb=0.00005)
    if topology is not None:
        ddp.register_comm_hook(None, gpubridge.ddp_comm_hook)
    report["initial_sha256"] = initial_hash
    if fingerprint(candidate.named_parameters()) != initial_hash:
        report["errors"].append("DDP initial state differs from frozen CPU initialization")
    opt, oracle_opt = optimizer(ddp.parameters(), opt_name), optimizer(reference.parameters(), opt_name)
    for step in range(STEPS):
        x, y = batch(step)
        rows = slice(rank * (BATCH // world), (rank + 1) * (BATCH // world))
        lx, ly = x[rows].clone(), y[rows].clone()
        want_hash = fingerprint([("x", lx), ("y", ly)])
        if control == "data" and rank == 0:
            lx.add_(1 / 8)
        got_hash = fingerprint([("x", lx), ("y", ly)])
        report["inputs"].append({"step": step, "expected": want_hash, "actual": got_hash})
        if got_hash != want_hash:
            report["errors"].append(f"input fingerprint mismatch at step {step}")
        opt.zero_grad(set_to_none=True)
        oracle_opt.zero_grad(set_to_none=True)
        loss = (ddp(lx.to(device)) * ly.to(device)).sum(dim=1).mean()
        (loss * (2 if control == "scale" else 1)).backward()
        oracle_loss = (reference(x.double()) * y.double()).sum(dim=1).mean()
        oracle_loss.backward()
        params, twins = list(candidate.parameters()), list(reference.parameters())
        gradients, oracle_gradients = values(p.grad for p in params), values(p.grad for p in twins)
        check(f"gradient/{step}", gradients, oracle_gradients)
        old, old_ref = [p.detach().clone() for p in params], [p.detach().clone() for p in twins]
        opt.step()
        oracle_opt.step()
        updates = values(p - q for p, q in zip(params, old, strict=True))
        ref_updates = values(p - q for p, q in zip(twins, old_ref, strict=True))
        check(f"update/{step}", updates, ref_updates)
        check(f"parameters/{step}", values(params), values(twins))
        states = values(optimizer_values(opt))
        check(f"optimizer_state/{step}", states, values(optimizer_values(oracle_opt)))
        report["snapshots"].append({"step": step, "gradients": gradients, "updates": updates,
                                    "parameters": values(params), "optimizer_state": states})
    report["final_sha256"] = fingerprint(candidate.named_parameters())
    if before is not None:
        after = bridge.telemetry()
        report.update(telemetry_before=before, telemetry_after=after)
        from gpubridge import verbs

        library = Path(verbs.load()._name).resolve()
        report["native_library"] = {"path": str(library),
                                    "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
        if not software_only:
            report["errors"].extend(provider_errors(before, after))
    report["passed"] = not report["errors"] and not any(c["violations"] for c in report["checks"])
    del ddp, candidate, opt, loss
    gc.collect()
    return report


def record_errors(record: dict) -> list[str]:
    """Recheck acceptance evidence; a cached passed=true is insufficient."""
    r, errors = record, []
    expected = {f"{op}/{n}" for op in ("sum", "avg") for n in r["probe_sizes"]}
    expected |= {f"{kind}/{s}" for kind in ("gradient", "update", "parameters", "optimizer_state")
                 for s in range(STEPS)}
    checks = r["checks"]
    if (len(checks) != len(expected) or {c["name"] for c in checks} != expected
            or any(c.get("violations") != 0 or c.get("elements", 0) <= 0 for c in checks)):
        errors.append("missing, duplicate or failing numerical checks")
    if r["errors"] or not r["passed"] or r["control"] != "none":
        errors.append("arm gate failed or negative control")
    if not r["source"]["imports_match"]:
        errors.append("imported source differs")
    frozen = {"seed": SEED, "steps": STEPS, "global_batch": BATCH, "shares": [8] * 4,
              "precision": "float32", "compression": "", "rtol": RTOL, "atol": ATOL,
              "update_rtol": UPDATE_RTOL, "update_atol": UPDATE_ATOL,
              "deterministic": True, "tf32": False}
    if any(r[k] != v for k, v in frozen.items()) or not r["run_id"]:
        errors.append("configuration differs from frozen gate")
    native = r["arm"] == "native"
    if (r["world"] != 4 or r["layout"] != ([[0, 1, 2, 3]] if native else [[0, 1], [2, 3]])
            or r["transport"] != r["arm"]
            or r["policy"] != ("native" if native else "reduce-bridge-broadcast")):
        errors.append("incorrect world, layout, transport or policy")
    backend = "gloo" if r["software_only"] else "nccl"
    if r["island_backend"] != backend or r["backend"] != (backend if native else "gloo"):
        errors.append("incorrect actual process-group backend")
    if (len(r["inputs"]) != STEPS or any(x["actual"] != x["expected"] for x in r["inputs"])):
        errors.append("incomplete or mismatched input fingerprints")
    if not r["software_only"]:
        if (r["source"]["dirty"] or not r.get("gpu_identity") or r["device"].split(":")[0] != "cuda"
                or r["probe_sizes"] != list(SIZES)):
            errors.append("hardware provenance incomplete")
        if r["arm"] == "verbs" and r["rank"] in (0, 2):
            if not all(k in r for k in ("telemetry_before", "telemetry_after", "native_library")):
                errors.append("missing actual verbs endpoint/binary evidence")
            else:
                errors.extend(provider_errors(r["telemetry_before"], r["telemetry_after"]))
                if not r["native_library"].get("path") or len(r["native_library"].get("sha256", "")) != 64:
                    errors.append("missing native library hash/path")
    return errors


def compare(records: list[dict]) -> dict:
    """Fail closed on absent arms/ranks, stale identities or changed inputs/settings."""
    errors, checks = [], []
    by = {(r["arm"], r["rank"]): r for r in records}
    if len(by) != 12 or len(records) != 12 or set(by) != {(a, r) for a in ARMS for r in range(4)}:
        return {"passed": False, "errors": ["need exactly three arms times four ranks"], "checks": []}
    identity = ("seed", "steps", "global_batch", "shares", "precision", "compression", "optimizer",
                "software_only", "rtol", "atol", "update_rtol", "update_atol", "torch", "cuda",
                "run_id", "deterministic", "tf32", "probe_sizes")
    anchor = by["native", 0]
    for (arm, rank), r in by.items():
        native = by["native", rank]
        errors.extend(f"{arm}/{rank}: {e}" for e in record_errors(r))
        if any(r[k] != anchor[k] for k in identity) or r["source"] != anchor["source"]:
            errors.append(f"{arm}/{rank}: configuration/source differs")
        if r["initial_sha256"] != anchor["initial_sha256"] or r["inputs"] != native["inputs"]:
            errors.append(f"{arm}/{rank}: initial state or per-rank data differs")
        if not r["software_only"] and (r.get("gpu_identity") != native.get("gpu_identity")
                                      or r["host"] != native["host"]):
            errors.append(f"{arm}/{rank}: physical rank mapping differs")
        if len(r["snapshots"]) != STEPS:
            errors.append(f"{arm}/{rank}: incomplete snapshots")
            continue
        for step, snapshot in enumerate(r["snapshots"]):
            # Compare across arms AND ranks; the FP64 oracle checks remain separate.
            for other in (native, anchor):
                for key in ("gradients", "updates", "parameters", "optimizer_state"):
                    result = metric(snapshot[key], other["snapshots"][step][key], update=key == "updates")
                    checks.append({"arm": arm, "rank": rank, "step": step, "tensor": key, **result})
    if not anchor["software_only"]:
        binaries = [by["verbs", r].get("native_library", {}).get("sha256") for r in (0, 2)]
        if not all(binaries) or len(set(binaries)) != 1:
            errors.append("verbs leaders used missing or different native binary hashes")
        if len({by["native", r].get("gpu_identity") for r in range(4)}) != 4:
            errors.append("four distinct physical GPUs were not recorded")
    return {"passed": not errors and not any(c["violations"] for c in checks),
            "software_only": anchor["software_only"], "errors": errors, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--negative-control", choices=CONTROLS, default="none")
    parser.add_argument("--software-only", action="store_true")
    parser.add_argument("--run-id", help="fresh comparison identifier shared by the three arms")
    parser.add_argument("--compare", nargs="+", type=Path, help="twelve rank JSON files")
    args = parser.parse_args(argv)
    if args.compare:
        result = compare([json.loads(p.read_text()) for p in args.compare])
        with args.out.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return 0 if result["passed"] else 1
    if not args.arm:
        parser.error("--arm is required without --compare")
    if not args.run_id:
        parser.error("--run-id is required for a new run")
    if args.software_only:
        os.environ["GPUBRIDGE_CPU_ONLY"] = "1"
        os.environ["GPUBRIDGE_VENDOR"] = "nvidia"
        if args.arm == "verbs" and not os.environ.get("GPUBRIDGE_VERBS_TEST_FABRIC"):
            parser.error("software verbs requires an explicit mock test fabric")
    else:
        forbidden = ("GPUBRIDGE_CPU_ONLY", "GPUBRIDGE_VENDOR", "GPUBRIDGE_VERBS_TEST_FABRIC")
        if any(os.environ.get(k) for k in forbidden) or not torch.cuda.is_available():
            parser.error("hardware mode forbids simulation, mock fabrics and missing CUDA")
        if args.arm == "verbs" and not Path(os.environ.get("GPUBRIDGE_VERBS_LIB", "")).is_file():
            parser.error("hardware verbs requires an explicit existing GPUBRIDGE_VERBS_LIB")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["GPUBRIDGE_BRIDGE_COMPRESSION"] = ""
    os.environ["GPUBRIDGE_SPLIT_TEST"] = "half" if args.arm != "native" else ""
    if args.arm == "verbs":
        os.environ["GPUBRIDGE_VERBS_GPU_PATH"] = "staged"
    topology = None
    if args.arm == "native":
        if not args.software_only:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo" if args.software_only else "nccl", timeout=timedelta(seconds=120))
    else:
        topology = gpubridge.init(bridge=args.arm, policy="reduce-bridge-broadcast",
                                  timeout=timedelta(seconds=120))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        result = run_gate(topology, arm=args.arm, software_only=args.software_only,
                          opt_name=args.optimizer, control=args.negative_control, run_id=args.run_id)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / f"rank{result['rank']}.json").open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        status = torch.tensor(int(result["passed"]))
        dist.all_reduce(status, group=control_group)
        return 0 if status.item() == 4 else 1
    finally:
        dist.destroy_process_group(control_group)
        if topology is not None:
            gpubridge.destroy()
        else:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
