"""Data-parallel training on synthetic data, with gradients synced by gpubridge.

A small MLP learns a fixed random classification task. Each rank trains on its
own batches. The model is wrapped in DistributedDataParallel on the job's
default group (gpubridge's own), with gpubridge.ddp_comm_hook registered: DDP buckets the
gradients and overlaps their sync with backward, and each bucket is averaged
with an async gpubridge.all_reduce. So the same script trains on NVIDIA, AMD
or both at once. It reports throughput (samples/sec across all ranks) and
checks that every rank ends with identical parameters.

Between steps, outside the timed part, it also compares every rank's gradients
(as DDP left them, after the all_reduce) and parameters (after the optimizer
step), bit for bit. The record says the first step at which each differed, and
by how much: gradients that differ point at the sync (the bridge, compression),
gradients that match with parameters that differ point at the optimizer step.

On CPU, in simulation mode:

    GPUBRIDGE_VENDOR=nvidia torchrun --nproc-per-node=2 examples/train_synthetic.py

On GPUs, the same command without GPUBRIDGE_VENDOR. scripts/gpu uses it for
the scaling run (1 to 4 GPUs) with --json to record the result, along with
every rank's versions, GPU and settings as the kit records them.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import gpubridge


def make_model(dim: int, layers: int, classes: int) -> nn.Module:
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks += [nn.Linear(dim, dim), nn.ReLU()]
    return nn.Sequential(*blocks, nn.Linear(dim, classes))


def wait(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_BITS = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def fingerprint(tensor: torch.Tensor) -> tuple[int, int]:
    """Two integers that change if any bit of ``tensor`` changes.

    The sums are over the raw bits as integers, so they wrap identically
    whatever order a device adds them in: bit-identical tensors give the same
    fingerprint on CUDA, ROCm or CPU. (A sum of the values themselves can differ
    in its last bits between devices even for identical tensors.)
    """
    flat = tensor.detach().contiguous().reshape(-1)
    bits = flat.view(_BITS[flat.element_size()]).to(torch.int64)
    weights = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64)
    return int(bits.sum()), int((bits * weights).sum())


def first_mismatch(per_rank: list[list[tuple[int, int]]]) -> list[int]:
    """The indices of the tensors whose fingerprint differs between ranks."""
    return [i for i, prints in enumerate(zip(*per_rank)) if len(set(prints)) > 1]


def largest_difference(names: list[str], per_rank: list[list[torch.Tensor]]) -> dict:
    """The largest elementwise difference from rank 0 over the given tensors, and where."""
    worst = {"max_abs_diff": 0.0, "tensor": None, "index": None, "rank": None,
             "elements_differing": 0}
    for i, name in enumerate(names):
        ref = per_rank[0][i].double()
        for rank, tensors in enumerate(per_rank[1:], start=1):
            diff = (tensors[i].double() - ref).abs().reshape(-1)
            worst["elements_differing"] += int((tensors[i] != per_rank[0][i]).sum())
            value, index = (float(x) for x in diff.max(dim=0))
            if value > worst["max_abs_diff"]:
                worst.update(max_abs_diff=value, tensor=name, index=int(index), rank=rank)
    return worst


class SyncCheck:
    """Per step, whether every rank holds the same gradients and parameters."""

    def __init__(self, model: nn.Module, world: int, perturb_step: int | None = None) -> None:
        self.named = [(n, p) for n, p in model.named_parameters()]
        self.world = world
        self.perturb_step = perturb_step
        self.first: dict[str, dict | None] = {"grads": None, "params": None}

    def _compare(self, what: str, step: int, tensors: list[torch.Tensor]) -> None:
        mine = [fingerprint(t) for t in tensors]
        everyone: list[list[tuple[int, int]] | None] = [None] * self.world
        dist.all_gather_object(everyone, mine)
        differing = first_mismatch(everyone)
        if not differing or self.first[what] is not None:
            return
        # Only the first time: gather the differing tensors themselves for the size of it.
        mine_cpu = [tensors[i].detach().cpu() for i in differing]
        theirs: list[list[torch.Tensor] | None] = [None] * self.world
        dist.all_gather_object(theirs, mine_cpu)
        names = [self.named[i][0] for i in differing]
        self.first[what] = {"step": step, "tensors": names,
                            **largest_difference(names, theirs)}

    def after_step(self, step: int, rank: int) -> None:
        """Call after optimizer.step(): p.grad still holds this step's synced gradients."""
        self._compare("grads", step, [p.grad for _, p in self.named])
        if self.perturb_step == step and rank == self.world - 1:
            with torch.no_grad():  # test hook: one parameter one ulp off on the last rank
                p = self.named[0][1].reshape(-1)
                p[0] = torch.nextafter(p[0], p[0] + 1)
        self._compare("params", step, [p for _, p in self.named])


def environment() -> dict:
    """This process's versions, GPUs and settings: the GPU kit's record when the
    kit is in this checkout (scripts/gpu/gpukit.py: allowlisted env vars, no
    paths or user names), else the basics."""
    kit = Path(__file__).resolve().parents[1] / "scripts" / "gpu"
    if (kit / "gpukit.py").exists():
        sys.path.insert(0, str(kit))
        from gpukit import environment as kit_environment

        return kit_environment()
    return {"hostname": socket.gethostname(), "gpubridge": gpubridge.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "torch_hip": torch.version.hip}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=50, help="timed steps")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256, help="per rank")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--json", type=Path, help="rank 0 writes a result record here")
    parser.add_argument("--perturb-step", type=int, help=argparse.SUPPRESS)  # tests only
    parser.add_argument("--timeout", type=float,
                        default=float(os.environ.get("KIT_TIMEOUT") or 600),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 600")
    args = parser.parse_args()

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
    device, rank, world = topology.device, topology.rank, topology.world_size

    # Same seed everywhere; DDP also broadcasts rank 0's parameters when it wraps
    # the model. The default group is gpubridge's own, which spans both vendors;
    # the hook sends every gradient bucket through gpubridge as one all_reduce.
    torch.manual_seed(0)
    model = make_model(args.dim, args.layers, args.classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None)
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # The task: labels come from a fixed random projection. Each rank sees its own samples.
    teacher = torch.randn(args.dim, args.classes, generator=torch.Generator().manual_seed(1))
    data = torch.Generator().manual_seed(100 + rank)
    batches = []
    for _ in range(8):
        x = torch.randn(args.batch_size, args.dim, generator=data)
        batches.append((x.to(device), (x @ teacher).argmax(dim=1).to(device)))

    check = SyncCheck(model.module, world, args.perturb_step)
    losses, step_times = [], []
    for step in range(args.warmup_steps + args.steps):
        x, y = batches[step % len(batches)]
        wait(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        loss = loss_fn(model(x), y)
        loss.backward()  # DDP waits for every bucket's all_reduce before returning
        optimizer.step()
        wait(device)
        if step >= args.warmup_steps:
            step_times.append(time.perf_counter() - start)
        losses.append(loss.item())
        check.after_step(step, rank)  # untimed: after the step's wait

    # Slowest rank sets the pace; every rank should hold identical parameters.
    elapsed = torch.tensor([sum(step_times)], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    in_sync = check.first["params"] is None
    # The check before the bit-for-bit one: the sum of each rank's parameters taken on
    # its own device, which can differ in its last bits even for identical parameters.
    checksum = float(sum(p.detach().double().sum() for p in model.parameters()))
    checksums: list[float | None] = [None] * world
    dist.all_gather_object(checksums, checksum)
    checksums_equal = all(c == checksums[0] for c in checksums)
    samples_per_sec = world * args.batch_size * args.steps / float(elapsed)
    environments: list[dict | None] = [None] * world
    if args.json:
        dist.all_gather_object(environments, environment())

    if rank == 0:
        result = {
            "script": "examples/train_synthetic.py",
            "world_size": world,
            "run": {"kind": topology.run_kind, "split_test": topology.split_test,
                    "islands": [list(i.ranks) for i in topology.layout.islands]},
            "device": str(device),
            "grad_sync": "DistributedDataParallel + gpubridge.ddp_comm_hook",
            "policy": topology.policy_name,
            "batch_size_per_rank": args.batch_size,
            "model": {"dim": args.dim, "layers": args.layers, "classes": args.classes,
                      "parameters": sum(p.numel() for p in model.parameters())},
            "steps": args.steps,
            "samples_per_sec": round(samples_per_sec, 2),
            "step_ms_median": round(statistics.median(step_times) * 1e3, 3),
            "loss_first": round(losses[0], 4),
            "loss_last": round(losses[-1], 4),
            "params_in_sync": in_sync,
            # Step numbers count the warm-up steps too. None: never differed.
            "first_grad_mismatch": check.first["grads"],
            "first_param_mismatch": check.first["params"],
            "params_checksums_equal": checksums_equal,
            "warmup_steps": args.warmup_steps,
            "lr": args.lr,
            "timeout_s": args.timeout,
            # Rank 0's, in the kit's layout, then every rank's.
            "environment": environments[0],
            "environments": environments,
        }
        if topology.split_test:
            result["warning"] = "SPLIT TEST: not a mixed-vendor result."
        print(f"{world} rank(s), {topology.run_kind}: {samples_per_sec:.1f} samples/sec, "
              f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, params in sync: {in_sync}"
              f" (float64 checksums equal: {checksums_equal})")
        for what in ("grads", "params"):
            if check.first[what]:
                first = check.first[what]
                print(f"{what} first differed at step {first['step']} "
                      f"(in {len(first['tensors'])} tensors, {first['elements_differing']} "
                      f"elements; largest {first['max_abs_diff']:.3g} in {first['tensor']})")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2, default=str))
    gpubridge.destroy()
    if not in_sync:
        raise SystemExit("parameters diverged across ranks")


if __name__ == "__main__":
    main()
