"""All-reduce across a simulated 2 NVIDIA + 2 AMD cluster, on CPU.

Each process fakes its vendor with GPUBRIDGE_VENDOR, so gpubridge runs the
same discovery, leader election and reduce-bridge-broadcast as on real
hardware, with Gloo standing in for NCCL and RCCL inside the islands. The
bridge is the default one: without an RDMA device, the Rust TCP bridge, built
with cargo on first use.

    python examples/simulate_two_islands.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.multiprocessing as mp

import gpubridge

VENDORS = ["nvidia", "nvidia", "amd", "amd"]


def worker(rank: int, init_file: str) -> None:
    os.environ["GPUBRIDGE_VENDOR"] = VENDORS[rank]
    topology = gpubridge.init(
        init_method=f"file://{init_file}", rank=rank, world_size=len(VENDORS)
    )

    if rank == 0:
        for island in topology.layout.islands:
            print(f"island {island.vendor:<6} ranks={list(island.ranks)} leader={island.leader}")
        print(f"bridge ranks={list(topology.layout.bridge_ranks)}\n")
    gpubridge.barrier()

    # Rank r contributes r + 1, so every rank should end up with 1 + 2 + 3 + 4 = 10.
    tensor = torch.full((4,), float(rank + 1))
    gpubridge.all_reduce(tensor)

    # Print one rank at a time so the output is in rank order.
    for turn in range(topology.world_size):
        if turn == rank:
            role = "leader" if topology.is_leader else "member"
            print(f"rank {rank} ({topology.vendor}, {role}): {tensor.tolist()}", flush=True)
        gpubridge.barrier()

    expected = float(sum(range(1, topology.world_size + 1)))
    matches = [None] * topology.world_size
    torch.distributed.all_gather_object(matches, bool(torch.all(tensor == expected)))
    if rank == 0:
        verdict = "all ranks match" if all(matches) else "MISMATCH"
        print(f"\nexpected {expected} on every rank: {verdict}")
    gpubridge.destroy()
    if not all(matches):
        sys.exit(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(os.path.join(tmp, "rendezvous"),), nprocs=len(VENDORS))
