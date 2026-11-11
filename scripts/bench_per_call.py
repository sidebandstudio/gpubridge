"""Per-call time of small all_reduces through gpubridge, against torch.distributed doing the same.

    python scripts/bench_per_call.py --out per_call.json
    python scripts/bench_per_call.py --vendors nvidia,nvidia,nvidia,nvidia --policies native-only
    python scripts/bench_per_call.py --profile reduce-bridge-broadcast   # cProfile on rank 0

Runs one CPU process per rank in simulation mode (Gloo everywhere, ``GPUBRIDGE_VENDOR``
per rank), so the times are CPU times: what gpubridge's Python adds to each call, not
what a GPU collective costs. Each rank times every call on its own clock. For each
policy, sync and async (``async_op=True`` then ``wait()``), with no observer and with
one that does nothing, it compares:

- ``gpubridge``: ``gpubridge.all_reduce`` with that policy;
- ``bare, same calls``: the torch.distributed calls the policy makes, made directly
  on the same groups (reduce-bridge-broadcast and policies that carry small messages
  the same way: an island reduce, the leaders' bridge all_reduce, an island broadcast;
  native-only: the island all_reduce; flat-gloo: the world all_reduce);
- ``bare, one world call``: one ``torch.distributed.all_reduce`` on the world group.

``overhead`` is gpubridge's median minus the same calls' median. Cases are
interleaved over ``--rounds`` so that drift on the host spreads across all of them.

With ``--host-time`` (sync calls only), each call's time minus the time spent inside
torch.distributed calls on the calling thread: the host work around the collectives.

The ``GPUBRIDGE_*`` settings that change what a policy does (compression, pinned staging,
chunk size, thresholds, the split test) must be unset, so that "bare, same calls" stays the
same calls; the script refuses to run otherwise. Its JSON output records its setup.

With ``--stub-comm``, torch.distributed's collectives (reduce, broadcast, all_reduce,
all_gather and reduce_scatter, the tensor forms included) return at
once after init: what is left is gpubridge's own work per call (checks, policy,
phases, the worker thread for async), without Gloo's latency or its noise.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import statistics
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import gpubridge
from gpubridge.config import (
    BRIDGE_COMPRESSION_ENV,
    CHUNK_BYTES_ENV,
    PINNED_STAGING_ENV,
    SPLIT_TEST_ENV,
    THRESHOLDS_ENV,
    VENDOR_ENV,
)

CHANGES_POLICIES = (BRIDGE_COMPRESSION_ENV, PINNED_STAGING_ENV, CHUNK_BYTES_ENV, THRESHOLDS_ENV,
                    SPLIT_TEST_ENV)
"""Settings that change what a policy does: with one set, the bare calls aren't the same calls."""

MIXED = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge",
         "flat-gloo", "auto-tuned")
SAME_CALLS = {"reduce-bridge-broadcast": "rbb", "pipelined-reduce-bridge-broadcast": "rbb",
              "auto-tuned": "rbb", "native-only": "island", "flat-gloo": "world"}
"""The torch.distributed calls each policy makes for a small all_reduce (auto-tuned
without GPUBRIDGE_THRESHOLDS runs what auto picks: reduce-bridge-broadcast here)."""


class _Done:
    def wait(self, *args, **kwargs) -> bool:
        return True

    def is_completed(self) -> bool:
        return True


def _no_comm(*args, async_op: bool = False, **kwargs):
    return _Done() if async_op else None


STUBBED = ("reduce", "broadcast", "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
           "reduce_scatter_tensor")


_inside = threading.local()


def _timed(real):
    """``real``, adding the time spent in it to this thread's ``_inside.ns``."""
    def call(*args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return real(*args, **kwargs)
        finally:
            _inside.ns = getattr(_inside, "ns", 0) + time.perf_counter_ns() - start
    return call


class Quiet(gpubridge.CollectiveObserver):
    def on_collective(self, record: gpubridge.CollectiveRecord) -> None:
        pass


def same_calls(kind: str, topology: gpubridge.Topology):
    island, leader = topology.island_group, topology.island.leader
    bridge = topology.bridge.group if topology.bridge is not None else None

    def rbb(t: torch.Tensor, async_op: bool) -> None:
        if async_op:  # what the worker thread would run, one after another
            dist.reduce(t, dst=leader, group=island, async_op=True).wait()
            if bridge is not None:
                dist.all_reduce(t, group=bridge, async_op=True).wait()
            dist.broadcast(t, src=leader, group=island, async_op=True).wait()
            return
        dist.reduce(t, dst=leader, group=island)
        if bridge is not None:
            dist.all_reduce(t, group=bridge)
        dist.broadcast(t, src=leader, group=island)

    def one(group):
        def call(t: torch.Tensor, async_op: bool) -> None:
            if async_op:
                dist.all_reduce(t, group=group, async_op=True).wait()
            else:
                dist.all_reduce(t, group=group)
        return call

    return {"rbb": rbb, "island": one(island), "world": one(None)}[kind]


def gpubridge_call(t: torch.Tensor, async_op: bool) -> None:
    if async_op:
        gpubridge.all_reduce(t, async_op=True).wait()
    else:
        gpubridge.all_reduce(t)


def time_calls(fn, t: torch.Tensor, async_op: bool, calls: int) -> list[int]:
    """Each call's time in ns: all of it, or with --host-time the part outside
    torch.distributed calls on this thread."""
    out = []
    clock = time.perf_counter_ns
    for _ in range(calls):
        _inside.ns = 0
        start = clock()
        fn(t, async_op)
        out.append(clock() - start - _inside.ns)
    return out


def rank_main(rank: int, args: argparse.Namespace, init_file: str, result_file: str) -> None:
    os.environ[VENDOR_ENV] = args.vendors[rank]
    torch.set_num_threads(1)
    samples: dict[str, list[int]] = {}
    for policy in args.policies:
        # bridge="gloo": the "same calls" this compares gpubridge with are torch calls
        # on its groups, the bridge's Gloo group among them.
        topology = gpubridge.init(init_method=f"file://{init_file}-{policy}", rank=rank,
                                  world_size=len(args.vendors), policy=policy, bridge="gloo",
                                  timeout=timedelta(seconds=120))
        real = {name: getattr(dist, name) for name in STUBBED}
        try:
            if args.stub_comm:
                for name in STUBBED:
                    setattr(dist, name, _no_comm)
            elif args.host_time:
                for name in STUBBED:
                    setattr(dist, name, _timed(real[name]))
            run_policy(topology, policy, args, samples)
        finally:
            for name, fn in real.items():
                setattr(dist, name, fn)
            gpubridge.destroy()
    if rank == 0 or args.all_ranks:
        Path(f"{result_file}.{rank}").write_text(json.dumps(samples))


def run_policy(topology, policy: str, args, samples: dict[str, list[int]]) -> None:
    quiet = Quiet()
    cases = []
    for numel in args.numels:
        for async_op in [m == "async" for m in args.modes]:
            mode = "async" if async_op else "sync"
            for observed in (False, True):
                cases.append((f"{policy}|{numel}|{mode}|gpubridge|{'observer' if observed else '-'}",
                              gpubridge_call, numel, async_op, observed))
            if policy in SAME_CALLS:
                cases.append((f"{policy}|{numel}|{mode}|bare, same calls|-",
                              same_calls(SAME_CALLS[policy], topology), numel, async_op, False))
            cases.append((f"{policy}|{numel}|{mode}|bare, one world call|-",
                          same_calls("world", topology), numel, async_op, False))
    tensors = {numel: torch.zeros(numel) for numel in args.numels}
    if args.profile:
        if args.observer:
            gpubridge.add_observer(quiet)
        profile(topology, policy, tensors[args.numels[0]], args)
        return
    for _ in range(args.rounds):
        for key, fn, numel, async_op, observed in cases:
            if observed:
                gpubridge.add_observer(quiet)
            try:
                time_calls(fn, tensors[numel], async_op, args.warmup)
                samples.setdefault(key, []).extend(time_calls(fn, tensors[numel], async_op,
                                                              args.calls))
            finally:
                if observed:
                    gpubridge.remove_observer(quiet)


def profile(topology, policy: str, tensor: torch.Tensor, args) -> None:
    time_calls(gpubridge_call, tensor, False, args.warmup)
    prof = cProfile.Profile()
    prof.enable()
    time_calls(gpubridge_call, tensor, False, args.calls * args.rounds)
    prof.disable()
    if topology.rank == 0:
        out = io.StringIO()
        stats = pstats.Stats(prof, stream=out).sort_stats("tottime")
        stats.print_stats(25)
        print(f"== cProfile, rank 0, {policy}, {args.calls * args.rounds} sync calls of "
              f"{tensor.numel()} float32, by own time\n{out.getvalue()}")


def summarize(samples: dict[str, list[int]]) -> list[dict]:
    rows = []
    for key, ns in samples.items():
        policy, numel, mode, path, observer = key.split("|")
        us = sorted(x / 1000 for x in ns)
        rows.append({"policy": policy, "bytes": int(numel) * 4, "mode": mode, "path": path,
                     "observer": observer, "calls": len(us), "median_us": statistics.median(us),
                     "p99_us": us[min(len(us) - 1, int(0.99 * len(us)))]})
    same = {(r["policy"], r["bytes"], r["mode"]): r["median_us"] for r in rows
            if r["path"] == "bare, same calls"}
    for r in rows:
        base = same.get((r["policy"], r["bytes"], r["mode"]))
        r["overhead_us"] = (r["median_us"] - base if r["path"] == "gpubridge" and base is not None
                            else None)
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| policy | bytes | mode | path | observer | median µs | p99 µs | overhead µs |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        over = f"{r['overhead_us']:+.1f}" if r["overhead_us"] is not None else ""
        lines.append(f"| {r['policy']} | {r['bytes']} | {r['mode']} | {r['path']} | {r['observer']} "
                     f"| {r['median_us']:.1f} | {r['p99_us']:.1f} | {over} |")
    return "\n".join(lines)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vendors", default="nvidia,nvidia,amd,amd",
                   help="one vendor per rank (default: two islands of 2)")
    p.add_argument("--policies", default=None,
                   help=f"comma-separated (default: {','.join(MIXED)}; native-only alone "
                        "for a single vendor)")
    p.add_argument("--numels", default="1,256,4096", help="float32 elements per tensor")
    p.add_argument("--calls", type=int, default=400, help="timed calls per case per round")
    p.add_argument("--warmup", type=int, default=50, help="untimed calls before each case")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--modes", default="sync,async", help="sync, async or both (default)")
    p.add_argument("--all-ranks", action="store_true", help="pool every rank's times (default rank 0)")
    p.add_argument("--profile", metavar="POLICY", help="cProfile rank 0's sync calls instead")
    p.add_argument("--observer", action="store_true", help="with --profile: an observer attached")
    p.add_argument("--host-time", action="store_true",
                   help="sync calls: time outside torch.distributed calls, not the whole call")
    p.add_argument("--stub-comm", action="store_true",
                   help="torch.distributed's calls return at once: gpubridge's own work only")
    p.add_argument("--out", type=Path, help="write the rows as JSON here")
    args = p.parse_args(argv)
    args.vendors = args.vendors.split(",")
    single = len(set(args.vendors)) == 1
    if args.profile:
        args.policies = [args.profile]
    elif args.policies:
        args.policies = args.policies.split(",")
    else:
        args.policies = ["native-only"] if single else list(MIXED)
    args.numels = [int(x) for x in args.numels.split(",")]
    args.modes = args.modes.split(",")
    if not set(args.modes) <= {"sync", "async"}:
        p.error(f"--modes takes sync, async or both, not {','.join(args.modes)}")
    if args.observer and not args.profile:
        p.error("--observer goes with --profile")
    if args.host_time and args.stub_comm:
        p.error("--host-time and --stub-comm don't mix")
    if args.host_time:
        args.modes = ["sync"]  # an async call's collectives run on the worker thread
    return args


def main(argv: list[str]) -> int:
    args = parse(argv)
    settings = [k for k in CHANGES_POLICIES if os.environ.get(k)]
    if settings:
        print(f"bench_per_call.py: unset {', '.join(settings)} first. These settings change what the "
              "policies do, so the bare calls would no longer be the same calls", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        result = os.path.join(tmp, "samples")
        mp.spawn(rank_main, args=(args, os.path.join(tmp, "rendezvous"), result),
                 nprocs=len(args.vendors))
        if args.profile:
            return 0
        samples: dict[str, list[int]] = {}
        for f in sorted(Path(tmp).glob("samples.*")):
            for key, ns in json.loads(f.read_text()).items():
                samples.setdefault(key, []).extend(ns)
    rows = summarize(samples)
    print(f"{len(args.vendors)} ranks ({','.join(args.vendors)}), torch {torch.__version__}, "
          f"gpubridge {gpubridge.__version__}, CPU simulation"
          f"{', communication stubbed' if args.stub_comm else ''}"
          f"{', host time only' if args.host_time else ''}\n{table(rows)}")
    if args.out:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        args.out.write_text(json.dumps({"vendors": args.vendors, "torch": torch.__version__,
                                        "stub_comm": args.stub_comm, "host_time": args.host_time,
                                        "argv": argv, "cpus": cpus, "python": platform.python_version(),
                                        "platform": platform.platform(),
                                        "gpubridge": gpubridge.__version__, "rows": rows},
                                       indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
