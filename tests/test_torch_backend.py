"""gpubridge as a torch.distributed backend: init_process_group(backend="gpubridge").

Whole jobs in simulation mode: DDP and FSDP2 trained through the backend
match the same training through Gloo bit for bit (integer-valued data, so
every sum is exact in any order); every torch.distributed collective the
backend carries, coalesced and bucketed paths included; and every unsupported
call refused on every rank. docs/TORCH_BACKEND.md has the design.
"""

from __future__ import annotations

import gc
import importlib.metadata
import inspect
import os
import re
import subprocess
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from _harness import expected_sum, rank_tensor

import gpubridge
from gpubridge import topology, torch_backend
from gpubridge.config import BRIDGE_ENV, POLICY_ENV, VENDOR_ENV

TWO_TWO = ["nvidia", "nvidia", "amd", "amd"]
TIMEOUT = timedelta(seconds=60)
REPO = Path(__file__).resolve().parents[1]


def run_backend(vendors, fn, tmp_path: Path, *, backend: str = "gpubridge",
                env: dict[int, dict[str, str]] | None = None) -> None:
    """Run ``fn(rank, world)`` on one process per vendor, after
    ``init_process_group(backend)``; destroy the group after."""
    mp.spawn(_main, args=(tuple(vendors), fn, str(tmp_path / "rendezvous"), backend, env or {}),
             nprocs=len(vendors))


def _main(rank, vendors, fn, init_file, backend, env):
    os.environ[VENDOR_ENV] = vendors[rank]
    os.environ.update(env.get(rank, env.get(-1, {})))
    torch.set_num_threads(1)
    dist.init_process_group(backend=backend, init_method=f"file://{init_file}", rank=rank,
                            world_size=len(vendors), timeout=TIMEOUT)
    try:
        fn(rank, len(vendors))
    finally:
        # FSDP2 modules keep process groups alive in reference cycles. Destroyed later,
        # at interpreter exit, a Gloo group can abort the process ("terminate called
        # without an active exception"), with either backend. Collect them first.
        gc.collect()
        dist.destroy_process_group()
        gc.collect()
    assert not gpubridge.is_initialized()  # shutdown() released it


# ---- training -----------------------------------------------------------------------------

#: The 4 ranks' gradients are averaged, so a learning rate of 4 makes each update
#: their sum: parameters stay whole numbers (about 1e5 after 3 steps), and float64
#: adds them exactly in any order. Different summation orders (Gloo's, gpubridge's
#: islands and bridge) must then give the same bits.
LR = 4.0
STEPS = 3


def _integer_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randint(-3, 4, p.shape).double())
    return model


def _batch(rank: int, step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 * step + rank)
    return torch.randint(-2, 3, (4, 8), generator=g).double()


def _ddp(rank, world, out, bucket_mb, static):
    model = _integer_model()
    ddp = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_mb, static_graph=static)
    opt = torch.optim.SGD(ddp.parameters(), lr=LR)
    for step in range(STEPS):
        ddp(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    if rank == 0:
        torch.save([p.detach().clone() for p in model.parameters()], out)


@pytest.mark.parametrize(("bucket_mb", "static"), [(0.0001, False), (25, False), (0.0001, True)],
                         ids=["many-buckets", "one-bucket", "static-graph"])
def test_ddp_trains_bit_for_bit_like_gloo(bucket_mb, static, tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_ddp, out=str(out), bucket_mb=bucket_mb, static=static),
                    tmp_path / backend, backend=backend)
        results[backend] = torch.load(out)
    assert all(torch.equal(a, b) for a, b in zip(results["gloo"], results["gpubridge"],
                                                strict=True))
    # And the model really moved: the comparison isn't between two untouched models.
    assert not all(torch.equal(a, b) for a, b in zip(results["gloo"], _integer_model().parameters(),
                                                    strict=True))


def _fully_shard():
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:  # older releases
        from torch.distributed._composable.fsdp import fully_shard
    return fully_shard


def _fsdp2(rank, world, out):
    from torch.distributed.device_mesh import init_device_mesh

    fully_shard = _fully_shard()
    mesh = init_device_mesh("cpu", (world,))
    model = _integer_model()
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        model(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().clone() for p in model.parameters()]
    if rank == 0:
        torch.save(full, out)


def _fsdp2_float32(rank, world, out):
    """float32 gradients take FSDP2's AVG reduce-scatter (float64 takes SUM)."""
    from torch.distributed.device_mesh import init_device_mesh

    fully_shard = _fully_shard()
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    fully_shard(model[0], mesh=mesh)
    fully_shard(model[2], mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(STEPS):
        model(_batch(rank, step).float()).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().clone() for p in model.parameters()]
    if rank == 0:
        torch.save(full, out)


def test_fsdp2_in_float32_takes_the_avg_path_and_matches_a_single_process(tmp_path):
    out = tmp_path / "gpubridge.pt"
    run_backend(TWO_TWO, partial(_fsdp2_float32, out=str(out)), tmp_path)
    # One process on the whole batch, not Gloo: torch 2.8.0's Gloo reduce_scatter
    # returns the sum for AVG (measured), so FSDP2 on Gloo doesn't average there.
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(STEPS):
        loss = sum(model(_batch(r, step).float()).sum() for r in range(4)) / 4
        loss.backward()
        opt.step()
        opt.zero_grad()
    # Not bit for bit: float32 sums in a different order round differently.
    for a, b in zip(torch.load(out), model.parameters(), strict=True):
        torch.testing.assert_close(a, b.detach(), rtol=1e-5, atol=1e-6)


def test_fsdp2_trains_bit_for_bit_like_gloo_and_a_single_process(tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_fsdp2, out=str(out)), tmp_path / backend, backend=backend)
        results[backend] = torch.load(out)
    assert all(torch.equal(a, b) for a, b in zip(results["gloo"], results["gpubridge"],
                                                strict=True))
    # One process on the whole batch: the mean gradient over the 4 ranks' batches.
    model = _integer_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        loss = sum(model(_batch(r, step)).sum() for r in range(4)) / 4
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert all(torch.equal(a, b.detach()) for a, b in zip(results["gpubridge"],
                                                          model.parameters(), strict=True))


# ---- every collective the backend carries -----------------------------------------------------

def _carried(rank, world):
    expected = sum(rank_tensor((5, 7), torch.float32, r) for r in range(world))
    for op, want in [(dist.ReduceOp.SUM, expected), (dist.ReduceOp.AVG, expected / world),
                     (dist.ReduceOp.MAX, rank_tensor((5, 7), torch.float32, world - 1)),
                     (dist.ReduceOp.MIN, rank_tensor((5, 7), torch.float32, 0))]:
        t = rank_tensor((5, 7), torch.float32, rank)
        dist.all_reduce(t, op=op)
        assert torch.equal(t, want), op
    t = rank_tensor((5, 7), torch.int64, rank)
    work = dist.all_reduce(t, async_op=True)
    work.wait()
    assert torch.equal(t, expected.to(torch.int64))
    t = rank_tensor((5, 7), torch.float32, rank)
    future = dist.all_reduce(t, async_op=True).get_future()
    assert torch.equal(future.wait()[0], expected)
    for src in range(world):
        t = rank_tensor((9,), torch.float32, rank)
        dist.broadcast(t, src=src)
        assert torch.equal(t, rank_tensor((9,), torch.float32, src))
    out = torch.empty(world * 6)
    dist.all_gather_into_tensor(out, rank_tensor((6,), torch.float32, rank))
    assert torch.equal(out, torch.cat([rank_tensor((6,), torch.float32, r) for r in range(world)]))
    outs = [torch.empty(3, 2) for _ in range(world)]
    dist.all_gather(outs, rank_tensor((3, 2), torch.float32, rank))
    assert all(torch.equal(o, rank_tensor((3, 2), torch.float32, r)) for r, o in enumerate(outs))
    rows = rank_tensor((world, 6), torch.float32, rank)
    got = torch.empty(6)
    dist.reduce_scatter_tensor(got, rows)
    assert torch.equal(got, sum(rank_tensor((world, 6), torch.float32, r)
                                for r in range(world))[rank])
    got = torch.empty(6)
    dist.reduce_scatter(got, list(rank_tensor((world, 6), torch.float32, rank).unbind(0)))
    assert torch.equal(got, sum(rank_tensor((world, 6), torch.float32, r)
                                for r in range(world))[rank])
    dist.barrier()
    # Coalesced: the deprecated all_reduce_coalesced, and the coalescing manager.
    ts = [rank_tensor((4,), torch.float32, rank), rank_tensor((3,), torch.float32, rank)]
    dist.all_reduce_coalesced(ts)
    assert torch.equal(ts[1], sum(rank_tensor((3,), torch.float32, r) for r in range(world)))
    from torch.distributed.distributed_c10d import _coalescing_manager

    for device in (None, torch.device("cpu")):
        ts = [rank_tensor((4,), torch.float32, rank), rank_tensor((5,), torch.float32, rank)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            for t in ts:
                dist.all_reduce(t)
        cm.wait()
        assert torch.equal(ts[0], sum(rank_tensor((4,), torch.float32, r) for r in range(world)))
        assert torch.equal(ts[1], sum(rank_tensor((5,), torch.float32, r) for r in range(world)))
        outs = [torch.empty(world * 2), torch.empty(world * 3)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            dist.all_gather_into_tensor(outs[0], rank_tensor((2,), torch.float32, rank))
            dist.all_gather_into_tensor(outs[1], rank_tensor((3,), torch.float32, rank))
        cm.wait()
        assert torch.equal(outs[1], torch.cat([rank_tensor((3,), torch.float32, r)
                                               for r in range(world)]))
        got = [torch.empty(2), torch.empty(3)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            dist.reduce_scatter_tensor(got[0], rank_tensor((world * 2,), torch.float32, rank))
            dist.reduce_scatter_tensor(got[1], rank_tensor((world * 3,), torch.float32, rank))
        cm.wait()
        whole = sum(rank_tensor((world * 3,), torch.float32, r) for r in range(world))
        assert torch.equal(got[1], whole.view(world, 3)[rank])


def _collectives(rank, world):
    _carried(rank, world)
    # gpubridge's own API works on the same topology, and its observers see the calls.
    seen = []

    class Seen(gpubridge.CollectiveObserver):
        def on_collective(self, record):
            seen.append((record.op, record.policy))

    gpubridge.add_observer(Seen())
    t = rank_tensor((3,), torch.float32, rank)
    gpubridge.all_reduce(t)
    dist.all_reduce(rank_tensor((3,), torch.float32, rank))
    assert seen == [("all_reduce", "reduce-bridge-broadcast")] * 2, seen
    assert gpubridge.get_topology().layout.needs_bridge


@pytest.mark.parametrize("vendors", [TWO_TWO, ["amd", "nvidia", "nvidia"]], ids=["2+2", "1+2"])
def test_every_collective_the_backend_carries(vendors, tmp_path):
    run_backend(vendors, _collectives, tmp_path)


def _cpu_tensor_path(rank, world):
    """Simulated jobs run on CPU, where nothing takes the CPU-tensor path: force it."""
    torch_backend._cpu_path = lambda topology, tensors: True
    seen = []

    class Seen(gpubridge.CollectiveObserver):
        def on_collective(self, record):
            seen.append(record.op)

    dist.barrier()  # sets gpubridge up, so the observer can be added
    gpubridge.add_observer(Seen())
    _carried(rank, world)
    for refused in (lambda: dist.all_reduce(torch.ones(2), op=dist.ReduceOp.PRODUCT),
                    lambda: dist.reduce_scatter_tensor(torch.empty(1), torch.ones(world),
                                                       op=dist.ReduceOp.MAX)):
        with pytest.raises((ValueError, NotImplementedError), match="(?i)product|max"):
            refused()
    rows = rank_tensor((world, 3), torch.float32, rank)
    got = torch.empty(3)
    dist.reduce_scatter_tensor(got, rows, op=dist.ReduceOp.AVG)
    assert torch.equal(got, sum(rank_tensor((world, 3), torch.float32, r)
                                for r in range(world))[rank] / world)
    assert seen == ["barrier"], seen  # the rest on the backend's Gloo group, not gpubridge


def test_the_cpu_tensor_path_carries_every_collective_on_gloo(tmp_path):
    run_backend(TWO_TWO, _cpu_tensor_path, tmp_path)


def _policy_and_bridge(rank, world, policy):
    dist.barrier()
    assert gpubridge.get_topology().policy_name == policy
    t = rank_tensor((50,), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, sum(rank_tensor((50,), torch.float32, r) for r in range(world)))


@pytest.mark.parametrize("policy", ["pipelined-reduce-bridge-broadcast", "sharded-bridge",
                                    "flat-gloo"])
def test_the_policy_comes_from_gpubridge_policy(policy, tmp_path):
    run_backend(TWO_TWO, partial(_policy_and_bridge, policy=policy), tmp_path,
                env={-1: {POLICY_ENV: policy, "GPUBRIDGE_CHUNK_BYTES": "64"}})


# ---- refused, on every rank ---------------------------------------------------------------

def _refused(rank, world):
    def expect(fn, pattern):
        with pytest.raises((NotImplementedError, ValueError, RuntimeError), match=pattern):
            fn()

    dist.barrier()  # set up first, so the refusals below are the backend's
    expect(lambda: dist.all_reduce(torch.ones(3), op=dist.ReduceOp.PRODUCT),
           "gpubridge backend, allreduce: .*supports ReduceOp.SUM, AVG, MAX and MIN")
    expect(lambda: dist.all_to_all_single(torch.empty(world), torch.ones(world)),
           "the gpubridge torch.distributed backend doesn't support all_to_all")
    expect(lambda: dist.reduce(torch.ones(2), dst=0), "doesn't support reduce")
    expect(lambda: dist.irecv(torch.empty(2)), "doesn't support any-source recv")
    expect(lambda: dist.all_gather([torch.empty(2) for _ in range(world)], torch.ones(3)),
           "allgather with tensors of different sizes")
    expect(lambda: dist.reduce_scatter_tensor(torch.empty(2), torch.ones(5)),
           "gpubridge backend, reduce_scatter_tensor: .*times as many elements")
    # A group of every rank shares gpubridge's topology.
    again = dist.new_group(list(range(world)))
    t = rank_tensor((4,), torch.float32, rank)
    dist.all_reduce(t, group=again)
    assert torch.equal(t, sum(rank_tensor((4,), torch.float32, r) for r in range(world)))
    # Still healthy after every refusal.
    t = rank_tensor((4,), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, sum(rank_tensor((4,), torch.float32, r) for r in range(world)))


def test_unsupported_calls_fail_clearly_on_every_rank(tmp_path):
    run_backend(TWO_TWO, _refused, tmp_path)


def _objects_and_checkpoint_helpers(rank, world):
    # gather_object and scatter_object_list (torch.distributed.checkpoint plans with them)
    # use gather and scatter of CPU tensors: the backend's Gloo group carries them.
    got = [None] * world if rank == 0 else None
    dist.gather_object({"rank": rank}, got, dst=0)
    if rank == 0:
        assert got == [{"rank": r} for r in range(world)]
    out = [None]
    dist.scatter_object_list(out, [f"to {r}" for r in range(world)] if rank == 2 else None,
                             src=2)
    assert out == [f"to {rank}"]
    objs = [None] * world
    dist.all_gather_object(objs, rank * 10)
    assert objs == [r * 10 for r in range(world)]
    box = [rank] if rank == 1 else [None]
    dist.broadcast_object_list(box, src=1)
    assert box == [1]


def test_object_collectives_and_what_checkpointing_needs(tmp_path):
    run_backend(TWO_TWO, _objects_and_checkpoint_helpers, tmp_path)


def _ddp_buffers_unused_and_hook(rank, world, out):
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(8, 8).double(), nn.Linear(8, 4).double()
            self.unused = nn.Linear(4, 4).double()
            self.norm = nn.BatchNorm1d(8).double()  # buffers, broadcast every step

        def forward(self, x):
            return self.b(self.norm(self.a(x)).relu())

    net = Net()
    ddp = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True,
                                              broadcast_buffers=True)
    ddp.register_comm_hook(None, lambda state, bucket: dist.all_reduce(
        bucket.buffer().div_(world), async_op=True).get_future().then(lambda f: f.value()[0]))
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
    for step in range(3):
        ddp(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    if rank == 0:
        torch.save([t.detach().clone() for t in net.state_dict().values()], out)


def test_ddp_with_buffers_unused_parameters_and_a_comm_hook(tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_ddp_buffers_unused_and_hook, out=str(out)),
                    tmp_path / backend, backend=backend)
        results[backend] = torch.load(out)
    for a, b in zip(results["gloo"], results["gpubridge"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def _reinit_and_teardown_with_work_queued(rank, init_file, world):
    os.environ[VENDOR_ENV] = TWO_TWO[rank]
    torch.set_num_threads(1)
    for attempt in range(2):  # the same process, twice
        dist.init_process_group(backend="gpubridge", init_method=f"file://{init_file}{attempt}",
                                rank=rank, world_size=world, timeout=TIMEOUT)
        t = rank_tensor((5,), torch.float32, rank)
        dist.all_reduce(t)
        assert torch.equal(t, sum(rank_tensor((5,), torch.float32, r) for r in range(world)))
        assert gpubridge.is_initialized()
        if attempt == 1:  # destroy with collectives still queued: never hangs
            for _ in range(3):
                dist.all_reduce(torch.ones(1000), async_op=True)
        dist.destroy_process_group()
        assert not gpubridge.is_initialized() and not dist.is_initialized()


def test_init_again_in_the_same_process_and_destroy_with_work_queued(tmp_path):
    mp.spawn(_reinit_and_teardown_with_work_queued,
             args=(str(tmp_path / "rendezvous"), 4), nprocs=4)


def test_init_process_group_with_a_device_id_finds_no_backend_to_split():
    # With device_id, torch asks the group for that device's C++ backend and whether
    # it can split eagerly; a Python group has none. (device_id must be an
    # accelerator, so the real call is in the GPU tests.)
    pg = torch_backend.GpubridgeProcessGroup(0, 1, "test", None)
    assert pg._get_backend(torch.device("cuda", 0)).supports_splitting is False
    assert pg.group_name == "test" and pg.getBackendName() == "gpubridge"


def _collective_during_setup(rank, world):
    original = gpubridge.topology.build_topology

    def calls_back(*args, **kwargs):  # as a custom transport or policy might
        dist.all_reduce(torch.ones(1))
        return original(*args, **kwargs)

    gpubridge.topology.build_topology = calls_back
    with pytest.raises(RuntimeError, match="while gpubridge was setting itself up"):
        dist.barrier()


def test_a_collective_on_the_backend_during_its_own_setup_is_refused(tmp_path):
    run_backend(TWO_TWO, _collective_during_setup, tmp_path)


def _first_collective_fails(rank, world):
    with pytest.raises(RuntimeError, match="Ranks picked different collective policies"):
        dist.barrier()


def test_a_setting_that_differs_fails_the_first_collective_on_every_rank(tmp_path):
    run_backend(TWO_TWO, _first_collective_fails, tmp_path,
                env={0: {POLICY_ENV: "flat-gloo"}, -1: {}})


@gpubridge.register_policy
class Failing(gpubridge.policies.ReduceBridgeBroadcast):
    """A policy whose all_reduce fails on every rank, as a lost bridge peer would."""

    name = "test-failing"

    def all_reduce(self, tensor, topology, op=None):
        raise ZeroDivisionError("policy bug")


def _a_failed_collective_raises(rank, world):
    dist.barrier()
    t = torch.ones(3)
    # The call that failed raises, not only the one after it: torch's own Work for a
    # Python backend would return from wait() as if the collective had worked.
    with pytest.raises(RuntimeError, match="all_reduce failed: policy bug"):
        dist.all_reduce(t)
    # And every later collective on this rank refuses to start, sync or async, whatever
    # torch function asks.
    refused = "an earlier async gpubridge collective failed"
    with pytest.raises(RuntimeError, match=refused):
        dist.all_reduce(t)
    with pytest.raises(RuntimeError, match=refused):
        dist.all_gather([torch.zeros(1) for _ in range(world)], torch.ones(1))
    with pytest.raises(RuntimeError, match=refused):
        dist.broadcast(t, src=0, async_op=True)
    with pytest.raises(RuntimeError, match=refused):
        dist.barrier()


def test_a_failed_collective_raises_from_the_torch_call_that_made_it(tmp_path):
    run_backend(TWO_TWO, _a_failed_collective_raises, tmp_path,
                env={-1: {POLICY_ENV: "test-failing"}})


def _an_async_failure_raises_from_wait(rank, world):
    dist.barrier()
    work = dist.all_reduce(torch.ones(3), async_op=True)
    with pytest.raises(RuntimeError, match="all_reduce failed: policy bug"):
        work.wait()
    assert work.is_completed() and not work.is_success()
    assert isinstance(work.exception(), ZeroDivisionError)


def test_an_async_collectives_failure_raises_from_its_wait(tmp_path):
    run_backend(TWO_TWO, _an_async_failure_raises_from_wait, tmp_path,
                env={-1: {POLICY_ENV: "test-failing"}})


def _ddp_without_a_hook_fails_by_the_next_collective(rank, world):
    # DDP's reducer is C++: it gets the kind of Work C++ can hold, whose wait() doesn't
    # raise. The failure then stops the job at this rank's next collective at the
    # latest, never later.
    model = nn.parallel.DistributedDataParallel(_integer_model())
    with pytest.raises(RuntimeError, match="policy bug|an earlier async gpubridge collective"):
        model(_batch(rank, 0)).sum().backward()
        dist.barrier()


def test_ddp_without_a_comm_hook_fails_by_the_collective_after_a_failed_one(tmp_path):
    run_backend(TWO_TWO, _ddp_without_a_hook_fails_by_the_next_collective, tmp_path,
                env={-1: {POLICY_ENV: "test-failing"}})


def _two_jobs_in_a_row(rank, world):
    t = rank_tensor((3,), torch.float32, rank)
    dist.all_reduce(t)
    with pytest.raises(RuntimeError, match="already initialized"):
        gpubridge.init()


def test_gpubridge_init_and_the_backend_exclude_each_other(tmp_path):
    run_backend(TWO_TWO, _two_jobs_in_a_row, tmp_path)


def _backend_group_after_init(topology):
    group = dist.new_group(list(range(topology.world_size)), backend="gpubridge")
    with pytest.raises(RuntimeError, match="use either gpubridge.init\\(\\) or the gpubridge"):
        dist.all_reduce(torch.ones(2), group=group)


def test_a_backend_group_after_gpubridge_init_is_refused(tmp_path):
    from _harness import run_ranks

    run_ranks(TWO_TWO, _backend_group_after_init, tmp_path)


def test_bridge_from_gpubridge_bridge_is_checked(tmp_path):
    run_backend(["nvidia", "amd"], _first_collective_unknown_bridge, tmp_path,
                env={-1: {BRIDGE_ENV: "no-such-bridge"}})


def _first_collective_unknown_bridge(rank, world):
    # A setting one rank can't use fails every rank (here all of them), not just that one.
    with pytest.raises(RuntimeError, match="unknown bridge transport 'no-such-bridge'"):
        dist.barrier()


def _one_rank_with_a_bad_setting(rank, world):
    with pytest.raises(RuntimeError, match=r"rank 1: unknown collective policy 'no-such'"):
        dist.barrier()


def test_a_bad_setting_on_one_rank_fails_every_rank_instead_of_hanging(tmp_path):
    run_backend(TWO_TWO, _one_rank_with_a_bad_setting, tmp_path,
                env={1: {POLICY_ENV: "no-such"}, -1: {}})


# ---- subgroups and HSDP --------------------------------------------------------------------

def _sub(group) -> torch_backend.GpubridgeProcessGroup:
    assert isinstance(group, torch_backend.GpubridgeProcessGroup), group
    return group


def _subgroups(rank, world):
    # Every rank creates every group, in the same order; torch calls the creator on
    # members only, and the others get NON_GROUP_MEMBER (None on torch 2.8).
    nvidia, amd = dist.new_group([0, 1]), dist.new_group([2, 3])
    pair, triple = dist.new_group([0, 2]), dist.new_group([0, 1, 3])
    outside = (None, dist.GroupMember.NON_GROUP_MEMBER)
    # The first collective runs on every rank, each on its own island's subgroup: it
    # sets up the world's topology too (every rank takes part), then the subgroup's.
    island = nvidia if rank < 2 else amd
    t = rank_tensor((5,), torch.float32, rank)
    dist.all_reduce(t, group=island)
    mates = [0, 1] if rank < 2 else [2, 3]
    assert torch.equal(t, sum(rank_tensor((5,), torch.float32, r) for r in mates))
    # The world's topology belongs to the default group even so, with its timeout.
    assert torch_backend._owner is dist.distributed_c10d._get_default_group()
    topology = _sub(island)._topology()
    assert topology.policy.name == "native-only" and topology.bridge is None
    assert topology.layout.members == tuple(mates) and len(topology.layout.islands) == 1
    assert (nvidia in outside) == (rank >= 2) and (amd in outside) == (rank < 2)

    # Mixed members: bridged, between the subgroup's own leaders.
    if rank in (0, 2):
        topology = _sub(pair)._topology()
        assert topology.policy.name == "reduce-bridge-broadcast"
        assert topology.layout.bridge_ranks == (0, 2) and topology.bridge is not None
        t = rank_tensor((6,), torch.float64, rank)
        dist.all_reduce(t, group=pair)
        assert torch.equal(t, rank_tensor((6,), torch.float64, 0)
                           + rank_tensor((6,), torch.float64, 2))
        out = torch.empty(2, 3, dtype=torch.int64)
        dist.all_gather_into_tensor(out, torch.full((3,), rank), group=pair)
        assert out.tolist() == [[0, 0, 0], [2, 2, 2]]  # rows in group-rank order
        got = torch.empty(2, dtype=torch.int64)
        dist.reduce_scatter_tensor(got, torch.arange(4) + 10 * rank, group=pair)
        assert got.tolist() == ([20, 22] if rank == 0 else [24, 26])  # slice = group rank
        b = torch.full((4,), float(rank))
        dist.broadcast(b, src=2, group=pair)  # src is a global rank, as torch takes it
        assert torch.equal(b, torch.full((4,), 2.0))
        dist.barrier(group=pair)
    else:
        assert pair in outside
    # Members not contiguous, and an island of one: rows follow ascending global rank.
    if rank in (0, 1, 3):
        topology = _sub(triple)._topology()
        assert [i.ranks for i in topology.layout.islands] == [(0, 1), (3,)]
        out = torch.empty(3, 2, dtype=torch.float32)
        dist.all_gather_into_tensor(out, torch.full((2,), float(rank)), group=triple)
        assert out[:, 0].tolist() == [0.0, 1.0, 3.0]
        got = torch.empty(2, dtype=torch.float32)
        dist.reduce_scatter_tensor(got, torch.arange(6, dtype=torch.float32), group=triple)
        position = [0, 1, 3].index(rank)
        assert got.tolist() == [3.0 * (2 * position), 3.0 * (2 * position + 1)]
        objects = [None] * 3
        dist.all_gather_object(objects, f"r{rank}", group=triple)  # CPU tensors: Gloo
        assert objects == ["r0", "r1", "r3"]
    # Destroying a subgroup closes its own topology only: gpubridge stays up for the job.
    if rank in (0, 2):
        dist.destroy_process_group(pair)
        assert all(g is not pair for g in torch_backend._subgroups)
    assert gpubridge.is_initialized()
    # The world still works after its subgroups.
    t = rank_tensor((4,), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, sum(rank_tensor((4,), torch.float32, r) for r in range(world)))
    # Ranks out of order (new_group(..., sort_ranks=False)) are refused on the members.
    if "sort_ranks" in inspect.signature(dist.new_group).parameters:  # torch 2.14; not 2.8
        if rank in (1, 2):
            with pytest.raises(RuntimeError, match="ascending order"):
                dist.new_group([2, 1], sort_ranks=False, use_local_synchronization=True)


def test_subgroups_of_one_vendor_are_native_and_mixed_ones_bridged(tmp_path):
    run_backend(TWO_TWO, _subgroups, tmp_path)


def _interleaved(rank, world):
    # Vendors alternate, so a subgroup's islands laid end to end ({0, 2} then {1}) are
    # not its member order (0, 1, 2): rows and slices must follow the member order.
    dist.barrier()
    group = dist.new_group([0, 1, 2])
    if rank == 3:
        return
    topology = _sub(group)._topology()
    assert [i.ranks for i in topology.layout.islands] == [(0, 2), (1,)]
    out = torch.empty(3, 2, dtype=torch.float32)
    dist.all_gather_into_tensor(out, torch.full((2,), float(rank)), group=group)
    assert out[:, 0].tolist() == [0.0, 1.0, 2.0]
    got = torch.empty(2, dtype=torch.float32)
    dist.reduce_scatter_tensor(got, torch.arange(6, dtype=torch.float32), group=group)
    assert got.tolist() == [3.0 * (2 * rank), 3.0 * (2 * rank + 1)]


def test_a_subgroup_whose_member_order_is_not_its_islands_end_to_end(tmp_path):
    run_backend(["nvidia", "amd", "nvidia", "amd"], _interleaved, tmp_path)


def _subgroup_cpu_path(rank, world):
    """The CPU-tensor path, forced as in _cpu_tensor_path, on subgroups: their own Gloo
    group, and roots given as global ranks."""
    torch_backend._cpu_path = lambda topology, tensors: True
    dist.barrier()
    pair = dist.new_group([0, 2])
    if rank not in (0, 2):
        return
    t = torch.full((3,), float(rank + 1))
    dist.all_reduce(t, group=pair)
    assert t.tolist() == [4.0] * 3
    b = torch.full((2,), float(rank))
    dist.broadcast(b, src=2, group=pair)
    assert b.tolist() == [2.0, 2.0]
    objects = [None, None]
    dist.all_gather_object(objects, rank, group=pair)
    assert objects == [0, 2]
    gathered = [None, None] if rank == 2 else None
    dist.gather_object(f"r{rank}", gathered, dst=2, group=pair)
    assert gathered == (["r0", "r2"] if rank == 2 else None)
    received = [None]
    dist.scatter_object_list(received, ["to0", "to2"] if rank == 2 else None, src=2, group=pair)
    assert received == [f"to{rank}"]
    assert _sub(pair)._cpu() is _sub(pair)._sub[1]  # the subgroup's own Gloo group


def test_the_cpu_tensor_path_on_a_subgroup(tmp_path):
    run_backend(TWO_TWO, _subgroup_cpu_path, tmp_path)


class _Records(gpubridge.CollectiveObserver):
    def __init__(self):
        self.records = []

    def on_collective(self, record):
        self.records.append(record)


def _hsdp(rank, world, out):
    """HSDP on a 2-D mesh whose rows are the vendor islands: shard within an island,
    replicate across vendors. Records every gpubridge collective and bridge call."""
    from torch.distributed.device_mesh import init_device_mesh

    from gpubridge import transport

    fully_shard = _fully_shard()
    dist.barrier()  # sets gpubridge up, so an observer can attach before training
    records = _Records()
    gpubridge.add_observer(records)
    bridge_calls = []
    for name in ("all_reduce", "broadcast"):
        plain = getattr(transport.GlooTransport, name)

        def counted(self, *args, _plain=plain, _name=name, **kwargs):
            bridge_calls.append(_name)
            return _plain(self, *args, **kwargs)

        setattr(transport.GlooTransport, name, counted)
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("replicate", "shard"))
    shard, replicate = mesh.get_group("shard"), mesh.get_group("replicate")
    assert _sub(shard)._topology().layout.members == ((0, 1) if rank < 2 else (2, 3))
    assert _sub(replicate)._topology().layout.members == ((0, 2) if rank % 2 == 0 else (1, 3))
    model = _integer_model()
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        model(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().clone() for p in model.parameters()]
    gpubridge.remove_observer(records)
    calls = [(r.op, r.policy, tuple(p.name for p in r.phases)) for r in records.records]
    torch.save({"params": full, "calls": calls, "bridge_calls": bridge_calls},
               f"{out}.{rank}")


def test_hsdp_shards_within_islands_and_replicates_across_the_bridge(tmp_path):
    out = tmp_path / "hsdp"
    run_backend(TWO_TWO, partial(_hsdp, out=str(out)), tmp_path)
    # One process on the whole batch: the mean gradient over the 4 ranks' batches.
    model = _integer_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        loss = sum(model(_batch(r, step)).sum() for r in range(4)) / 4
        loss.backward()
        opt.step()
        opt.zero_grad()
    for rank in range(4):
        got = torch.load(f"{out}.{rank}")
        assert all(torch.equal(a, b.detach()) for a, b in zip(got["params"], model.parameters(),
                                                              strict=True)), rank
        calls = got["calls"]
        gathers = [c for c in calls if c[0] in ("all_gather_into_tensor", "reduce_scatter_tensor")]
        scatters = [c for c in gathers if c[0] == "reduce_scatter_tensor"]
        reduces = [c for c in calls if c[0] == "all_reduce"]
        # Every all-gather and reduce-scatter ran native-only, inside an island: no bridge.
        assert gathers and all(c[1] == "native-only" and "bridge" not in c[2] for c in gathers)
        # Each reduce-scattered gradient shard crossed vendors once, on the replicate group.
        assert scatters and len(reduces) == len(scatters) and len(reduces) % STEPS == 0
        assert all(c[1] == "reduce-bridge-broadcast" and "bridge" in c[2] for c in reduces)
        # Bridge calls: only on the leaders of the replicate groups (every rank is one
        # here: one rank per island), one per all_reduce, none for the rest.
        assert got["bridge_calls"] == ["all_reduce"] * len(reduces), rank


# A torch whose private group naming isn't what subgroup setup patches
# (topology._members_only): subgroups are refused where they are created, on
# their members, naming the torch version; groups of every rank still work.

def _changed_naming(plain):
    return lambda ranks, use_hashed_name, group_desc=None: plain(ranks, use_hashed_name)


NAMING_CHANGES = {
    "a parameter added": (lambda c10d: ("_process_group_name", _changed_naming(
        c10d._process_group_name)), r"takes \(ranks, use_hashed_name, group_desc=None\)"),
    "a parameter renamed": (lambda c10d: ("_process_group_name", lambda ranks, hashed: "0"),
                            r"takes \(ranks, hashed\), not \(ranks, use_hashed_name\)"),
    "keyword-only": (lambda c10d: ("_process_group_name", lambda ranks, *, use_hashed_name: "0"),
                     r"takes \(ranks, \*, use_hashed_name\)"),
    "gone": (lambda c10d: ("_process_group_name", None), "has no _process_group_name"),
    "new_group names groups otherwise": (
        lambda c10d: ("new_group", lambda ranks=None, use_local_synchronization=False: None),
        "doesn't name groups with _process_group_name"),
    "new_group without local synchronization": (
        lambda c10d: ("new_group", lambda ranks=None, timeout=None, backend=None: None),
        "has no use_local_synchronization argument"),
}


def _different_group_counts_before_setup(rank, world):
    # torch 2.9.1's DeviceMesh creates a group for a (2, 1) mesh's replicate dimension
    # where 2.14.1's reuses the default group, so the two ranks reach gpubridge's first
    # collective having created different numbers of groups. gpubridge's world setup
    # must still meet: its groups can't take names from torch's counter of each
    # process's groups (they did, and HSDP hung across those two releases).
    if rank == world - 1:
        dist.new_group(list(range(world)))
    x = torch.ones(3)
    dist.all_reduce(x)
    assert torch.equal(x, torch.full((3,), float(world)))


def test_the_world_setup_meets_after_ranks_created_different_numbers_of_groups(tmp_path):
    run_backend(["nvidia", "amd"], _different_group_counts_before_setup, tmp_path)


def _repeated_world_setup_inside_gloo(rank, world):
    for _ in range(2):
        group = dist.new_group(backend="gpubridge")
        x = torch.ones(3)
        dist.all_reduce(x, group=group)
        assert torch.equal(x, torch.full((3,), float(world)))
        dist.destroy_process_group(group)
        assert not gpubridge.is_initialized()


def test_world_setup_names_are_fresh_after_its_group_is_destroyed(tmp_path):
    # Destroying a Python backend inside a Gloo world keeps its internal Gloo
    # groups registered. The next setup must not reuse their names or link keys.
    run_backend(["nvidia", "amd"], _repeated_world_setup_inside_gloo, tmp_path, backend="gloo")


def test_world_setups_use_fresh_transport_store_names():
    from gpubridge.transport import scoped_name

    names = []
    for _ in range(2):  # as setup() and gpubridge.init() name each world setup
        with topology.setup_names(f"gpubridge:world{next(torch_backend._world_setups)}"):
            names.append(scoped_name([0, 1]))
    assert all(names) and names[0] != names[1]
    assert scoped_name([0, 1]) is None


def test_transport_names_do_not_depend_on_members_only_torch_group_counts():
    from gpubridge.transport import scoped_name

    names = []
    for member in (True, False):
        with topology._members_only("gpubridge:test"):
            if member:  # torch never names a locally synchronized group on non-members
                dist.distributed_c10d._process_group_name([0, 1], True)
            names.append([scoped_name([0, 1]), scoped_name([0, 1])])
    assert names[0] == names[1]
    assert names[0][0] != names[0][1]


def _world_setup_with_changed_naming(rank, world):
    c10d = dist.distributed_c10d
    plain = c10d._process_group_name
    c10d._process_group_name = _changed_naming(plain)
    try:
        x = torch.ones(3)
        with pytest.warns(RuntimeWarning, match="every rank must have created the same number"):
            dist.all_reduce(x)
        assert torch.equal(x, torch.full((3,), float(world)))
        # The fallback must restore the caller's naming function too.
        assert c10d._process_group_name is not plain
    finally:
        c10d._process_group_name = plain


def test_world_setup_warns_and_uses_torch_naming_when_its_private_api_changes(tmp_path):
    run_backend(["nvidia", "amd"], _world_setup_with_changed_naming, tmp_path)


def test_this_torch_has_the_private_group_naming_subgroups_patch():
    assert topology._subgroup_naming_problem() is None, torch.__version__


@pytest.mark.parametrize("change", list(NAMING_CHANGES))
def test_subgroups_are_refused_if_torch_changed_its_private_group_naming(change, monkeypatch):
    patch, why = NAMING_CHANGES[change]
    c10d = dist.distributed_c10d
    name, replacement = patch(c10d)
    owner = dist if name == "new_group" else c10d
    if replacement is None:
        monkeypatch.delattr(owner, name)
    else:
        monkeypatch.setattr(owner, name, replacement)
    refused = rf"can't create subgroups on torch {re.escape(str(torch.__version__))}: .*{why}"

    def options(ranks, size):
        return SimpleNamespace(global_ranks_in_group=ranks, group_size=size, group_rank=0,
                               group_id="1", timeout=TIMEOUT, store=dist.HashStore())

    world = dist.get_world_size() if dist.is_initialized() else 4
    with pytest.raises(RuntimeError, match=refused):
        torch_backend.create(options([0, 2], 2))
    with pytest.raises(RuntimeError, match=refused):  # and nothing is patched
        with topology._members_only("gpubridge:1"):
            pass
    assert getattr(owner, name, None) is replacement
    assert torch_backend.create(options([], world)).size() == world  # the world group
    assert torch_backend.create(options(list(range(world)), world))._members == ()


def _naming_changed_mid_job(rank, world):
    t = rank_tensor((3,), torch.float32, rank)
    dist.all_reduce(t)  # gpubridge is set up
    c10d = dist.distributed_c10d
    plain = c10d._process_group_name
    c10d._process_group_name = _changed_naming(plain)
    try:
        if rank in (0, 2):
            with pytest.raises(RuntimeError, match="can't create subgroups on torch "
                               + re.escape(str(torch.__version__))):
                dist.new_group([0, 2], backend="gpubridge")
        else:
            assert dist.new_group([0, 2], backend="gpubridge") == dist.GroupMember.NON_GROUP_MEMBER
    finally:
        c10d._process_group_name = plain
    dist.all_reduce(t)  # the job goes on, on every rank
    assert torch.equal(t, expected_sum((3,), torch.float32, world) * world)


def test_a_refused_subgroup_fails_its_members_where_created_and_the_job_goes_on(tmp_path):
    run_backend(TWO_TWO, _naming_changed_mid_job, tmp_path)


# ---- registration and pure logic -----------------------------------------------------------

def test_the_backend_is_registered_by_import_gpubridge():
    assert "gpubridge" in dist.Backend.backend_list
    torch_backend.register()  # again: nothing happens
    assert dist.Backend.backend_list.count("gpubridge") == 1


def test_import_torch_never_runs_gpubridge():
    # No eager torch.backends entry point: anything that goes wrong in one breaks
    # `import torch` itself. Only the lazy torch.distributed.backends group, if any.
    code = ("import sys, torch, torch.distributed as dist; "
            "print('gpubridge' in dist.Backend.backend_list, "
            "any(m.split('.')[0] == 'gpubridge' for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["False", "False"]
    pyproject = (REPO / "pyproject.toml").read_text()
    assert '[project.entry-points."torch.backends"]' not in pyproject
    assert '[project.entry-points."torch.distributed.backends"]' in pyproject


def _lazy_entry_point() -> bool:
    """Installed gpubridge, on a torch with the lazy group (2.14 and later)."""
    return hasattr(dist.Backend, "_ensure_backend_registered") and any(
        ep.name == "gpubridge"
        for ep in importlib.metadata.entry_points(group="torch.distributed.backends"))


@pytest.mark.skipif(not _lazy_entry_point(), reason="needs torch 2.14+ and gpubridge installed "
                    "with its entry point (not a source tree)")
def test_newer_torch_registers_it_when_the_name_is_first_asked_for(tmp_path):
    asked = ("import torch.distributed as dist; "
             "print('gpubridge' in dist.Backend.backend_list, dist.is_backend_available('gpubridge'), "
             "'gpubridge' in dist.Backend.backend_list)")
    out = subprocess.run([sys.executable, "-c", asked], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["False", "True", "True"]
    # A script whose only gpubridge line is the backend name.
    script = ("import torch.distributed as dist; "
              f"dist.init_process_group('gpubridge', init_method='file://{tmp_path / 'init'}', "
              "rank=0, world_size=1); print(dist.get_backend()); dist.destroy_process_group()")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["gpubridge"]


def test_cpu_tensors_go_to_gloo_only_on_gpu_jobs():
    gpu = SimpleNamespace(device=torch.device("cuda", 0))
    cpu = SimpleNamespace(device=torch.device("cpu"))
    assert torch_backend._cpu_path(gpu, [torch.ones(1)])
    assert not torch_backend._cpu_path(cpu, [torch.ones(1)])  # simulation: CPU is gpubridge's
    assert not torch_backend._cpu_path(gpu, [torch.ones(1), torch.ones(1, device="meta")])
    with pytest.raises(NotImplementedError, match="gather of GPU tensors"):
        torch_backend._cpu_only("gather", [torch.ones(1, device="meta")])


def test_shutdown_never_raises_even_if_releasing_gpubridge_fails(monkeypatch):
    def fails(drain=True):
        raise RuntimeError("boom")

    pg = torch_backend.GpubridgeProcessGroup(0, 1, "test", None)
    monkeypatch.setattr(torch_backend, "_owner", pg)
    monkeypatch.setattr(gpubridge.topology, "is_initialized", lambda: True)
    monkeypatch.setattr(gpubridge, "_release", fails)
    with pytest.warns(RuntimeWarning, match="releasing gpubridge's state failed: boom"):
        pg.shutdown()
    assert torch_backend._owner is None
    pg.shutdown()  # once released, again does nothing


# ---- on GPUs (written; skipped without one) ------------------------------------------------

def _gpu_device_id(rank, init_file, world):
    torch.cuda.set_device(rank)
    os.environ["GPUBRIDGE_SPLIT_TEST"] = "half"
    dist.init_process_group(backend="gpubridge", init_method=f"file://{init_file}", rank=rank,
                            world_size=world, timeout=TIMEOUT,
                            device_id=torch.device("cuda", rank))
    t = torch.full((4,), float(rank + 1), device="cuda")
    dist.all_reduce(t)
    assert t.cpu().tolist() == [float(sum(range(1, world + 1)))] * 4
    gc.collect()
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_gpu_init_process_group_with_a_device_id(tmp_path):
    mp.spawn(_gpu_device_id, args=(str(tmp_path / "rendezvous"), 2), nprocs=2)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_gpu_ddp_on_two_gpus_split_into_two_islands(tmp_path):
    run_backend(["nvidia", "nvidia"], _gpu_ddp, tmp_path, env={-1: {
        VENDOR_ENV: "", "GPUBRIDGE_SPLIT_TEST": "half"}})


def _gpu_ddp(rank, world):
    torch.cuda.set_device(rank)
    model = _integer_model().cuda()
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ddp(_batch(rank, 0).cuda()).sum().backward()
    for p in model.parameters():
        g = p.grad.clone()
        dist.all_reduce(g, op=dist.ReduceOp.MAX)
        assert torch.equal(g, p.grad)  # identical on every rank
    flags = torch.ones(1)  # a CPU tensor on a GPU job: Gloo carries it
    dist.all_reduce(flags)
    assert flags.item() == world
