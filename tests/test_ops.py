"""Reductions, all_gather_into_tensor, reduce_scatter_tensor and async_op, in every layout.

Every result is checked bit for bit against FlatGloo, the reference policy, and
against the answer computed locally from every rank's input.
"""

import warnings
from functools import partial

import pytest
import torch
from _harness import expected_sum, gather, rank_tensor, run_ranks  # noqa: F401 - gather: others import it from here
from _transports import STORE_ENV, StoreTransport
from test_policies import BUILTIN, LAYOUTS, layout_of

import gpubridge
from gpubridge import ReduceOp
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import SPLIT_TEST_ENV
from gpubridge.policies import AUTO, FlatGloo, choose_policy, resolve_policy
from gpubridge.work import Work

FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
OP_NAMES = {ReduceOp.SUM: "sum", ReduceOp.AVG: "avg", ReduceOp.MAX: "max", ReduceOp.MIN: "min"}


def cases():
    """Every (layout, policy) pair where the policy applies, plus auto everywhere."""
    for layout_id, (vendors, split) in LAYOUTS.items():
        layout = layout_of(vendors, split)
        names = [AUTO, *sorted(n for n, cls in BUILTIN.items() if cls.applies_to(layout))]
        for name in names:
            expected = choose_policy(resolve_policy(name), layout).name
            yield pytest.param(vendors, split, name, expected, id=f"{layout_id}-{name}")


def expected_reduce(shape, dtype, world, op):
    inputs = [rank_tensor(shape, dtype, r) for r in range(world)]
    if op == ReduceOp.MAX:
        return torch.stack(inputs).amax(0)
    if op == ReduceOp.MIN:
        return torch.stack(inputs).amin(0)
    total = expected_sum(shape, dtype, world)
    return total.div(world) if op == ReduceOp.AVG else total


def reference_all_reduce(tensor, op, topology):
    """What FlatGloo gives, with AVG done the way gpubridge defines it."""
    base = ReduceOp.SUM if op == ReduceOp.AVG else op
    FlatGloo().all_reduce(tensor, topology, op=base)
    if op == ReduceOp.AVG:
        tensor.div_(topology.world_size)


def _check_everything(topology, expected_policy):
    # On CPU, staging returns the tensor itself, so the copies GPUs need would
    # never run. Force real copies so they do.
    gpubridge.policies._stage_to_host = torch.Tensor.clone
    assert topology.policy_name == expected_policy
    rank, world = topology.rank, topology.world_size
    reference = FlatGloo()

    for dtype in SUPPORTED_DTYPES:
        ops = [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN]
        if dtype in FLOATS:
            ops.append(ReduceOp.AVG)
        for op in ops:
            tensor = rank_tensor((3, 5), dtype, rank)
            want = tensor.clone()
            gpubridge.all_reduce(tensor, op=op)
            reference_all_reduce(want, op, topology)
            assert torch.equal(tensor, want), (dtype, OP_NAMES[op])
            assert torch.equal(tensor, expected_reduce((3, 5), dtype, world, op)), (dtype, op)

        # all_gather_into_tensor: rows in global rank order, input untouched.
        piece = rank_tensor((2, 3), dtype, rank)
        before = piece.clone()
        output = torch.full((world * 2, 3), -1, dtype=dtype)
        gpubridge.all_gather_into_tensor(output, piece)
        want = torch.empty_like(output)
        reference.all_gather_into_tensor(want, piece, topology)
        assert torch.equal(output, want), dtype
        assert torch.equal(output, torch.cat([rank_tensor((2, 3), dtype, r)
                                              for r in range(world)]))
        assert torch.equal(piece, before)

        # reduce_scatter_tensor: this rank's slice of the sum (or average), input untouched.
        for op in [ReduceOp.SUM] + ([ReduceOp.AVG] if dtype in FLOATS else []):
            full = rank_tensor((world * 2, 3), dtype, rank)
            before = full.clone()
            output = torch.full((2, 3), -1, dtype=dtype)
            gpubridge.reduce_scatter_tensor(output, full, op=op)
            want = torch.empty_like(output)
            reference.reduce_scatter_tensor(want, full, topology)
            if op == ReduceOp.AVG:
                want.div_(world)
            assert torch.equal(output, want), (dtype, OP_NAMES[op])
            total = expected_reduce((world * 2, 3), dtype, world, op)
            assert torch.equal(output, total[rank * 2:(rank + 1) * 2]), (dtype, op)
            assert torch.equal(full, before)

    # A 1-D tensor that doesn't split evenly by island, and a 0-d input to all_gather.
    scalar = torch.tensor(float(rank + 1))
    gathered = torch.empty(world)
    gpubridge.all_gather_into_tensor(gathered, scalar)
    assert gathered.tolist() == [float(r + 1) for r in range(world)]

    _check_async(topology)


def _check_async(topology):
    rank, world = topology.rank, topology.world_size
    keep = Keep()
    gpubridge.add_observer(keep)
    a = rank_tensor((4,), torch.float32, rank)
    b = rank_tensor((5,), torch.int64, rank)
    out = torch.empty(world * 3)
    work_a = gpubridge.all_reduce(a, op=ReduceOp.MAX, async_op=True)
    work_b = gpubridge.broadcast(b, src=world - 1, async_op=True)
    work_c = gpubridge.all_gather_into_tensor(out, torch.full((3,), float(rank)), async_op=True)
    work_d = gpubridge.barrier(async_op=True)
    assert all(isinstance(w, gpubridge.Work) for w in (work_a, work_b, work_c, work_d))
    c = rank_tensor((2,), torch.float64, rank)
    gpubridge.all_reduce(c, op=ReduceOp.AVG)  # synchronous: runs after everything queued
    assert all(w.is_completed() for w in (work_a, work_b, work_c, work_d))
    for work in (work_a, work_b, work_c, work_d):
        assert work.wait() is True and work.exception() is None
    assert torch.equal(a, expected_reduce((4,), torch.float32, world, ReduceOp.MAX))
    assert torch.equal(b, rank_tensor((5,), torch.int64, world - 1))
    assert work_c.get_future().wait() is out
    assert out.tolist() == [float(r) for r in range(world) for _ in range(3)]
    assert work_d.get_future().wait() is None
    assert torch.equal(c, expected_reduce((2,), torch.float64, world, ReduceOp.AVG))

    # Records arrive in submission order, with the same sequence numbers everywhere.
    summary = [(r.seq, r.op, r.reduce_op) for r in keep.records]
    assert [op for _, op, _ in summary] == ["all_reduce", "broadcast", "all_gather_into_tensor",
                                            "barrier", "all_reduce"]
    assert [reduce_op for _, _, reduce_op in summary] == ["max", None, None, None, "avg"]
    assert len({tuple(s) for s in gather(summary)}) == 1


class Keep(gpubridge.CollectiveObserver):
    def __init__(self):
        self.records = []

    def on_collective(self, record):
        self.records.append(record)


@pytest.mark.parametrize(("vendors", "split", "policy", "expected"), list(cases()))
def test_every_op_matches_flat_gloo(vendors, split, policy, expected, tmp_path, monkeypatch):
    if split:
        monkeypatch.setenv(SPLIT_TEST_ENV, split)
    run_ranks(vendors, partial(_check_everything, expected_policy=expected), tmp_path,
              init_kwargs={"policy": policy})


# ---- calls that are rejected on every rank, before any communication -------------------

def _check_rejections(topology):
    world = topology.world_size
    ints = torch.int64
    rejected = [
        (ValueError, "PRODUCT is not supported yet",
         lambda a: gpubridge.all_reduce(torch.ones(2), op=ReduceOp.PRODUCT, async_op=a)),
        (ValueError, "AVG needs a floating-point tensor",
         lambda a: gpubridge.all_reduce(torch.ones(2, dtype=ints), op=ReduceOp.AVG, async_op=a)),
        (ValueError, "AVG needs a floating-point tensor",
         lambda a: gpubridge.reduce_scatter_tensor(torch.ones(1, dtype=ints),
                                                   torch.ones(world, dtype=ints),
                                                   op=ReduceOp.AVG, async_op=a)),
        (ValueError, "supports SUM and AVG",
         lambda a: gpubridge.reduce_scatter_tensor(torch.ones(1), torch.ones(world),
                                                   op=ReduceOp.MAX, async_op=a)),
        (ValueError, r"output must have world_size \(3\) times as many elements",
         lambda a: gpubridge.all_gather_into_tensor(torch.ones(4), torch.ones(2), async_op=a)),
        (ValueError, r"input must have world_size \(3\) times as many elements",
         lambda a: gpubridge.reduce_scatter_tensor(torch.ones(2), torch.ones(5), async_op=a)),
        (ValueError, "output is torch.float16 but input is torch.float32",
         lambda a: gpubridge.all_gather_into_tensor(torch.ones(world, dtype=torch.float16),
                                                    torch.ones(1), async_op=a)),
        (ValueError, "doesn't support torch.bool",
         lambda a: gpubridge.all_gather_into_tensor(torch.ones(world, dtype=torch.bool),
                                                    torch.ones(1, dtype=torch.bool), async_op=a)),
        (ValueError, "contiguous",
         lambda a: gpubridge.all_gather_into_tensor(torch.ones(2, world).t(), torch.ones(2),
                                                    async_op=a)),
        (ValueError, "src must be a rank",
         lambda a: gpubridge.broadcast(torch.ones(2), world, async_op=a)),
    ]
    for error, match, call in rejected:
        for async_op in (False, True):  # async calls are validated before they are queued
            with pytest.raises(error, match=match):
                call(async_op)
    # Nobody communicated, so every rank still agrees on what comes next.
    tensor = torch.ones(2)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.full((2,), float(world)))


def test_bad_calls_are_rejected_on_every_rank(tmp_path):
    run_ranks(["nvidia", "amd", "amd"], _check_rejections, tmp_path)


class SumOnly(FlatGloo):
    """A policy written before 0.2: SUM all_reduce and broadcast, nothing else."""

    name = "test-sum-only"
    reduce_ops = frozenset({"sum"})
    all_gather_into_tensor = gpubridge.CollectivePolicy.all_gather_into_tensor
    reduce_scatter_tensor = gpubridge.CollectivePolicy.reduce_scatter_tensor

    def all_reduce(self, tensor, topology):  # no op parameter, like old subclasses
        super().all_reduce(tensor, topology)


def _check_what_a_policy_offers(topology):
    world = topology.world_size
    with pytest.raises(NotImplementedError, match="supports only SUM"):
        gpubridge.all_reduce(torch.ones(2), op=ReduceOp.MAX)
    with pytest.raises(NotImplementedError,
                       match="'test-sum-only' doesn't implement all_gather_into_tensor. Policies "
                             "that do and fit this cluster: auto-tuned, flat-gloo, "
                             "pipelined-reduce-bridge-broadcast, reduce-bridge-broadcast"):
        gpubridge.all_gather_into_tensor(torch.ones(world), torch.ones(1))
    with pytest.raises(NotImplementedError, match="doesn't implement reduce_scatter_tensor"):
        gpubridge.reduce_scatter_tensor(torch.ones(1), torch.ones(world), async_op=True)
    tensor = torch.ones(3)
    gpubridge.all_reduce(tensor, op=ReduceOp.AVG)  # AVG is SUM then a division: works
    assert torch.equal(tensor, torch.ones(3))


def test_a_policy_offers_only_what_it_implements(tmp_path):
    run_ranks(["nvidia", "amd"], _check_what_a_policy_offers, tmp_path,
              init_kwargs={"policy": SumOnly})


def _check_sum_only_transport(topology):
    with pytest.raises(NotImplementedError,
                       match="bridge transport 'test-store' supports only SUM"):
        gpubridge.all_reduce(torch.ones(2), op=ReduceOp.MIN)
    tensor = rank_tensor((3,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((3,), torch.float32, topology.world_size))


def test_max_and_min_need_a_transport_that_lists_them(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    run_ranks(["nvidia", "nvidia", "amd"], _check_sum_only_transport, tmp_path,
              init_kwargs={"bridge": StoreTransport})


def test_new_collectives_need_init():
    with pytest.raises(RuntimeError, match="init"):
        gpubridge.all_gather_into_tensor(torch.ones(2), torch.ones(1))
    with pytest.raises(RuntimeError, match="init"):
        gpubridge.reduce_scatter_tensor(torch.ones(1), torch.ones(2), async_op=True)


# ---- async failures -------------------------------------------------------------------

class Exploding(FlatGloo):
    """Fails every all_reduce on every rank, before communicating."""

    name = "test-exploding"

    def all_reduce(self, tensor, topology, op=ReduceOp.SUM):
        raise ZeroDivisionError("policy bug")


def _check_async_failure(topology):
    work = gpubridge.all_reduce(torch.ones(2), async_op=True)
    with pytest.raises(RuntimeError, match="async all_reduce failed: policy bug") as info:
        work.wait()
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    assert isinstance(work.exception(), ZeroDivisionError)
    with pytest.raises(ZeroDivisionError):
        work.get_future().wait()
    # Every later collective refuses to start: the groups may be inconsistent.
    with pytest.raises(RuntimeError, match="an earlier async gpubridge collective failed"):
        gpubridge.broadcast(torch.ones(2), 0)
    with pytest.raises(RuntimeError, match="an earlier async gpubridge collective failed"):
        gpubridge.barrier(async_op=True)


def test_a_failed_async_collective_stops_later_ones(tmp_path):
    run_ranks(["nvidia", "amd"], _check_async_failure, tmp_path,
              init_kwargs={"policy": Exploding})


def test_is_completed_waits_for_the_gpu_work_too():
    """The worker finishes once it has queued the last kernels; is_completed() also asks their event."""

    class Event:
        passed = False

        def query(self):
            return self.passed

    work = Work("all_reduce", torch.device("cpu"), None)
    assert not work.is_completed()
    work._event = event = Event()
    work._finish(None)
    assert not work.is_completed()
    event.passed = True
    assert work.is_completed()


class TalksAsync(Keep):
    def on_collective(self, record):
        super().on_collective(record)
        gpubridge.all_reduce(torch.ones(1), async_op=True)


def _check_async_observer_cannot_communicate(topology):
    talker = TalksAsync()
    gpubridge.add_observer(talker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        work = gpubridge.all_reduce(torch.ones(2), async_op=True)
        work.wait()  # no deadlock: the callback's call is refused on the worker thread
    assert any("must not communicate" in str(w.message) for w in caught)
    tensor = torch.ones(2)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.full((2,), float(topology.world_size)))


def test_an_observer_of_an_async_collective_cannot_communicate(tmp_path):
    run_ranks(["nvidia", "amd"], _check_async_observer_cannot_communicate, tmp_path)


def _check_destroy_runs_the_queue(topology):
    tensors = [torch.full((3,), float(topology.rank)) for _ in range(5)]
    works = [gpubridge.all_reduce(t, async_op=True) for t in tensors]
    gpubridge.destroy()
    assert all(w.is_completed() and w.exception() is None for w in works)
    total = float(sum(range(topology.world_size)))
    assert all(t.tolist() == [total] * 3 for t in tensors)


def test_destroy_runs_queued_collectives_first(tmp_path):
    run_ranks(["nvidia", "amd", "nvidia"], _check_destroy_runs_the_queue, tmp_path)
