"""Collective observers: records match across ranks, never communicate, cost nothing unwatched."""

import functools
import threading
import time
import warnings

import pytest
import torch
from _harness import TIMEOUT, gather, run_ranks

import gpubridge
from gpubridge import observe
from gpubridge.policies import FlatGloo

MIXED = ["nvidia", "nvidia", "amd"]


class Keep(gpubridge.CollectiveObserver):
    """Stores every record; remembers why it was detached."""

    def __init__(self):
        self.records = []
        self.detached = None

    def on_collective(self, record):
        self.records.append(record)

    def on_detach(self, error):
        self.detached = error


def run_some_collectives(topology):
    gpubridge.all_reduce(torch.ones(8))
    gpubridge.broadcast(torch.ones(3, dtype=torch.float16), src=topology.world_size - 1)
    gpubridge.barrier()
    gpubridge.all_reduce(torch.ones(2, dtype=torch.int64))


# ---- what a record says ---------------------------------------------------------------

def _check_records(topology):
    keep = Keep()
    gpubridge.add_observer(keep)
    run_some_collectives(topology)
    records = keep.records
    assert all(r.ready() for r in records)
    summary = [(r.seq, r.op, r.nbytes, str(r.dtype), r.src, r.policy) for r in records]
    assert summary == [
        (0, "all_reduce", 32, "torch.float32", None, "reduce-bridge-broadcast"),
        (1, "broadcast", 6, "torch.float16", 2, "reduce-bridge-broadcast"),
        (2, "barrier", 0, "None", None, None),
        (3, "all_reduce", 16, "torch.int64", None, "reduce-bridge-broadcast"),
    ]
    assert len(set(map(tuple, gather(summary)))) == 1  # identical on every rank

    leader, src_island = topology.is_leader, topology.island == topology.layout.island_of(2)
    phases = [tuple(p.name for p in r.phases) for r in records]
    reduce_phases = ("island-reduce", "bridge", "island-broadcast") if leader else (
        "island-reduce", "island-broadcast")
    if src_island:
        broadcast_phases = ("island-broadcast", "bridge") if leader else ("island-broadcast",)
    else:
        broadcast_phases = ("bridge", "island-broadcast") if leader else ("island-broadcast",)
    assert phases == [reduce_phases, broadcast_phases, (), reduce_phases]

    for record in records:
        total = record.elapsed_ms()
        assert total >= 0
        assert sum(p.elapsed_ms() for p in record.phases) <= total + 1e-3
    assert (records[0].phase("bridge") is not None) == leader
    assert records[0].phase("no-such-phase") is None


def test_records_match_across_ranks_and_show_each_ranks_phases(tmp_path):
    run_ranks(MIXED, _check_records, tmp_path)


def _check_late_attach(topology):
    keep = Keep()
    for step in range(4):
        if step == topology.rank:  # each rank starts watching at a different time
            gpubridge.add_observer(keep)
        gpubridge.all_reduce(torch.ones(4))
    assert [r.seq for r in keep.records] == list(range(topology.rank, 4))


def test_sequence_numbers_match_even_when_ranks_attach_at_different_times(tmp_path):
    run_ranks(MIXED, _check_late_attach, tmp_path)


def _check_rejected_calls(topology):
    keep = Keep()
    gpubridge.add_observer(keep)
    gpubridge.all_reduce(torch.ones(2))
    with pytest.raises(ValueError):
        gpubridge.all_reduce(torch.ones(2, dtype=torch.bool))  # rejected on every rank
    gpubridge.all_reduce(torch.ones(2))
    assert [r.seq for r in keep.records] == [0, 1]


def test_rejected_calls_do_not_use_a_sequence_number(tmp_path):
    run_ranks(MIXED, _check_rejected_calls, tmp_path)


# ---- timing -----------------------------------------------------------------------------

NAP_S = 0.05


class Napping(FlatGloo):
    """A policy with one marked step that takes at least NAP_S."""

    name = "test-napping"

    def all_reduce(self, tensor, topology):
        with observe.phase("nap"):
            time.sleep(NAP_S)
        super().all_reduce(tensor, topology)


def _check_timing(topology):
    keep = Keep()
    gpubridge.add_observer(keep)
    gpubridge.all_reduce(torch.ones(4))
    (record,) = keep.records
    nap = record.phase("nap")
    assert [p.name for p in record.phases] == ["nap"]
    assert record.policy == "test-napping"
    assert NAP_S * 1e3 * 0.9 <= nap.elapsed_ms() <= record.elapsed_ms()
    assert observe.elapsed_ms(record.start, record.end) == record.elapsed_ms()


def test_marks_time_the_call_and_each_marked_phase(tmp_path):
    run_ranks(MIXED, _check_timing, tmp_path, init_kwargs={"policy": Napping})


# ---- observers can't change what any rank communicates ---------------------------------

class FailsOnThird(Keep):
    def on_collective(self, record):
        super().on_collective(record)
        if len(self.records) == 3:
            raise ZeroDivisionError("observer bug")


def _check_failing_observer(topology):
    keep = FailsOnThird() if topology.rank == 1 else Keep()
    gpubridge.add_observer(keep)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(8):  # carries on past the failure on rank 1
            gpubridge.all_reduce(torch.ones(4))
    if topology.rank == 1:
        assert len(keep.records) == 3
        assert isinstance(keep.detached, ZeroDivisionError)
        assert any("detached collective observer" in str(w.message) for w in caught)
    else:
        assert len(keep.records) == 8 and keep.detached is None
        assert not caught
    tensor = torch.ones(2)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.full((2,), float(topology.world_size)))


def test_an_observer_that_raises_is_detached_on_its_rank_only(tmp_path):
    run_ranks(MIXED, _check_failing_observer, tmp_path)


class Talks(Keep):
    def on_collective(self, record):
        super().on_collective(record)
        gpubridge.all_reduce(torch.ones(1))


def _check_talking_observer(topology):
    keep = Keep()
    gpubridge.add_observer(keep)
    talker = Talks()
    if topology.rank == 0:
        gpubridge.add_observer(talker)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(3):
            gpubridge.all_reduce(torch.ones(4))
    if topology.rank == 0:
        assert isinstance(talker.detached, RuntimeError)
        assert "must not communicate" in str(talker.detached)
    # The refused call took no sequence number, so ranks still agree.
    assert [r.seq for r in keep.records] == [0, 1, 2]


def test_an_observer_that_communicates_is_refused_and_detached(tmp_path):
    run_ranks(MIXED, _check_talking_observer, tmp_path)


def _check_lifecycle(topology):
    with pytest.raises(TypeError):
        gpubridge.add_observer(lambda record: None)
    keep = Keep()
    gpubridge.add_observer(keep)
    gpubridge.add_observer(keep)  # attaching twice delivers once
    gpubridge.all_reduce(torch.ones(1))
    gpubridge.remove_observer(keep)
    gpubridge.remove_observer(keep)  # removing twice is fine
    gpubridge.all_reduce(torch.ones(1))
    assert [r.seq for r in keep.records] == [0]
    gpubridge.add_observer(keep)
    gpubridge.destroy()
    assert observe._observers == [] and observe._seq == 0


def test_add_remove_and_destroy(tmp_path):
    run_ranks(MIXED, _check_lifecycle, tmp_path)


def test_add_observer_needs_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        gpubridge.add_observer(Keep())


# ---- in_flight: the collective each thread is inside right now ----------------------------

TWO_TWO = ["nvidia", "nvidia", "amd", "amd"]
LATE = 2  # the rank that reaches all_reduce #1 late


def _check_in_flight_while_a_rank_is_late(topology, watched):
    if watched:
        gpubridge.add_observer(Keep())
    tensor = torch.ones(4)
    gpubridge.all_reduce(tensor)  # #0
    assert gpubridge.in_flight() == []
    seen = []
    watcher = threading.Thread(target=lambda: (time.sleep(1.0), seen.append(gpubridge.in_flight())))
    watcher.start()
    if topology.rank == LATE:
        time.sleep(2.0)
    gpubridge.all_reduce(tensor)  # #1
    watcher.join()
    assert gpubridge.in_flight() == []
    (snapshot,) = seen
    if topology.rank == LATE:
        assert snapshot == []  # still outside gpubridge
        return
    (flight,) = snapshot
    assert (flight.seq, flight.op, flight.policy, flight.thread) == (
        1, "all_reduce", "reduce-bridge-broadcast", "MainThread")
    assert 0.5 < flight.seconds < 2.0
    # The nvidia island reduced and its leader waits on the bridge, the other
    # nvidia rank for the leader's broadcast; amd can't reduce without rank 2.
    expected = {0: ("bridge", ("island-reduce",)), 1: ("island-broadcast", ("island-reduce",)),
                3: ("island-reduce", ())}
    assert (flight.phase, flight.phases_done) == expected[topology.rank]


@pytest.mark.parametrize("watched", [False, True], ids=["no observers", "observed"])
def test_in_flight_shows_each_ranks_collective_and_phase(watched, tmp_path):
    run_ranks(TWO_TWO, functools.partial(_check_in_flight_while_a_rank_is_late, watched=watched),
              tmp_path)


def _check_in_flight_async(topology):
    tensor = torch.ones(4)
    if topology.rank == 1:
        time.sleep(1.0)
        gpubridge.all_reduce(tensor)
        return
    work = gpubridge.all_reduce(tensor, async_op=True)
    deadline = time.monotonic() + 5.0
    while not (flights := gpubridge.in_flight()) and time.monotonic() < deadline:
        time.sleep(0.01)
    (flight,) = flights
    assert (flight.seq, flight.op, flight.thread, flight.phase) == (
        0, "all_reduce", "gpubridge-async", "bridge")
    work.wait()
    assert gpubridge.in_flight() == []


def test_in_flight_shows_async_collectives_on_the_worker_thread(tmp_path):
    run_ranks(["nvidia", "amd"], _check_in_flight_async, tmp_path)


class NestedThenFails(FlatGloo):
    """flat-gloo that marks nested phases, looks at in_flight inside them, then fails."""

    name = "test-nested-then-fails"
    seen: list = []

    def all_reduce(self, tensor, topology, op=None):
        with observe.phase("outer"):
            with observe.phase("inner"):
                NestedThenFails.seen.append(gpubridge.in_flight())
            NestedThenFails.seen.append(gpubridge.in_flight())
            raise RuntimeError("failed inside the policy")


def _check_in_flight_nested_and_failing(topology):
    with pytest.raises(RuntimeError, match="failed inside the policy"):
        gpubridge.all_reduce(torch.ones(4))
    inner, outer = (snapshot[0] for snapshot in NestedThenFails.seen)
    assert (inner.phase, inner.phases_done) == ("inner", ())
    assert (outer.phase, outer.phases_done) == ("outer", ("inner",))
    assert inner.policy == "test-nested-then-fails"
    assert gpubridge.in_flight() == []  # a failed collective leaves no entry behind
    gpubridge.barrier()
    assert gpubridge.in_flight() == []


def test_in_flight_tracks_nested_phases_and_forgets_failed_calls(tmp_path):
    run_ranks(["nvidia"], _check_in_flight_nested_and_failing, tmp_path,
              init_kwargs={"policy": NestedThenFails})


def test_in_flight_is_empty_without_init():
    assert gpubridge.in_flight() == []


# ---- no GPU syncs ------------------------------------------------------------------------

def forbidden(*args, **kwargs):
    """Stands in for torch.cuda functions and classes; any use fails the test."""
    raise AssertionError("touched torch.cuda")


class FakeEvent:
    """A CUDA event that logs what is done with it."""

    calls: list = []

    def __init__(self, **kwargs):
        FakeEvent.calls.append(("create", kwargs))

    def record(self, stream=None):
        FakeEvent.calls.append(("record", stream))

    def query(self):
        FakeEvent.calls.append(("query", None))
        return True

    def elapsed_time(self, other):
        FakeEvent.calls.append(("elapsed_time", None))
        return 1.0

    def synchronize(self):
        raise AssertionError("synchronized an event")

    def wait(self, stream=None):
        raise AssertionError("made a stream wait")


def _check_no_cuda(topology):
    # Pretend this rank times on a GPU, then forbid every CUDA call.
    observe._event_device = lambda topology: torch.device("cuda", 0)
    for name in ("Event", "Stream", "current_stream", "synchronize"):
        setattr(torch.cuda, name, forbidden)

    # No observers: the whole path runs without creating a single event.
    for size in (4, 1024):
        gpubridge.all_reduce(torch.ones(size))
        gpubridge.broadcast(torch.ones(size), src=0)
    gpubridge.barrier()

    # With an observer, collectives only create and record events: no sync,
    # no query, no elapsed_time. Reading the record afterwards may query.
    torch.cuda.Event = FakeEvent
    torch.cuda.current_stream = lambda device: f"stream:{device}"
    keep = Keep()
    gpubridge.add_observer(keep)
    run_some_collectives(topology)
    kinds = {kind for kind, _ in FakeEvent.calls}
    assert kinds == {"create", "record"}
    assert all(args == {"enable_timing": True} for kind, args in FakeEvent.calls
               if kind == "create")
    assert {arg for kind, arg in FakeEvent.calls if kind == "record"} == {"stream:cuda:0"}
    assert all(r.ready() for r in keep.records)
    keep.records[0].elapsed_ms()


@pytest.mark.parametrize(
    ("vendors", "policy"),
    [(MIXED, "auto"), (MIXED, "flat-gloo"), (["amd", "amd"], "native-only")],
)
def test_no_observer_path_touches_no_cuda_and_observers_never_sync(vendors, policy, tmp_path):
    run_ranks(vendors, _check_no_cuda, tmp_path, init_kwargs={"policy": policy})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_no_sync_on_a_real_gpu(tmp_path):
    """One rank, native NCCL/RCCL: neither path may trigger a CUDA sync."""
    gpubridge.init(init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1,
                   timeout=TIMEOUT)
    try:
        tensor = torch.ones(1 << 20, device=gpubridge.get_topology().device)
        gpubridge.all_reduce(tensor)  # warm up: creates the communicator
        torch.cuda.synchronize()
        keep = Keep()
        torch.cuda.set_sync_debug_mode("error")
        try:
            gpubridge.all_reduce(tensor)
            gpubridge.add_observer(keep)
            gpubridge.all_reduce(tensor)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        assert keep.records[0].ready() and keep.records[0].elapsed_ms() > 0
    finally:
        gpubridge.destroy()
