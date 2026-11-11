"""A job that doesn't ask for Gloo has no Gloo group (docs/RUNTIME_INVENTORY.md, section 6).

On a bridge other than Gloo, discovery and ``barrier()`` go through the job's c10d
store, the default group is gpubridge's own, and a one-rank island in CPU simulation
is a :class:`~gpubridge.torch_backend.SoloGroup`. So a job of one-rank islands holds
no Gloo group and none of Gloo's threads: checked here from inside each rank, against
torch's own table of groups everywhere and against the kernel's thread names on Linux
(``/proc/self/task/*/comm``), with a ``bridge="gloo"`` job as the control that shows
the check can see them.
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _harness import (
    GLOO_THREADS,
    TIMEOUT,
    assert_no_gloo,
    expected_sum,
    gloo_groups,
    rank_tensor,
    run_failing_init,
    run_ranks,
    thread_names,
)

import gpubridge
from gpubridge import multitcp, rendezvous, torch_backend
from gpubridge.config import BRIDGE_ENV, SOLO_BACKEND, VENDOR_ENV
from gpubridge.multitcp.transport import IFNAME_ENV, STREAMS_ENV
from gpubridge.policies import FlatGloo

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"the Rust TCP bridge is unavailable: {UNAVAILABLE}")

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
LOCAL = {IFNAME_ENV: LOOPBACK, STREAMS_ENV: "2"}
#: Four one-rank islands: every rank a bridge leader, no island of several ranks.
FOUR = (["nvidia"] * 4, ["nvidia", "amd", "third", "fourth"])


def _islands(labels, env, rank):
    """Before init: the settings, and (simulation has two vendors only) more islands
    than vendors, by labelling the ranks; a subgroup's layout takes its members' labels."""
    os.environ.update(env)
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda vendors, ranks=(): original(
            [labels[r] for r in ranks] if ranks else labels, ranks)


def _every_collective(topology) -> None:
    """Each kind of collective once, sync and async, checked against the local answer."""
    rank, world = topology.rank, topology.world_size
    for dtype in (torch.float32, torch.int64, torch.bfloat16):
        tensor = rank_tensor((7,), dtype, rank)
        gpubridge.all_reduce(tensor)
        assert torch.equal(tensor, expected_sum((7,), dtype, world))
    tensor = rank_tensor((5,), torch.float64, rank)
    gpubridge.all_reduce(tensor, op=gpubridge.ReduceOp.MAX)
    assert torch.equal(tensor, rank_tensor((5,), torch.float64, world - 1))
    tensor = rank_tensor((3,), torch.int32, rank)
    gpubridge.broadcast(tensor, src=world - 1)
    assert torch.equal(tensor, rank_tensor((3,), torch.int32, world - 1))
    out = torch.empty(world * 2)
    gpubridge.all_gather_into_tensor(out, rank_tensor((2,), torch.float32, rank))
    assert torch.equal(out, torch.cat([rank_tensor((2,), torch.float32, r) for r in range(world)]))
    got = torch.empty(3)
    gpubridge.reduce_scatter_tensor(got, rank_tensor((world, 3), torch.float32, rank))
    assert torch.equal(got, expected_sum((world, 3), torch.float32, world)[rank])
    work = gpubridge.all_reduce(rank_tensor((9,), torch.float32, rank), async_op=True)
    gpubridge.barrier()
    assert work.wait() is True
    assert torch.equal(work.get_future().value(), expected_sum((9,), torch.float32, world))


def _check_no_gloo(topology, leaders):
    assert topology.gloo_free and topology.world_group is None
    assert dist.get_backend() == torch_backend.NAME
    assert dist.get_backend(topology.island_group) == SOLO_BACKEND
    assert isinstance(topology.island_group, torch_backend.SoloGroup)
    assert len(topology.layout.islands) == leaders and topology.is_leader
    _every_collective(topology)
    # torch.distributed on the default group runs through gpubridge too.
    everyone: list = [None] * topology.world_size
    dist.all_gather_object(everyone, {"rank": topology.rank})
    assert everyone == [{"rank": r} for r in range(topology.world_size)]
    tensor = rank_tensor((4,), torch.float32, topology.rank)
    dist.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((4,), torch.float32, topology.world_size))
    dist.barrier()
    assert_no_gloo()  # the last thing a rank does before it returns


@needs_library
@pytest.mark.parametrize(("vendors", "labels"), [(["nvidia", "amd"], None), FOUR],
                         ids=["two-ranks", "four-ranks"])
def test_a_job_of_one_rank_islands_on_the_tcp_bridge_has_no_gloo_group_or_thread(
        vendors, labels, tmp_path):
    run_ranks(vendors, partial(_check_no_gloo, leaders=len(vendors)), tmp_path,
              before_init=partial(_islands, labels, LOCAL), init_kwargs={"bridge": "multitcp"})


#: torch names its Gloo threads from some release after 2.8.0 (2.8.0's are all called
#: after the process; 2.15.0 has pt_gloo_runloop and gloo_tcp_loop). The control below
#: insists on the names from 2.15, the release the exit abort was diagnosed on.
NAMES_GLOO_THREADS = tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 15)


def _check_gloo_is_seen(topology, seen):
    assert not topology.gloo_free and dist.get_backend() == "gloo"
    _every_collective(topology)
    # Its default group, its one-rank island and its bridge: all Gloo groups.
    assert len(gloo_groups()) == 3
    names = sorted(name for name in thread_names() if name in GLOO_THREADS)
    Path(seen, f"rank{topology.rank}").write_text(" ".join(names))
    if NAMES_GLOO_THREADS and Path("/proc/self/task").is_dir():
        assert "pt_gloo_runloop" in names and "gloo_tcp_loop" in names


def test_the_check_sees_gloo_in_a_job_that_asks_for_it(tmp_path):
    # The control: the same job on bridge="gloo" has the groups everywhere, and on
    # Linux, on a torch that names its threads, the threads the check looks for.
    seen = tmp_path / "seen"
    seen.mkdir()
    run_ranks(FOUR[0], partial(_check_gloo_is_seen, seen=str(seen)), tmp_path,
              before_init=partial(_islands, FOUR[1], {}), init_kwargs={"bridge": "gloo"})
    found = [(seen / f"rank{r}").read_text() for r in range(4)]
    if NAMES_GLOO_THREADS and Path("/proc/self/task").is_dir():
        assert all("pt_gloo_runloop" in names for names in found), found


def _check_single_island(topology):
    # One vendor, one rank: no bridge, a solo island, and still nothing of Gloo's.
    assert topology.bridge is None and topology.policy_name == "native-only"
    assert isinstance(topology.island_group, torch_backend.SoloGroup)
    tensor = torch.arange(6.0)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.arange(6.0))
    out = torch.empty(4)
    gpubridge.all_gather_into_tensor(out, torch.arange(4.0))
    assert torch.equal(out, torch.arange(4.0))
    gpubridge.barrier()
    assert_no_gloo()


def test_a_single_rank_job_has_no_gloo_group_or_thread(tmp_path):
    run_ranks(["amd"], _check_single_island, tmp_path, init_kwargs={"bridge": "multitcp"})


def _check_multi_rank_islands(topology):
    # Simulation's islands of several ranks are Gloo groups (a test device): the job
    # runs, on gpubridge's own default group, with the bridge on TCP.
    assert topology.gloo_free and dist.get_backend() == torch_backend.NAME
    assert dist.get_backend(topology.island_group) == "gloo"
    assert topology.bridge_group is None
    _every_collective(topology)
    assert len(gloo_groups()) == 1  # this rank's island, and nothing else


@needs_library
def test_islands_of_several_ranks_in_simulation_are_still_gloo_groups(tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check_multi_rank_islands, tmp_path,
              before_init=partial(_islands, None, LOCAL), init_kwargs={"bridge": "multitcp"})


def _check_flat_gloo_asks_for_gloo(topology):
    assert gloo_groups() == []
    tensor = rank_tensor((5,), torch.float32, topology.rank)
    FlatGloo().all_reduce(tensor, topology)  # the reference of the bit-for-bit tests
    assert torch.equal(tensor, expected_sum((5,), torch.float32, topology.world_size))
    assert len(gloo_groups()) == 1  # made once, at the first use
    FlatGloo().broadcast(tensor, 0, topology)
    assert len(gloo_groups()) == 1


@needs_library
def test_flat_gloo_makes_its_own_gloo_group_when_it_is_first_used(tmp_path):
    run_ranks(FOUR[0], _check_flat_gloo_asks_for_gloo, tmp_path,
              before_init=partial(_islands, FOUR[1], LOCAL), init_kwargs={"bridge": "multitcp"})


def _check_flat_gloo_policy(topology):
    assert topology.policy_name == "flat-gloo" and len(gloo_groups()) == 1  # made by init()
    _every_collective(topology)
    assert len(gloo_groups()) == 1


@needs_library
def test_the_flat_gloo_policy_creates_its_group_at_init(tmp_path):
    run_ranks(FOUR[0], _check_flat_gloo_policy, tmp_path,
              before_init=partial(_islands, FOUR[1], LOCAL),
              init_kwargs={"bridge": "multitcp", "policy": "flat-gloo"})


# ---- the barrier ------------------------------------------------------------------------

def _check_barrier_waits(topology, marks):
    path = Path(marks)
    if topology.rank == 0:
        time.sleep(0.5)
        (path / "late").write_text("arrived")
    gpubridge.barrier()
    assert (path / "late").exists(), "barrier() returned before the last rank reached it"
    for _ in range(20):  # each barrier retires the one before it
        gpubridge.barrier()
    work = gpubridge.barrier(async_op=True)
    assert work.wait() is True
    assert_no_gloo()


@needs_library
def test_barrier_returns_only_once_every_rank_has_reached_it(tmp_path):
    run_ranks(FOUR[0], partial(_check_barrier_waits, marks=str(tmp_path)), tmp_path,
              before_init=partial(_islands, FOUR[1], LOCAL), init_kwargs={"bridge": "multitcp"})


def test_the_store_barrier_times_out_naming_who_arrived(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    rendezvous.barrier(store, "b/0", 1, time.monotonic() + 5)  # alone: at once
    store.add("b/1", 1)  # one other rank has arrived, the third never does
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="2 of 3 ranks reached it"):
        rendezvous.barrier(store, "b/1", 3, time.monotonic() + 0.3)
    assert 0.25 < time.monotonic() - start < 5


def test_a_barrier_retires_the_previous_barriers_key(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    rendezvous.barrier(store, "b/0", 1, time.monotonic() + 5)
    assert store.check(["b/0"])
    rendezvous.barrier(store, "b/1", 1, time.monotonic() + 5, retire="b/0")
    assert store.check(["b/1"]) and not store.check(["b/0"])


# ---- discovery --------------------------------------------------------------------------

def test_discovery_names_the_ranks_that_never_reported(tmp_path, monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "nvidia")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    local, _ = gpubridge.topology.describe_local("multitcp")
    with pytest.raises(TimeoutError, match=r"nothing from ranks \[0, 2\] of 3"):
        gpubridge.topology.discover(store, "world0", 1, 3, local, timedelta(seconds=0.3))


def test_a_rank_that_cannot_describe_itself_fails_discovery_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "nvidia")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    local, _ = gpubridge.topology.describe_local("multitcp")
    good = json.dumps({"v": rendezvous.RECORD_VERSION, "peer": local.to_wire()})
    store.set("gpubridge/w/peer/0", good)
    with pytest.raises(RuntimeError, match="1 of 2 ranks reported problems.\n  rank 1: bad"):
        gpubridge.topology.discover(store, "w", 1, 2, None, TIMEOUT, problem="bad setting")
    with pytest.raises(RuntimeError, match="1 of 2 ranks reported problems.\n  rank 1: bad"):
        gpubridge.topology.discover(store, "w", 0, 2, local, TIMEOUT)  # the healthy rank too
    # A record from a gpubridge that writes another version is a problem too, not a guess.
    store.set("gpubridge/v/peer/0", '{"v": 99, "peer": {}}')
    with pytest.raises(RuntimeError, match="rank 0: its gpubridge writes discovery record "
                                           "version 99"):
        gpubridge.topology.discover(store, "v", 1, 2, local, TIMEOUT)


def test_a_peer_record_survives_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "amd")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    local, _ = gpubridge.topology.describe_local("multitcp")
    (peer,) = gpubridge.topology.discover(store, "world0", 0, 1, local, TIMEOUT)
    assert peer == local


@needs_library
def test_ranks_that_ask_for_different_bridges_fail_init_before_any_group_exists(tmp_path):
    # One rank asks for Gloo, the other doesn't: they would otherwise wait in two
    # different rendezvous until the timeout. Discovery comes first, so both refuse.
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", **LOCAL}, "init_kwargs": {"bridge": "gloo"}},
         {"env": {VENDOR_ENV: "amd", **LOCAL}, "init_kwargs": {"bridge": "multitcp"}}],
        tmp_path, [r"Ranks picked different bridge transports: 'gloo' on ranks \[0\]; "
                   r"'tcp' on ranks \[1\]"])


# ---- the solo group ---------------------------------------------------------------------

def test_a_solo_groups_collectives_leave_one_ranks_data_as_it_is():
    group = torch_backend.SoloGroup(0, 1, "solo")
    assert group.size() == 1 and group.rank() == 0
    assert group.getBackendName() == SOLO_BACKEND and group.group_name == "solo"
    tensor = torch.arange(6.0)
    for call in (group.allreduce, group.broadcast, group.reduce):
        work = call([tensor])
        assert work.wait() and torch.equal(tensor, torch.arange(6.0))
    assert group.barrier().wait()
    out = torch.empty(2, 3)
    assert group._allgather_base(out, tensor).wait()
    assert torch.equal(out.view(-1), tensor)
    rows = [torch.empty(6)]
    assert group.allgather([rows], [tensor]).wait() and torch.equal(rows[0], tensor)
    out = torch.empty(6)
    assert group._reduce_scatter_base(out, tensor.view(1, 6)).wait()
    assert torch.equal(out, tensor)
    assert group.gather([[out.zero_()]], [tensor]).wait() and torch.equal(out, tensor)
    assert group.scatter([out.zero_()], [[tensor]]).wait() and torch.equal(out, tensor)
    with pytest.raises(NotImplementedError, match="one-rank island"):
        group.send([tensor], 0, 0)
    with pytest.raises(RuntimeError, match="exactly one rank"):
        torch_backend.SoloGroup(0, 2, "two")


# ---- the torch.distributed backend ------------------------------------------------------

def _backend_main(rank, vendors, init_file, env, fn, labels=None):
    os.environ[VENDOR_ENV] = vendors[rank]
    _islands(labels, env, rank)
    torch.set_num_threads(1)
    dist.init_process_group("gpubridge", init_method=f"file://{init_file}", rank=rank,
                            world_size=len(vendors), timeout=TIMEOUT)
    try:
        fn(rank, len(vendors))
    finally:
        dist.destroy_process_group()
    assert not gpubridge.is_initialized()


def _check_backend_object_collectives(rank, world):
    dist.barrier()  # the first collective builds the topology
    assert gpubridge.get_topology().gloo_free
    everyone: list = [None] * world
    dist.all_gather_object(everyone, f"rank {rank}" * (rank + 1))
    assert everyone == [f"rank {r}" * (r + 1) for r in range(world)]
    objects = [{"from": 1, "big": "x" * 500}] if rank == 1 else [None]
    dist.broadcast_object_list(objects, src=1)
    assert objects == [{"from": 1, "big": "x" * 500}]
    gathered: list | None = [None] * world if rank == 0 else None
    dist.gather_object({"r": rank}, gathered, dst=0)
    if rank == 0:
        assert gathered == [{"r": r} for r in range(world)]
    received = [None]
    dist.scatter_object_list(received, [f"for {r}" for r in range(world)] if rank == 1 else None,
                             src=1)
    assert received == [f"for {rank}"]
    # Tensor gather and scatter of a dtype no reduction takes: they travel as bytes.
    rows = [torch.empty(3, dtype=torch.bool) for _ in range(world)] if rank == 0 else None
    dist.gather(torch.tensor([rank % 2 == 0, True, False]), rows, dst=0)
    if rank == 0:
        assert all(torch.equal(row, torch.tensor([r % 2 == 0, True, False]))
                   for r, row in enumerate(rows))
    mine = torch.empty(2, dtype=torch.int16)
    dist.scatter(mine, [torch.tensor([r, -r], dtype=torch.int16) for r in range(world)]
                 if rank == world - 1 else None, src=world - 1)
    assert torch.equal(mine, torch.tensor([rank, -rank], dtype=torch.int16))
    assert_no_gloo()


@needs_library
def test_the_backends_object_collectives_need_no_gloo_group(tmp_path):
    mp.spawn(_backend_main, args=(("nvidia",) * 3, str(tmp_path / "rendezvous"),
                                  {**LOCAL, BRIDGE_ENV: "multitcp"},
                                  _check_backend_object_collectives, FOUR[1][:3]), nprocs=3)


def _check_cpu_tensors_through_the_device(rank, world):
    """What a GPU job does with CPU tensors on a bridge other than Gloo: staged onto the
    rank's device for the collective and copied back. Here the "device" copy is a clone
    and every tensor takes that path, so the staging and the copies back run on CPU."""
    torch_backend._cpu_path = lambda topology, tensors: True
    staged = []

    def on_device(tensor, topology):
        staged.append(tensor.numel())
        return tensor.contiguous().clone()

    torch_backend._on_device = on_device
    dist.barrier()
    t = rank_tensor((2, 3), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, expected_sum((2, 3), torch.float32, world))
    t = rank_tensor((4,), torch.float64, rank).view(2, 2).t()  # not contiguous
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    assert torch.equal(t, rank_tensor((4,), torch.float64, world - 1).view(2, 2).t())
    flags = torch.tensor([rank == 1, True, False])  # bool: carried as bytes
    dist.broadcast(flags, src=1)
    assert torch.equal(flags, torch.tensor([True, True, False]))
    out = torch.empty(world, 3, dtype=torch.int64)
    dist.all_gather_into_tensor(out, rank_tensor((3,), torch.int64, rank))
    assert torch.equal(out, torch.stack([rank_tensor((3,), torch.int64, r) for r in range(world)]))
    rows = [torch.empty(3) for _ in range(world)]
    dist.all_gather(rows, rank_tensor((3,), torch.float32, rank))
    assert all(torch.equal(row, rank_tensor((3,), torch.float32, r)) for r, row in enumerate(rows))
    got = torch.empty(2)
    dist.reduce_scatter_tensor(got, rank_tensor((world, 2), torch.float32, rank).view(-1))
    assert torch.equal(got, expected_sum((world, 2), torch.float32, world)[rank])
    assert len(staged) >= 6
    with pytest.raises(ValueError, match="gpubridge doesn't support torch.bool"):
        dist.all_reduce(torch.tensor([True]))  # a reduction needs a dtype gpubridge reduces
    assert_no_gloo()


@needs_library
def test_cpu_tensors_on_a_gpu_job_go_through_the_topology_not_a_gloo_group(tmp_path):
    mp.spawn(_backend_main, args=(("nvidia", "amd"), str(tmp_path / "rendezvous"),
                                  {**LOCAL, BRIDGE_ENV: "multitcp"},
                                  _check_cpu_tensors_through_the_device), nprocs=2)


def _check_subgroup(rank, world):
    dist.barrier()
    pair = dist.new_group([0, 2])  # one rank of each vendor: a bridged subgroup
    if rank in (0, 2):
        t = rank_tensor((5,), torch.float32, rank)
        dist.all_reduce(t, group=pair)
        assert torch.equal(t, rank_tensor((5,), torch.float32, 0) + rank_tensor(
            (5,), torch.float32, 2))
        dist.barrier(group=pair)
        objects: list = [None, None]
        dist.all_gather_object(objects, rank, group=pair)
        assert objects == [0, 2]
    dist.barrier()
    assert_no_gloo()


@needs_library
def test_a_bridged_subgroup_creates_no_gloo_group(tmp_path):
    mp.spawn(_backend_main, args=(tuple(FOUR[0]), str(tmp_path / "rendezvous"),
                                  {**LOCAL, BRIDGE_ENV: "multitcp"}, _check_subgroup, FOUR[1]),
             nprocs=4)
