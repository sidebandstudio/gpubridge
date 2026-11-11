"""torch's debug level DETAIL would bring Gloo back into a job that didn't ask for it.

At ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` torch's ``new_group`` wraps every group of a C++
backend (NCCL, RCCL, Gloo) in ``_ProcessGroupWrapper``, whose helper is a
``ProcessGroupGloo`` of the same ranks. A GPU job on ``bridge="auto"``, ``"tcp"``,
``"verbs"`` or ``"ucx"`` would then have Gloo groups and threads without ever naming
Gloo. So such a job is refused, on every rank with one message, before any island group
exists, unless it says ``allow_debug_gloo`` (``GPUBRIDGE_ALLOW_DEBUG_GLOO=1``).

What this file establishes, all on CPU:

- the rule (``debug_gloo_problem``) refuses a layout of native (NCCL) islands, as a
  function of what the ranks reported, with no process group involved;
- in real multi-process jobs the refusal reaches every rank alike, through both init
  APIs, for a level set by the environment and by ``set_debug_level``, for one rank of
  four at DETAIL, and leaves no group behind;
- what torch itself does at DETAIL to a group of a C++ backend, with simulation's
  Gloo islands standing in for NCCL ones: it wraps it, on the torch under test;
- that torch does not wrap a group defined in Python, so a job of one-rank simulated
  islands starts at DETAIL with no Gloo.

What it does not establish: that torch wraps an NCCL group the same way. No test here
creates one. That rests on reading ``_new_process_group_helper`` in torch 2.8.0 and
2.15.0, where NCCL and Gloo take the same branch; it was not run on a GPU.
"""

from __future__ import annotations

import os
import re
import socket
import time
from dataclasses import replace

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
    thread_names,
)

import gpubridge
from gpubridge import multitcp
from gpubridge.config import (
    ALLOW_DEBUG_GLOO_ENV,
    BRIDGE_ENV,
    GPU_BACKEND,
    VENDOR_ENV,
    Config,
)
from gpubridge.multitcp.transport import IFNAME_ENV, STREAMS_ENV
from gpubridge.topology import debug_gloo_problem, describe_local, plan_topology, validate_peers
from gpubridge.transport import GlooTransport, resolve_transport

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"the Rust TCP bridge is unavailable: {UNAVAILABLE}")

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
LOCAL = {IFNAME_ENV: LOOPBACK, STREAMS_ENV: "2"}
DETAIL = dist.DebugLevel.DETAIL
#: Two islands of two ranks: in simulation each is a group of a C++ backend (Gloo), the
#: kind torch wraps at DETAIL, as it would an NCCL island.
TWO_BY_TWO = ("nvidia", "nvidia", "amd", "amd")
REFUSAL = [r"torch's distributed debug level is DETAIL", r"Gloo helper group",
           r"pt_gloo_runloop", r"TORCH_DISTRIBUTED_DEBUG=INFO or OFF",
           r"allow_debug_gloo=True", rf"{ALLOW_DEBUG_GLOO_ENV}=1",
           r"doesn't change the debug level itself"]
#: Well under the jobs' 60 s timeout: a rank that waited for one would not make it.
TOGETHER = 30.0


# ---- the rule, with no process group ------------------------------------------------------

def _peers(monkeypatch, count, detail=(), allow=False):
    """What ``count`` GPU ranks would report: this machine's description with the
    simulation mark taken off (a native island's backend is then NCCL)."""
    monkeypatch.setenv(VENDOR_ENV, "nvidia")
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    local, _ = describe_local("tcp")
    assert local.debug_detail is False and local.allow_debug_gloo is False
    return tuple(replace(local, simulated=False, debug_detail=rank in detail,
                         allow_debug_gloo=allow) for rank in range(count))


GPUS = Config(simulated=False, island_backend=GPU_BACKEND, device=torch.device("cpu"))


@pytest.mark.parametrize("bridge", ["auto", "tcp", "verbs", "ucx"])
def test_native_islands_at_detail_are_refused_on_any_bridge_but_gloo(monkeypatch, bridge):
    layout = plan_topology(list(TWO_BY_TWO))
    transport = bridge if bridge == "auto" else resolve_transport(bridge)
    assert debug_gloo_problem(_peers(monkeypatch, 4), layout, GPUS, transport) is None
    problem = debug_gloo_problem(_peers(monkeypatch, 4, detail=(2,)), layout, GPUS, transport)
    assert problem is not None and "on ranks [2] " in problem
    for pattern in REFUSAL:
        assert re.search(pattern, problem), pattern
    everyone = _peers(monkeypatch, 4, detail=range(4))
    assert "on ranks [0, 1, 2, 3] " in debug_gloo_problem(everyone, layout, GPUS, transport)
    # One GPU per vendor: each island is still an NCCL group, of one rank.
    pair = plan_topology(["nvidia", "amd"])
    assert debug_gloo_problem(_peers(monkeypatch, 2, detail=(0, 1)), pair, GPUS, transport)
    # One vendor, no bridge at all: the island is the whole job, and NCCL.
    single = plan_topology(["nvidia"] * 4)
    assert debug_gloo_problem(everyone, single, GPUS, transport)


def test_a_job_that_asked_for_gloo_or_accepted_the_helpers_is_not_refused(monkeypatch):
    layout = plan_topology(list(TWO_BY_TWO))
    everyone = _peers(monkeypatch, 4, detail=range(4))
    assert debug_gloo_problem(everyone, layout, GPUS, GlooTransport) is None
    accepted = _peers(monkeypatch, 4, detail=range(4), allow=True)
    assert debug_gloo_problem(accepted, layout, GPUS, resolve_transport("tcp")) is None
    # Accepted, but not every rank is at DETAIL: torch's helper group needs every member.
    some = _peers(monkeypatch, 4, detail=(1, 3), allow=True)
    problem = debug_gloo_problem(some, layout, GPUS, resolve_transport("tcp"))
    assert "Ranks disagree on torch's distributed debug level: DETAIL on ranks [1, 3], " \
           "not on ranks [0, 2]" in problem
    # And the opt-in itself must be the same everywhere.
    mixed = (*_peers(monkeypatch, 2, allow=True), *_peers(monkeypatch, 2))
    with pytest.raises(RuntimeError, match=rf"Ranks disagree on {ALLOW_DEBUG_GLOO_ENV} "
                                           r"\(allow_debug_gloo\): on for ranks \[0, 1\], "
                                           r"off for ranks \[2, 3\]"):
        validate_peers(mixed, warn=False)


def test_a_rank_reports_the_debug_level_however_it_was_set_and_the_opt_in(monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "amd")
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    before = dist.get_debug_level()
    try:
        dist.set_debug_level(DETAIL)
        assert describe_local("tcp")[0].debug_detail is True
        dist.set_debug_level(dist.DebugLevel.INFO)
        assert describe_local("tcp")[0].debug_detail is False
    finally:
        dist.set_debug_level(before)
    assert dist.get_debug_level() == before  # describing a rank never changes the level
    assert describe_local("tcp", allow_debug_gloo=True)[0].allow_debug_gloo is True
    monkeypatch.setenv(ALLOW_DEBUG_GLOO_ENV, "1")
    assert describe_local("tcp")[0].allow_debug_gloo is True
    assert describe_local("tcp", allow_debug_gloo=False)[0].allow_debug_gloo is False
    monkeypatch.setenv(ALLOW_DEBUG_GLOO_ENV, "maybe")
    local, _ = describe_local("tcp")
    assert local.allow_debug_gloo is False
    assert local.problems == (f"{ALLOW_DEBUG_GLOO_ENV}='maybe' is not a boolean; use 1 or 0",)


# ---- real jobs ---------------------------------------------------------------------------

def _rank(rank, vendors, init_file, setups, api, expect, check, labels=None):
    """One rank. ``setups[rank]``: ``detail`` (call ``set_debug_level(DETAIL)`` first),
    ``at_detail`` (the level this rank must already be at, from the environment),
    ``kwargs`` for ``gpubridge.init``, ``env``. ``expect``: the patterns of the
    refusal every rank must raise, or None if the job must start (then ``check``)."""
    setup = setups[rank]
    os.environ.update({VENDOR_ENV: vendors[rank], **LOCAL, **setup.get("env", {})})
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda v, ranks=(): original(
            [labels[r] for r in ranks] if ranks else labels, ranks)
    torch.set_num_threads(1)
    if "at_detail" in setup:
        assert (dist.get_debug_level() == DETAIL) is setup["at_detail"]
    if setup.get("detail"):
        dist.set_debug_level(DETAIL)
    level = dist.get_debug_level()
    started = time.monotonic()
    try:
        if api == "init":
            topology = gpubridge.init(init_method=f"file://{init_file}", rank=rank,
                                      world_size=len(vendors), timeout=TIMEOUT,
                                      **{"bridge": "tcp", **setup.get("kwargs", {})})
        else:
            os.environ.setdefault(BRIDGE_ENV, "tcp")
            dist.init_process_group("gpubridge", init_method=f"file://{init_file}", rank=rank,
                                    world_size=len(vendors), timeout=TIMEOUT)
            dist.barrier()  # the backend sets itself up at its first collective
            topology = gpubridge.get_topology()
    except RuntimeError as exc:
        assert expect is not None, f"rank {rank}: {exc}"
        for pattern in expect:
            assert re.search(pattern, str(exc)), f"rank {rank}: {pattern!r} not in:\n{exc}"
        assert time.monotonic() - started < TOGETHER  # with the others, not at a timeout
        assert gloo_groups() == [] and not gpubridge.is_initialized()  # no island was made
        if api == "backend":
            dist.destroy_process_group()
        assert not dist.is_initialized()
        assert dist.get_debug_level() == level  # and the level is the user's still
        return
    try:
        assert expect is None, f"rank {rank}: started, but {expect[0]!r} was expected"
        assert dist.get_debug_level() == level
        check(topology)
    finally:
        if api == "init":
            gpubridge.destroy()
        else:
            dist.destroy_process_group()


def _job(tmp_path, setups, api="init", expect=None, check=None, vendors=TWO_BY_TWO, labels=None):
    mp.spawn(_rank, args=(tuple(vendors), str(tmp_path / "rendezvous"), tuple(setups), api,
                          expect, check, labels), nprocs=len(vendors))


def _sum_is_exact(topology):
    tensor = rank_tensor((7,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((7,), torch.float32, topology.world_size))


@pytest.mark.parametrize("api", ["init", "backend"])
def test_detail_from_the_environment_refuses_the_job_on_every_rank(api, tmp_path, monkeypatch):
    # Set where the ranks are started, as a launcher would: each rank's torch reads it
    # when it is imported.
    monkeypatch.setenv("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    _job(tmp_path, [{"at_detail": True}] * 4, api,
         expect=[r"on ranks \[0, 1, 2, 3\] ", *REFUSAL])


@pytest.mark.parametrize("api", ["init", "backend"])
def test_detail_set_by_one_ranks_program_refuses_the_job_on_every_rank(api, tmp_path,
                                                                        monkeypatch):
    monkeypatch.delenv("TORCH_DISTRIBUTED_DEBUG", raising=False)
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    setups = [{"at_detail": False, "detail": rank == 2} for rank in range(4)]
    _job(tmp_path, setups, api, expect=[r"on ranks \[2\] ", *REFUSAL])


def _helpers_are_there(topology):
    assert topology.gloo_free and topology.debug_gloo
    backend = topology.island_group._get_backend(torch.device("cpu"))
    assert isinstance(backend, torch._C._distributed_c10d._ProcessGroupWrapper), type(backend)
    _sum_is_exact(topology)
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 15):
        assert "pt_gloo_runloop" in thread_names()  # torch names them from some release on


@needs_library
@pytest.mark.parametrize(("api", "how"), [("init", "argument"), ("init", "environment"),
                                          ("backend", "environment")])
def test_the_opt_in_starts_the_job_and_torch_wraps_its_c_groups(api, how, tmp_path,
                                                                 monkeypatch):
    # init_process_group takes no gpubridge argument: the variable is the backend's way.
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    opt_in = ({"kwargs": {"allow_debug_gloo": True}} if how == "argument"
              else {"env": {ALLOW_DEBUG_GLOO_ENV: "1"}})
    _job(tmp_path, [{"detail": True, **opt_in}] * 4, api, check=_helpers_are_there)


def _no_helper(topology):
    assert topology.gloo_free and not topology.debug_gloo
    backend = topology.island_group._get_backend(torch.device("cpu"))
    assert not isinstance(backend, torch._C._distributed_c10d._ProcessGroupWrapper)
    _sum_is_exact(topology)


@needs_library
def test_below_detail_torch_wraps_nothing_and_the_opt_in_changes_nothing(tmp_path, monkeypatch):
    # The control of the test above: the same job at torch's default level.
    monkeypatch.delenv("TORCH_DISTRIBUTED_DEBUG", raising=False)
    _job(tmp_path, [{"at_detail": False, "kwargs": {"allow_debug_gloo": True}}] * 4,
         check=_no_helper)


def _python_islands(topology):
    assert topology.gloo_free and not topology.debug_gloo
    assert len(topology.layout.islands) == 4
    _sum_is_exact(topology)
    assert_no_gloo()
    assert [name for name in thread_names() if name in GLOO_THREADS] == []


@needs_library
def test_python_groups_are_not_wrapped_so_one_rank_simulated_islands_start_at_detail(
        tmp_path, monkeypatch):
    # torch returns a group defined in Python before its wrapper branch: nothing of
    # Gloo's appears, so nothing is refused. (This is why the one-rank-island tests of
    # tests/test_no_gloo.py cannot say anything about native islands at DETAIL.)
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    _job(tmp_path, [{"detail": True}] * 4, check=_python_islands, vendors=["nvidia"] * 4,
         labels=["nvidia", "amd", "third", "fourth"])


def test_the_opt_in_must_be_the_same_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    setups = [{"kwargs": {"allow_debug_gloo": rank < 2}} for rank in range(4)]
    _job(tmp_path, setups, expect=[rf"Ranks disagree on {ALLOW_DEBUG_GLOO_ENV} "
                                   r"\(allow_debug_gloo\): on for ranks \[0, 1\], off for "
                                   r"ranks \[2, 3\]"])


def test_a_malformed_opt_in_fails_every_rank(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    setups = [{"env": {ALLOW_DEBUG_GLOO_ENV: "maybe"} if rank == 3 else {}} for rank in range(4)]
    _job(tmp_path, setups, expect=[r"1 of 4 ranks reported problems",
                                   rf"rank 3 \([^)]*\): {ALLOW_DEBUG_GLOO_ENV}='maybe' is not "
                                   r"a boolean"])


def test_the_opt_in_with_detail_on_some_ranks_only_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    setups = [{"detail": rank == 0, "kwargs": {"allow_debug_gloo": True}} for rank in range(4)]
    _job(tmp_path, setups, expect=[r"Ranks disagree on torch's distributed debug level: "
                                   r"DETAIL on ranks \[0\], not on ranks \[1, 2, 3\]"])


def _gloo_job(topology):
    assert not topology.gloo_free and not topology.debug_gloo
    _sum_is_exact(topology)


def test_a_job_on_the_gloo_bridge_starts_at_detail_as_it_always_did(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    _job(tmp_path, [{"detail": True, "kwargs": {"bridge": "gloo"}}] * 4, check=_gloo_job)


# ---- a level raised after init: seen when a subgroup is set up ---------------------------

def _subgroup_after_init(rank, vendors, init_file, allowed):
    os.environ.update({VENDOR_ENV: vendors[rank], **LOCAL, BRIDGE_ENV: "tcp"})
    if allowed:
        os.environ[ALLOW_DEBUG_GLOO_ENV] = "1"
    torch.set_num_threads(1)
    dist.init_process_group("gpubridge", init_method=f"file://{init_file}", rank=rank,
                            world_size=len(vendors), timeout=TIMEOUT)
    try:
        dist.barrier()  # the world's topology, at torch's default level
        assert not gpubridge.get_topology().debug_gloo
        dist.set_debug_level(DETAIL)  # every rank's program raises it, after init
        trio = dist.new_group([0, 1, 2])  # its nvidia island has two ranks: a C++ group
        if rank < 3:
            tensor = rank_tensor((5,), torch.float32, rank)
            if allowed:
                dist.all_reduce(tensor, group=trio)
                assert torch.equal(tensor, expected_sum((5,), torch.float32, 3))
            else:
                with pytest.raises(RuntimeError) as refusal:
                    dist.all_reduce(tensor, group=trio)
                for pattern in [rf"DETAIL on rank {rank}, raised after init\(\), when "
                                r"subgroup '[^']+' of ranks \[0, 1, 2\] was set up",
                                *REFUSAL[1:]]:
                    assert re.search(pattern, str(refusal.value)), (pattern, refusal.value)
                assert torch.equal(tensor, rank_tensor((5,), torch.float32, rank))
        assert dist.get_debug_level() == DETAIL
        dist.barrier()  # the world still works
    finally:
        dist.destroy_process_group()


@needs_library
@pytest.mark.parametrize("allowed", [False, True], ids=["refused", "opted-in"])
def test_a_subgroup_set_up_at_detail_is_refused_unless_the_job_opted_in(allowed, tmp_path,
                                                                        monkeypatch):
    monkeypatch.delenv("TORCH_DISTRIBUTED_DEBUG", raising=False)
    monkeypatch.delenv(ALLOW_DEBUG_GLOO_ENV, raising=False)
    mp.spawn(_subgroup_after_init, args=(TWO_BY_TWO, str(tmp_path / "rendezvous"), allowed),
             nprocs=4)
