"""The raw-verbs bridge transport, ``bridge="verbs"`` (docs/VERBS_TRANSPORT.md).

There is no RDMA device here, so these run ``bridge="verbs"`` between processes on
the MOCK test fabric (``GPUBRIDGE_VERBS_TEST_FABRIC``: a test-only build of the
library whose provider carries RC queue pairs over Unix sockets). Everything above
the provider is the real code: the endpoint, the wire protocol with its credits and
chunking, the completion thread, the handshake through the c10d store and the
collectives. Every bridge policy must match ``flat-gloo`` bit for bit. This is a
software test: nothing here touches RDMA hardware or GPUs. The tests need cargo and
skip where the test library can't be built; the pure-Python parts run everywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _harness import rank_tensor, run_failing_init, run_ranks
from test_ops import FLOATS, OP_NAMES, expected_reduce, reference_all_reduce
from torch.distributed import ReduceOp

import gpubridge
from gpubridge import verbs
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import BRIDGE_COMPRESSION_ENV, VENDOR_ENV
from gpubridge.config import CHUNK_BYTES_ENV as PIPELINE_CHUNK_ENV
from gpubridge.policies import FlatGloo
from gpubridge.ucx.transport import tag
from gpubridge.verbs import transport as verbs_transport
from gpubridge.verbs.transport import (
    CHUNK_BYTES_ENV,
    CREDITS_ENV,
    SLOT_BYTES_ENV,
    VerbsTransport,
)

try:
    verbs.build(test_fabric=True)
    UNAVAILABLE = None
except verbs.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"verbs test library unavailable: {UNAVAILABLE}")

#: (vendors, island labels if patched): 2, 3 and 4 leaders.
LAYOUTS = {
    "2+2": (["nvidia", "nvidia", "amd", "amd"], None),
    "1+3": (["amd", "nvidia", "nvidia", "nvidia"], None),
    "three-islands-2+2+1": (["nvidia"] * 5, ["nvidia", "amd", "third", "nvidia", "amd"]),
    "four-islands": (["nvidia"] * 4, ["nvidia", "amd", "third", "fourth"]),
}
POLICIES = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge")
SIZES = (1, 3, 7, 13)


@pytest.fixture
def fabric():
    """A test fabric directory (short: Unix socket paths are limited to ~100 bytes)."""
    path = tempfile.mkdtemp(prefix="gbvf-", dir="/tmp")
    yield {verbs.TEST_FABRIC_ENV: path}
    shutil.rmtree(path, ignore_errors=True)


def _label_islands(labels, env, rank):
    os.environ.update(env)
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda vendors: original(labels)


# ---- pure Python ----------------------------------------------------------------------

def test_settings(monkeypatch):
    for env in (CHUNK_BYTES_ENV, SLOT_BYTES_ENV, CREDITS_ENV):
        monkeypatch.delenv(env, raising=False)
    assert verbs_transport.chunk_bytes_setting() == 4 * 2**20
    defaults = verbs_transport.endpoint_settings()
    assert defaults.pop("rndv_bytes") == verbs_transport.DEFAULT_RNDV_BYTES
    assert set(defaults.values()) == {None}
    monkeypatch.setenv(CHUNK_BYTES_ENV, "64K")
    monkeypatch.setenv(SLOT_BYTES_ENV, "8K")
    monkeypatch.setenv(CREDITS_ENV, "16")
    assert verbs_transport.chunk_bytes_setting() == 65536
    settings = verbs_transport.endpoint_settings()
    assert (settings["slot_bytes"], settings["credits"]) == (8192, 16)
    monkeypatch.setenv(CREDITS_ENV, "many")
    with pytest.raises(ValueError, match=CREDITS_ENV):
        verbs_transport.endpoint_settings()


def test_verbs_is_a_known_bridge_and_what_auto_chooses_only_with_rdma_on_every_leader():
    assert gpubridge.resolve_transport("verbs") is VerbsTransport
    assert gpubridge.transport.DEFAULT_TRANSPORT == "auto"
    ok = {"ok": True, "link_layer": "InfiniBand"}
    assert gpubridge.transport.choose_bridge([0, 1], {0: ok, 1: ok})[0] == "verbs"
    assert gpubridge.transport.choose_bridge(
        [0, 1], {0: ok, 1: {"ok": False, "reason": "no RDMA device"}})[0] == "tcp"
    assert VerbsTransport.device_tensors is True


def test_the_test_build_and_the_real_build_are_different_libraries():
    assert verbs.features(True) == ["test-fabric"]
    assert verbs.features(False) == ["ibverbs"]
    assert verbs.library_path(test_fabric=True) != verbs.library_path(test_fabric=False)


def test_the_linker_directory_is_not_where_the_library_goes(monkeypatch, tmp_path):
    # GPUBRIDGE_VERBS_LIB_DIR is where libibverbs.so.1 is (build.rs, VERBS_TRANSPORT.md):
    # it reaches the build and its cache key, never the cache directory itself.
    rdma = tmp_path / "rdma-core" / "lib"
    monkeypatch.delenv(verbs.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(verbs.LINK_DIR_ENV, raising=False)
    default = verbs.library_path()
    monkeypatch.setenv(verbs.LINK_DIR_ENV, str(rdma))
    linked = verbs.library_path()
    assert linked.parent == default.parent == Path.home() / ".cache" / "gpubridge"
    assert linked != default  # another library to link against: another build
    monkeypatch.setenv(verbs.CACHE_DIR_ENV, str(tmp_path / "cache"))
    assert verbs.library_path().parent == tmp_path / "cache"
    assert verbs.library_path(test_fabric=True).parent == tmp_path / "cache"


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="the refusal is for other systems")
def test_off_linux_the_real_library_says_why(monkeypatch):
    monkeypatch.delenv(verbs.LIB_ENV, raising=False)
    with pytest.raises(verbs.Unavailable, match="needs Linux with rdma-core"):
        verbs.build()


@needs_library
def test_the_test_library_reports_its_providers():
    lib = verbs.load(test_fabric=True)
    assert verbs.protocol(lib) == 2
    assert "test-fabric" in verbs.build_info(lib)
    assert "libibverbs" not in verbs.build_info(lib)


# ---- collectives against flat-gloo ----------------------------------------------------

def _check_matches_flat_gloo(topology, policy):
    gpubridge.policies._stage_to_host = torch.Tensor.clone  # run the copy back too
    assert topology.policy_name == policy
    if topology.is_leader:
        assert isinstance(topology.bridge, VerbsTransport)
    rank, world = topology.rank, topology.world_size
    for numel in SIZES:
        for dtype in SUPPORTED_DTYPES:
            ops = [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN]
            if dtype in FLOATS:
                ops.append(ReduceOp.AVG)
            for op in ops:
                tensor = rank_tensor((numel,), dtype, rank)
                want = tensor.clone()
                gpubridge.all_reduce(tensor, op=op)
                reference_all_reduce(want, op, topology)
                assert torch.equal(tensor, want), (numel, dtype, OP_NAMES[op])
                assert torch.equal(tensor, expected_reduce((numel,), dtype, world, op))
        for src in range(world):
            tensor = rank_tensor((numel,), torch.float32, rank)
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            FlatGloo().broadcast(want, src, topology)
            assert torch.equal(tensor, want), (numel, src)
    rows = rank_tensor((world, 5), torch.float32, rank)
    got, want = torch.empty(5), torch.empty(5)
    gpubridge.reduce_scatter_tensor(got, rows)
    FlatGloo().reduce_scatter_tensor(want, rows, topology)
    assert torch.equal(got, want)
    a = rank_tensor((3, 7), torch.bfloat16, rank)
    work = gpubridge.all_reduce(a, async_op=True)
    b = rank_tensor((4,), torch.int64, rank)
    gpubridge.broadcast(b, src=world - 1)
    assert work.wait() is True
    assert torch.equal(a, expected_reduce((3, 7), torch.bfloat16, world, ReduceOp.SUM))
    assert torch.equal(b, rank_tensor((4,), torch.int64, world - 1))


@needs_library
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("vendors", "labels"), list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_every_policy_on_the_verbs_bridge_matches_flat_gloo(vendors, labels, policy, fabric,
                                                           tmp_path):
    # 16-byte collective chunks and 128-byte slots (80 bytes of payload): most messages
    # are several chunks, and several RC messages each.
    env = {**fabric, CHUNK_BYTES_ENV: "16", SLOT_BYTES_ENV: "128", CREDITS_ENV: "4"}
    run_ranks(vendors, partial(_check_matches_flat_gloo, policy=policy), tmp_path,
              before_init=partial(_label_islands, labels, env),
              init_kwargs={"bridge": "verbs", "policy": policy})


def _check_compressed_on_verbs(topology, mode):
    from test_compression import _check_compressed

    if topology.is_leader:
        assert isinstance(topology.bridge, VerbsTransport)
    _check_compressed(topology, mode)  # bit for bit the model's result, on every rank


@needs_library
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_bridge_compression_on_the_verbs_bridge(mode, policy, fabric, tmp_path):
    vendors, labels = LAYOUTS["three-islands-2+2+1"]
    env = {**fabric, BRIDGE_COMPRESSION_ENV: mode, PIPELINE_CHUNK_ENV: "64"}
    run_ranks(vendors, partial(_check_compressed_on_verbs, mode=mode), tmp_path,
              init_kwargs={"policy": policy, "bridge": "verbs"},
              before_init=partial(_label_islands, labels, env))


def _check_large(topology):
    rank, world = topology.rank, topology.world_size
    for dtype, numel in ((torch.float32, 2**20 + 3), (torch.int64, 3 * 2**19 + 1),
                         (torch.float32, 4099)):
        for op in (ReduceOp.SUM, ReduceOp.MAX):
            tensor = rank_tensor((numel,), dtype, rank)
            gpubridge.all_reduce(tensor, op=op)
            assert torch.equal(tensor, expected_reduce((numel,), dtype, world, op)), (dtype, op)
        tensor = rank_tensor((numel,), dtype, rank)
        gpubridge.broadcast(tensor, world - 1)
        assert torch.equal(tensor, rank_tensor((numel,), dtype, world - 1))
    if topology.bridge is not None:
        t = topology.bridge.telemetry()
        assert t["chunks_sent"] > t["sends"] > 0  # messages went as several RC messages


@needs_library
@pytest.mark.parametrize("layout", ["2+2", "three-islands-2+2+1"])
def test_large_tensors_in_many_chunks(layout, fabric, tmp_path):
    vendors, labels = LAYOUTS[layout]
    env = {**fabric, CHUNK_BYTES_ENV: "64K", SLOT_BYTES_ENV: "16K"}
    run_ranks(vendors, _check_large, tmp_path,
              before_init=partial(_label_islands, labels, env),
              init_kwargs={"bridge": "verbs"})


def _check_empty_and_bad_tensors(topology):
    bridge = topology.bridge
    if bridge is not None:
        for _ in range(2):  # an empty tensor's data pointer is null: nothing may read it
            empty = torch.empty(0)
            bridge.all_reduce(empty)
            bridge.broadcast(empty, bridge.members[0])
        with pytest.raises(ValueError, match="contiguous host or GPU tensors"):
            bridge.all_reduce(torch.ones(4, 2).t())
        tensor = torch.ones(3)
        bridge.all_reduce(tensor)  # a bad argument leaves the bridge working
        assert torch.equal(tensor, torch.full((3,), float(len(bridge.members))))
    dist.barrier()


@needs_library
def test_empty_tensors_and_bad_arguments(fabric, tmp_path):
    run_ranks(LAYOUTS["2+2"][0], _check_empty_and_bad_tensors, tmp_path,
              before_init=partial(_label_islands, None, fabric),
              init_kwargs={"bridge": "verbs"})


def _check_telemetry(topology):
    bridge = topology.bridge
    tensor = rank_tensor((1000,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)  # every rank: the island steps involve them all
    if bridge is not None:
        t = bridge.telemetry()
        assert t["mock"] is True and t["provider"].startswith("MOCK"), t
        assert (t["protocol"], t["path"], t["state"]) == (2, "host", "ok"), t
        assert t["operations"] == "RC SEND/RECV, RDMA WRITE (rendezvous)"
        assert t["rndv_bytes"] == verbs_transport.DEFAULT_RNDV_BYTES
        assert t["device"] == "mockfab0" and t["link_layer"] == "InfiniBand"
        assert t["bytes_sent"] >= 4000 and t["bytes_received"] >= 4000
        assert [p["connected"] for p in t["peers"]] == [True] * (len(bridge.members) - 1)
        assert bridge.state == "ok"
        # Nothing key-like is reported: the protocol has no remote keys.
        assert "key" not in json.dumps(t).lower()
    dist.barrier()


@needs_library
def test_telemetry_says_mock_and_counts(fabric, tmp_path):
    run_ranks(LAYOUTS["2+2"][0], _check_telemetry, tmp_path,
              before_init=partial(_label_islands, None, fabric),
              init_kwargs={"bridge": "verbs"})


def _check_a_receiver_after_the_root_left(topology):
    bridge = topology.bridge
    if bridge is not None:
        tensor = torch.arange(4, dtype=torch.float32) if bridge.index == 0 else torch.zeros(4)
        if bridge.index == 1:
            time.sleep(0.5)  # the root has sent, finished and closed by now
        bridge.broadcast(tensor, bridge.members[0])
        if bridge.index == 0:
            bridge.close()  # done: it leaves before the other leader has posted a receive
        assert torch.equal(tensor, torch.arange(4, dtype=torch.float32))
    dist.barrier()


@needs_library
def test_a_receiver_still_gets_what_a_leader_sent_before_it_closed(fabric, tmp_path):
    run_ranks(["nvidia", "amd"], _check_a_receiver_after_the_root_left, tmp_path,
              before_init=partial(_label_islands, None, {**fabric, CHUNK_BYTES_ENV: "8"}),
              init_kwargs={"bridge": "verbs"})


@needs_library
def test_subgroups_on_the_verbs_bridge(fabric, tmp_path):
    # A subgroup's members build its links on their own: their store keys must not
    # depend on how many links each process made before (three mixed groups here).
    from test_torch_backend import TWO_TWO, _subgroups, run_backend

    from gpubridge.config import BRIDGE_ENV

    run_backend(TWO_TWO, _subgroups, tmp_path, env={-1: {BRIDGE_ENV: "verbs", **fabric}})


# ---- failures -------------------------------------------------------------------------

def test_without_verbs_init_fails_everywhere_and_nothing_falls_back(tmp_path):
    # No test fabric: the real library (none on macOS; no device in CI) can't run, and
    # the bridge fails with the reason instead of moving data some other way.
    good = {VENDOR_ENV: "nvidia", verbs.TEST_FABRIC_ENV: "", verbs.LIB_ENV: ""}
    run_failing_init(
        [{"env": good, "init_kwargs": {"bridge": "verbs"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "verbs"}}],
        tmp_path,
        [r"The verbs bridge between ranks \[0, 1\] could not be set up",
         r"never replaced by another transport", r'bridge="tcp"', r"rank 0: \w+"],
    )


@needs_library
def test_a_bad_setting_on_one_leader_fails_init_on_every_rank(fabric, tmp_path):
    good = {VENDOR_ENV: "nvidia", **fabric}
    run_failing_init(
        [{"env": {**good, CHUNK_BYTES_ENV: "12"}, "init_kwargs": {"bridge": "verbs"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "verbs"}}],
        tmp_path,
        [r"The verbs bridge between ranks \[0, 1\] could not be set up",
         rf"rank 0: ValueError: {CHUNK_BYTES_ENV}='12' must be"],
    )


@needs_library
def test_leaders_with_different_protocol_settings_fail_init_on_every_rank(fabric, tmp_path):
    good = {VENDOR_ENV: "nvidia", **fabric}
    run_failing_init(
        [{"env": {**good, CREDITS_ENV: "8"}, "init_kwargs": {"bridge": "verbs"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "verbs"}}],
        tmp_path,
        [r"could not be set up", r"use the same settings on every leader"],
    )


def _check_deadline(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        bridge.timeout = 1.0
        tensor = torch.ones(10)
        start = time.monotonic()
        with pytest.raises(verbs.TransportError, match=r"timed out: \d+ of \d+ operation") as failure:
            bridge.all_reduce(tensor)  # the other leader never joins
        assert failure.value.timed_out is True
        assert time.monotonic() - start < 10
        with pytest.raises(verbs.TransportError, match="already failed"):
            bridge.all_reduce(tensor)
        assert bridge.state == "failed"
    dist.barrier()  # on the world Gloo group, not the bridge


@needs_library
def test_a_leader_that_never_joins_trips_the_deadline(fabric, tmp_path):
    run_ranks(["nvidia", "amd"], _check_deadline, tmp_path,
              before_init=partial(_label_islands, None, fabric),
              init_kwargs={"bridge": "verbs"})


def _check_teardown(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        never = torch.empty(4)
        bridge.worker.recv(1, never.data_ptr(), 16, tag(99, 0, 1, 0))
    dist.barrier()
    start = time.monotonic()
    gpubridge.destroy()  # closes the bridge with a receive still posted
    assert time.monotonic() - start < 15
    gpubridge.init(init_method=f"file://{os.environ['GB_RDZV']}-again", rank=topology.rank,
                   world_size=topology.world_size, bridge="verbs",
                   timeout=timedelta(seconds=60))
    tensor = rank_tensor((5,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)  # a new bridge works after the old one closed
    assert torch.equal(tensor, expected_reduce((5,), torch.float32, topology.world_size,
                                               ReduceOp.SUM))


@needs_library
def test_teardown_with_an_operation_outstanding(fabric, tmp_path):
    env = {**fabric, "GB_RDZV": str(tmp_path / "rdzv")}
    run_ranks(["nvidia", "amd"], _check_teardown, tmp_path,
              before_init=partial(_label_islands, None, env), init_kwargs={"bridge": "verbs"})


#: Two endpoints on the test fabric and a receive nothing will meet; ``waiting()`` runs
#: ``a.wait`` on it in a thread, then goes on using ``a`` as a failed call does (abort,
#: outcomes, another post). Run in a process of its own: a freed handle crashes it.
_CLOSE_RACE = """
import ctypes, json, threading, time
from gpubridge import verbs
a, b = verbs.Endpoint(2, 0, {}), verbs.Endpoint(2, 1, {})
a.connect(1, b.local_info())
b.connect(0, a.local_info())
buf = (ctypes.c_char * 16)()
op = a.recv(1, ctypes.addressof(buf), 16, 5)
got = {}

def waiting():
    try:
        a.wait([op], time.monotonic() + 60)
        got["wait"] = "returned"
    except verbs.TransportError as exc:
        got["wait"] = str(exc)
    a.abort("the waiter gave up")
    got["outcome"] = a.operation_outcome(op)
    try:
        a.send(1, ctypes.addressof(buf), 16, 6)
    except verbs.TransportError as exc:
        got["send"] = str(exc)

waiter = threading.Thread(target=waiting)
"""


def _run_close_race(script: str, fabric: dict) -> dict:
    result = subprocess.run([sys.executable, "-c", _CLOSE_RACE + script],
                            env={**os.environ, **fabric}, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, (result.returncode, result.stderr[-3000:])
    return json.loads(result.stdout.splitlines()[-1])


@needs_library
def test_closing_while_another_thread_waits_aborts_it_first_and_frees_nothing_it_uses(
        fabric):
    got = _run_close_race("""
waiter.start()
time.sleep(0.5)  # inside gb_verbs_wait, the GIL let go
start = time.monotonic()
got["drained"] = a.close(timeout=5)
got["close_s"] = time.monotonic() - start
waiter.join(10)
got["alive"] = waiter.is_alive()
b.close(timeout=1)
print(json.dumps(got))
""", fabric)
    assert got["alive"] is False and got["drained"] is True
    assert "closed while another call was in the library" in got["wait"], got
    assert got["close_s"] < 4, got  # aborted at once, not left to its deadline
    assert got["outcome"] == {"status": "cancelled"}
    assert "already failed" in got["send"] or "closed" in got["send"], got


@needs_library
def test_a_call_that_stays_in_the_library_past_the_bound_leaks_the_endpoint(fabric):
    got = _run_close_race("""
class Stuck:
    # The library, with gb_verbs_wait held until released: a call that doesn't return
    # after the abort.
    def __init__(self, lib):
        self.lib, self.entered, self.release, self.closes = lib, threading.Event(), \
            threading.Event(), 0
    def __getattr__(self, name):
        return getattr(self.lib, name)
    def gb_verbs_wait(self, *args):
        self.entered.set()
        self.release.wait()
        return self.lib.gb_verbs_wait(*args)
    def gb_verbs_close(self, *args):
        self.closes += 1
        return self.lib.gb_verbs_close(*args)

stuck = a._lib = Stuck(a._lib)
waiter.start()
stuck.entered.wait(10)
start = time.monotonic()
got["drained"] = a.close(timeout=0.5)
got["close_s"] = time.monotonic() - start
got["closes"] = stuck.closes
stuck.release.set()  # it enters the library now, with the handle that was kept
waiter.join(10)
got["alive"] = waiter.is_alive()
got["telemetry"] = a.telemetry()
b.close(timeout=1)
print(json.dumps(got))
""", fabric)
    assert got["drained"] is False and got["closes"] == 0, got
    assert got["close_s"] < 3, got
    assert got["alive"] is False
    assert "closed while another call was in the library" in got["wait"], got
    assert got["telemetry"] == {"state": "closed"}
