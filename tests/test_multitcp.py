"""The multi-stream TCP bridge transport, ``bridge="tcp"`` (docs/RUST_TRANSPORT.md), under
its older name ``"multitcp"``, which the jobs here ask for: both name one transport.

On CPU, between processes on one machine, over loopback TCP: the same code that
runs between two hosts. Every bridge policy runs on it and must match
``flat-gloo`` bit for bit. The transport tests need the Rust library and skip
where it can't be built (no cargo); the pure-Python parts run everywhere.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _harness import rank_tensor, run_failing_init, run_ranks
from test_ops import FLOATS, OP_NAMES, expected_reduce, reference_all_reduce
from torch.distributed import ReduceOp

import gpubridge
from gpubridge import multitcp
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import BRIDGE_COMPRESSION_ENV, VENDOR_ENV
from gpubridge.config import CHUNK_BYTES_ENV as PIPELINE_CHUNK_ENV
from gpubridge.multitcp import transport as tcp_transport
from gpubridge.multitcp.transport import (
    CHUNK_BYTES_ENV,
    IFNAME_ENV,
    STREAMS_ENV,
    MultiTcpTransport,
)
from gpubridge.policies import FlatGloo

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"multitcp bridge unavailable: {UNAVAILABLE}")

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
#: Over loopback, three connections per pair: most messages land on different ones.
LOCAL = {IFNAME_ENV: LOOPBACK, STREAMS_ENV: "3"}

#: (vendors, island labels if patched): 2, 3 and 4 leaders.
LAYOUTS = {
    "2+2": (["nvidia", "nvidia", "amd", "amd"], None),
    "1+3": (["amd", "nvidia", "nvidia", "nvidia"], None),
    "three-islands-2+2+1": (["nvidia"] * 5, ["nvidia", "amd", "third", "nvidia", "amd"]),
    "four-islands": (["nvidia"] * 4, ["nvidia", "amd", "third", "fourth"]),
}
POLICIES = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge")
#: Element counts for every dtype: one element, fewer than leaders, odd. Small
#: enough that every sum stays exact in float16, bfloat16 and int8 (see
#: rank_tensor); with 16-byte chunks most are still several messages.
SIZES = (1, 3, 7, 13)


# ---- pure Python ----------------------------------------------------------------------

def test_pieces_cover_every_element_once():
    assert tcp_transport._pieces(0, 4) == [(0, 0)]
    assert tcp_transport._pieces(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert tcp_transport._pieces(8, 4) == [(0, 4), (4, 4)]


def test_settings(monkeypatch):
    monkeypatch.delenv(CHUNK_BYTES_ENV, raising=False)
    monkeypatch.delenv(STREAMS_ENV, raising=False)
    assert tcp_transport.chunk_bytes_setting() == 4 * 2**20
    assert tcp_transport.streams_setting() == 4
    monkeypatch.setenv(CHUNK_BYTES_ENV, "64K")
    monkeypatch.setenv(STREAMS_ENV, "8")
    assert tcp_transport.chunk_bytes_setting() == 65536
    assert tcp_transport.streams_setting() == 8
    for bad in ("0", "65", "two"):
        monkeypatch.setenv(STREAMS_ENV, bad)
        with pytest.raises(ValueError, match=STREAMS_ENV):
            tcp_transport.streams_setting()


def test_the_listening_address_follows_the_interface_settings(monkeypatch):
    monkeypatch.setenv(IFNAME_ENV, LOOPBACK)
    assert tcp_transport.local_address() == "127.0.0.1"
    monkeypatch.delenv(IFNAME_ENV)
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", LOOPBACK)
    assert tcp_transport.local_address() == "127.0.0.1"
    monkeypatch.delenv("GLOO_SOCKET_IFNAME")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    assert tcp_transport.local_address() == "127.0.0.1"
    monkeypatch.setenv(IFNAME_ENV, "no-such-interface0")
    with pytest.raises(RuntimeError, match="no IPv4 address on interface"):
        tcp_transport.local_address()


def test_multitcp_is_a_known_bridge():
    assert gpubridge.resolve_transport("multitcp") is MultiTcpTransport
    assert gpubridge.resolve_transport("tcp") is MultiTcpTransport
    assert MultiTcpTransport.name == "tcp"


def test_without_cargo_or_a_library_the_bridge_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(multitcp, "_library", None)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CARGO", raising=False)
    monkeypatch.delenv(multitcp.LIB_ENV, raising=False)
    monkeypatch.setenv(multitcp.LIB_DIR_ENV, str(tmp_path))
    with pytest.raises(multitcp.Unavailable, match="no cargo"):
        multitcp.load()
    monkeypatch.setenv(multitcp.LIB_ENV, str(tmp_path / "missing.so"))
    with pytest.raises(multitcp.Unavailable, match="no multitcp bridge library"):
        multitcp.load()


@needs_library
def test_a_library_built_elsewhere_is_used_as_given(monkeypatch, tmp_path):
    built = multitcp.build()
    copy = tmp_path / built.name
    copy.write_bytes(built.read_bytes())
    monkeypatch.setattr(multitcp, "_library", None)
    monkeypatch.setenv("PATH", str(tmp_path))  # no cargo: nothing to build with
    monkeypatch.setenv(multitcp.LIB_ENV, str(copy))
    assert multitcp.protocol() >= 1
    assert multitcp.load()._name == str(copy)
    from gpubridge.multitcp.__main__ import main
    assert main(["build"]) == 0  # no cargo here: build only loads the given library


# ---- collectives against flat-gloo ----------------------------------------------------

def _label_islands(labels, env, rank):
    os.environ.update(env)
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda vendors: original(labels)


def _check_matches_flat_gloo(topology, policy):
    gpubridge.policies._stage_to_host = torch.Tensor.clone  # run the copy back too
    assert topology.policy_name == policy
    if topology.is_leader:
        assert isinstance(topology.bridge, MultiTcpTransport)
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
def test_every_policy_on_the_multitcp_bridge_matches_flat_gloo(vendors, labels, policy, tmp_path):
    # 16-byte chunks: most messages are several chunks, over several connections.
    env = {**LOCAL, CHUNK_BYTES_ENV: "16"}
    run_ranks(vendors, partial(_check_matches_flat_gloo, policy=policy), tmp_path,
              before_init=partial(_label_islands, labels, env),
              init_kwargs={"bridge": "multitcp", "policy": policy})


def _check_compressed_on_multitcp(topology, mode):
    from test_compression import _check_compressed

    if topology.is_leader:
        assert isinstance(topology.bridge, MultiTcpTransport)
    _check_compressed(topology, mode)  # bit for bit the model's result, on every rank


@needs_library
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
@pytest.mark.parametrize(("vendors", "labels"), [LAYOUTS["2+2"], LAYOUTS["three-islands-2+2+1"]],
                         ids=["2+2", "three-islands"])
def test_bridge_compression_on_the_multitcp_bridge(vendors, labels, mode, policy, tmp_path):
    env = {**LOCAL, BRIDGE_COMPRESSION_ENV: mode, PIPELINE_CHUNK_ENV: "64"}
    run_ranks(vendors, partial(_check_compressed_on_multitcp, mode=mode), tmp_path,
              init_kwargs={"policy": policy, "bridge": "multitcp"},
              before_init=partial(_label_islands, labels, env))


def _check_large(topology):
    # float32 and int64 hold these sums exactly; odd sizes leave a short last chunk.
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


@needs_library
@pytest.mark.parametrize("layout", ["2+2", "three-islands-2+2+1"])
def test_large_tensors_in_many_chunks(layout, tmp_path):
    vendors, labels = LAYOUTS[layout]
    run_ranks(vendors, _check_large, tmp_path,
              before_init=partial(_label_islands, labels, {**LOCAL, CHUNK_BYTES_ENV: "64K"}),
              init_kwargs={"bridge": "multitcp"})


def _check_thread_count_restored(topology):
    torch.set_num_threads(3)
    tensor = rank_tensor((257,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.get_num_threads() == 3  # one thread during the collective, then back
    assert torch.equal(tensor, expected_reduce((257,), torch.float32, topology.world_size,
                                               ReduceOp.SUM))


@needs_library
def test_the_reductions_leave_torchs_thread_count_as_it_was(tmp_path):
    run_ranks(LAYOUTS["2+2"][0], _check_thread_count_restored, tmp_path,
              before_init=partial(_label_islands, None, LOCAL),
              init_kwargs={"bridge": "multitcp"})


def _check_empty_and_bad_tensors(topology):
    bridge = topology.bridge
    if bridge is not None:
        for _ in range(2):  # an empty tensor's data pointer is null: nothing may read it
            empty = torch.empty(0)
            bridge.all_reduce(empty)
            bridge.broadcast(empty, bridge.members[0])
        with pytest.raises(ValueError, match="contiguous host tensors"):
            bridge.all_reduce(torch.ones(4, 2).t())
        tensor = torch.ones(3)
        bridge.all_reduce(tensor)  # a bad argument leaves the bridge working
        assert torch.equal(tensor, torch.full((3,), float(len(bridge.members))))
    dist.barrier()


@needs_library
def test_empty_tensors_and_bad_arguments(tmp_path):
    run_ranks(LAYOUTS["2+2"][0], _check_empty_and_bad_tensors, tmp_path,
              before_init=partial(_label_islands, None, LOCAL),
              init_kwargs={"bridge": "multitcp"})


def _check_a_receiver_after_the_root_left(topology):
    bridge = topology.bridge
    if bridge is not None:
        tensor = torch.arange(4, dtype=torch.float32) if bridge.index == 0 else torch.zeros(4)
        if bridge.index == 1:
            time.sleep(0.5)  # the root has sent, finished and closed by now
        bridge.broadcast(tensor, bridge.members[0])
        if bridge.index == 0:
            bridge.close()  # done: it leaves before the other leader has read a byte
        assert torch.equal(tensor, torch.arange(4, dtype=torch.float32))
    dist.barrier()


@needs_library
def test_a_receiver_still_gets_what_a_leader_sent_before_it_closed(tmp_path):
    # Four connections: the frames sent before the root closed sit on some of them while
    # others already show the close; the receive must still be met.
    env = {**LOCAL, STREAMS_ENV: "4", CHUNK_BYTES_ENV: "8"}
    run_ranks(["nvidia", "amd"], _check_a_receiver_after_the_root_left, tmp_path,
              before_init=partial(_label_islands, None, env), init_kwargs={"bridge": "multitcp"})


@needs_library
def test_subgroups_on_the_multitcp_bridge(tmp_path):
    # A subgroup's members build its links on their own: their store keys must not
    # depend on how many links each process made before (three mixed groups here).
    from test_torch_backend import TWO_TWO, _subgroups, run_backend

    from gpubridge.config import BRIDGE_ENV

    run_backend(TWO_TWO, _subgroups, tmp_path, env={-1: {BRIDGE_ENV: "multitcp", **LOCAL}})


# ---- failures -------------------------------------------------------------------------

@needs_library
def test_a_bad_setting_on_one_leader_fails_init_on_every_rank(tmp_path):
    good = {VENDOR_ENV: "nvidia", **LOCAL}
    run_failing_init(
        [{"env": {**good, CHUNK_BYTES_ENV: "12"}, "init_kwargs": {"bridge": "multitcp"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "multitcp"}}],
        tmp_path,
        [r"The tcp bridge between ranks \[0, 1\] could not be set up",
         r"python -m gpubridge.multitcp build",
         rf"rank 0: ValueError: {CHUNK_BYTES_ENV}='12' must be"],
    )


@needs_library
def test_leaders_with_different_settings_fail_init_on_every_rank(tmp_path):
    good = {VENDOR_ENV: "nvidia", **LOCAL}
    run_failing_init(
        [{"env": {**good, STREAMS_ENV: "2"}, "init_kwargs": {"bridge": "multitcp"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "multitcp"}},
         {"env": good, "init_kwargs": {"bridge": "multitcp"}}],
        tmp_path,
        [r"could not be set up", r"\|4194304\|2 here but \d+\|4194304\|3 on rank 1"],
    )


def _check_deadline(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        bridge.timeout = 1.0
        tensor = torch.ones(10)
        start = time.monotonic()
        with pytest.raises(multitcp.TransportError, match=r"timed out: \d+ of \d+ operation"):
            bridge.all_reduce(tensor)  # the other leader never joins
        assert time.monotonic() - start < 10
        with pytest.raises(multitcp.TransportError, match="already failed"):
            bridge.all_reduce(tensor)
    dist.barrier()  # through the c10d store, not the bridge


@needs_library
def test_a_leader_that_never_joins_trips_the_deadline(tmp_path):
    run_ranks(["nvidia", "amd"], _check_deadline, tmp_path,
              before_init=partial(_label_islands, None, LOCAL),
              init_kwargs={"bridge": "multitcp"})


def _check_teardown(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        never = torch.empty(4)
        bridge.worker.recv(1, never.data_ptr(), 16, tcp_transport.tag(99, 0, 1, 0))
    dist.barrier()
    start = time.monotonic()
    gpubridge.destroy()  # closes the bridge with a receive still posted
    assert time.monotonic() - start < 15
    gpubridge.init(init_method=f"file://{os.environ['GB_RDZV']}-again", rank=topology.rank,
                   world_size=topology.world_size, bridge="multitcp",
                   timeout=timedelta(seconds=60))
    tensor = rank_tensor((5,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)  # a new bridge works after the old one closed
    assert torch.equal(tensor, expected_reduce((5,), torch.float32, topology.world_size,
                                               ReduceOp.SUM))


@needs_library
def test_teardown_with_an_operation_outstanding(tmp_path):
    env = {**LOCAL, "GB_RDZV": str(tmp_path / "rdzv")}
    run_ranks(["nvidia", "amd"], _check_teardown, tmp_path,
              before_init=partial(_label_islands, None, env), init_kwargs={"bridge": "multitcp"})


def _wrong_size(rank, path):
    worker = multitcp.Worker(2, rank, 2, "127.0.0.1")
    with open(f"{path}.{rank}", "w") as f:
        f.write(worker.address())
    other = f"{path}.{1 - rank}"
    deadline = time.monotonic() + 30
    while not os.path.exists(other) and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    with open(other) as f:
        worker.connect(1 - rank, f.read(), 10.0)
    buffer = torch.zeros(4, dtype=torch.int32)
    try:
        if rank == 0:  # 8 bytes, where rank 1 posted 16
            worker.wait([worker.send(1, buffer.data_ptr(), 8, 77)], time.monotonic() + 10)
            time.sleep(1.0)  # leave the connection open while rank 1 reads the header
        else:
            request = worker.recv(0, buffer.data_ptr(), 16, 77)
            with pytest.raises(multitcp.TransportError, match="sent 8 bytes for tag 0x4d"):
                worker.wait([request], time.monotonic() + 10)
    finally:
        worker.close(timeout=1.0)


@needs_library
def test_a_message_of_the_wrong_size_is_an_error_not_garbage(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_wrong_size, args=(str(tmp_path / "addr"),), nprocs=2)
