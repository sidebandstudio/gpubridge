"""P2P routing, matching, concurrency and failure on CPU processes."""

from __future__ import annotations

import collections
import gc
import json
import os
import threading
import time
import weakref
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _staging import LazyStaging
from test_ucx import TCP, needs_ucx

from gpubridge import topology
from gpubridge import p2p
from gpubridge import torch_backend
from gpubridge import batch_isend_irecv
from gpubridge.config import VENDOR_ENV
from gpubridge.torch_backend import GpubridgeProcessGroup


def _main(rank, vendors, labels, fn, rendezvous, seconds):
    os.environ[VENDOR_ENV] = vendors[rank]
    torch.set_num_threads(1)
    if labels:
        original = topology.plan_topology
        topology.plan_topology = lambda vendors: original(labels)
    dist.init_process_group("gpubridge", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=len(vendors), timeout=timedelta(seconds=seconds))
    try:
        fn(rank, len(vendors))
    finally:
        if dist.is_initialized():  # some tests time teardown themselves
            dist.destroy_process_group()


def run(vendors, fn, tmp_path, *, labels=None, seconds=15):
    mp.spawn(_main, nprocs=len(vendors), args=(vendors, labels, fn,
                                             str(tmp_path / "rendezvous"), seconds))


DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
          torch.float16, torch.bfloat16, torch.float32, torch.float64,
          torch.complex64, torch.complex128)


def _payload(n, dtype, src, seq=0):
    return ((torch.arange(n) % 97) + src * 3 + seq).to(dtype)


def _all_pairs(rank, world):
    dist.barrier()
    for src in range(world):
        for dst in range(world):
            if src == dst:
                continue
            for i, dtype in enumerate(DTYPES):
                for n in (0, 1, 31, 4097):
                    t = _payload(n, dtype, src) if rank == src else torch.empty(n, dtype=dtype)
                    if rank == src:
                        dist.send(t, dst, tag=i)
                    elif rank == dst:
                        assert dist.recv(t, src, tag=i) == src
                        assert torch.equal(t, _payload(n, dtype, src)), (src, dst, dtype, n)
    # Gloo native path and bridge path both ran, including non-leader ranks.
    dist.barrier()


@pytest.mark.parametrize(("vendors", "labels"), [
    (["nvidia", "amd"], None),
    (["nvidia", "nvidia", "amd", "amd"], None),
    (["nvidia", "amd", "nvidia", "nvidia"], None),
    (["nvidia"] * 4, ["nvidia", "amd", "third", "nvidia"]),
], ids=["1+1", "2+2", "uneven", "three-islands"])
def test_every_pair_sizes_and_dtypes(vendors, labels, tmp_path):
    run(vendors, _all_pairs, tmp_path, labels=labels)


def _tags_and_batches(rank, world, helper=False):
    dist.barrier()
    peer = 1 - rank
    # Different tags may be received in the opposite order. A single worker
    # waiting for the first receive would deadlock this exchange.
    if rank == 0:
        sends = [dist.isend(_payload(4097, torch.int64, rank, tag), peer, tag=tag)
                 for tag in (2**31 - 1, 12, 3)]
        for work in sends:
            work.wait()
    else:
        for tag in (3, 12, 2**31 - 1):
            t = torch.empty(4097, dtype=torch.int64)
            dist.recv(t, peer, tag=tag)
            assert torch.equal(t, _payload(4097, torch.int64, 0, tag))
    # Same tag, same pair: each message must match in program order.
    receive = [torch.empty(257, dtype=torch.int64) for _ in range(12)]
    ops = []
    for seq in range(12):
        ops.extend([dist.P2POp(dist.irecv, receive[seq], peer, tag=99),
                    dist.P2POp(dist.isend, _payload(257, torch.int64, rank, seq), peer, tag=99)])
    for work in (batch_isend_irecv if helper else dist.batch_isend_irecv)(ops):
        work.wait()
    for seq, t in enumerate(receive):
        assert torch.equal(t, _payload(257, torch.int64, peer, seq)), seq
    dist.barrier()


def test_out_of_order_tags_and_full_duplex_batch_fifo(tmp_path):
    run(["nvidia", "amd"], _tags_and_batches, tmp_path)


def test_explicit_batch_helper_on_old_and_new_torch(tmp_path):
    run(["nvidia", "amd"], partial(_tags_and_batches, helper=True), tmp_path)


@pytest.mark.parametrize("settings", [{"GPUBRIDGE_PINNED_STAGING": "1"},
                                      {p2p.CHUNK_BYTES_ENV: "128"}], ids=["pinned", "chunks"])
def test_staging_options_keep_tag_and_fifo_matching(settings, tmp_path, monkeypatch):
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    run(["nvidia", "amd"], _tags_and_batches, tmp_path)


@pytest.mark.parametrize("raw", ["0", "7", "12", "-8", "bad"])
def test_invalid_p2p_chunk_setting_is_refused(raw, monkeypatch):
    monkeypatch.setenv(p2p.CHUNK_BYTES_ENV, raw)
    with pytest.raises(ValueError):
        p2p.chunk_bytes_setting()


class _MemoryLink:
    def __init__(self, incoming):
        self.incoming = iter(incoming.split(16))
        self.sent = []

    def post(self, host, peer, tag, sending):
        if sending:
            self.sent.append(host.clone())
        else:
            host.copy_(next(self.incoming))
        return self

    def wait(self, timeout):
        pass


def _check_lazy_pipeline(monkeypatch, sending, mutation):
    expected = torch.arange(117, dtype=torch.uint8)  # 8 chunks, both slots reused
    link = _MemoryLink(expected)
    staging = LazyStaging(16, 2)
    if mutation == "skip-host-wait":
        staging.wait_host = lambda event: None
    elif mutation == "skip-final-stream-wait":
        staging.end = lambda event: None
    monkeypatch.setattr(p2p, "make_staging", lambda *args: staging)
    lane = object.__new__(p2p._Lane)
    lane.engine = SimpleNamespace(link=link, chunk_bytes=16)
    lane.peer, lane.tag, lane.sending = 1, 0, sending
    target = expected.clone() if sending else torch.full_like(expected, 254)
    lane._pipeline(target, time.monotonic() + 2)
    got = torch.cat(link.sent) if sending else target
    assert torch.equal(got, expected), "lazy-copy protocol returned incorrect bytes"


@pytest.mark.parametrize("sending", [False, True], ids=["receive", "send"])
def test_lazy_staging_preserves_bytes(sending, monkeypatch):
    _check_lazy_pipeline(monkeypatch, sending, "")


@pytest.mark.parametrize(("sending", "mutation"), [
    (True, "skip-host-wait"),
    (False, "skip-host-wait"),
    (False, "skip-final-stream-wait"),
], ids=["copy-out-before-send", "copy-back-before-slot-reuse", "output-stream-wait"])
def test_staging_ordering_mutations_are_detected(sending, mutation, monkeypatch):
    with pytest.raises(AssertionError, match="incorrect bytes"):
        _check_lazy_pipeline(monkeypatch, sending, mutation)


def _concurrent(rank, world):
    dist.barrier()
    receives, works = [], []
    for peer in reversed(range(world)):
        if peer == rank:
            continue
        incoming = torch.empty(2**18, dtype=torch.int32)
        receives.append((peer, incoming))
        works.append(dist.irecv(incoming, peer, tag=41))
        works.append(dist.isend(_payload(2**18, torch.int32, rank), peer, tag=41))
    for work in reversed(works):
        work.wait()
    for peer, incoming in receives:
        assert torch.equal(incoming, _payload(2**18, torch.int32, peer))
    dist.barrier()


def test_all_pairs_in_flight_native_and_bridged(tmp_path):
    run(["nvidia", "nvidia", "amd", "amd"], _concurrent, tmp_path)


@needs_ucx
@pytest.mark.parametrize("chunks", [False, True], ids=["whole", "chunked"])
def test_ucx_p2p_reverse_tags_and_duplex(chunks, tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBRIDGE_BRIDGE", "ucx")
    for key, value in TCP.items():
        monkeypatch.setenv(key, value)
    if chunks:
        monkeypatch.setenv(p2p.CHUNK_BYTES_ENV, "128")
    run(["nvidia", "amd"], _tags_and_batches, tmp_path)


@needs_ucx
def test_ucx_p2p_includes_nonleaders_and_multiple_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBRIDGE_BRIDGE", "ucx")
    for key, value in TCP.items():
        monkeypatch.setenv(key, value)
    run(["nvidia", "nvidia", "amd", "amd"], _concurrent, tmp_path)


def _slow_receiver(rank, world):
    dist.barrier()
    if rank == 0:
        slow = dist.isend(torch.ones(11), 1, tag=7)
        fast = dist.isend(torch.full((9,), 2.), 2, tag=8)
        fast.wait()
        assert not slow.is_completed(), "the slow receive must still be waiting"
        slow.wait()
    elif rank == 1:
        time.sleep(0.6)
        t = torch.empty(11)
        dist.recv(t, 0, tag=7)
        assert torch.equal(t, torch.ones(11))
    else:
        t = torch.empty(9)
        dist.recv(t, 0, tag=8)
        assert torch.equal(t, torch.full((9,), 2.))
    dist.barrier()


def test_a_slow_peer_does_not_block_other_pairs(tmp_path):
    run(["nvidia", "amd", "amd"], _slow_receiver, tmp_path)


def _subgroup(rank, world):
    dist.barrier()
    group = dist.new_group([1, 3])
    if rank in (1, 3):
        dist.barrier(group=group)
        t = torch.full((17,), 42) if rank == 3 else torch.empty(17, dtype=torch.int64)
        if rank == 3:
            dist.send(t, 1, group=group, tag=5)
        else:
            dist.recv(t, 3, group=group, tag=5)
            assert torch.equal(t, torch.full((17,), 42))
        dist.barrier(group=group)
    dist.barrier()


def test_subgroup_peers_are_global_ranks(tmp_path):
    run(["nvidia", "nvidia", "amd", "amd"], _subgroup, tmp_path)


def _refusals(rank, world):
    with pytest.raises(RuntimeError, match="call dist.barrier"):
        dist.isend(torch.ones(1), 1 - rank)
    dist.barrier()
    with pytest.raises(ValueError, match="contiguous"):
        dist.isend(torch.ones(2, 3).T, 1 - rank)
    with pytest.raises(ValueError, match="tag"):
        dist.isend(torch.ones(1), 1 - rank, tag=-1)
    with pytest.raises(NotImplementedError, match="any-source"):
        dist.irecv(torch.empty(1))
    dist.barrier()


def test_refusals_do_not_start_communication(tmp_path):
    run(["nvidia", "amd"], _refusals, tmp_path)


def _mismatch(rank, world, dtype):
    dist.barrier()
    t = torch.ones(4 if rank == 0 or dtype else 3,
                   dtype=torch.int64 if rank == 0 or not dtype else torch.float64)
    work = dist.isend(t, 1) if rank == 0 else dist.irecv(t, 0)
    with pytest.raises(RuntimeError, match="payload mismatch"):
        work.wait()
    with pytest.raises(RuntimeError, match="payload mismatch"):
        dist.barrier()


@pytest.mark.parametrize("dtype", [False, True], ids=["size", "dtype"])
def test_payload_mismatch_errors_on_both_ranks_without_gloo_abort(dtype, tmp_path):
    run(["nvidia", "amd"], partial(_mismatch, dtype=dtype), tmp_path)


def _missing(rank, world):
    dist.barrier()
    if rank == 1:
        # This is a graceful process exit with no posted receive. The sender's
        # metadata deadline must fail, and teardown must not wait for a peer.
        return
    start = time.monotonic()
    work = dist.isend(torch.ones(13), 1)
    with pytest.raises(RuntimeError, match="timed out"):
        work.wait()
    assert time.monotonic() - start < 5


def test_missing_peer_times_out_and_teardown_is_bounded(tmp_path):
    run(["nvidia", "amd"], _missing, tmp_path, seconds=1)


def _dead_peer(rank, world):
    dist.barrier()
    if rank == 1:
        os._exit(0)  # no cleanup: simulate a peer disappearing during the job
    if rank == 0:
        work = dist.isend(torch.ones(13), 1)
    else:
        time.sleep(0.35)  # its own deadline is later: it must see rank 0's failure
        work = dist.irecv(torch.empty(13), 0, tag=19)
    with pytest.raises((RuntimeError, TimeoutError), match="rank 1.*timed out"):
        work.get_future().wait()
    with pytest.raises(RuntimeError, match="timed out"):
        dist.barrier()


def test_dead_peer_failure_reaches_other_pending_pairs(tmp_path):
    run(["nvidia", "amd", "amd"], _dead_peer, tmp_path, seconds=1)


class _BlockedLink:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def post(self, tensor, peer, tag, sending):
        self.entered.set()
        return self

    def wait(self, timeout):
        self.release.wait(2)

    def close(self):
        pass


def test_teardown_fails_handle_without_waiting_for_inflight_transport(monkeypatch):
    link = _BlockedLink()
    engine = p2p.Engine(link, torch.device("cpu"), 0, dist.HashStore(), timedelta(seconds=5))
    monkeypatch.setattr(engine, "match", lambda *args: None)
    work = engine.submit(torch.ones(3), 1, 0, True)
    assert link.entered.wait(2)
    start = time.monotonic()
    engine.close(timeout=0.02)
    assert time.monotonic() - start < 0.2
    with pytest.raises(RuntimeError, match="destroyed"):
        work.wait()
    link.release.set()


def test_torch_work_wait_and_future_both_report_transfer_failure(monkeypatch):
    link = _BlockedLink()

    def fail(timeout):
        raise ValueError("injected transfer failure")

    monkeypatch.setattr(link, "wait", fail)
    engine = p2p.Engine(link, torch.device("cpu"), 0, dist.HashStore(), timedelta(seconds=5))
    monkeypatch.setattr(engine, "match", lambda *args: None)
    work = p2p.TorchWork(engine.submit(torch.ones(3), 1, 0, True), 0)
    try:
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            work.wait()
        with pytest.raises(ValueError, match="injected transfer failure"):
            work.get_future().wait()
        assert work.is_completed() and not work.is_success()
    finally:
        engine.close()


def _check_lane_fifo(monkeypatch, mutation):
    link = _BlockedLink()
    posted = []
    original_post = link.post

    def post(tensor, peer, tag, sending):
        posted.append(int(tensor[0]))
        return original_post(tensor, peer, tag, sending)

    monkeypatch.setattr(link, "post", post)
    engine = p2p.Engine(link, torch.device("cpu"), 0, dist.HashStore(), timedelta(seconds=5))
    monkeypatch.setattr(engine, "match", lambda *args: None)
    try:
        works = [engine.submit(torch.tensor([0], dtype=torch.uint8), 1, 9, True)]
        assert link.entered.wait(2)
        for i in (1, 2):
            if mutation:
                # Remove lane reuse, the fix that keeps repeated same-tag
                # requests behind their predecessor rather than posting at once.
                engine._lanes.clear()
            works.append(engine.submit(torch.tensor([i], dtype=torch.uint8), 1, 9, True))
        time.sleep(0.03)
        assert posted == [0], "a later same-tag request overtook its predecessor"
        link.release.set()
        for work in works:
            work.wait()
        assert posted == [0, 1, 2]
    finally:
        link.release.set()
        engine.close()


def test_same_tag_fifo_and_mutation(monkeypatch):
    _check_lane_fifo(monkeypatch, False)
    with pytest.raises(AssertionError, match="overtook"):
        _check_lane_fifo(monkeypatch, True)


def _callback_dependency(rank, world, tags):
    dist.barrier()
    store = dist.distributed_c10d._get_default_store()
    if rank == 1:
        a = dist.isend(torch.ones(5), 3, tag=tags[0])
        b = dist.isend(torch.full((5,), 2.), 3, tag=tags[1])

        def callback(future):
            store.set("callback-entered", "1")
            # b completes while this callback runs; it must not need a lock
            # or a lane thread that the callback is holding.
            return b.wait(timedelta(seconds=3))

        chained = a.get_future().then(callback)
        store.set("callback-registered", "1")
        assert chained.wait() is True
        b.wait()
    elif rank == 3:
        store.wait(["callback-registered"])
        t = torch.empty(5)
        dist.recv(t, 1, tag=tags[0])
        assert torch.equal(t, torch.ones(5))
        store.wait(["callback-entered"])
        dist.recv(t, 1, tag=tags[1])
        assert torch.equal(t, torch.full((5,), 2.))
    dist.barrier()


@pytest.mark.parametrize("tags", [(0, 1), (4, 4)], ids=["another-tag", "same-tag"])
def test_future_callback_can_wait_for_a_later_message(tags, tmp_path):
    run(["nvidia", "nvidia", "amd", "amd"], partial(_callback_dependency, tags=tags), tmp_path)


def _blocked_callback_teardown(rank, world):
    dist.barrier()
    entered, release = threading.Event(), threading.Event()
    chained = None
    if rank == 0:
        work = dist.isend(torch.ones(7), 1, tag=3)
        chained = work.get_future().then(lambda future: (entered.set(), release.wait(10)))
        work.wait()
    else:
        dist.recv(torch.empty(7), 0, tag=3)
    # Neither a later collective nor teardown may wait for the callback.
    start = time.monotonic()
    dist.barrier()
    if rank == 0:
        assert entered.wait(5)
    dist.destroy_process_group()
    elapsed = time.monotonic() - start
    release.set()
    if chained is not None:
        # Exiting while a daemon thread is inside torch's Future aborts on Linux.
        chained.wait()
        for thread in threading.enumerate():
            if thread.name == "gpubridge-p2p-completion":
                thread.join(5)
    assert elapsed < 3, f"barrier and teardown waited {elapsed:.1f} s for a blocked callback"


def test_teardown_does_not_wait_for_a_blocked_callback(tmp_path):
    run(["nvidia", "amd"], _blocked_callback_teardown, tmp_path)


def _released_payloads(rank, world):
    dist.barrier()
    references = []
    peer = 1 - rank
    for tag in range(8):
        tensor = (_payload(2**20, torch.float32, rank, tag) if rank == 0
                  else torch.empty(2**20))
        references.append(weakref.ref(tensor))
        work = dist.isend(tensor, peer, tag=tag) if rank == 0 else dist.irecv(tensor, peer, tag=tag)
        if tag % 2:
            work.get_future().wait()
        else:
            work.wait()
        if rank == 1:
            assert torch.equal(tensor, _payload(2**20, torch.float32, 0, tag))
        del work, tensor
    dist.barrier()
    deadline = time.monotonic() + 3
    while any(ref() is not None for ref in references) and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
    alive = sum(ref() is not None for ref in references)
    assert alive == 0, f"{alive}/8 completed payloads retained after the caller dropped them"
    dist.barrier()


def test_completed_send_and_recv_payloads_are_released(tmp_path):
    run(["nvidia", "amd"], _released_payloads, tmp_path)


def _lane_threads():
    return sum(thread.name == "gpubridge-p2p" for thread in threading.enumerate())


def _fresh_tags_and_retired_lanes(rank, world):
    dist.barrier()
    peer = 1 - rank
    most = 0
    for tag in range(100, 140):
        t = torch.full((3,), tag) if rank == 0 else torch.empty(3, dtype=torch.int64)
        (dist.send if rank == 0 else dist.recv)(t, peer, tag=tag)
        assert torch.equal(t, torch.full((3,), tag))
        most = max(most, _lane_threads())
    assert most <= 2, f"{most} lane threads alive during 40 sequential fresh tags"
    # Same tag after its lane retired: the sender's lane exits between
    # messages while the receiver's stays busy, so their sequences must agree.
    if rank == 0:
        for seq in range(4):
            dist.send(_payload(9, torch.int64, rank, seq), peer, tag=7)
            time.sleep(0.05)
    else:
        receive = [torch.empty(9, dtype=torch.int64) for _ in range(4)]
        works = [dist.irecv(t, peer, tag=7) for t in receive]
        for work in works:
            work.wait()
        for seq, t in enumerate(receive):
            assert torch.equal(t, _payload(9, torch.int64, 0, seq)), seq
    deadline = time.monotonic() + 2
    while _lane_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _lane_threads() == 0
    dist.barrier()


def test_fresh_tags_do_not_accumulate_threads_and_sequences_survive(tmp_path):
    run(["nvidia", "amd"], _fresh_tags_and_retired_lanes, tmp_path)


def _self_wait_callback(rank, world):
    dist.barrier()
    peer = 1 - rank
    t = torch.full((4,), 6.) if rank == 0 else torch.empty(4)
    work = dist.isend(t, peer, tag=2) if rank == 0 else dist.irecv(t, peer, tag=2)
    chained = work.get_future().then(
        lambda future: (work.wait(timedelta(seconds=2)), work.is_completed()))
    assert chained.wait() == (True, True)
    assert torch.equal(t, torch.full((4,), 6.))
    dist.barrier()


def test_future_callback_sees_its_own_work_completed(tmp_path):
    run(["nvidia", "amd"], _self_wait_callback, tmp_path)


def test_cpu_request_on_a_lane_that_has_a_stream_uses_no_stream(monkeypatch):
    engine = p2p.Engine(_BlockedLink(), torch.device("cpu"), 0, dist.HashStore(),
                        timedelta(seconds=5))
    entered, release = threading.Event(), threading.Event()

    def match(*args):
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(engine, "match", match)
    engine.link.release.set()
    try:
        first = engine.submit(torch.ones(2), 1, 0, True)
        assert entered.wait(2)
        # As if an earlier CUDA request on this (peer, tag, direction) created it.
        engine._lanes[1, 0, True].stream = object()
        second = engine.submit(torch.ones(2), 1, 0, True)
        release.set()
        assert first.wait(2) and second.wait(2)
        assert second._event is None and second.is_completed()
    finally:
        release.set()
        engine.close()


def test_tag_range_is_checked_without_a_process_group():
    pg = GpubridgeProcessGroup(0, 2, "unit", None)
    with pytest.raises(ValueError, match="tag"):
        pg.send([torch.empty(1)], 1, 2**31)


def _check_native_grouping(monkeypatch, mutation):
    log = []

    class Native:
        active = False

        def _start_coalescing(self, device):
            log.append("start")
            self.active = not mutation

        def _end_coalescing(self, device):
            log.append("end")
            self.active = False
            return torch_backend._completed([])

    native = Native()
    device = torch.device("cuda:0")
    tensor = SimpleNamespace(device=device, layout=torch.strided, is_contiguous=lambda: True)
    pg = GpubridgeProcessGroup(0, 2, "unit-native", None)
    topo = SimpleNamespace(device=device, island=SimpleNamespace(ranks=(0, 1)),
                           island_group=native)
    monkeypatch.setattr(torch_backend, "_owner", pg)
    monkeypatch.setattr(pg, "_topology", lambda: topo)

    def post(kind, *args, **kwargs):
        assert native.active, "native P2P was posted outside its NCCL/RCCL batch"
        log.append(kind)
        return torch_backend._completed([])

    monkeypatch.setattr(dist, "isend", partial(post, "send"))
    monkeypatch.setattr(dist, "irecv", partial(post, "recv"))
    pg._start_coalescing(device)
    pg.send([tensor], 1, 7)
    pg.recv([tensor], 1, 7)
    pg._end_coalescing(device).wait()
    assert log == ["start", "send", "recv", "end"]


def test_native_gpu_posts_are_grouped_and_mutation_is_detected(monkeypatch):
    _check_native_grouping(monkeypatch, False)
    with pytest.raises(AssertionError, match="outside its NCCL/RCCL batch"):
        _check_native_grouping(monkeypatch, True)


# ---- batches and callbacks (review round 2 of the lifecycle fix) -----------------------------

def _within(fn, seconds: float) -> bool:
    """Run ``fn`` on a helper thread; whether it returned within ``seconds``."""
    done = threading.Event()
    threading.Thread(target=lambda: (fn(), done.set()), daemon=True).start()
    return done.wait(seconds)


def _callback_waits_on_a_batch(rank, world):
    dist.barrier()
    store = dist.distributed_c10d._get_default_store()
    if rank == 0:
        a = dist.isend(torch.ones(5), 1, tag=0)
        b = batch_isend_irecv([dist.P2POp(dist.isend, torch.full((5,), 2.0), 1, tag=1)])[0]
        seen = {}

        def callback(future):
            store.set("entered", "1")
            store.wait(["received-b"], timedelta(seconds=10))  # the peer has b's payload
            seen["b"] = _within(b.wait, 5)

        chained = a.get_future().then(callback)
        store.set("registered", "1")
        chained.wait()
        assert seen["b"], "a callback on a's future couldn't see batch b complete"
    else:
        store.wait(["registered"])
        t = torch.empty(5)
        dist.recv(t, 0, tag=0)
        store.wait(["entered"])
        dist.recv(t, 0, tag=1)
        assert torch.equal(t, torch.full((5,), 2.0))
        store.set("received-b", "1")
    dist.barrier()


def test_a_callback_can_wait_on_a_batch_handle(tmp_path):
    # Before: a batch's Work waited on its members' futures, which the group's one
    # completion thread resolves, and that thread was running this very callback.
    run(["nvidia", "amd"], _callback_waits_on_a_batch, tmp_path)


def _slow_callback_and_a_batch(rank, world):
    dist.barrier()
    if rank == 0:
        a = dist.isend(torch.ones(3), 1, tag=0)
        a.get_future().then(lambda _: time.sleep(2))
        a.wait()
        start = time.monotonic()
        for work in batch_isend_irecv([dist.P2POp(dist.isend, torch.ones(3), 1, tag=1)]):
            work.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 1, f"a batch's wait() took {elapsed:.2f} s behind a slow callback"
    else:
        dist.recv(torch.empty(3), 0, tag=0)
        dist.recv(torch.empty(3), 0, tag=1)
    dist.barrier()


def test_a_slow_callback_does_not_delay_a_batch_wait(tmp_path):
    run(["nvidia", "amd"], _slow_callback_and_a_batch, tmp_path)


def _batch_wait_with_a_timeout(rank, world):
    dist.barrier()
    if rank == 0:
        (work,) = batch_isend_irecv([dist.P2POp(dist.irecv, torch.empty(3), 1, tag=5)])
        with pytest.raises(TimeoutError):  # the peer sends only after this
            work.wait(timedelta(seconds=0.2))
        dist.barrier()
        assert work.wait(timedelta(seconds=10)) and work.is_completed()
        assert work.exception() is None
    else:
        dist.barrier()
        dist.send(torch.ones(3), 0, tag=5)
    dist.barrier()


def test_a_batch_wait_takes_a_finite_timeout(tmp_path):
    run(["nvidia", "amd"], _batch_wait_with_a_timeout, tmp_path)


def _batch_with_a_same_island_peer(rank, world):
    # Ranks 0 and 1 are one island: on Gloo in simulation, whose P2P Works have no future
    # and can be waited on only once. Rank 2 is the other island, across the bridge.
    dist.barrier()
    if rank == 0:
        start = time.monotonic()
        works = batch_isend_irecv([dist.P2POp(dist.isend, torch.ones(3), 1, tag=2),
                                   dist.P2POp(dist.isend, torch.full((3,), 2.0), 2, tag=2)])
        for work in works:
            work.wait()
        assert time.monotonic() - start < 5, "a batch with a same-island peer stalled"
        assert all(work.is_completed() and work.exception() is None for work in works)
    elif rank in (1, 2):
        t = torch.empty(3)
        dist.recv(t, 0, tag=2)
        assert torch.equal(t, torch.full((3,), float(rank)))
    dist.barrier()


def test_a_batch_with_a_same_island_peer_completes(tmp_path):
    # A batch Work once waited a second time on the same-island Gloo send, which then
    # waited for a transfer that never came, until the group's timeout.
    run(["nvidia", "nvidia", "amd"], _batch_with_a_same_island_peer, tmp_path, seconds=15)


# ---- the descriptor handshake on a shared Store ---------------------------------------------

class _InstantLink:
    """Transfers that complete at once: what is left to time is the Store handshake."""

    def post(self, tensor, peer, tag, sending):
        return self

    def wait(self, timeout):
        pass

    def close(self):
        pass


def _two_clients(kind, tmp_path):
    """Two clients of one Store, one per engine, as each rank has its own."""
    if kind == "tcp":
        server = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=10))
        return server, dist.TCPStore("127.0.0.1", server.port, 1, False, timedelta(seconds=10))
    return dist.FileStore(str(tmp_path / "store"), 2), dist.FileStore(str(tmp_path / "store"), 2)


@pytest.mark.parametrize("kind", ["tcp", "file"])
def test_waiting_lanes_do_not_hold_up_ready_ones_and_every_wait_ends(kind, tmp_path, capfd):
    # Every lane of an engine shares its Store client. Lanes waiting for peers that never
    # come (there is no rank 2) or come late must not delay pairs that are ready, on
    # either Store; then each wait ends by its deadline, and no Store timeout is logged.
    mine, theirs = _two_clients(kind, tmp_path)
    a = p2p.Engine(_InstantLink(), torch.device("cpu"), 0, mine, timedelta(seconds=3))
    b = p2p.Engine(_InstantLink(), torch.device("cpu"), 1, theirs, timedelta(seconds=3))
    try:
        missing = [a.submit(torch.ones(1), 2, tag, tag % 2 == 0) for tag in range(8)]
        late = a.submit(torch.ones(3), 1, 20, True)
        time.sleep(0.05)  # the missing and late lanes are waiting
        start = time.monotonic()
        ready = [a.submit(torch.ones(2), 1, 10 + i, i % 2 == 0) for i in range(4)]
        peers = [b.submit(torch.empty(2), 0, 10 + i, i % 2 == 1) for i in range(4)]
        for work in ready + peers:
            assert work.wait(timedelta(seconds=2))
        elapsed = time.monotonic() - start
        assert elapsed < 1, f"ready pairs took {elapsed:.2f} s behind waiting lanes"
        b.submit(torch.empty(3), 0, 20, False).wait(timedelta(seconds=2))  # a late peer
        assert late.wait(timedelta(seconds=2))
        assert not any(work.is_completed() for work in missing)
        for work in missing:  # rank 2 never comes: each fails by its deadline
            with pytest.raises(RuntimeError, match="timed out"):
                work.wait(timedelta(seconds=4))
        assert time.monotonic() - start < 4.5
    finally:
        a.close()
        b.close()
    assert "waitForInput" not in capfd.readouterr().err


@pytest.mark.parametrize("kind", ["tcp", "file"])
def test_close_ends_waiting_lanes_at_once(kind, tmp_path):
    mine, _ = _two_clients(kind, tmp_path)
    engine = p2p.Engine(_InstantLink(), torch.device("cpu"), 0, mine, timedelta(seconds=30))
    works = [engine.submit(torch.ones(1), 1, tag, True) for tag in range(4)]
    time.sleep(0.05)
    lanes = list(engine._lanes.values())
    assert len(lanes) == 4
    start = time.monotonic()
    engine.close(timeout=0.5)
    for work in works:
        with pytest.raises(RuntimeError, match="destroyed"):
            work.wait(timedelta(seconds=1))
    assert not any(lane.thread.is_alive() for lane in lanes), "a waiting lane outlived close()"
    assert time.monotonic() - start < 0.5


class _SlowGet:
    """A Store whose get() takes ``delay`` seconds, like a busy server."""

    def __init__(self, store, delay):
        self.store, self.delay = store, delay

    def get(self, key):
        time.sleep(self.delay)
        return self.store.get(key)

    def __getattr__(self, name):
        return getattr(self.store, name)


def test_a_deadline_passed_in_a_store_call_fails_even_when_the_peer_was_ready():
    store = dist.HashStore()
    engine = p2p.Engine(_InstantLink(), torch.device("cpu"), 0, _SlowGet(store, 0.2),
                        timedelta(seconds=5))
    try:
        store.set("0/1/7/0/recv", json.dumps([4, str(torch.float32), 0]))  # the peer is ready
        with pytest.raises(TimeoutError, match="timed out"):
            engine.match(torch.ones(1), 1, 7, 0, True, time.monotonic() + 0.1)
    finally:
        engine.close()


class _Counted:
    """A Store that counts each method's calls."""

    def __init__(self, store):
        self.store, self.calls = store, collections.Counter()

    def __getattr__(self, name):
        method = getattr(self.store, name)

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return method(*args, **kwargs)

        return counted


def test_an_idle_watcher_asks_nothing_and_wakes_for_a_posted_request(monkeypatch):
    # With nothing pending the health watcher makes no Store call. A request submitted
    # then, already posted to the link, still learns of a remote failure within the
    # watch interval: submit() wakes the watcher.
    store = _Counted(dist.HashStore())
    link = _BlockedLink()
    engine = p2p.Engine(link, torch.device("cpu"), 0, store, timedelta(seconds=5))
    try:
        time.sleep(0.55)  # eleven of the 50 ms intervals at which the watcher used to poll
        assert store.calls["check"] == 0, "an idle watcher polled the Store"
        monkeypatch.setattr(engine, "match", lambda *args: None)
        work = engine.submit(torch.ones(1), 1, 0, True)
        assert link.entered.wait(2)  # the transfer is posted and blocks
        store.store.set("failure", "remote failure")
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="remote failure"):
            work.wait(timedelta(seconds=2))
        elapsed = time.monotonic() - start
        assert elapsed < 0.3, f"the watcher noticed the failure only after {elapsed:.2f} s"
    finally:
        link.release.set()
        engine.close()


class _SlowCheck:
    """A Store whose ``check`` takes a while, as a busy server's does, and that says
    when the health watcher is inside it."""

    def __init__(self, store, seconds: float) -> None:
        self.store, self.seconds = store, seconds
        self.watcher_inside = threading.Event()
        self.watcher_left = threading.Event()

    def __getattr__(self, name):
        return getattr(self.store, name)

    def check(self, keys):
        watcher = threading.current_thread().name == "gpubridge-p2p-health"
        if watcher:
            self.watcher_inside.set()
            time.sleep(self.seconds)
        found = self.store.check(keys)
        if watcher:
            self.watcher_left.set()
        return found


def test_close_waits_for_the_watcher_to_leave_its_store_call(monkeypatch):
    # A script often exits right after it destroys its groups. A thread still inside a
    # Store call when the interpreter finalizes is ended there, inside torch's C++, and
    # that terminates the process: close() must not return while the watcher is in one.
    store = _SlowCheck(dist.HashStore(), 0.3)
    link = _BlockedLink()
    engine = p2p.Engine(link, torch.device("cpu"), 0, store, timedelta(seconds=5))
    try:
        monkeypatch.setattr(engine, "match", lambda *args: None)
        engine.submit(torch.ones(1), 1, 0, True)  # something pending: the watcher polls
        assert store.watcher_inside.wait(2), "the watcher never asked the Store"
        assert not store.watcher_left.is_set()
        engine.close()
        assert store.watcher_left.is_set(), "close() returned with the watcher in a Store call"
        assert not engine._watcher.is_alive()
    finally:
        link.release.set()
        engine.close()


def test_watcher_exits_when_closed_between_stop_check_and_lock():
    # If close() runs after the watcher has evaluated _stop.is_set() outside the lock,
    # the watcher must check for stop/closed under the lock before clearing _wake,
    # exiting cleanly without clearing close()'s wake or leaking the daemon thread.
    engine = p2p.Engine(_InstantLink(), torch.device("cpu"), 0, dist.HashStore(),
                        timedelta(seconds=2))
    entered = threading.Event()
    release = threading.Event()
    original_is_set = engine._stop.is_set
    paused = False

    def paused_is_set():
        nonlocal paused
        value = original_is_set()
        if threading.current_thread() is engine._watcher and not value and not paused:
            paused = True
            entered.set()
            assert release.wait(2), "test timed out waiting to release watcher"
        return value

    engine._stop.is_set = paused_is_set
    engine._wake.set()
    assert entered.wait(2), "watcher did not reach between stop check and lock"
    engine.close()
    release.set()
    engine._watcher.join(0.5)
    assert not engine._watcher.is_alive(), "watcher thread leaked after close()"
    assert engine._wake.is_set(), "close()'s wake signal was cleared by watcher"
