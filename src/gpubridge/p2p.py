"""Independent FIFO lanes for the torch backend's cross-island P2P operations.

The collective worker cannot run these: a receive waiting for one peer must
not prevent posting a send to another peer (or in the opposite direction).
Each (peer, tag, direction) lane has its own queue and stream. It keeps the
input alive until completion, and never reorders messages in that lane.

A lane's thread exits as soon as its queue is empty, so resident threads
follow the keys with queued or running operations, not every tag ever used;
sequence numbers live in the engine and survive the lane. User futures are
resolved on one completion thread per engine, never on a lane thread or under
the engine lock.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from gpubridge.work import Work
from gpubridge.config import parse_size
from gpubridge.staging import make_staging

CHUNK_BYTES_ENV = "GPUBRIDGE_P2P_CHUNK_BYTES"

#: How long ``Engine.match`` pauses between looks for the peer's descriptor: from
#: ``MATCH_POLL_FIRST`` seconds, doubling up to ``MATCH_POLL_MAX``, the fixed interval it
#: used to have, so a long wait costs the Store no more calls than before. Each look is one
#: short Store call: a lane never holds the Store client that every lane shares while it
#: waits, as a blocking ``store.wait`` would (TCPStore serializes a client's requests).
MATCH_POLL_FIRST = 50e-6
MATCH_POLL_MAX = 0.002
#: How often the health watcher asks the Store for a published failure while operations
#: are pending. With none pending it asks nothing: ``submit`` wakes it again.
WATCH_INTERVAL = 0.05
#: How long ``Engine.close`` waits for the watcher to leave its last Store call.
WATCHER_JOIN = 1.0


def chunk_bytes_setting() -> int:
    """Zero means one whole message; a positive setting uses two staging slots."""
    raw = os.environ.get(CHUNK_BYTES_ENV, "").strip()
    if not raw:
        return 0
    size = parse_size(raw)
    if size < 8 or size % 8:
        raise ValueError(f"{CHUNK_BYTES_ENV} must be a positive multiple of 8 bytes")
    return size


class TorchWork(dist.Work):
    """Keep Python exceptions and stream ordering through torch's P2P entrypoints.

    ``_create_work_from_future`` erases the Python Future's exception unwrap
    callback on torch 2.14.1: wait() would report success for a failed send.
    Delegating explicitly also preserves Work.wait's optional host timeout.
    """

    def __init__(self, work: Work, source: int) -> None:
        super().__init__()
        self.work, self.source = work, source

    def wait(self, timeout=None) -> bool:
        return self.work.wait(timeout)

    def is_completed(self) -> bool:
        return self.work.is_completed()

    def is_success(self) -> bool:
        return self.is_completed() and self.work.exception() is None

    def get_future(self):
        return self.work.get_future()

    def exception(self):
        return self.work.exception()

    def synchronize(self) -> None:
        self.wait()

    def _source_rank(self) -> int:
        return self.source


class _P2PWork(Work):
    """Publishes host completion only; the engine's completer resolves the future.

    ``wait()`` and ``is_completed()`` therefore see completion before any
    callback runs, including a callback waiting on this same handle.
    """

    def _finish(self, error: BaseException | None) -> None:
        self._error = error
        self._done.set()


class _Completer:
    """Resolves futures one at a time, in completion order, on its own thread.

    Future callbacks run inline in ``set_result``. A callback may wait on any
    P2P handle, since lanes publish completion without this thread, but a
    callback that blocks delays the later futures of this group.
    """

    def __init__(self) -> None:
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.streams: dict[torch.device, torch.cuda.Stream] = {}
        self.thread = threading.Thread(target=self._run, name="gpubridge-p2p-completion",
                                       daemon=True)
        self.thread.start()

    def put(self, work: _P2PWork) -> None:
        self.queue.put(work)

    def close(self) -> None:
        self.queue.put(None)  # resolves what is queued first; never joined

    def _run(self) -> None:
        while True:
            work = self.queue.get()
            if work is None:
                return
            self._resolve(work)
            del work

    def _resolve(self, work: _P2PWork) -> None:
        context = contextlib.nullcontext()
        if work._event is not None:
            # The device-aware future records on the current stream; use a
            # private one ordered after the transfer, not the caller's.
            if work._device not in self.streams:
                self.streams[work._device] = torch.cuda.Stream(work._device)
            stream = self.streams[work._device]
            stream.wait_event(work._event)
            context = torch.cuda.stream(stream)
        with context:
            if work._error is None:
                work._future.set_result(work._result)
            else:
                work._future.set_exception(work._error)


class GlooLink:
    """Raw bytes over an already-created Gloo group of all members."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group

    def post(self, tensor: torch.Tensor, peer: int, tag: int, sending: bool):
        call = dist.isend if sending else dist.irecv
        arg = {"dst" if sending else "src": peer}
        return call(tensor, group=self.group, tag=tag, **arg)

    def close(self) -> None:
        pass  # torch owns the process group


class Engine:
    """One rank's P2P state. Creation is local; only submitted pairs communicate.

    A small Store handshake checks byte count and dtype before Gloo sees the
    payload: Gloo can terminate the process on a size mismatch. Metadata keys
    include the user tag and lane sequence, and are removed by their reader.
    Failures are shared with all members under this backend group's namespace.
    """

    def __init__(self, link, device: torch.device, rank: int, store,
                 timeout: timedelta | None, *, pinned: bool = False,
                 chunk_bytes: int = 0) -> None:
        self.link, self.device, self.rank, self.store = link, device, rank, store
        self.timeout = (timeout or timedelta(minutes=30)).total_seconds()
        self.pinned, self.chunk_bytes = pinned, chunk_bytes
        self._lock = threading.RLock()
        self._lanes: dict[tuple[int, int, bool], _Lane] = {}
        self._sequences: dict[tuple[int, int, bool], int] = {}
        self._pending: dict[_P2PWork, None] = {}  # ordered: failures resolve in posting order
        self._closed = False
        self._failed: BaseException | None = None
        self._completer = _Completer()
        self._stop = threading.Event()
        #: Set by submit() and close(): wakes a watcher parked with nothing pending.
        self._wake = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="gpubridge-p2p-health",
                                         daemon=True)
        self._watcher.start()

    def _watch(self) -> None:
        """While operations are pending, look for a failure every WATCH_INTERVAL seconds;
        with none, park until submit() or close(). A request already posted to the link
        learns of a remote failure only from here."""
        while not self._stop.is_set():
            with self._lock:  # the lock submit() adds under: no wake-up is lost
                if self._closed or self._stop.is_set():
                    return
                idle = not self._pending
                if idle:
                    self._wake.clear()
            if idle:
                self._wake.wait()
                continue
            # On a fixed grid of the monotonic clock, not from the wake-up: a failure
            # published right after submit() is seen as soon as it was when this thread
            # polled all the time, at most WATCH_INTERVAL later.
            if self._stop.wait(WATCH_INTERVAL - time.monotonic() % WATCH_INTERVAL):
                return
            try:
                self.check()
            except BaseException as exc:  # noqa: BLE001 - fail pending handles on store loss too
                self._fail(exc, publish=False)
                return

    def check(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("gpubridge point-to-point group was destroyed")
            if self._failed is not None:
                raise RuntimeError(f"gpubridge point-to-point group failed: {self._failed}")
        if self.store.check(["failure"]):
            error = RuntimeError(self.store.get("failure").decode())
            self._fail(error, publish=False)
            raise error

    def submit(self, tensor: torch.Tensor, peer: int, tag: int, sending: bool) -> Work:
        self.check()
        ready = None
        if tensor.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(tensor.device))
        work = _P2PWork("send" if sending else "recv", tensor.device, [tensor])
        with self._lock:
            self.check()
            key = peer, tag, sending
            sequence = self._sequences.get(key, 0)
            self._sequences[key] = sequence + 1
            if key not in self._lanes:
                self._lanes[key] = _Lane(self, peer, tag, sending)
            self._pending[work] = None
            self._lanes[key].queue.put((work, tensor, ready, time.monotonic() + self.timeout,
                                        sequence))
        self._wake.set()  # an idle watcher starts watching this operation
        return work

    def _complete(self, works, error: BaseException | None) -> None:
        # No user code runs here: callers may hold the engine lock.
        for work in works:
            work._finish(error)
            self._completer.put(work)

    def _finish(self, work: _P2PWork, error: BaseException | None = None) -> None:
        with self._lock:
            if work not in self._pending:
                return
            del self._pending[work]
        self._complete([work], error)

    def _fail(self, error: BaseException, *, publish: bool = True) -> None:
        with self._lock:
            if self._failed is not None or self._closed:
                return
            self._failed = error
            if publish:
                with contextlib.suppress(Exception):
                    self.store.compare_set("failure", "", f"rank {self.rank}: {error}")
            pending = list(self._pending)
            self._pending.clear()
        self._complete(pending, error)

    def match(self, tensor: torch.Tensor, peer: int, tag: int, seq: int,
              sending: bool, deadline: float) -> None:
        self.check()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"point-to-point operation with rank {peer} timed out in queue")
        src, dst = (self.rank, peer) if sending else (peer, self.rank)
        prefix = f"{src}/{dst}/{tag}/{seq}"
        mine, theirs = (("send", "recv") if sending else ("recv", "send"))
        descriptor = json.dumps([tensor.numel() * tensor.element_size(), str(tensor.dtype),
                                 self.chunk_bytes])
        self.store.set(f"{prefix}/{mine}", descriptor)
        key = f"{prefix}/{theirs}"
        timed_out = f"point-to-point {mine} with rank {peer}, tag {tag} timed out"
        pause = MATCH_POLL_FIRST
        while not self.store.check([key]):
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(timed_out)
            # time.sleep, not an Event wait, which costs about twice the CPU per pause;
            # check() sees close() one pause later, as before.
            time.sleep(min(pause, remaining))
            pause = min(2 * pause, MATCH_POLL_MAX)
        other = self.store.get(key).decode()
        self.store.delete_key(key)
        if time.monotonic() >= deadline:  # Store calls can block (a slow or busy server)
            raise TimeoutError(timed_out)
        if descriptor != other:
            raise ValueError(f"point-to-point payload mismatch with rank {peer}, tag {tag}: "
                             f"local {descriptor}, peer {other}")

    def close(self, timeout: float = 0.2) -> None:
        """Fail queued handles immediately; never wait for a missing peer or a callback.

        In-flight Gloo requests retain their buffers in daemon lane threads
        until torch destroys the group or its timeout passes.

        The health watcher is joined (for at most ``WATCHER_JOIN`` seconds: it ends
        after the Store call it may be in). A script often exits right after it
        destroys its groups, and a thread still inside a Store call when the
        interpreter finalizes is ended there, in the middle of torch's C++, which
        terminates the process ("terminate called without an active exception").
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            self._wake.set()
            pending = list(self._pending)
            self._pending.clear()
            lanes = list(self._lanes.values())
            self._lanes.clear()
            for lane in lanes:
                lane.queue.put(None)
        self._complete(pending, RuntimeError("point-to-point group was destroyed"))
        self._completer.close()
        deadline = time.monotonic() + timeout
        for lane in lanes:
            lane.thread.join(max(0, deadline - time.monotonic()))
        if threading.current_thread() is not self._watcher:
            self._watcher.join(WATCHER_JOIN)
        self.link.close()


class _Lane:
    def __init__(self, engine: Engine, peer: int, tag: int, sending: bool) -> None:
        self.engine, self.peer, self.tag, self.sending = engine, peer, tag, sending
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.stream: torch.cuda.Stream | None = None
        self.thread = threading.Thread(target=self._run, name="gpubridge-p2p", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                if self._retire():
                    return
                continue
            if item is None:
                return
            self._run_one(*item)
            del item  # an idle lane keeps no completed payload or Work

    def _retire(self) -> bool:
        # Under the lock that submit holds while queueing, so nothing is lost.
        with self.engine._lock:
            if not self.queue.empty():
                return False
            key = self.peer, self.tag, self.sending
            if self.engine._lanes.get(key) is self:
                del self.engine._lanes[key]
            return True

    def _run_one(self, work: _P2PWork, tensor: torch.Tensor, ready, deadline: float,
                 sequence: int) -> None:
        try:
            self.engine.check()
            stream = None
            if tensor.device.type == "cuda":
                torch.cuda.set_device(tensor.device)
                if self.stream is None or self.stream.device != tensor.device:
                    self.stream = torch.cuda.Stream(tensor.device)
                stream = self.stream
                stream.wait_event(ready)
                tensor.record_stream(stream)
            context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with context:
                self.engine.match(tensor, self.peer, self.tag, sequence, self.sending, deadline)
                self._transfer(tensor, deadline)
                if stream is not None:
                    work._event = torch.cuda.Event()
                    work._event.record(stream)
            self.engine._finish(work)
        except BaseException as exc:  # noqa: BLE001 - delivered through every pending Work
            self.engine._fail(exc)

    def _transfer(self, tensor: torch.Tensor, deadline: float) -> None:
        # Byte views cover every dense dtype, including types Gloo cannot carry
        # natively. The GPU copy runs after the caller's readiness event.
        flat = tensor.reshape(-1).view(torch.uint8)
        if self.engine.pinned or self.engine.chunk_bytes:
            self._pipeline(flat, deadline)
            return
        host = flat.cpu() if self.sending else torch.empty(flat.numel(), dtype=torch.uint8)
        request = self.engine.link.post(host, self.peer, self.tag, self.sending)
        request.wait(timedelta(seconds=max(0.001, deadline - time.monotonic())))
        if not self.sending:
            flat.copy_(host)

    def _pipeline(self, flat: torch.Tensor, deadline: float) -> None:
        """Two slots overlap copies and wire transfers; slots are never reused early."""
        size = self.engine.chunk_bytes or max(8, flat.numel())
        chunks = list(flat.split(size)) if flat.numel() else [flat]
        staging = make_staging(flat.device, size, min(2, len(chunks)))
        staging.begin(flat)
        try:
            if self.sending:
                prepared = {}

                def prepare(i):
                    host = staging.slot(i % 2, chunks[i])
                    prepared[i] = host, staging.to_host(host, chunks[i], None)

                for i in range(min(2, len(chunks))):
                    prepare(i)
                for i in range(len(chunks)):
                    host, ready = prepared.pop(i)
                    staging.wait_host(ready)  # the network may now read the host slot
                    request = self.engine.link.post(host, self.peer, self.tag, True)
                    request.wait(timedelta(seconds=max(0.001, deadline - time.monotonic())))
                    if i + 2 < len(chunks):
                        prepare(i + 2)  # the previous send no longer reads this slot
            else:
                posted, copied = {}, [None, None]

                def receive(i):
                    staging.wait_host(copied[i % 2])  # before the network overwrites it
                    host = staging.slot(i % 2, chunks[i])
                    posted[i] = host, self.engine.link.post(host, self.peer, self.tag, False)

                for i in range(min(2, len(chunks))):
                    receive(i)
                for i, chunk in enumerate(chunks):
                    host, request = posted.pop(i)
                    request.wait(timedelta(seconds=max(0.001, deadline - time.monotonic())))
                    copied[i % 2] = staging.to_device(chunk, host)
                    if i + 2 < len(chunks):
                        receive(i + 2)
                staging.end(copied[(len(chunks) - 1) % 2])
        finally:
            staging.close()
