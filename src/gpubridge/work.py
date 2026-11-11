"""Async collectives: a :class:`Work` handle, and one worker thread per rank that runs them.

``async_op=True`` validates the call on the calling thread, queues it on this
rank's worker thread and returns a :class:`Work` at once. The worker runs
queued collectives one at a time, first in first out. A synchronous collective
first waits for every queued one. So on every rank collectives run in program
order, the same order on every rank, which is what keeps them from hanging.

Stream semantics on GPUs, matching ``torch.distributed`` with NCCL:

- **Inputs.** At submission an event is recorded on the caller's current
  stream, and the worker's stream waits on it before the collective starts.
  Work queued on the caller's stream before the call is therefore done before
  the collective reads the tensor. Tensors are marked with ``record_stream``
  for the worker's stream, so the caching allocator won't reuse them early.
- **Outputs.** :meth:`Work.wait` blocks the host until the worker has run the
  collective, then makes the caller's current stream wait on an event recorded
  after it on the worker's stream. Kernels queued on the caller's stream after
  ``wait()`` see the result. :meth:`Work.is_completed` turns True only once
  that event has passed, i.e. the GPU work is done too, so the result can then
  be used on any stream without ``wait()``, as with NCCL. The future from
  :meth:`Work.get_future` is created with the rank's device, so
  ``future.wait()`` and ``future.value()`` synchronize the same way.
- **Between call and wait**, don't read or write the tensors.

The bridge steps of a collective block the worker thread on host copies, not
the caller, so the caller's thread can keep queueing GPU work.

Submissions are bounded (GPUBRIDGE_ASYNC_QUEUE, 1024 waiting calls by default).
Overflow rejects the new call; accepted work keeps its FIFO order. Wait for it
before retrying, preserving the same collective sequence on every rank.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import torch

QUEUE_ENV = "GPUBRIDGE_ASYNC_QUEUE"
DEFAULT_QUEUE = 1024


def queue_limit() -> int:
    """Maximum queued (not yet running) async collectives per rank."""
    raw = os.environ.get(QUEUE_ENV, "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_QUEUE
    except ValueError:
        raise ValueError(f"{QUEUE_ENV} must be a positive whole number") from None
    if limit <= 0:
        raise ValueError(f"{QUEUE_ENV} must be a positive whole number")
    return limit


class Work:
    """Handle for one async gpubridge collective. Mirrors ``torch.distributed.Work``."""

    def __init__(self, op: str, device: torch.device, result: Any) -> None:
        self.op = op
        self._device = device
        self._result = result
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._event: torch.cuda.Event | None = None
        devices = [device] if device.type == "cuda" else []
        self._future: torch.futures.Future = torch.futures.Future(devices=devices)

    def is_completed(self) -> bool:
        """Whether the collective has finished on this rank, GPU work included. Never blocks.

        The worker thread finishes as soon as it has queued the last kernels
        (the island broadcast), so on GPUs this also asks the event recorded
        after them, like ``torch.distributed`` with NCCL.
        """
        if not self._done.is_set():
            return False
        return self._event is None or self._event.query()

    def wait(self, timeout: timedelta | float | None = None) -> bool:
        """Block until the collective finishes; on GPUs, order the current stream after it.

        Raises:
            TimeoutError: if ``timeout`` (a timedelta or seconds) passes first.
            RuntimeError: if the collective failed; the original error is its cause.
        """
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else timeout
        if not self._done.wait(seconds):
            raise TimeoutError(f"gpubridge async {self.op} did not finish within {seconds} s")
        if self._error is not None:
            raise RuntimeError(f"gpubridge async {self.op} failed: {self._error}") from self._error
        if self._event is not None:
            torch.cuda.current_stream(self._device).wait_event(self._event)
        return True

    def exception(self) -> BaseException | None:
        """The error the collective raised, or None. Only meaningful once completed."""
        return self._error

    def get_future(self) -> torch.futures.Future:
        """A future holding the output tensor (None for a barrier) once the collective ends."""
        return self._future

    def _finish(self, error: BaseException | None) -> None:
        self._error = error
        if error is None:
            self._future.set_result(self._result)
        else:
            self._future.set_exception(error)
        self._done.set()


class _Worker:
    """Runs queued collectives in FIFO order on one thread, on its own CUDA stream."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.limit = queue_limit()
        self.queue: queue.Queue[tuple[Work, Callable[[], None], Any, tuple] | None] = (
            queue.Queue(self.limit))
        self.failed: BaseException | None = None
        self.stream: torch.cuda.Stream | None = None
        self.thread = threading.Thread(target=self._run, name="gpubridge-async", daemon=True)
        self.thread.start()

    def submit(self, op: str, fn: Callable[[], None], tensors: Sequence[torch.Tensor],
               result: Any) -> Work:
        work = Work(op, self.device, result)
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        try:
            self.queue.put_nowait((work, fn, ready, tuple(tensors)))
        except queue.Full:
            raise RuntimeError(
                f"gpubridge async queue is full ({self.limit} queued collectives); "
                "wait for accepted work before retrying this collective, preserving "
                f"the same collective order on every rank ({QUEUE_ENV})") from None
        return work

    def drain(self) -> None:
        """Wait until every queued collective has run."""
        self.queue.join()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def abandon(self, timeout: float) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0]._finish(RuntimeError(
                    f"gpubridge async {item[0].op} not run: its process groups were destroyed"))
            self.queue.task_done()
        self.queue.put(None)
        self.thread.join(timeout)

    def _run(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)  # the current device is per thread
            self.stream = torch.cuda.Stream(self.device)
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                work, fn, ready, tensors = item
                if self.failed is not None:
                    work._finish(RuntimeError(
                        f"not run: an earlier async collective failed on this rank "
                        f"({self.failed})"))
                    continue
                try:
                    self._run_one(work, fn, ready, tensors)
                except BaseException as error:  # noqa: BLE001 - reported through the Work
                    self.failed = error
                    work._finish(error)
            finally:
                self.queue.task_done()

    def _run_one(self, work: Work, fn: Callable[[], None], ready: torch.cuda.Event | None,
                 tensors: tuple[torch.Tensor, ...]) -> None:
        if self.stream is None:
            fn()
            work._finish(None)
            return
        assert ready is not None
        self.stream.wait_event(ready)
        for tensor in tensors:
            tensor.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            fn()
            work._event = torch.cuda.Event()
            work._event.record(self.stream)
            work._finish(None)  # the device-aware future records on this stream


_worker: _Worker | None = None


def submit(device: torch.device, op: str, fn: Callable[[], None],
           tensors: Sequence[torch.Tensor], result: Any) -> Work:
    """Queue ``fn`` on this rank's worker thread, starting it on first use."""
    global _worker
    check_healthy()
    if _worker is None:
        _worker = _Worker(device)
    return _worker.submit(op, fn, tensors, result)


def check_healthy() -> None:
    """Raise if an earlier async collective failed on this rank.

    After a failed collective the process groups may be in any state, so
    every later collective refuses to start rather than risk a hang.
    """
    if _worker is not None and _worker.failed is not None:
        raise RuntimeError(
            "an earlier async gpubridge collective failed on this rank "
            f"({type(_worker.failed).__name__}: {_worker.failed}); the process groups may "
            "be inconsistent. Call gpubridge.destroy() and init() again (under the "
            "torch.distributed backend: destroy_process_group() and init_process_group())."
        )


def drain() -> None:
    """Wait for every queued collective, then raise if one failed. Synchronous calls do this."""
    if _worker is not None:
        _worker.drain()
    check_healthy()


def shutdown() -> None:
    """Run what is queued, then stop the worker thread. Called by init() and destroy()."""
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.close()


def forget() -> None:
    """Drop the worker without touching its thread or queue: for a forked child, which
    inherited the object but not the thread."""
    global _worker
    _worker = None


def abandon(timeout: float = 10.0) -> None:
    """Stop the worker thread without running what is queued: each queued collective
    fails. For teardown once the process groups are gone (the torch.distributed
    backend's ``shutdown()``); waits at most ``timeout`` seconds for the thread."""
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.abandon(timeout)
