"""``bridge="ucx"``: the island leaders' bridge over UCX (docs/RDMA_TRANSPORT.md).

Leaders' tensors move with UCX tagged messages, over whatever UCX picks
(InfiniBand or RoCE verbs, TCP). The reductions run here, with torch ops on
the received data. Three paths, chosen per call and recorded in
:attr:`UcxTransport.last_path`:

- ``host``: a host tensor (as Gloo gets it; the pipelined and sharded
  policies always stage to host memory).
- ``direct``: a GPU tensor, when this rank's UCX was built for that memory
  type (``cuda`` or ``rocm``). UCX reads and writes GPU memory itself, with
  GPUDirect RDMA where the NIC and drivers allow it. The host waits on an
  event recorded on the caller's stream before each step posts its messages,
  so no send reads data a queued kernel (the island reduce, a reduction) is
  still writing. After each step's receives complete, the transport flushes
  the NIC's writes into the GPU (:mod:`gpubridge.ucx.gdr`) before any kernel
  can read them; :attr:`UcxTransport.flush_method` says how.
- ``staged``: a GPU tensor when UCX can't take that memory type, or when that
  flush is needed and impossible on this GPU: the transport stages it through
  one pinned host buffer (:class:`gpubridge.staging.PinnedBridgeStaging`) and
  runs the host path.

- all_reduce: with two leaders, an exchange and one local reduction (both get
  the same bits, since addition, max and min are commutative); with more, a
  ring (reduce-scatter, then all-gather), so every element is reduced in one
  fixed order and then copied.
- broadcast: the source leader sends to each of the others.

Every message is split into chunks of ``GPUBRIDGE_UCX_CHUNK_BYTES`` (4 MiB by
default), all posted at once. Every wait has the deadline ``init()`` gave its
groups; a timeout or a failed peer raises, and the transport refuses later
calls.
"""

from __future__ import annotations

import base64
import itertools
import os
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import ucx
from gpubridge.config import parse_size
from gpubridge.rendezvous import DEFAULT_TIMEOUT  # noqa: F401 - this module's name for it
from gpubridge.rendezvous import wait_for as _wait_for
from gpubridge.staging import PinnedBridgeStaging
from gpubridge.tagged import _REDUCE, _segments, tag  # noqa: F401 - shared with the Rust bridges
from gpubridge.transport import BridgeTransport, register_transport, scoped_name
from gpubridge.ucx import gdr

#: Chunk size of every UCX message, e.g. ``4M`` or ``65536``.
CHUNK_BYTES_ENV = "GPUBRIDGE_UCX_CHUNK_BYTES"
DEFAULT_CHUNK_BYTES = 4 * 2**20
#: The tag step of the connection handshake.
HANDSHAKE = 0xFFF

def chunk_bytes_setting() -> int:
    raw = os.environ.get(CHUNK_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_CHUNK_BYTES
    try:
        size = parse_size(raw)
    except ValueError as exc:
        raise ValueError(f"{CHUNK_BYTES_ENV}: {exc}") from None
    if size < 8 or size % 8:
        raise ValueError(f"{CHUNK_BYTES_ENV}={raw!r} must be a positive multiple of 8 bytes")
    return size


def _pieces(nbytes: int, chunk: int) -> list[tuple[int, int]]:
    """(offset, length) of each chunk of a ``nbytes`` buffer; one empty piece for 0 bytes."""
    return [(offset, min(chunk, nbytes - offset)) for offset in range(0, nbytes, chunk)] or [
        (0, 0)]


def choose_path(device_type: str, hip: bool, memory_types: set[str]) -> tuple[str, int]:
    """(path, UCX memory type) for a tensor on ``device_type``: see the module docstring.

    ``hip`` says whether this is a ROCm build (its GPU tensors are ``cuda``
    tensors to torch but ROCm memory to UCX); ``memory_types`` is what this
    rank's UCX supports.
    """
    if device_type == "cpu":
        return "host", ucx.HOST
    kind, memory = ("rocm", ucx.ROCM) if hip else ("cuda", ucx.CUDA)
    if kind in memory_types:
        return "direct", memory
    return "staged", ucx.HOST


@register_transport
class UcxTransport(BridgeTransport):
    """The bridge between island leaders over UCX. One instance per leader per link."""

    name = "ucx"
    reduce_ops = frozenset({"sum", "max", "min"})
    device_tensors = True

    #: Links created in this process, the same count on every rank (creation is
    #: collective and in the same order everywhere), so store keys never collide.
    #: A subgroup's links, built by its members only, take scoped_name() instead.
    _links: ClassVar[itertools.count] = itertools.count()

    def __init__(self, worker: ucx.Worker, members: Sequence[int], rank: int,
                 timeout: timedelta, chunk_bytes: int) -> None:
        self.worker = worker
        self.members = list(members)
        self.index = self.members.index(rank)
        self.timeout = timeout.total_seconds()
        self.chunk_bytes = chunk_bytes
        self._seq = 0
        #: Memory types this rank's UCX supports ("host", "cuda", "rocm").
        self.memory_types = worker.memory_types()
        #: The path of the last collective: "host", "direct" or "staged".
        self.last_path: str | None = None
        #: For the last collective's GPU, when UCX has its memory type: how the NIC's writes
        #: into it are flushed (gdr.Flush.method), or why it was staged instead. Else None.
        self.flush_method: str | None = None
        self._memory = ucx.HOST
        self._flushes: dict[int, gdr.Flush] = {}
        self._flush: gdr.Flush | None = None
        self._staging: PinnedBridgeStaging | None = None

    # ---- creation ---------------------------------------------------------------------

    @classmethod
    def create(cls, ranks: Sequence[int], *, timeout: timedelta | None = None
               ) -> UcxTransport | None:
        timeout = timeout or DEFAULT_TIMEOUT
        link = scoped_name(ranks) or next(cls._links)
        store = dist.distributed_c10d._get_default_store()
        key = f"gpubridge/ucx/{link}"
        members, rank = list(ranks), dist.get_rank()
        deadline = time.monotonic() + timeout.total_seconds()
        transport, error = None, None
        if rank in members:
            try:
                transport = cls._connect(store, key, members, rank, timeout, deadline)
            except Exception as exc:  # noqa: BLE001 - reported to every rank below
                error = f"{type(exc).__name__}: {exc}"
            store.set(f"{key}/status/{rank}", "ok" if error is None else error)
        # Every rank, member or not, learns whether every member connected.
        statuses = _wait_for(store, [f"{key}/status/{m}" for m in members], deadline,
                             "every bridge member to report on its UCX connection")
        failed = [(m, s) for m, s in zip(members, statuses, strict=True) if s != "ok"]
        if failed:
            if transport is not None:
                transport.close()
            detail = "\n".join(f"  rank {m}: {s}" for m, s in failed)
            raise RuntimeError(f"The UCX bridge between ranks {members} could not be set up; "
                               f'use bridge="gloo" to run without it.\n{detail}')
        return transport

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> UcxTransport:
        chunk = chunk_bytes_setting()
        worker = ucx.Worker(len(members))
        try:
            major, minor, release = ucx.version()
            # The chunk size decides how many messages each step sends: leaders that
            # disagreed would match the wrong messages, so it is checked like the version.
            mine = (f"{major}.{minor}.{release}|{chunk}|"
                    + base64.b64encode(worker.address()).decode())
            store.set(f"{key}/addr/{rank}", mine)
            others = [m for m in members if m != rank]
            entries = _wait_for(store, [f"{key}/addr/{m}" for m in others], deadline,
                                "the other bridge members' UCX addresses",
                                failures=[f"{key}/status/{m}" for m in others])
            for member, entry in zip(others, entries, strict=True):
                version, their_chunk, address = entry.split("|", 2)
                if version.split(".")[:2] != [str(major), str(minor)]:
                    raise RuntimeError(f"UCX {major}.{minor}.{release} here but {version} on "
                                       f"rank {member}; the bridge needs the same UCX major "
                                       "and minor version on every leader")
                if int(their_chunk) != chunk:
                    raise RuntimeError(f"{CHUNK_BYTES_ENV} is {chunk} here but {their_chunk} "
                                       f"on rank {member}; set the same value on every rank")
                worker.connect(members.index(member), base64.b64decode(address))
            transport = cls(worker, members, rank, timeout, chunk)
            transport._handshake(deadline)
            return transport
        except BaseException:
            worker.close(timeout=1.0)
            raise

    def _handshake(self, deadline: float) -> None:
        """One small message each way with every other member, so a link that can't
        carry data fails now, not in the first collective."""
        hello = torch.tensor([self.index], dtype=torch.int64)
        replies = {}
        ids = []
        for peer in self._peers():
            replies[peer] = torch.full((1,), -1, dtype=torch.int64)
            ids.append(self.worker.recv(replies[peer].data_ptr(), 8,
                                        tag(0, HANDSHAKE, peer, 0)))
            ids.append(self.worker.send(peer, hello.data_ptr(), 8,
                                        tag(0, HANDSHAKE, self.index, 0)))
        self.worker.wait(ids, deadline)
        for peer, reply in replies.items():
            if int(reply.item()) != peer:
                raise RuntimeError(f"UCX handshake with bridge member {peer} returned "
                                   f"{int(reply.item())}")

    # ---- collectives ------------------------------------------------------------------

    def _peers(self) -> list[int]:
        return [i for i in range(len(self.members)) if i != self.index]

    def _start(self, tensor: torch.Tensor) -> tuple[int, float]:
        if not tensor.is_contiguous():
            raise ValueError("the UCX bridge carries contiguous tensors")
        path, memory = choose_path(tensor.device.type, bool(torch.version.hip),
                                   self.memory_types)
        self._flush = self.flush_method = None
        if path == "direct":
            self._flush = self._flush_for(tensor.device)
            self.flush_method = self._flush.method
            if not self._flush.available:
                path, memory = "staged", ucx.HOST
        self.last_path, self._memory = path, memory
        self._seq += 1
        return self._seq, time.monotonic() + self.timeout

    def _flush_for(self, device: torch.device) -> gdr.Flush:
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in self._flushes:
            self._flushes[index] = gdr.for_device(index, bool(torch.version.hip))
        return self._flushes[index]

    def _ready(self, tensor: torch.Tensor) -> None:
        """Before posting messages on GPU memory: wait for the work queued on the
        caller's stream (only up to now, not the whole device)."""
        if self.last_path == "direct":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
            event.synchronize()

    def _wait(self, ids: list[int], deadline: float, *, received: bool = True) -> None:
        """Wait for one step's messages. After receives on the direct path, flush the NIC's
        writes into GPU memory, so the kernels launched next (a reduction, the island
        broadcast, the caller's) can't read stale data (docs/RDMA_TRANSPORT.md, section 4)."""
        self.worker.wait(ids, deadline)
        if received and self.last_path == "direct":
            self._flush()

    def _staged(self, tensor: torch.Tensor, run) -> None:
        """Run ``run(host_tensor)`` on a pinned host copy of a GPU ``tensor``, then copy back."""
        if self._staging is None:
            self._staging = PinnedBridgeStaging(tensor.device)
        host = self._staging.to_host(tensor)
        run(host)
        self._staging.to_device(tensor, host)

    def _send(self, peer: int, view: torch.Tensor, seq: int, step: int) -> list[int]:
        nbytes = view.numel() * view.element_size()
        return [self.worker.send(peer, view.data_ptr() + offset, length,
                                 tag(seq, step, self.index, i), self._memory)
                for i, (offset, length) in enumerate(_pieces(nbytes, self.chunk_bytes))]

    def _recv(self, peer: int, view: torch.Tensor, seq: int, step: int) -> list[int]:
        nbytes = view.numel() * view.element_size()
        return [self.worker.recv(view.data_ptr() + offset, length, tag(seq, step, peer, i),
                                 self._memory)
                for i, (offset, length) in enumerate(_pieces(nbytes, self.chunk_bytes))]

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        if len(self.members) == 1:
            return
        seq, deadline = self._start(tensor)
        if self.last_path == "staged":
            self._staged(tensor, lambda host: self._all_reduce(host, op, seq, deadline))
        else:
            self._all_reduce(tensor, op, seq, deadline)

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType, seq: int,
                    deadline: float) -> None:
        size = len(self.members)
        reduce = _REDUCE[op]
        flat = tensor.view(-1)
        if size == 2:
            (peer,) = self._peers()
            scratch = torch.empty_like(flat)
            self._ready(flat)
            ids = self._recv(peer, scratch, seq, 0) + self._send(peer, flat, seq, 0)
            self._wait(ids, deadline)
            reduce(flat, scratch)
            return
        right, left = (self.index + 1) % size, (self.index - 1) % size
        segments = _segments(flat.numel(), size)
        scratch = torch.empty(max(length for _, length in segments), dtype=flat.dtype,
                              device=flat.device)

        def segment(i: int) -> torch.Tensor:
            start, length = segments[i % size]
            return flat[start:start + length]

        for step in range(size - 1):  # reduce-scatter
            incoming = segment(self.index - step - 1)
            buffer = scratch[:incoming.numel()]
            self._ready(flat)  # the last step's reduction is done before its result is sent
            ids = self._recv(left, buffer, seq, step) + self._send(
                right, segment(self.index - step), seq, step)
            self._wait(ids, deadline)
            reduce(incoming, buffer)
        for step in range(size - 1):  # all-gather: this member now owns segment index + 1
            self._ready(flat)
            ids = self._recv(left, segment(self.index - step), seq, size + step) + self._send(
                right, segment(self.index + 1 - step), seq, size + step)
            self._wait(ids, deadline)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if len(self.members) == 1:
            return
        seq, deadline = self._start(tensor)
        if self.last_path == "staged":
            self._staged(tensor, lambda host: self._broadcast(host, src, seq, deadline))
        else:
            self._broadcast(tensor, src, seq, deadline)

    def _broadcast(self, tensor: torch.Tensor, src: int, seq: int, deadline: float) -> None:
        root = self.members.index(src)
        flat = tensor.view(-1)
        self._ready(flat)
        if self.index == root:
            ids = [i for peer in self._peers() for i in self._send(peer, flat, seq, 0)]
        else:
            ids = self._recv(root, flat, seq, 0)
        self._wait(ids, deadline, received=self.index != root)

    def close(self) -> None:
        if self._staging is not None:
            self._staging.close()
            self._staging = None
        self.worker.close(timeout=min(10.0, self.timeout))

    @classmethod
    def create_p2p(cls, group, *, store=None, timeout=None):
        from gpubridge.ucx.p2p import UcxLink

        return UcxLink(dist.get_rank(), store, timeout or DEFAULT_TIMEOUT)
