"""Collectives over a worker that moves tagged byte ranges between bridge members.

The multitcp (:mod:`gpubridge.multitcp.transport`) and verbs
(:mod:`gpubridge.verbs.transport`) bridges differ only in how bytes move: each
has a worker with ``send(member, pointer, nbytes, tag)`` and
``recv(...)`` returning operation ids, ``wait(ids, deadline)``, ``abort()`` and
``close()``. The collectives, their reduction order and so their numerics are
this one class:

- all_reduce: with two leaders, an exchange in chunks of ``chunk_bytes``; each
  chunk is reduced as soon as it has arrived and gone out, while the rest are
  still in flight. Both leaders get the same bits, since addition, max and min
  are commutative. With more leaders, a ring (reduce-scatter, then all-gather),
  so every element is reduced in one fixed order and then copied.
- broadcast: the source leader sends to each of the others, in chunks.

The reductions run on one torch thread (the thread count is set for the
collective and restored after), so torch's intra-op threads don't compete with
the worker's own threads. If anything raises inside a collective, the worker is
aborted before the error propagates, so no posted operation can still read or
write the collective's tensors once the caller sees the error.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge.rendezvous import DEFAULT_TIMEOUT, default_store
from gpubridge.rendezvous import wait_for as _wait_for
from gpubridge.transport import BridgeTransport, scoped_name

#: The tag step of the connection handshake.
HANDSHAKE = 0xFFF

_REDUCE = {ReduceOp.SUM: torch.Tensor.add_,
           ReduceOp.MAX: lambda a, b: torch.maximum(a, b, out=a),
           ReduceOp.MIN: lambda a, b: torch.minimum(a, b, out=a)}


def tag(seq: int, step: int, sender: int, chunk: int) -> int:
    """A 64-bit message tag: collective number, step, sending member, chunk."""
    return ((seq & 0xFFFFFF) << 40) | ((step & 0xFFF) << 28) | ((sender & 0xFF) << 20) | (
        chunk & 0xFFFFF)


def _segments(numel: int, parts: int) -> list[tuple[int, int]]:
    """(start, length) in elements of ``parts`` near-equal segments."""
    base, extra = divmod(numel, parts)
    bounds = list(itertools.accumulate([base + (i < extra) for i in range(parts)], initial=0))
    return [(bounds[i], bounds[i + 1] - bounds[i]) for i in range(parts)]


def _pieces(numel: int, chunk: int) -> list[tuple[int, int]]:
    """(start, length) in elements of each chunk of ``numel``; one empty piece for 0."""
    return [(start, min(chunk, numel - start)) for start in range(0, numel, chunk)] or [(0, 0)]


class TaggedBridge(BridgeTransport):
    """A bridge whose worker moves tagged byte ranges; see the module docstring.

    Subclasses set ``name`` and ``_links`` (their own ``itertools.count()``), and
    implement :meth:`_connect`, which makes the worker, exchanges addresses through
    the store under ``key`` and returns a connected transport.
    """

    reduce_ops = frozenset({"sum", "max", "min"})
    #: Links created in this process, the same count on every rank (creation is
    #: collective and in the same order everywhere), so store keys never collide.
    #: A subgroup's links, built by its members only, take scoped_name() instead.
    _links: ClassVar[itertools.count]
    #: What the setup failure tells the user to do instead.
    _instead: ClassVar[str] = 'use bridge="gloo" to run without it'

    def __init__(self, worker: Any, members: Sequence[int], rank: int, timeout: timedelta,
                 chunk_bytes: int) -> None:
        self.worker = worker
        self.members = list(members)
        self.index = self.members.index(rank)
        self.timeout = timeout.total_seconds()
        self.chunk_bytes = chunk_bytes
        self._seq = 0

    # ---- creation ---------------------------------------------------------------------

    @classmethod
    def create(cls, ranks: Sequence[int], *, timeout: timedelta | None = None
               ) -> TaggedBridge | None:
        timeout = timeout or DEFAULT_TIMEOUT
        link = scoped_name(ranks) or next(cls._links)
        store = default_store()
        key = f"gpubridge/{cls.name}/{link}"
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
                             f"every bridge member to report on its {cls.name} connections")
        failed = [(m, s) for m, s in zip(members, statuses, strict=True) if s != "ok"]
        if failed:
            if transport is not None:
                transport.close()
            detail = "\n".join(f"  rank {m}: {s}" for m, s in failed)
            raise RuntimeError(f"The {cls.name} bridge between ranks {members} could not be set "
                               f"up; {cls._instead}.\n{detail}")
        return transport

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> TaggedBridge:
        raise NotImplementedError

    def _handshake(self, deadline: float) -> None:
        """One small message each way with every other member, so a link that can't
        carry data fails now, not in the first collective."""
        hello = torch.tensor([self.index], dtype=torch.int64)
        replies = {}
        ids = []
        for peer in self._peers():
            replies[peer] = torch.full((1,), -1, dtype=torch.int64)
            ids.append(self.worker.recv(peer, replies[peer].data_ptr(), 8,
                                        tag(0, HANDSHAKE, peer, 0)))
            ids.append(self.worker.send(peer, hello.data_ptr(), 8,
                                        tag(0, HANDSHAKE, self.index, 0)))
        self.worker.wait(ids, deadline)
        for peer, reply in replies.items():
            if int(reply.item()) != peer:
                raise RuntimeError(f"{self.name} handshake with bridge member {peer} returned "
                                   f"{int(reply.item())}")

    # ---- collectives ------------------------------------------------------------------

    def _peers(self) -> list[int]:
        return [i for i in range(len(self.members)) if i != self.index]

    def _check(self, tensor: torch.Tensor) -> None:
        if not tensor.is_contiguous() or tensor.device.type != "cpu":
            raise ValueError(f"the {self.name} bridge carries contiguous host tensors")

    def _scratch(self, numel: int, like: torch.Tensor) -> torch.Tensor:
        """A buffer of ``numel`` elements like ``like`` for incoming data (a subclass may
        hand out one its worker has registered)."""
        return torch.empty(numel, dtype=like.dtype, device=like.device)

    def _start(self, tensor: torch.Tensor) -> tuple[int, float, int]:
        self._seq += 1
        chunk = max(1, self.chunk_bytes // tensor.element_size())
        return self._seq, time.monotonic() + self.timeout, chunk

    def _send(self, peer: int, view: torch.Tensor, seq: int, step: int, chunk: int) -> list[int]:
        size = view.element_size()
        return [self.worker.send(peer, view.data_ptr() + start * size, length * size,
                                 tag(seq, step, self.index, i))
                for i, (start, length) in enumerate(_pieces(view.numel(), chunk))]

    def _recv(self, peer: int, view: torch.Tensor, seq: int, step: int, chunk: int) -> list[int]:
        size = view.element_size()
        return [self.worker.recv(peer, view.data_ptr() + start * size, length * size,
                                 tag(seq, step, peer, i))
                for i, (start, length) in enumerate(_pieces(view.numel(), chunk))]

    def _guarded(self, run) -> None:
        """``run()`` on one torch thread (see the module docstring); if anything raises,
        stop the worker before it propagates, so no posted operation outlives the tensors
        it points into."""
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            run()
        except BaseException:
            self.worker.abort()
            raise
        finally:
            torch.set_num_threads(threads)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)  # a bad argument fails the call, not the bridge
        self._guarded(lambda: self._all_reduce(tensor, op))

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType) -> None:
        seq, deadline, chunk = self._start(tensor)
        size = len(self.members)
        reduce = _REDUCE[op]
        flat = tensor.view(-1)
        if size == 2:
            (peer,) = self._peers()
            scratch = self._scratch(flat.numel(), flat)
            received = self._recv(peer, scratch, seq, 0, chunk)
            sent = self._send(peer, flat, seq, 0, chunk)
            # Chunk i is reduced once it has arrived and this side's copy of it has gone
            # out (the send reads the same memory), while later chunks are in flight.
            for (start, length), got, out in zip(_pieces(flat.numel(), chunk), received, sent,
                                                 strict=True):
                self.worker.wait([got, out], deadline)
                reduce(flat[start:start + length], scratch[start:start + length])
            return
        right, left = (self.index + 1) % size, (self.index - 1) % size
        segments = _segments(flat.numel(), size)
        scratch = self._scratch(max(length for _, length in segments), flat)

        def segment(i: int) -> torch.Tensor:
            start, length = segments[i % size]
            return flat[start:start + length]

        for step in range(size - 1):  # reduce-scatter
            incoming = segment(self.index - step - 1)
            buffer = scratch[:incoming.numel()]
            ids = self._recv(left, buffer, seq, step, chunk) + self._send(
                right, segment(self.index - step), seq, step, chunk)
            self.worker.wait(ids, deadline)
            reduce(incoming, buffer)
        for step in range(size - 1):  # all-gather: this member now owns segment index + 1
            ids = self._recv(left, segment(self.index - step), seq, size + step, chunk) + \
                self._send(right, segment(self.index + 1 - step), seq, size + step, chunk)
            self.worker.wait(ids, deadline)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)
        self._guarded(lambda: self._broadcast(tensor, src))

    def _broadcast(self, tensor: torch.Tensor, src: int) -> None:
        seq, deadline, chunk = self._start(tensor)
        root = self.members.index(src)
        flat = tensor.view(-1)
        if self.index == root:
            ids = [i for peer in self._peers() for i in self._send(peer, flat, seq, 0, chunk)]
        else:
            ids = self._recv(root, flat, seq, 0, chunk)
        self.worker.wait(ids, deadline)
