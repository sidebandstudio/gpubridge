"""``bridge="multitcp"``: the island leaders' bridge over several TCP connections per pair.

Opt-in (docs/RUST_TRANSPORT.md). The Rust library in :mod:`gpubridge.multitcp`
moves tagged byte ranges between leaders over ``GPUBRIDGE_MULTITCP_STREAMS``
connections per pair (4 by default), each with its own sending and receiving
thread; this module runs the collectives and the reductions, with torch ops,
like the UCX transport. It takes host tensors only (``device_tensors`` is
False), so GPU tensors reach it through the policies' usual staging and their
stream ordering is untouched. The collectives are :class:`gpubridge.tagged.TaggedBridge`'s
(shared with ``bridge="verbs"``), in chunks of ``GPUBRIDGE_MULTITCP_CHUNK_BYTES``
(4 MiB by default) spread over the connections.

The reductions run on one torch thread (the thread count is set for the collective and
restored after). On a node with few CPUs, torch's intra-op threads would otherwise
compete with this transport's network threads: on Explorer's 4-CPU jobs that made each
chunk's reduction 10 to 25 times slower than on one thread. Gloo's own bridge reduces
on one thread too.

The leaders connect on the address of ``GPUBRIDGE_MULTITCP_IFNAME``, else of
``GLOO_SOCKET_IFNAME``, else the one this machine reaches ``MASTER_ADDR`` from.
Every wait has the deadline ``init()`` gave its groups; a timeout or a failed
peer raises, and the transport refuses later calls. Before anything raises out of
a collective, the library's threads are stopped, so none of them can still write
into (or read from) the collective's tensors once the caller sees the error.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import socket
import struct
import sys
import time
from datetime import timedelta
from typing import ClassVar

from gpubridge import multitcp
from gpubridge.config import parse_size
from gpubridge.rendezvous import wait_for as _wait_for
from gpubridge.tagged import (  # noqa: F401 - tag: the tests' name for it
    DEFAULT_TIMEOUT,
    HANDSHAKE,
    TaggedBridge,
    _pieces,
    tag,
)
from gpubridge.transport import register_transport

#: Chunk size of every message, e.g. ``4M`` or ``65536``.
CHUNK_BYTES_ENV = "GPUBRIDGE_MULTITCP_CHUNK_BYTES"
DEFAULT_CHUNK_BYTES = 4 * 2**20
#: TCP connections per pair of leaders.
STREAMS_ENV = "GPUBRIDGE_MULTITCP_STREAMS"
DEFAULT_STREAMS = 4
#: The interface whose address the leaders connect on (else GLOO_SOCKET_IFNAME's).
IFNAME_ENV = "GPUBRIDGE_MULTITCP_IFNAME"

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


def streams_setting() -> int:
    raw = os.environ.get(STREAMS_ENV, "").strip()
    if not raw:
        return DEFAULT_STREAMS
    if not raw.isdigit() or not 1 <= int(raw) <= 64:
        raise ValueError(f"{STREAMS_ENV}={raw!r} must be a whole number from 1 to 64")
    return int(raw)


def _interface_address(ifname: str) -> str:
    """The IPv4 address of interface ``ifname`` (SIOCGIFADDR)."""
    request = 0xC0206921 if sys.platform == "darwin" else 0x8915
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            answer = fcntl.ioctl(probe.fileno(), request, struct.pack("256s", ifname[:15].encode()))
        except OSError as exc:
            raise RuntimeError(f"no IPv4 address on interface {ifname!r} for the multitcp "
                               f"bridge ({exc}); set {IFNAME_ENV}") from None
    return socket.inet_ntoa(answer[20:24])


def local_address() -> str:
    """The address this leader listens on: see the module docstring."""
    ifname = (os.environ.get(IFNAME_ENV) or os.environ.get("GLOO_SOCKET_IFNAME") or "").strip()
    if ifname:
        return _interface_address(ifname.split(",")[0])
    master = os.environ.get("MASTER_ADDR", "").strip()
    if master:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((master, 9))  # picks the route; sends nothing
            return probe.getsockname()[0]
    return socket.gethostbyname(socket.gethostname())


@register_transport
class MultiTcpTransport(TaggedBridge):
    """The bridge between island leaders over several TCP connections per pair."""

    name = "multitcp"
    _links: ClassVar[itertools.count] = itertools.count()

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> MultiTcpTransport:
        chunk, streams = chunk_bytes_setting(), streams_setting()
        index = members.index(rank)
        worker = multitcp.Worker(len(members), index, streams, local_address())
        try:
            # The chunk size and stream count decide which messages each step sends and
            # where: leaders that disagreed would wait for the wrong ones, so they are
            # checked like the protocol.
            settings = f"{multitcp.protocol()}|{chunk}|{streams}"
            store.set(f"{key}/addr/{rank}", f"{settings}|{worker.address()}")
            others = [m for m in members if m != rank]
            entries = _wait_for(store, [f"{key}/addr/{m}" for m in others], deadline,
                                "the other bridge members' multitcp addresses",
                                failures=[f"{key}/status/{m}" for m in others])
            for member, entry in zip(others, entries, strict=True):
                theirs, address = entry.rsplit("|", 1)
                if theirs != settings:
                    raise RuntimeError(
                        f"multitcp protocol|{CHUNK_BYTES_ENV}|{STREAMS_ENV} is {settings} here "
                        f"but {theirs} on rank {member}; use the same gpubridge and settings "
                        "on every leader")
                worker.connect(members.index(member), address,
                               max(0.0, deadline - time.monotonic()))
            transport = cls(worker, members, rank, timeout, chunk)
            transport._handshake(deadline)
            return transport
        except BaseException:
            worker.close(timeout=1.0)
            raise

    def close(self) -> None:
        self.worker.close(timeout=min(10.0, self.timeout))
