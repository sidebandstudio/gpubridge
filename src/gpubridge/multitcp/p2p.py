"""Host-staged TCP links for the torch backend's cross-island point-to-point lanes.

``bridge="tcp"``'s answer to :mod:`gpubridge.ucx.p2p`, in the same shape. The
collective bridge's worker links island leaders only and one thread drives it, so
it can't carry point-to-point messages between arbitrary ranks. Instead each matched
(source, destination, tag) lane of the common engine (:mod:`gpubridge.p2p`) connects
a worker of its own: two members, one TCP connection, the sender dialing the
receiver. The two ends find each other through the backend group's c10d store, and
only the two ranks of a lane ever rendezvous, so an unrelated slow peer can't hold
up another pair or tag.

Payloads are host bytes: the engine stages GPU tensors (pageable, pinned or in
chunks) before it posts them here. A lane is FIFO, so its messages are numbered in
the order both ends post them, and each travels as one tagged frame.

One thread uses a lane's worker at a time (its lock): the library's calls on one
worker are not thread-safe. Closing is therefore left to a helper thread that takes
each lane's lock in turn, so it never frees a worker under a wait still in the
library, and the caller's teardown waits for it only briefly.
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta

from gpubridge import multitcp
from gpubridge.rendezvous import wait_for

#: How long ``close()`` waits for its helper thread: idle lanes close at once, and a
#: lane still inside a transfer keeps its worker until that transfer ends or times out.
CLOSE_WAIT = 2.0


class _Request:
    def __init__(self, connection: _Connection, identifier: int, tensor) -> None:
        # The tensor: its memory must outlive the posted operation.
        self.connection, self.identifier, self.tensor = connection, identifier, tensor
        self.done = False

    def wait(self, timeout: timedelta) -> None:
        connection = self.connection
        with connection.lock:
            if self.done:
                return
            connection.worker.wait([self.identifier],
                                   time.monotonic() + timeout.total_seconds())
            self.done = True


class _Connection:
    """One lane's end: a two-member worker, index 0 on the sending rank."""

    def __init__(self, worker: multitcp.Worker, sending: bool) -> None:
        self.worker, self.sending = worker, sending
        self.lock = threading.Lock()
        self.posted = 0  # the next message's number, the same at both ends

    def post(self, tensor) -> _Request:
        with self.lock:
            nbytes = tensor.numel() * tensor.element_size()
            number, self.posted = self.posted, self.posted + 1
            if self.sending:
                identifier = self.worker.send(1, tensor.data_ptr(), nbytes, number)
            else:
                identifier = self.worker.recv(0, tensor.data_ptr(), nbytes, number)
            return _Request(self, identifier, tensor)

    def close(self) -> None:
        with self.lock:  # after any wait still in the library
            self.worker.close(timeout=0.1)


class TcpLink:
    """Local factory; only the two endpoints of a submitted lane rendezvous."""

    def __init__(self, rank: int, store, timeout: timedelta) -> None:
        self.rank, self.store, self.timeout = rank, store, timeout.total_seconds()
        self.connections: dict[tuple[int, int, bool], _Connection] = {}
        self.lock = threading.Lock()
        self.closed = False

    def _connect(self, peer: int, tag: int, sending: bool) -> _Connection:
        from gpubridge.multitcp.transport import local_address

        index = 0 if sending else 1  # the lower index dials: the sender
        worker = multitcp.Worker(2, index, 1, local_address())
        src, dst = (self.rank, peer) if sending else (peer, self.rank)
        prefix = f"tcp/{src}/{dst}/{tag}"
        mine, other = ("send", "recv") if sending else ("recv", "send")
        deadline = time.monotonic() + self.timeout
        try:
            protocol = multitcp.protocol()
            self.store.set(f"{prefix}/{mine}", f"{protocol}|{worker.address()}")
            key = f"{prefix}/{other}"
            entry, = wait_for(self.store, [key], deadline, "the P2P peer's TCP address",
                              stop=lambda: self.closed)
            self.store.delete_key(key)
            theirs, address = entry.split("|", 1)
            if theirs != str(protocol):
                raise RuntimeError(f"TCP P2P protocols differ: {protocol} here, {theirs} on "
                                   f"rank {peer}; use the same gpubridge on every rank")
            worker.connect(1 - index, address, max(0.0, deadline - time.monotonic()))
            return _Connection(worker, sending)
        except BaseException:
            worker.close(timeout=0.1)
            raise

    def post(self, tensor, peer: int, tag: int, sending: bool) -> _Request:
        key = peer, tag, sending
        with self.lock:
            if self.closed:
                raise RuntimeError("TCP point-to-point link was closed")
            connection = self.connections.get(key)
        if connection is None:
            connection = self._connect(peer, tag, sending)  # no link lock while waiting
            with self.lock:
                if self.closed:
                    connection.close()
                    raise RuntimeError("TCP point-to-point link was closed while connecting")
                self.connections[key] = connection
        return connection.post(tensor)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            connections = list(self.connections.values())
            self.connections.clear()

        def cleanup() -> None:
            for connection in connections:
                connection.close()

        # A worker can't be closed under a wait another thread has in the library: the
        # helper takes each lane's lock, and teardown doesn't wait for a stuck lane.
        helper = threading.Thread(target=cleanup, name="gpubridge-p2p-tcp-close", daemon=True)
        helper.start()
        helper.join(CLOSE_WAIT)
