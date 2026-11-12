"""Startup and barriers through torch's c10d store, with no process group.

Nothing here creates a process group, of Gloo's or any other backend's. Discovery runs
before any group exists, and a job that didn't ask for Gloo has none to carry a
collective before its islands and bridge exist. (Which jobs do ask, by bridge, policy
or simulation, and the one way torch itself can add Gloo, its debug level DETAIL, are
in docs/RUNTIME_INVENTORY.md, section 6.) What has to cross ranks before then goes
through the job's c10d store, the
key-value server ``torch.distributed`` starts for its own rendezvous (a TCPStore
under torchrun, a FileStore for ``file://``), the way the bridges already exchange
their addresses:

- :func:`open_store` makes the store ``init_process_group`` would have made, so
  discovery can run before the default group exists and decide what it is;
- :func:`exchange` publishes one record per rank and reads every rank's, with a
  deadline, and then says that it has read them;
- :func:`await_readers` keeps the rank that hosts the store there until every rank has
  said so, when what they read makes the job fail;
- :func:`barrier` is a counter every rank adds to and then watches.

Every wait has a deadline and names what it was waiting for. Nothing here starts a
thread or holds the store while it waits: each look is one short call, because a
client serializes its requests and other threads share it (the point-to-point lanes,
a monitoring tool).
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import torch.distributed as dist

#: The deadline when ``init()`` gave none: torch's default group timeout.
DEFAULT_TIMEOUT = timedelta(minutes=30)
#: A wait's pauses between looks at the store: from the first, doubling up to the cap.
WAIT_POLL_FIRST, WAIT_POLL_MAX = 0.0005, 0.01
BARRIER_POLL_FIRST, BARRIER_POLL_MAX = 50e-6, 0.002
#: The version of the discovery record. Ranks on a gpubridge that writes another fail
#: ``init()`` on every rank that reads it, instead of misreading each other.
RECORD_VERSION = 1

#: A seam for tests, never set by gpubridge: called by :func:`exchange` as
#: ``_after_publish(prefix, rank)`` once this rank's record is in the store and before
#: it reads anyone's. A test holds a rank there to pin an interleaving of two ranks.
_after_publish: Callable[[str, int], None] | None = None


def open_store(init_method: str | None, rank: int | None, world_size: int | None,
               timeout: timedelta | None) -> tuple[Any, int, int]:
    """The job's c10d store, this process's rank and the world size, from the same
    rendezvous ``dist.init_process_group(init_method=...)`` runs (``env://`` when
    ``init_method`` is None). Pass the store on to ``init_process_group(store=...)``.

    Raises:
        whatever torch's rendezvous handler raises for a bad URL or missing variables.
    """
    from torch.distributed.rendezvous import rendezvous

    timeout = timeout or DEFAULT_TIMEOUT
    handler = rendezvous(init_method or "env://", -1 if rank is None else rank,
                         -1 if world_size is None else world_size, timeout=timeout)
    store, rank, world_size = next(handler)
    store.set_timeout(timeout)
    return store, rank, world_size


def default_store() -> Any:
    """The default process group's store (what the bridges exchange addresses through)."""
    return dist.distributed_c10d._get_default_store()


def wait_for(store, keys: list[str], deadline: float, what: str,
             failures: Sequence[str] = (), stop: Callable[[], bool] | None = None) -> list[str]:
    """The values of ``keys`` once all exist. Gives up at ``deadline``
    (``time.monotonic()``), as soon as one of ``failures`` (a member's status key)
    holds an error, or once ``stop()`` is true (what the wait was for was closed: a
    thread must not keep asking the store while its job is being torn down).

    Raises:
        TimeoutError: naming ``what`` and the keys still missing.
        RuntimeError: with the first failure found, or because ``stop()`` said so.
    """
    pause = WAIT_POLL_FIRST
    while True:
        if stop is not None and stop():
            raise RuntimeError(f"stopped waiting for {what}: it was closed")
        if store.check(keys):
            return [store.get(k).decode() for k in keys]
        for failure in failures:
            if store.check([failure]) and store.get(failure).decode() != "ok":
                raise RuntimeError(f"a bridge member failed first: {store.get(failure).decode()}")
        left = deadline - time.monotonic()
        if left <= 0:
            missing = [k for k in keys if not store.check([k])]
            raise TimeoutError(f"timed out waiting for {what} (still missing: "
                               f"{', '.join(missing) or 'nothing'})")
        time.sleep(min(pause, left))
        pause = min(2 * pause, WAIT_POLL_MAX)


def exchange(store, prefix: str, rank: int, size: int, record: dict[str, Any],
             deadline: float, what: str) -> list[dict[str, Any]]:
    """Publish ``record`` (plain JSON values) as rank ``rank``'s under ``prefix``, and
    return every rank's, indexed by rank.

    Collective over the ``size`` ranks: each calls it once per ``prefix``.

    Once it holds every record, the rank says so with a marker of its own
    (``<prefix>/read/<rank>``): one more ``set``, which waits for nobody. From then on
    it needs nothing more from the store to know what the others reported, which is
    what :func:`await_readers` waits for.

    Raises:
        TimeoutError: if a rank's record isn't there by ``deadline``, naming the ranks.
    """
    store.set(f"{prefix}/{rank}", json.dumps(record))
    if _after_publish is not None:
        _after_publish(prefix, rank)
    keys = [f"{prefix}/{r}" for r in range(size)]
    try:
        records = [json.loads(value) for value in wait_for(store, keys, deadline, what)]
    except TimeoutError:
        missing = [r for r, key in enumerate(keys) if not store.check([key])]
        raise TimeoutError(f"timed out waiting for {what}: nothing from ranks {missing} of "
                           f"{size}. Did every rank start, with the same gpubridge?") from None
    store.set(f"{prefix}/read/{rank}", "1")
    return records


def await_readers(store, prefix: str, rank: int, size: int, deadline: float) -> list[int]:
    """Stay until every other rank has read the records of ``prefix`` (:func:`exchange`'s
    markers); the ranks that hadn't by ``deadline``, normally none.

    For the rank that hosts the job's store, before it raises because of what the
    records say. torch's rendezvous starts a TCP store's server inside rank 0 (unless a
    launcher's agent hosts it), and the server goes when that process does. A rank 0
    that read the records, raised and left would take the store from a rank that had
    published but not yet read: that rank would end with a network error from the
    store instead of the reason every rank is meant to raise. A rank that has written
    its marker has every record and reads the store no more to decide, so once all
    markers are in, rank 0 may go.

    Never raises: the caller is about to raise the real reason. A rank that vanished
    between its record and its marker holds this rank until ``deadline``, the wait the
    exchange itself had, and no longer.
    """
    keys = [f"{prefix}/read/{r}" for r in range(size) if r != rank]
    pause = WAIT_POLL_FIRST
    try:
        while keys and not store.check(keys):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(pause, left))
            pause = min(2 * pause, WAIT_POLL_MAX)
        return [int(key.rsplit("/", 1)[1]) for key in keys if not store.check([key])]
    except Exception:  # noqa: BLE001 - a store that is gone has nobody left to wait for
        return []


def barrier(store, key: str, size: int, deadline: float, *, retire: str | None = None) -> None:
    """Return once ``size`` callers have reached this call with the same ``key``.

    Each caller adds one to the counter at ``key`` and watches it reach ``size``.
    ``retire`` names the previous barrier's key, to delete: every caller is in this
    barrier only after it returned from that one, so once this one is complete nobody
    reads that key again. One caller passes it (so a job that calls barrier() every
    step keeps one key in the store, not one per step).

    Raises:
        TimeoutError: at ``deadline``, saying how many callers had arrived.
    """
    arrived = store.add(key, 1)
    pause = BARRIER_POLL_FIRST
    while arrived < size:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"gpubridge.barrier timed out: {arrived} of {size} ranks "
                               "reached it")
        time.sleep(min(pause, left))
        pause = min(2 * pause, BARRIER_POLL_MAX)
        arrived = store.add(key, 0)
    if retire is not None:
        with contextlib.suppress(Exception):  # a store without delete only keeps the key
            store.delete_key(retire)
