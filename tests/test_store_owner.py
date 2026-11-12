"""The rank that hosts the store leaves last when discovery makes the job fail.

Under a ``tcp://`` rendezvous (and ``env://`` without a launcher's agent store) torch
starts the job's TCP store inside rank 0, and the server goes when that process does.
Every rank is meant to raise the same reason when discovery refuses the job. So rank 0
must not raise and exit while another rank has published its record but not yet read
the others': that rank would get a network error from a store that is gone, not the
reason. Each rank therefore marks that it has read, and rank 0 waits for the marks
before it raises (``gpubridge.rendezvous.await_readers``).

These jobs pin that interleaving with events between the test and its two ranks, not
with a sleep, on a real TCP store that rank 0 hosts:

1. rank 1 publishes its discovery record, says so (``published``) and stops, through the
   test seam ``gpubridge.rendezvous._after_publish``, before it reads anything;
2. the test waits until rank 0 has gone as far as the code lets it while rank 1 is
   stopped: it is waiting for rank 1's mark (``holding``, set as rank 0 enters
   ``await_readers``), or, on code without that wait, its process has ended and taken
   the store with it;
3. only then does the test release rank 1, which reads, or tries to.

So the order is the same in every run: nothing depends on how long anything takes.
Every wait is bounded and fails the test when it runs out. Both ranks must then report
the intended reason, with no group left; that both are done in ``PROMPT`` seconds is a
watchdog, not the evidence of the order. The file-store tests of the same refusals
(tests/test_no_gloo.py, tests/test_debug_gloo.py) cannot see any of this: a file
outlives its ranks.
"""

from __future__ import annotations

import json
import multiprocessing
import multiprocessing.connection
import os
import random
import re
import socket
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _harness import TIMEOUT, gloo_groups

import gpubridge
from gpubridge.config import ALLOW_DEBUG_GLOO_ENV, BRIDGE_ENV, VENDOR_ENV

#: The longest the test waits for each step of the schedule before it gives up.
STEP = 60.0
#: A watchdog: both ranks must be done well inside the job's 60 s timeout once rank 1
#: is released. It says nothing about the order of events; the events do.
PROMPT = 30.0


def _port() -> int:
    """A port nobody listens on, below the range the kernel hands to outgoing
    connections (so no connection of this job can take it in between)."""
    for _ in range(200):
        port = random.randint(20000, 30000)
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port between 20000 and 30000")


def _rank(rank, port, case, out, published, holding, release):
    """One rank of two. Writes what its ``init`` raised and what was left, then returns:
    the process ends at once, and with rank 0 the store."""
    os.environ[VENDOR_ENV] = case["vendors"][rank]
    os.environ.pop(ALLOW_DEBUG_GLOO_ENV, None)
    os.environ.update(case.get("env", [{}, {}])[rank])
    torch.set_num_threads(1)
    if case.get("detail"):
        dist.set_debug_level(dist.DebugLevel.DETAIL)
    stopped = []
    if rank == 1:
        def stop(prefix, who):
            if prefix.endswith("/peer") and not stopped:  # the discovery record, once
                published.set()
                stopped.append(release.wait(2 * STEP))  # True: the test released it
        gpubridge.rendezvous._after_publish = stop
    elif hasattr(gpubridge.rendezvous, "await_readers"):
        # Rank 0 says when it starts to wait for the others' marks. (Code that has no
        # such wait never says so: the test then sees its process end instead.)
        plain = gpubridge.rendezvous.await_readers

        def waiting(*args, **kwargs):
            holding.set()
            return plain(*args, **kwargs)
        gpubridge.rendezvous.await_readers = waiting
    url = f"tcp://127.0.0.1:{port}"
    started = time.monotonic()
    result = {"rank": rank}
    try:
        if case["api"] == "init":
            gpubridge.init(init_method=url, rank=rank, world_size=2, timeout=TIMEOUT,
                           **{"bridge": "tcp", **case.get("kwargs", [{}, {}])[rank]})
        else:
            os.environ[BRIDGE_ENV] = "tcp"
            dist.init_process_group("gpubridge", init_method=url, rank=rank, world_size=2,
                                    timeout=TIMEOUT)
            dist.barrier()  # the backend sets itself up at its first collective
        result.update(error=None)
    except Exception as exc:  # noqa: BLE001 - which one is what the test checks
        result.update(error=type(exc).__name__, message=str(exc))
    result.update(seconds=time.monotonic() - started, released=stopped == [True],
                  gloo_groups=gloo_groups(), gpubridge=gpubridge.is_initialized())
    Path(out, f"rank{rank}.json").write_text(json.dumps(result))
    # And nothing else. No destroy, no wait: rank 0 is gone as soon as it has raised.


def _run(case, tmp_path):
    """Run the two ranks through the schedule; their results, and what the test saw
    rank 0 do before it released rank 1 (``"waited for the marks"`` or ``"exited"``)."""
    ctx = multiprocessing.get_context("spawn")
    port = _port()
    published, holding, release = ctx.Event(), ctx.Event(), ctx.Event()
    ranks = [ctx.Process(target=_rank, args=(rank, port, case, str(tmp_path), published,
                                             holding, release)) for rank in (0, 1)]
    for process in ranks:
        process.start()
    try:
        # 1. Rank 1's record is in the store, and rank 1 has stopped before reading.
        assert published.wait(STEP), "rank 1 never published its discovery record"
        # 2. Rank 0 goes as far as it can without rank 1: into the wait for the marks,
        #    or out of existence.
        seen, deadline = None, time.monotonic() + STEP
        while seen is None and time.monotonic() < deadline:
            if holding.is_set():
                seen = "waited for the marks"
            elif multiprocessing.connection.wait([ranks[0].sentinel], timeout=0.02):
                seen = "exited"
        assert seen is not None, "rank 0 neither waited for the marks nor ended"
        alive = ranks[0].is_alive()
        # 3. Only now may rank 1 read.
        release.set()
        released = time.monotonic()
        for process in ranks:
            process.join(max(0.0, released + 2 * PROMPT - time.monotonic()))
        took = time.monotonic() - released
    finally:
        release.set()
        for process in ranks:
            if process.is_alive():
                process.kill()
            process.join()
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    return results, {"rank 0": seen, "rank 0 alive at the release": alive, "seconds": took}


def _check(run, own, shared):
    """The schedule ran as intended: rank 0 was alive and waiting for the marks when
    rank 1 was let go. Then rank 0 raised ``own[0]`` and rank 1 ``own[1]`` (exception
    type names), and both messages carry every pattern of ``shared``."""
    results, saw = run
    assert results[1]["released"] is True, results[1]  # rank 1 did stop, and was let go
    for result, kind in zip(results, own, strict=True):
        rank = result["rank"]
        assert result["error"] == kind, f"rank {rank}: {result}; the test saw: {saw}"
        for pattern in shared:
            assert re.search(pattern, result["message"]), \
                f"rank {rank}: {pattern!r} not in:\n{result['message']}"
        assert result["gloo_groups"] == [] and result["gpubridge"] is False, result
    assert saw["rank 0"] == "waited for the marks" and saw["rank 0 alive at the release"], saw
    assert saw["seconds"] < PROMPT, saw  # the watchdog


def test_a_malformed_bridge_name_on_the_rank_that_hosts_the_store_reaches_the_other(tmp_path):
    case = {"api": "init", "vendors": ("nvidia", "amd"), "kwargs": [{"bridge": "typo"}, {}]}
    run = _run(case, tmp_path)
    _check(run, ("ValueError", "RuntimeError"), [r"unknown bridge transport 'typo'"])
    assert "1 of 2 ranks reported problems.\n  rank 0: " in run[0][1]["message"]


#: Two ranks of one vendor: in simulation one island of two ranks, a group of a C++
#: backend, the kind torch would wrap at DETAIL as it would an NCCL island.
ONE_ISLAND = ("nvidia", "nvidia")


@pytest.mark.parametrize("api", ["init", "backend"])
def test_a_detail_refusal_reaches_both_ranks_when_rank_0_hosts_the_store(api, tmp_path):
    run = _run({"api": api, "vendors": ONE_ISLAND, "detail": True}, tmp_path)
    _check(run, ("RuntimeError", "RuntimeError"),
           [r"torch's distributed debug level is DETAIL on ranks \[0, 1\] ",
            r"allow_debug_gloo=True", r"doesn't change the debug level itself"])


def test_an_opt_in_mismatch_reaches_both_ranks_when_rank_0_hosts_the_store(tmp_path):
    case = {"api": "init", "vendors": ONE_ISLAND,
            "kwargs": [{"allow_debug_gloo": True}, {}]}
    run = _run(case, tmp_path)
    _check(run, ("RuntimeError", "RuntimeError"),
           [rf"Ranks disagree on {ALLOW_DEBUG_GLOO_ENV} \(allow_debug_gloo\): on for ranks "
            r"\[0\], off for ranks \[1\]"])


def test_the_marks_cost_a_starting_job_one_set_and_no_wait(tmp_path):
    # The success path: each rank writes its mark and nobody waits for one.
    store = dist.FileStore(str(tmp_path / "store"), 1)
    sets, checks = [], []
    plain_set, plain_check = store.set, store.check

    class Counting:
        def set(self, key, value):
            sets.append(key)
            return plain_set(key, value)

        def check(self, keys):
            checks.append(list(keys))
            return plain_check(keys)

        def __getattr__(self, name):
            return getattr(store, name)

    deadline = time.monotonic() + 5
    records = gpubridge.rendezvous.exchange(Counting(), "p", 0, 1, {"v": 1}, deadline, "it")
    assert records == [{"v": 1}] and sets == ["p/0", "p/read/0"]
    assert not any("read" in key for keys in checks for key in keys)
    # await_readers: nobody else in a job of one; a rank that never marks is named at
    # the deadline, and nothing is raised.
    assert gpubridge.rendezvous.await_readers(store, "p", 0, 1, deadline) == []
    began = time.monotonic()
    assert gpubridge.rendezvous.await_readers(store, "p", 0, 3, began + 0.3) == [1, 2]
    assert 0.25 < time.monotonic() - began < 2
    store.set("p/read/1", "1")
    store.set("p/read/2", "1")
    assert gpubridge.rendezvous.await_readers(store, "p", 0, 3, time.monotonic() + 5) == []
    # Only rank 0 holds: any other rank returns at once, whatever is missing.
    began = time.monotonic()
    gpubridge.topology.hold_for_readers(store, "nobody-read-this", 1, 3, TIMEOUT)
    assert time.monotonic() - began < 0.5
