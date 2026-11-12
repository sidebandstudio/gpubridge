"""The rank that hosts the store leaves last when discovery makes the job fail.

Under a ``tcp://`` rendezvous (and ``env://`` without a launcher's agent store) torch
starts the job's TCP store inside rank 0, and the server goes when that process does.
Every rank is meant to raise the same reason when discovery refuses the job. So rank 0
must not raise and exit while another rank has published its record but not yet read
the others': that rank would get a network error from a store that is gone, not the
reason. Each rank therefore marks that it has read, and rank 0 waits for the marks
before it raises (``gpubridge.rendezvous.await_readers``).

These jobs pin that interleaving instead of hoping for it: two ranks on a real TCP
store that rank 0 hosts; rank 1 is held, through the test seam
``gpubridge.rendezvous._after_publish``, after its discovery record is in the store
and before it reads; rank 0 raises as soon as the code lets it and its process ends at
once. Both ranks must report the intended reason, promptly, with no group left. The
file-store tests of the same refusals (tests/test_no_gloo.py, tests/test_debug_gloo.py)
cannot see this: a file outlives its ranks.
"""

from __future__ import annotations

import json
import multiprocessing
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

#: How long rank 1 is held between publishing and reading: far longer than rank 0 needs
#: to read both records, raise and end.
HELD = 2.0
#: Both ranks must be done well inside the job's 60 s timeout.
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


def _rank(rank, port, case, out):
    """One rank of two. Writes what its ``init`` raised, how long it took and what was
    left, then returns: the process ends at once, and with rank 0 the store."""
    os.environ[VENDOR_ENV] = case["vendors"][rank]
    os.environ.pop(ALLOW_DEBUG_GLOO_ENV, None)
    os.environ.update(case.get("env", [{}, {}])[rank])
    torch.set_num_threads(1)
    if case.get("detail"):
        dist.set_debug_level(dist.DebugLevel.DETAIL)
    held = []
    if rank == 1:
        def hold(prefix, who):
            if prefix.endswith("/peer") and not held:  # the discovery record, once
                held.append(time.monotonic())
                time.sleep(HELD)
        gpubridge.rendezvous._after_publish = hold
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
    result.update(seconds=time.monotonic() - started, held=bool(held),
                  gloo_groups=gloo_groups(), gpubridge=gpubridge.is_initialized())
    Path(out, f"rank{rank}.json").write_text(json.dumps(result))
    # And nothing else. No destroy, no wait: rank 0 is gone as soon as it has raised.


def _run(case, tmp_path):
    ctx = multiprocessing.get_context("spawn")
    port = _port()
    ranks = [ctx.Process(target=_rank, args=(rank, port, case, str(tmp_path))) for rank in (0, 1)]
    for process in ranks:
        process.start()
    for process in ranks:
        process.join(TIMEOUT.total_seconds() * 2)
        if process.is_alive():
            process.kill()
            process.join()
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def _check(results, own, shared):
    """Rank 0 raised ``own[0]`` and rank 1 ``own[1]`` (exception type names); both
    messages carry every pattern of ``shared``."""
    for result, kind in zip(results, own, strict=True):
        rank = result["rank"]
        assert result["error"] == kind, f"rank {rank}: {result}"
        for pattern in shared:
            assert re.search(pattern, result["message"]), \
                f"rank {rank}: {pattern!r} not in:\n{result['message']}"
        assert result["seconds"] < PROMPT, f"rank {rank} took {result['seconds']:.1f} s"
        assert result["gloo_groups"] == [] and result["gpubridge"] is False, result
    assert results[1]["held"] is True  # the interleaving the test is about did happen
    # Rank 0 was still there when rank 1 came to read: it stayed for at least the hold.
    assert results[0]["seconds"] >= HELD * 0.9, f"rank 0 left after {results[0]['seconds']:.2f} s"


def test_a_malformed_bridge_name_on_the_rank_that_hosts_the_store_reaches_the_other(tmp_path):
    case = {"api": "init", "vendors": ("nvidia", "amd"), "kwargs": [{"bridge": "typo"}, {}]}
    results = _run(case, tmp_path)
    _check(results, ("ValueError", "RuntimeError"), [r"unknown bridge transport 'typo'"])
    assert "1 of 2 ranks reported problems.\n  rank 0: " in results[1]["message"]


#: Two ranks of one vendor: in simulation one island of two ranks, a group of a C++
#: backend, the kind torch would wrap at DETAIL as it would an NCCL island.
ONE_ISLAND = ("nvidia", "nvidia")


@pytest.mark.parametrize("api", ["init", "backend"])
def test_a_detail_refusal_reaches_both_ranks_when_rank_0_hosts_the_store(api, tmp_path):
    results = _run({"api": api, "vendors": ONE_ISLAND, "detail": True}, tmp_path)
    _check(results, ("RuntimeError", "RuntimeError"),
           [r"torch's distributed debug level is DETAIL on ranks \[0, 1\] ",
            r"allow_debug_gloo=True", r"doesn't change the debug level itself"])


def test_an_opt_in_mismatch_reaches_both_ranks_when_rank_0_hosts_the_store(tmp_path):
    case = {"api": "init", "vendors": ONE_ISLAND,
            "kwargs": [{"allow_debug_gloo": True}, {}]}
    results = _run(case, tmp_path)
    _check(results, ("RuntimeError", "RuntimeError"),
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
