"""Leaving a job: ``destroy()``, the exit handler for a script that forgot it, and a fork.

The exit handler (``gpubridge._at_exit``, registered with ``atexit``) closes the bridge
a script left open, which joins the bridge's native threads, and on a job without Gloo
destroys the process groups as ``destroy()`` would have. It does nothing after
``destroy()`` and nothing in a forked child, whose sockets are its parent's.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _harness import TIMEOUT, expected_sum, rank_tensor, thread_names

import gpubridge
from gpubridge import multitcp
from gpubridge.config import VENDOR_ENV
from gpubridge.multitcp.transport import IFNAME_ENV, STREAMS_ENV

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
pytestmark = pytest.mark.skipif(UNAVAILABLE is not None,
                                reason=f"the Rust TCP bridge is unavailable: {UNAVAILABLE}")

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
VENDORS = ("nvidia", "amd")
PROC = Path("/proc/self/task").is_dir()


def _native_threads() -> list[str]:
    return [name for name in thread_names() if name.startswith("gb-tcp-")] if PROC else []


def _main(rank, init_file, bridge, fn):
    os.environ.update({VENDOR_ENV: VENDORS[rank], IFNAME_ENV: LOOPBACK, STREAMS_ENV: "2"})
    torch.set_num_threads(1)
    topology = gpubridge.init(init_method=f"file://{init_file}", rank=rank, world_size=2,
                              timeout=TIMEOUT, bridge=bridge)
    fn(topology)
    # No destroy() here: each check ends the way its name says.


def run(fn, tmp_path, bridge="tcp"):
    """Two ranks, one of each vendor; mp.spawn fails the test if a rank's exit code isn't 0."""
    mp.spawn(_main, args=(str(tmp_path / "rendezvous"), bridge, fn), nprocs=2)


def _collective(topology, shape=(33,)):
    tensor = rank_tensor(shape, torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum(shape, torch.float32, topology.world_size))


def _forgets_destroy(topology, queued):
    _collective(topology)
    work = gpubridge.all_reduce(rank_tensor((5,), torch.float32, topology.rank), async_op=True)
    assert work.wait()  # the async worker thread exists now
    if queued:  # and leave collectives queued or running when the rank returns
        for _ in range(3):
            gpubridge.all_reduce(torch.ones(1000), async_op=True)


@pytest.mark.parametrize("queued", [False, True], ids=["idle", "work-queued"])
def test_a_rank_that_never_calls_destroy_exits_cleanly(queued, tmp_path):
    run(partial(_forgets_destroy, queued=queued), tmp_path)


def test_a_gloo_job_that_never_calls_destroy_still_exits(tmp_path):
    # With bridge="gloo" the handler releases gpubridge's own state and leaves torch's
    # Gloo groups as such a script always left them. (Whether such a rank can abort at
    # exit is torch's and Python's doing: scripts/exit_abort.py counts it.)
    run(partial(_forgets_destroy, queued=False), tmp_path, bridge="gloo")


def _handler_closes_everything(topology):
    _collective(topology)
    gpubridge.all_reduce(rank_tensor((5,), torch.float32, topology.rank), async_op=True).wait()
    bridge = topology.bridge
    assert bridge.telemetry()["state"] == "ok"
    if PROC:  # two connections to the other leader, each with a sender and a receiver
        # (the accepting thread ends once the lower-index leaders have dialed)
        assert sorted(_native_threads()) == ["gb-tcp-recv"] * 2 + ["gb-tcp-send"] * 2
    assert any(t.name == "gpubridge-async" for t in threading.enumerate())
    gpubridge._at_exit()
    assert not gpubridge.is_initialized() and not dist.is_initialized()
    assert bridge.telemetry()["state"] == "closed"
    assert _native_threads() == []  # joined, not merely asked to stop
    assert not any(t.name == "gpubridge-async" for t in threading.enumerate())
    gpubridge._at_exit()  # again, as atexit will: nothing left to do
    gpubridge.destroy()   # and a late destroy() is harmless too


def test_the_exit_handler_closes_the_bridge_and_joins_its_threads(tmp_path):
    run(_handler_closes_everything, tmp_path)


def _destroy_then_handler(topology):
    _collective(topology)
    gpubridge.destroy()
    assert _native_threads() == []
    gpubridge._at_exit()
    assert not gpubridge.is_initialized() and not dist.is_initialized()


def test_the_exit_handler_does_nothing_after_destroy(tmp_path):
    run(_destroy_then_handler, tmp_path)


def _fork_in_the_middle(topology):
    _collective(topology)
    if topology.rank == 0:
        pid = os.fork()
        if pid == 0:  # the child: every way it might "clean up" must leave the job alone
            code = 1
            try:
                assert gpubridge.is_initialized() and not gpubridge.topology.owned()
                gpubridge._at_exit()  # what atexit would run in a child that exits normally
                assert gpubridge.is_initialized()  # it didn't even forget
                gpubridge.destroy()   # an explicit destroy() forgets, and closes nothing
                assert not gpubridge.is_initialized()
                assert dist.is_initialized()  # the process groups are the parent's
                code = 0
            finally:
                os._exit(code)
        assert os.waitpid(pid, 0)[1] == 0
    # The parent's bridge is intact: its sockets were not shut down by the child.
    assert topology.bridge.telemetry()["state"] == "ok"
    _collective(topology, shape=(4099,))
    gpubridge.barrier()
    gpubridge.destroy()
    assert not gpubridge.is_initialized()


def test_a_forked_child_closes_nothing_of_its_parents(tmp_path):
    run(_fork_in_the_middle, tmp_path)


def test_the_exit_handler_is_registered_once_and_is_safe_with_nothing_initialized():
    assert not gpubridge.is_initialized()
    gpubridge._at_exit()
    gpubridge.destroy()
    # In a fresh interpreter: importing gpubridge registers the handler, exactly once.
    code = ("import atexit\n"
            "seen, plain = [], atexit.register\n"
            "atexit.register = lambda f, *a, **k: (seen.append(f), plain(f, *a, **k))[1]\n"
            "import gpubridge, gpubridge.torch_backend\n"
            "print(sum(f is gpubridge._at_exit for f in seen))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "1", out.stderr
