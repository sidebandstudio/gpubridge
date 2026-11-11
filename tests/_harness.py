"""Run simulated ranks as CPU processes for multi-process tests."""

from __future__ import annotations

import itertools
import math
import os
import pickle
import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import gpubridge
from gpubridge.config import (
    CHUNK_BYTES_ENV,
    CPU_ONLY_ENV,
    SPLIT_TEST_ENV,
    THRESHOLDS_ENV,
    VENDOR_ENV,
)

# Fail a deadlocked test instead of hanging the suite.
TIMEOUT = timedelta(seconds=60)
#: The names of torch's Gloo threads: the worker that can abort a rank at interpreter
#: exit on Python 3.12 (``pt_gloo_runloop``), and Gloo's TCP event loop.
GLOO_THREADS = ("pt_gloo_runloop", "gloo_tcp_loop")


#: The kernel's flag for a task that has begun to exit (``PF_EXITING``, include/linux/sched.h).
_PF_EXITING = 0x4


def thread_names() -> list[str]:
    """The kernel's name for every thread of this process that can still run code
    (Linux; empty elsewhere).

    A thread that has just been joined can stay listed for a moment: the kernel wakes
    the joiner before it takes the task out of ``/proc``. Such a task is past its last
    instruction of ours and has ``PF_EXITING`` set, so it is left out. Measured: 2 of
    3000 closes of a TCP worker pair still listed a joined ``gb-tcp-send`` thread, both
    with the flag, gone within 2 ms. A thread merely asked to stop doesn't have it."""
    names: list[str] = []
    if not Path("/proc/self/task").is_dir():
        return names
    for task in Path("/proc/self/task").iterdir():
        try:
            name = (task / "comm").read_text().strip()
            stat = (task / "stat").read_text()
        except OSError:  # the thread ended while we were reading
            continue
        # "pid (comm) state ppid pgrp session tty tpgid flags ...": comm may hold spaces.
        flags = int(stat[stat.rindex(")") + 2:].split()[6])
        if not flags & _PF_EXITING:
            names.append(name)
    return names


def gloo_groups() -> list[str]:
    """Every process group torch knows in this process whose backend is Gloo."""
    world = dist.distributed_c10d._world
    return [f"{world.pg_names.get(pg)} ({backend})" for pg, (backend, _) in world.pg_map.items()
            if str(backend) == "gloo"]


def assert_no_gloo() -> None:
    """This process holds no Gloo group, by torch's own table of groups (on every
    platform and torch release), and on Linux no thread with one of Gloo's names
    (which torch gives its Gloo threads from some release after 2.8.0)."""
    assert gloo_groups() == []
    if Path("/proc/self/task").is_dir():
        found = [name for name in thread_names() if name in GLOO_THREADS]
        assert found == [], f"Gloo threads in a job that didn't ask for Gloo: {found}"


_gathers = itertools.count()


def gather(value: Any) -> list[Any]:
    """Every rank's ``value``, by rank: the tests' own exchange between ranks.

    Through the job's c10d store, not a collective. On a job without Gloo the default
    process group is gpubridge's own, so ``dist.all_gather_object`` there is itself a
    gpubridge collective: observers would record the test's bookkeeping among the
    collectives under test, and it would take a sequence number. Collective all the
    same: every rank calls it at the same point."""
    store = dist.distributed_c10d._get_default_store()
    number, rank = next(_gathers), dist.get_rank()
    store.set(f"gpubridge-test/gather/{number}/{rank}", pickle.dumps(value))
    return [pickle.loads(store.get(f"gpubridge-test/gather/{number}/{r}"))
            for r in range(dist.get_world_size())]


def default_bridge() -> str:
    """The bridge a job that names none asks for: ``GPUBRIDGE_BRIDGE`` if the suite runs
    with one (``gloo``, to run every test on the Gloo bridge), else ``auto``."""
    return os.environ.get("GPUBRIDGE_BRIDGE", "").strip() or "auto"


def run_ranks(
    vendors: Sequence[str],
    fn: Callable[[gpubridge.Topology], None],
    tmp_path: Path,
    *,
    fake_builds: bool = False,
    init_kwargs: dict[str, Any] | None = None,
    before_init: Callable[[int], None] | None = None,
) -> None:
    """Run ``fn(topology)`` on one CPU process per entry in ``vendors``.

    By default each process fakes its vendor with ``GPUBRIDGE_VENDOR``. With
    ``fake_builds`` it instead patches ``torch.version`` to look like a CUDA or
    ROCm build and sets ``GPUBRIDGE_CPU_ONLY``, so the vendor comes from build
    detection. ``init_kwargs`` are passed to ``gpubridge.init`` on every rank.
    ``before_init(rank)``, if given, runs in each process just before init,
    e.g. to patch something init uses.

    ``fn`` and ``before_init`` must be picklable: module-level functions or
    ``functools.partial`` of one. A failure on any rank is re-raised in the
    calling process.
    """
    init_file = tmp_path / "rendezvous"
    mp.spawn(
        _rank_main,
        args=(tuple(vendors), fn, str(init_file), fake_builds, init_kwargs or {}, before_init),
        nprocs=len(vendors),
    )


def _rank_main(
    rank: int,
    vendors: tuple[str, ...],
    fn: Callable[[gpubridge.Topology], None],
    init_file: str,
    fake_builds: bool,
    init_kwargs: dict[str, Any],
    before_init: Callable[[int], None] | None = None,
) -> None:
    if fake_builds:
        os.environ[CPU_ONLY_ENV] = "1"
        torch.version.cuda = "13.0" if vendors[rank] == "nvidia" else None
        torch.version.hip = "7.2.26015" if vendors[rank] == "amd" else None
    else:
        os.environ[VENDOR_ENV] = vendors[rank]
    torch.set_num_threads(1)
    if before_init is not None:
        before_init(rank)
    topology = gpubridge.init(
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(vendors),
        timeout=TIMEOUT,
        **init_kwargs,
    )
    try:
        fn(topology)
    finally:
        gpubridge.destroy()


def rank_tensor(shape: tuple[int, ...], dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Return a tensor that differs per rank and per element.

    Values are small integers, so sums are exact even in float16 and do not
    depend on the order the bridge adds them in.
    """
    base = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return (base + rank + 1).to(dtype)


def expected_sum(shape: tuple[int, ...], dtype: torch.dtype, world_size: int) -> torch.Tensor:
    """Return the sum of :func:`rank_tensor` over ranks ``0..world_size-1``."""
    total = sum(rank_tensor(shape, torch.float64, rank) for rank in range(world_size))
    return torch.as_tensor(total).to(dtype)


def run_failing_init(setups: Sequence[dict[str, Any]], tmp_path: Path, patterns: Sequence[str]):
    """Check that ``gpubridge.init`` raises RuntimeError on every rank, without hanging.

    Each entry of ``setups`` configures one rank before it calls init:

    - ``env``: environment variables to set (gpubridge's own are cleared first),
    - ``cuda`` / ``hip``: values for ``torch.version.cuda`` / ``.hip``, to fake a build,
    - ``gpu``: what ``torch.cuda.is_available()`` should return,
    - ``gpus``: what ``torch.cuda.device_count()`` should return,
    - ``gpu_identity``: the physical GPU this rank reports driving,
    - ``init_kwargs``: extra arguments for ``gpubridge.init``.

    Every rank's error message must match every regex in ``patterns``, and
    nothing may stay initialized afterwards.
    """
    init_file = tmp_path / "rendezvous"
    mp.spawn(
        _failing_rank_main,
        args=(tuple(setups), str(init_file), tuple(patterns)),
        nprocs=len(setups),
    )


def _failing_rank_main(
    rank: int, setups: tuple[dict[str, Any], ...], init_file: str, patterns: tuple[str, ...]
) -> None:
    setup = setups[rank]
    for name in (VENDOR_ENV, CPU_ONLY_ENV, SPLIT_TEST_ENV, CHUNK_BYTES_ENV, THRESHOLDS_ENV):
        os.environ.pop(name, None)
    os.environ.update(setup.get("env", {}))
    for attr in ("cuda", "hip"):
        if attr in setup:
            setattr(torch.version, attr, setup[attr])
    if "gpu" in setup:
        torch.cuda.is_available = lambda: setup["gpu"]
    if "gpus" in setup:
        torch.cuda.device_count = lambda: setup["gpus"]
    if "gpu_identity" in setup:
        gpubridge.topology.gpu_identity = lambda device: setup["gpu_identity"]
    torch.set_num_threads(1)
    try:
        gpubridge.init(
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=len(setups),
            timeout=TIMEOUT,
            **setup.get("init_kwargs", {}),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        gpubridge.destroy()
        raise AssertionError(f"rank {rank}: gpubridge.init() unexpectedly succeeded")
    for pattern in patterns:
        assert re.search(pattern, message), f"rank {rank}: {pattern!r} not found in:\n{message}"
    assert not gpubridge.is_initialized()
    assert not torch.distributed.is_initialized()
