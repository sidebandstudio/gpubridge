"""gpubridge: write collective communication once, run it on mixed NVIDIA and AMD clusters.

Ranks are grouped into one island per vendor, each using its native backend
(NCCL on NVIDIA, RCCL on AMD). Island leaders are linked by a CPU bridge (Gloo by
default, swappable via :mod:`gpubridge.transport`), and cluster-wide collectives
are composed from island collectives and the bridge by a collective policy
(:mod:`gpubridge.policies`). Monitoring or profiling tools can watch every
collective through observers (:mod:`gpubridge.observe`), :func:`in_flight`
shows the collective each thread is inside (a debugging aid for hangs), and
DistributedDataParallel can sync gradients through gpubridge with
:func:`ddp_comm_hook` (:mod:`gpubridge.ddp`).
"""

from __future__ import annotations

import atexit
import logging
import os
from datetime import timedelta
from typing import Any

import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import observe as _observe
from gpubridge import rendezvous as _rendezvous
from gpubridge import topology as _topology
from gpubridge import work as _work
from gpubridge.collectives import (
    all_gather_into_tensor,
    all_reduce,
    barrier,
    broadcast,
    reduce_scatter_tensor,
)
from gpubridge.config import BRIDGE_ENV, CPU_BACKEND
from gpubridge.ddp import ddp_comm_hook
from gpubridge.detect import Probe, probe
from gpubridge.observe import (
    CollectiveObserver,
    CollectiveRecord,
    InFlight,
    Mark,
    Phase,
    add_observer,
    elapsed_ms,
    in_flight,
    remove_observer,
)
from gpubridge.policies import (
    DEFAULT_POLICY,
    CollectivePolicy,
    register_policy,
    resolve_policy,
)
from gpubridge.split_test import SplitTestWarning
from gpubridge.topology import Island, Layout, PeerInfo, Topology, get_topology, is_initialized
from gpubridge.transport import (
    DEFAULT_TRANSPORT,
    BridgeTransport,
    GlooTransport,
    asks_for_gloo,
    register_transport,
    resolve_transport,
    transport_name,
)
from gpubridge.work import Work

# bridge="tcp", "verbs" and "ucx" register themselves when a job first runs on them
# (gpubridge.transport._BUILT_IN): importing gpubridge doesn't load them.
from gpubridge import torch_backend as _torch_backend  # noqa: E402

_torch_backend.register()  # backend="gpubridge" for torch.distributed, and solo islands
batch_isend_irecv = _torch_backend.batch_isend_irecv

__version__ = "0.4.0a0"
_log = logging.getLogger("gpubridge")

__all__ = [
    "batch_isend_irecv",
    "BridgeTransport",
    "CollectiveObserver",
    "CollectivePolicy",
    "CollectiveRecord",
    "GlooTransport",
    "InFlight",
    "Island",
    "Layout",
    "Mark",
    "PeerInfo",
    "Phase",
    "Probe",
    "ReduceOp",
    "SplitTestWarning",
    "Topology",
    "Work",
    "add_observer",
    "all_gather_into_tensor",
    "all_reduce",
    "barrier",
    "broadcast",
    "ddp_comm_hook",
    "destroy",
    "elapsed_ms",
    "get_topology",
    "in_flight",
    "init",
    "is_initialized",
    "probe",
    "reduce_scatter_tensor",
    "register_policy",
    "register_transport",
    "remove_observer",
]


def init(
    init_method: str | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    timeout: timedelta | None = None,
    bridge: str | type[BridgeTransport] | None = None,
    policy: str | type[CollectivePolicy] = DEFAULT_POLICY,
    allow_debug_gloo: bool | None = None,
) -> Topology:
    """Join the cluster and build the island and bridge groups.

    Call once on every rank. The arguments mirror
    ``torch.distributed.init_process_group``. With none given, the ``env://``
    variables set by torchrun are used (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE).

    gpubridge owns the default process group. Before it exists, every rank shares
    its :func:`probe` results and any problems that would stop it from joining
    through the job's c10d store (discovery), so a rank without a GPU or a vendor
    fails ``init()`` on the other ranks too, with its message, instead of leaving them
    waiting. The default group is then gpubridge's own
    (:class:`gpubridge.torch_backend.GpubridgeProcessGroup`: ``torch.distributed``
    calls on it run through gpubridge), and :func:`barrier` counts through the store.

    The job then has no Gloo group unless it asks for Gloo or accepts torch's own
    debug helpers. It asks for Gloo with ``bridge="gloo"`` or a ``GlooTransport``
    subclass (the default group is then a Gloo group of every rank, as every job's
    was through 0.4.0-alpha); with the ``flat-gloo`` policy, by name or as an
    ``auto-tuned`` rule (the policy makes a Gloo group of its own on any bridge); and
    in CPU simulation or CPU-only mode, where an island of several ranks is a Gloo
    group (a test device). It accepts torch's helpers with ``allow_debug_gloo``: at
    torch's distributed debug level DETAIL (``TORCH_DISTRIBUTED_DEBUG=DETAIL``, or
    ``torch.distributed.set_debug_level``), torch wraps every native island group in a
    checker that runs on a Gloo helper group, so a job that didn't ask for Gloo is
    refused at that level before any island group exists, unless it says
    ``allow_debug_gloo``. gpubridge never changes the debug level, and a level raised
    after ``init()`` is seen only when a subgroup is set up.

    What a refusal reaches, and what that rests on. Refusals decided from the
    discovery records (a rank's reported problem, ranks that disagree, the DETAIL
    refusal, two ranks on one GPU, a policy that doesn't fit the layout) are raised
    before any island group exists and have the same ``RuntimeError`` text on ranks
    that read those records; a rank with a malformed local setting (its own bridge
    name, policy name or ``GPUBRIDGE_*`` value) keeps its original exception, a
    ``ValueError``. Rank 0, which hosts a TCP store, waits for other ranks' read marks
    only until its original discovery deadline. If that deadline expires first, it
    logs the unread ranks when the store permits, then raises the original refusal; a
    reader still delayed at that point can instead receive a store error when the
    owner exits, even if it is alive and connected. The deadline bounds polling, not
    synchronous store calls or total init time. Failures later in ``init()`` (a
    bridge that cannot connect) are published through the bridge's own keys and are
    not part of this. It also needs discovery keys no earlier job wrote: a
    fresh store for the job, or a namespace of its own for each job or attempt, as a
    launcher gives. Starting a new job on a persistent store under a prefix an
    earlier job used is not supported: that job's records and read marks would be
    taken for this one's.

    The bridge between island leaders is one of gpubridge's Rust bridges unless the
    job names another: ``"auto"`` resolves, once and the same on every rank, to
    ``"verbs"`` (RDMA) if every island leader has a usable RDMA device, else to
    ``"tcp"``. :attr:`Topology.selection` records the choice and its reason, the
    ``gpubridge`` logger reports it at INFO, and it never changes during the job. A
    bridge asked for by name is never replaced: ``bridge="verbs"`` where verbs can't
    run fails ``init()`` on every rank.

    Args:
        init_method: Rendezvous URL, e.g. ``env://``, ``tcp://host:port`` or ``file:///path``.
        rank: This process's global rank. Read from the environment if omitted.
        world_size: Total number of ranks. Read from the environment if omitted.
        timeout: Timeout for every group gpubridge creates, and for discovery. None
            keeps PyTorch's default (30 minutes).
        bridge: Transport between island leaders: ``"auto"``, a registered name
            (``"tcp"``, ``"verbs"``, ``"ucx"``, ``"gloo"``) or a
            :class:`BridgeTransport` subclass. Must be the same on every rank. None
            (the default) reads ``GPUBRIDGE_BRIDGE``, else ``"auto"``.
        policy: How collectives are put together: ``"auto"``, a registered
            name (``"reduce-bridge-broadcast"``, ``"native-only"``,
            ``"flat-gloo"``) or a :class:`CollectivePolicy` subclass. Must be the
            same on every rank. ``"auto"`` picks ``native-only`` for one island
            and ``reduce-bridge-broadcast`` otherwise.
        allow_debug_gloo: Accept torch's Gloo helper groups at debug level DETAIL
            (above), and with them Gloo's worker threads and their exit-abort
            exposure, instead of being refused. Must be the same on every rank. None
            (the default) reads ``GPUBRIDGE_ALLOW_DEBUG_GLOO``, else False. It does
            nothing at any other debug level.

    Returns:
        This rank's :class:`Topology`.

    Raises:
        RuntimeError: if gpubridge or torch.distributed is already initialized,
            if any rank reported a problem (no vendor, no GPU, no NCCL/RCCL
            backend), if ranks disagree on CPU mode, the bridge transport or the
            collective policy, if the policy doesn't apply to the cluster, or if
            torch's debug level is DETAIL on a job that didn't ask for Gloo and
            didn't say ``allow_debug_gloo``.
        TimeoutError: if a rank's discovery record doesn't arrive within ``timeout``.
        ValueError: for an unknown bridge or policy name, or invalid gpubridge env vars,
            on the rank that has them. That rank first publishes the error as its
            discovery record, so every other rank raises RuntimeError with its
            message instead of waiting for it.
    """
    if is_initialized():
        raise RuntimeError("gpubridge is already initialized")
    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized; gpubridge.init() creates the "
            "default process group itself"
        )
    try:
        if bridge is None:
            bridge = os.environ.get(BRIDGE_ENV, "").strip() or DEFAULT_TRANSPORT
        transport = resolve_transport(bridge)
        chosen = resolve_policy(policy)
        # torch.cuda.set_device waits until discovery has checked every rank's
        # device (build_topology): a bad index then fails init() on every rank.
        local, config = _topology.describe_local(transport_name(transport), chosen,
                                                 allow_debug_gloo)
    except Exception as exc:
        # A name this rank can't resolve, a module it can't import, a setting it can't
        # read: the other ranks are about to wait for this rank's discovery record.
        _publish_failure(exc, init_method, rank, world_size, timeout)
        raise

    # The store torch's own rendezvous would make, first: discovery runs on it before
    # any process group exists, so what every rank reports decides the groups.
    store, rank, world_size = _rendezvous.open_store(init_method, rank, world_size, timeout)
    scope = f"world{next(_torch_backend._world_setups)}"
    deadline = _topology.discovery_deadline(timeout)
    peers = _topology.discover(store, scope, rank, world_size, local, timeout,
                               deadline=deadline)
    try:
        # Everything the ranks' reports alone can refuse, before any group exists: no
        # store call lies between a rank's reading the reports and its raising here.
        _topology.validate_peers(peers, warn=rank == 0)
        _topology.check_plan(peers, config, chosen, transport)
    except RuntimeError:
        # If this rank hosts the store, the others must have read the reports first.
        # Within discovery's own deadline, not a new one.
        _topology.hold_for_readers(store, scope, rank, world_size, deadline)
        raise

    kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
    gloo = asks_for_gloo(transport)
    dist.init_process_group(backend=CPU_BACKEND if gloo else _torch_backend.NAME,
                            store=dist.PrefixStore("default_pg", store), rank=rank,
                            world_size=world_size, **kwargs)
    try:
        if not gloo:
            return _torch_backend.adopt(peers, config, transport, chosen, timeout, scope)
        topology = _topology.build_topology(peers, config, transport=transport, policy=chosen,
                                            timeout=timeout, scope=scope)
    except BaseException:
        dist.destroy_process_group()
        raise
    _work.shutdown()
    _observe._reset()
    _topology._set_topology(topology)
    return topology


def _publish_failure(exc: Exception, init_method: str | None, rank: int | None,
                     world_size: int | None, timeout: timedelta | None) -> None:
    """Tell the other ranks that this one cannot even describe itself, before
    :func:`init` raises ``exc`` here: its discovery record is the error, so they fail
    with this rank's message instead of waiting for a record until their timeout (the
    way the torch.distributed backend's setup has always reported a bad setting).

    This rank takes part in discovery like any other: it stays until every rank has
    published, and if it is rank 0, which hosts a TCP store, until every rank has also
    read (:func:`gpubridge.topology.hold_for_readers`: a rank 0 that left sooner would
    take the store, and the reason, with it), or until the timeout if the others
    never come. Nothing is raised from here: if the job's
    store can't be reached (no rendezvous given, as in a script run by hand), there is
    nobody to tell, and ``init()`` raises ``exc`` either way.
    """
    problem = _topology.failure_text(exc)
    try:
        store, rank, world_size = _rendezvous.open_store(init_method, rank, world_size, timeout)
        scope = f"world{next(_torch_backend._world_setups)}"
        _topology.discover(store, scope, rank, world_size, None, timeout, problem=problem)
    except Exception as told:  # noqa: BLE001 - the error to raise is the caller's
        _log.debug("gpubridge: after %s, discovery ended with %s: %s", problem,
                   type(told).__name__, told)


def destroy() -> None:
    """Tear down every process group created by :func:`init`, and close the bridge.

    Queued async collectives run first. Does nothing if gpubridge is not initialized.
    A script that never calls it is covered at interpreter exit: an exit handler then
    closes the bridge, which joins its native threads, without running queued work.

    In a forked child (a ``DataLoader`` worker, say) the job belongs to the parent:
    this only forgets the child's copy of gpubridge's state. It closes nothing and
    destroys no process group, since the sockets and device handles are the parent's.
    """
    if not is_initialized():
        return
    if not _topology.owned():
        _forget()
        return
    try:
        _release()
    finally:
        # Why torch 2.8 is the floor: before it, this could deadlock. A Gloo
        # worker dropping the last reference to a Python tensor waits for the
        # GIL while holding the group's work lock, and the group's destructor
        # ran with the GIL held, waiting for that lock. torch 2.8 releases the
        # GIL first (https://github.com/pytorch/pytorch/pull/154976).
        dist.destroy_process_group()


def _forget() -> None:
    """Drop this process's gpubridge state without closing anything: a forked child's
    copy of what its parent set up."""
    _work.forget()
    _topology._set_topology(None)
    _observe._reset()


def _release(drain: bool = True) -> None:
    """Release gpubridge's state, but not the process groups: :func:`destroy` destroys
    them after, and under the torch.distributed backend ``destroy_process_group()``
    does (before, so there queued collectives fail instead: ``drain=False``)."""
    if not _topology.owned():  # a forked child: nothing here is its own to close
        _forget()
        return
    topology = get_topology()
    try:
        if drain:
            _work.shutdown()  # queued async collectives run first
        else:
            _work.abandon()
    finally:
        _topology._set_topology(None)
        _observe._reset()
    topology.policy.close()
    if topology.bridge is not None:
        topology.bridge.close()


def _at_exit() -> None:
    """Close what a script that never called :func:`destroy` (or
    ``destroy_process_group()``) left open, before the interpreter finalizes.

    Closing the bridge joins its native threads, and stopping the async worker ends
    gpubridge's own thread, so none of them is alive while the interpreter is torn
    down. Queued async collectives are not run (their peers may be gone): they fail.

    Best effort, and not bounded in time:

    - Some waits here have deadlines: the async worker is given 10 s, the TCP bridge's
      accepting thread and the verbs bridge's drain of pending work at most 10 s (less
      with a shorter job timeout), a point-to-point watcher 1 s. The joins after them
      have none, on purpose: the TCP bridge's sending and receiving threads and the
      verbs bridge's completion thread can still touch memory the caller owns, so
      they are joined, never detached, however long that takes. Shutting the sockets
      down and the stop flag are what end them; a device or driver call that never
      returned would hold the exit. That has not been seen, and no teardown fault was
      injected on hardware.
    - Python exceptions raised while closing are logged, not raised; a blocked native
      call can't be caught. If the policy's ``close()`` raises, the bridge after it is
      not closed.
    - It runs at a normal interpreter exit, an uncaught Python exception included.
      ``os._exit``, SIGKILL and a fatal abort run no exit handler at all. Handlers
      registered after gpubridge was imported run before this one, and Python's and
      torch's own teardown after it; no order against every such hook is promised.

    - After :func:`destroy`, or with nothing initialized: nothing to do.
    - In a forked child: nothing. The job is the parent's (see :func:`destroy`).
    - On a job without Gloo, whose default group is gpubridge's own, the process
      groups are destroyed too, as ``destroy()`` would have: subgroups' bridges and
      point-to-point lanes close with them.
    - With ``bridge="gloo"`` only gpubridge's own state is released: torch's Gloo
      groups are left as a script that forgot ``destroy()`` always left them.
    """
    if not _topology.owned():
        return
    try:
        topology = get_topology()
        ours = dist.is_initialized() and isinstance(
            dist.distributed_c10d._get_default_group(), _torch_backend.GpubridgeProcessGroup)
        if ours and topology.gloo_free:
            dist.destroy_process_group()  # shutdown() releases gpubridge's state
        if is_initialized():
            _release(drain=False)
    except Exception as exc:  # noqa: BLE001 - never turn a clean exit into a failed one
        _log.warning("gpubridge: closing at interpreter exit failed: %s: %s",
                     type(exc).__name__, exc)


atexit.register(_at_exit)
