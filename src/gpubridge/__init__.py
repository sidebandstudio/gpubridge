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
from gpubridge.config import CPU_BACKEND
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
    register_transport,
    resolve_transport,
)
from gpubridge.work import Work

# bridge="ucx", "multitcp" and "verbs" register themselves when a job first names them
# (gpubridge.transport._BUILT_IN): importing gpubridge doesn't load them.
from gpubridge import torch_backend as _torch_backend  # noqa: E402

_torch_backend.register()  # backend="gpubridge" for torch.distributed, and solo islands
batch_isend_irecv = _torch_backend.batch_isend_irecv

__version__ = "0.4.0a0"

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
    bridge: str | type[BridgeTransport] = DEFAULT_TRANSPORT,
    policy: str | type[CollectivePolicy] = DEFAULT_POLICY,
) -> Topology:
    """Join the cluster and build the island and bridge groups.

    Call once on every rank. The arguments mirror
    ``torch.distributed.init_process_group``. With none given, the ``env://``
    variables set by torchrun are used (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE).

    gpubridge owns the default process group. Before it exists, every rank shares
    its :func:`probe` results and any problems that would stop it from joining
    through the job's c10d store (discovery), so a rank without a GPU or a vendor
    fails ``init()`` on every rank with one message instead of leaving the others
    waiting. The default group is then gpubridge's own
    (:class:`gpubridge.torch_backend.GpubridgeProcessGroup`: ``torch.distributed``
    calls on it run through gpubridge), and :func:`barrier` counts through the
    store: the job creates no Gloo group. Only ``bridge="gloo"`` makes the default
    group a Gloo group of every rank, as every job's was before.

    Args:
        init_method: Rendezvous URL, e.g. ``env://``, ``tcp://host:port`` or ``file:///path``.
        rank: This process's global rank. Read from the environment if omitted.
        world_size: Total number of ranks. Read from the environment if omitted.
        timeout: Timeout for every group gpubridge creates, and for discovery. None
            keeps PyTorch's default (30 minutes).
        bridge: Transport between island leaders: a registered name or a
            :class:`BridgeTransport` subclass. Must be the same on every rank.
        policy: How collectives are put together: ``"auto"``, a registered
            name (``"reduce-bridge-broadcast"``, ``"native-only"``,
            ``"flat-gloo"``) or a :class:`CollectivePolicy` subclass. Must be the
            same on every rank. ``"auto"`` picks ``native-only`` for one island
            and ``reduce-bridge-broadcast`` otherwise.

    Returns:
        This rank's :class:`Topology`.

    Raises:
        RuntimeError: if gpubridge or torch.distributed is already initialized,
            if any rank reported a problem (no vendor, no GPU, no NCCL/RCCL
            backend), if ranks disagree on CPU mode, the bridge transport or the
            collective policy, or if the policy doesn't apply to the cluster.
        TimeoutError: if a rank's discovery record doesn't arrive within ``timeout``.
        ValueError: for an unknown bridge or policy name, or invalid gpubridge env vars.
    """
    if is_initialized():
        raise RuntimeError("gpubridge is already initialized")
    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized; gpubridge.init() creates the "
            "default process group itself"
        )
    transport = resolve_transport(bridge)
    chosen = resolve_policy(policy)
    # torch.cuda.set_device waits until discovery has checked every rank's
    # device (build_topology): a bad index then fails init() on every rank.
    local, config = _topology.describe_local(transport.name, chosen)

    # The store torch's own rendezvous would make, first: discovery runs on it before
    # any process group exists, so what every rank reports decides the groups.
    store, rank, world_size = _rendezvous.open_store(init_method, rank, world_size, timeout)
    scope = f"world{next(_torch_backend._world_setups)}"
    peers = _topology.discover(store, scope, rank, world_size, local, timeout)
    _topology.validate_peers(peers, warn=rank == 0)

    kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
    gloo = issubclass(transport, GlooTransport)
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


def destroy() -> None:
    """Tear down every process group created by :func:`init`.

    Queued async collectives run first. Does nothing if gpubridge is not initialized.
    """
    if not is_initialized():
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


def _release(drain: bool = True) -> None:
    """Release gpubridge's state, but not the process groups: :func:`destroy` destroys
    them after, and under the torch.distributed backend ``destroy_process_group()``
    does (before, so there queued collectives fail instead: ``drain=False``)."""
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
