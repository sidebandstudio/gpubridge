"""Collective policies: how a cluster-wide collective is put together.

A policy decides which process groups carry an all_reduce or broadcast, and in
which order. It never changes what the collective computes. ``collectives.py``
validates each call and hands it to the active policy, which ``init()`` picks
once and checks is the same on every rank. Policies are registered by name,
the same way ``transport.py`` registers bridge transports.

Built in:

- ``reduce-bridge-broadcast``: island reduce onto each leader, island leaders
  over the bridge, island broadcast. The default for clusters with two or
  more islands.
- ``native-only``: one native collective on the single island. The default for
  single-vendor clusters.
- ``flat-gloo``: every rank stages its tensor to CPU and uses the world Gloo
  group. Slow but obviously correct: the reference in tests, and a fallback for
  debugging that never touches NCCL/RCCL communicators.
- ``pipelined-reduce-bridge-broadcast``: reduce-bridge-broadcast with the
  leaders' bridge step split into chunks, so copies to and from the GPU
  overlap the bridge (``GPUBRIDGE_CHUNK_BYTES``, default 4 MiB).
- ``sharded-bridge``: k ranks per island share the bridge step, one segment
  each, over k bridge links (k is the size of the smallest island).
- ``auto-tuned``: picks one of the others by message size, from a thresholds
  file measured on the cluster (``GPUBRIDGE_THRESHOLDS``); without one, it
  does what ``auto`` does.

``auto`` (the default) picks ``native-only`` for one island and
``reduce-bridge-broadcast`` otherwise. It never picks the pipelined, sharded or
auto-tuned policies: they are opt-in until validated on GPUs.

Bridge compression (``GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8``, off by default)
is the one exception to "policies never change what a collective computes":
the three bridged policies (reduce-bridge-broadcast, pipelined, sharded) then
compress the bridge step of floating-point SUM reductions (AVG included), as
:mod:`gpubridge.compression` describes. Their island steps, every other op and
dtype, and the native-only and flat-gloo policies are unaffected.

What each built-in policy carries:

==========================  =====================  ==========================
collective                  reductions             policies
==========================  =====================  ==========================
all_reduce                  SUM, AVG, MAX, MIN     all built-in
broadcast                   (none)                 all built-in
all_gather_into_tensor      (none)                 all built-in
reduce_scatter_tensor       SUM, AVG               all built-in
==========================  =====================  ==========================

AVG is always a SUM followed by a division in ``collectives.py``, so policies
never see it. MAX and MIN over the bridge also need the bridge transport to
list them (the default Gloo transport does). A custom policy that doesn't
override ``all_gather_into_tensor`` or ``reduce_scatter_tensor`` simply
doesn't offer them: the call fails on every rank before any communication.
"""

from __future__ import annotations

import abc
import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import compression as _compression
from gpubridge import staging as _staging
from gpubridge.config import GPU_BACKEND, THRESHOLDS_ENV, chunk_bytes_setting
from gpubridge.observe import phase

if TYPE_CHECKING:
    from gpubridge.topology import Layout, Topology
    from gpubridge.transport import BridgeTransport

AUTO = "auto"


def _stage_to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` to CPU for the bridge or Gloo.

    Returns the tensor itself when it is already on CPU (simulation mode), so
    the copy back is skipped there. Tests replace this with ``clone`` to run the
    copy-out-and-back path that GPUs take.
    """
    return tensor.cpu()


def _host_like(tensor: torch.Tensor) -> torch.Tensor:
    """A CPU buffer to receive ``tensor``'s data into.

    On CPU it is whatever :func:`_stage_to_host` returns (the tensor itself,
    or a copy in tests); on GPUs a new buffer, since its old contents aren't
    needed.
    """
    if tensor.device.type == "cpu":
        return _stage_to_host(tensor)
    return torch.empty(tensor.shape, dtype=tensor.dtype)


def _bridge_all_reduce(bridge: BridgeTransport, tensor: torch.Tensor, op: ReduceOp.RedOpType
                       ) -> None:
    # SUM goes without op, so transports written before reductions existed still work.
    if op == ReduceOp.SUM:
        bridge.all_reduce(tensor)
    else:
        bridge.all_reduce(tensor, op=op)


def _world(topology: Topology):
    """A Gloo group of every member: the job's own with ``bridge="gloo"``, else one made
    the first time ``flat-gloo`` is used (:func:`gpubridge.topology.gloo_world`)."""
    from gpubridge.topology import gloo_world  # topology imports this module

    return gloo_world(topology)


def _native(topology: Topology) -> bool:
    """Whether islands run NCCL/RCCL (GPUs), which have the tensor collectives built in."""
    return topology.config.island_backend == GPU_BACKEND


def _all_gather(output: torch.Tensor, input: torch.Tensor, group, native: bool) -> None:  # noqa: A002
    """all_gather ``input`` into ``output`` (rows in group-rank order) on ``group``."""
    if native:
        dist.all_gather_into_tensor(output, input, group=group)
    else:  # Gloo: the list form works on every supported torch release
        size = dist.get_world_size(group)
        dist.all_gather(list(output.view(-1).chunk(size)), input.reshape(-1), group=group)


def _reduce_scatter(output: torch.Tensor, input: torch.Tensor, group, rank: int,  # noqa: A002
                    native: bool, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
    """Reduce ``input`` over ``group`` and keep slice ``rank`` (a group rank) in ``output``."""
    if native:
        dist.reduce_scatter_tensor(output, input, op=op, group=group)
    else:  # Gloo: all_reduce a copy, then slice it; the input stays unmodified
        work = input.clone()
        dist.all_reduce(work, op=op, group=group)
        size = dist.get_world_size(group)
        output.view(-1).copy_(work.view(size, -1)[rank])


class CollectivePolicy(abc.ABC):
    """How all_reduce and broadcast travel between ranks. One instance per rank.

    ``init()`` refuses a job whose ranks asked for different policies, and the
    public collectives require the same shape and dtype on every rank. So every
    rank always takes the same path, which is what keeps a policy from hanging.
    """

    name: ClassVar[str]

    #: Reductions :meth:`all_reduce` can carry: always ``"sum"``, plus ``"max"``
    #: and ``"min"`` if listed. Only a policy that lists one is called with it,
    #: as ``all_reduce(tensor, topology, op=ReduceOp.MAX)``; SUM always comes
    #: without ``op``, so policies written before reductions existed still work.
    reduce_ops: ClassVar[frozenset[str]] = frozenset({"sum"})

    @classmethod
    @abc.abstractmethod
    def applies_to(cls, layout: Layout) -> bool:
        """Whether this policy can run on a cluster with this island layout."""

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        """Sum an already-validated tensor across every rank, in place.

        A policy that lists ``"max"`` or ``"min"`` in :attr:`reduce_ops` also
        takes ``op=ReduceOp.MAX`` / ``ReduceOp.MIN``.
        """

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        """Copy an already-validated tensor from global rank ``src`` to every rank, in place."""

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        """Gather every rank's ``input`` into ``output``, ordered by global rank.

        Optional. Policies that don't override it don't offer the collective;
        ``gpubridge.all_gather_into_tensor`` checks that before communicating.
        """
        raise NotImplementedError(
            f"collective policy {self.name!r} doesn't implement all_gather_into_tensor")

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        """SUM every rank's ``input`` and leave this rank's slice in ``output``.

        Optional, like :meth:`all_gather_into_tensor`. Must not modify ``input``.
        """
        raise NotImplementedError(
            f"collective policy {self.name!r} doesn't implement reduce_scatter_tensor")

    def supported_reduce_ops(self, topology: Topology) -> frozenset[str]:
        """The reductions this policy can carry on this cluster.

        Depends only on state that is identical on every rank, so a call it
        rejects is rejected everywhere. The collectives ask once per policy
        instance and topology, at the first collective after ``init()``, and
        remember the answer, so it must not change after that (a policy object
        that can't hold the answer, one with ``__slots__``, is asked every call).
        The default is :attr:`reduce_ops`.
        """
        return self.reduce_ops

    @classmethod
    def settings(cls) -> dict[str, Any]:
        """Settings this policy reads from the environment, which every rank must share.

        Called on every rank before discovery. Discovery refuses a job whose
        ranks report different settings, because a setting such as a chunk
        size decides how many messages a collective sends: ranks that disagree
        would hang. Return plain JSON values. Raise ``ValueError`` for a bad
        value: ``init()`` then fails on every rank with that message. The
        policy reads the agreed values in :meth:`setup` from
        ``topology.policy_settings``. The default has none.
        """
        return {}

    def setup(self, topology: Topology) -> None:  # noqa: B027 - optional hook
        """Called once by ``init()`` after every group exists, e.g. to allocate staging buffers."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Called by ``destroy()``. Release anything :meth:`setup` acquired."""

    def select_policy(self, nbytes: int) -> CollectivePolicy:
        """Return the policy that carries a collective of ``nbytes`` bytes.

        The hook for choosing by message size, e.g. one path for small messages
        and a pipelined one for large messages. The default uses this policy for
        every size. An override may depend only on ``nbytes`` and on state that
        is identical on every rank, so that every rank picks the same path.
        """
        return self


class ReduceBridgeBroadcast(CollectivePolicy):
    """Reduce in each island, combine island sums over the bridge, broadcast back.

    all_reduce:

    1. reduce within each island onto its leader, on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    Step 1 is a reduce, not an all_reduce, because only the leader uses the
    island sum: the other ranks' copies would be overwritten by step 3 anyway.
    A ring reduce moves about half the bytes of a ring all_reduce. The other
    ranks' tensors hold unspecified values between steps 1 and 3. MAX and MIN
    take the same three steps.

    broadcast: from ``src`` to its island, then from that island's leader to the
    other leaders over the bridge, then from each leader to its island.

    all_gather_into_tensor: island gather onto the leader, then each leader in
    turn broadcasts its island's rows over the bridge, then each leader
    broadcasts the assembled output to its island. Bridge broadcasts are pure
    copies, so the result is exact for any data.

    reduce_scatter_tensor: island reduce onto the leader, bridge all_reduce of
    the whole input, then each leader scatters its island members' slices.

    With a bridge transport that takes GPU tensors (``device_tensors``, e.g.
    ``bridge="ucx"``), GPU leaders hand the tensor straight to it. Otherwise
    leaders copy to host memory with ``.cpu()`` (pageable) by default. With
    ``GPUBRIDGE_PINNED_STAGING=1`` the bridge steps of all_reduce, broadcast and
    reduce_scatter_tensor go through one pinned buffer per leader instead
    (:class:`gpubridge.staging.PinnedBridgeStaging`), so one allocation can
    compare the two. all_gather_into_tensor keeps pageable buffers.
    """

    name = "reduce-bridge-broadcast"
    reduce_ops = frozenset({"sum", "max", "min"})
    #: Pinned staging, when ``GPUBRIDGE_PINNED_STAGING`` is on and this rank is a GPU
    #: leader; None otherwise, and always None in subclasses that set up their own.
    staging_buffer: _staging.BridgeStaging | None = None

    #: True on a GPU leader whose bridge transport takes GPU tensors itself
    #: (``device_tensors``): the bridge step then makes no host copy at all.
    direct_bridge: bool = False

    #: Bridge compression for this instance: None follows ``GPUBRIDGE_BRIDGE_COMPRESSION``
    #: (``topology.bridge_compression``), ``""`` turns it off and ``"bf16"`` or ``"fp8"``
    #: picks a mode, e.g. to compare modes in one job (``bench_all_reduce.py
    #: --compression``). Every rank must construct the same.
    compression: str | None = None
    #: Set up on bridge members when compression is on; None otherwise.
    compressor: _compression.BridgeCompressor | None = None

    def __init__(self, compression: str | None = None) -> None:
        self.compression = compression

    def _setup_compression(self, topology: Topology, member: bool) -> None:
        mode = (getattr(topology, "bridge_compression", "") if self.compression is None
                else self.compression)
        self.compressor = (_compression.BridgeCompressor(mode)
                           if mode and member and topology.layout.needs_bridge else None)

    def _compresses(self, tensor: torch.Tensor, op: ReduceOp.RedOpType) -> bool:
        """Whether this bridge step is compressed: the same answer on every bridge member."""
        return (self.compressor is not None and op == ReduceOp.SUM and tensor.numel() > 0
                and self.compressor.applies(tensor.dtype))

    def setup(self, topology: Topology) -> None:
        self._setup_compression(topology, member=topology.is_leader)
        if not (topology.layout.needs_bridge and topology.is_leader):
            return
        self.direct_bridge = (topology.device.type == "cuda"
                              and getattr(topology.bridge, "device_tensors", False))
        if topology.pinned_staging and not self.direct_bridge:
            self.staging_buffer = _staging.make_bridge_staging(topology.device)

    def close(self) -> None:
        if self.staging_buffer is not None:
            self.staging_buffer.close()
            self.staging_buffer = None
        self.compressor = None

    def _to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.direct_bridge:
            return tensor
        if self.staging_buffer is not None:
            return self.staging_buffer.to_host(tensor)
        return _stage_to_host(tensor)

    def _to_device(self, tensor: torch.Tensor, host: torch.Tensor) -> None:
        if host is tensor:
            return
        if self.staging_buffer is not None:
            self.staging_buffer.to_device(tensor, host)
        else:
            tensor.copy_(host)

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return layout.needs_bridge

    def supported_reduce_ops(self, topology: Topology) -> frozenset[str]:
        assert topology.transport is not None
        return self.reduce_ops & topology.transport.reduce_ops

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        leader = topology.island.leader
        with phase("island-reduce"):
            dist.reduce(tensor, dst=leader, op=op, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            if self._compresses(tensor, op):
                self._compressed_bridge(tensor, tensor, topology)
            else:
                with phase("bridge"):
                    staged = self._to_host(tensor)
                    _bridge_all_reduce(topology.bridge, staged, op)
                    self._to_device(tensor, staged)
        with phase("island-broadcast"):
            dist.broadcast(tensor, src=leader, group=topology.island_group)

    def _compressed_bridge(self, tensor: torch.Tensor, buffer: torch.Tensor,
                           topology: Topology) -> None:
        """The leaders' bridge step, compressed: phases compress, bridge, decompress.
        ``buffer`` is the caller's tensor, whose error feedback residual this uses."""
        assert self.compressor is not None and topology.bridge is not None
        self.compressor.all_reduce(tensor.view(-1), _compression.buffer_key(buffer),
                                   topology.bridge, topology.layout.bridge_ranks,
                                   rank=topology.rank, direct=self.direct_bridge)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        src_island = topology.layout.island_of(src)
        in_src_island = topology.island == src_island
        if in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=src, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            with phase("bridge"):
                staged = self._to_host(tensor)
                topology.bridge.broadcast(staged, src=src_island.leader)
                if not in_src_island:
                    self._to_device(tensor, staged)
        if not in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        island, world, n = topology.island, topology.world_size, input.numel()
        rows: torch.Tensor | None = None
        with phase("island-gather"):
            if topology.is_leader:
                rows = torch.empty((len(island.ranks), n), dtype=input.dtype, device=input.device)
            dist.gather(input.reshape(-1), list(rows.unbind(0)) if rows is not None else None,
                        dst=island.leader, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None and rows is not None
            with phase("bridge"):
                assembled = _host_like(output).view(world, n)
                mine = _stage_to_host(rows)
                for other in topology.layout.islands:  # same order on every leader
                    staged = mine if other == island else torch.empty(
                        (len(other.ranks), n), dtype=input.dtype)
                    topology.bridge.broadcast(staged, src=other.leader)
                    assembled[[topology.layout.index(r) for r in other.ranks]] = staged
                if assembled.data_ptr() != output.data_ptr():
                    output.view(world, n).copy_(assembled)
        with phase("island-broadcast"):
            dist.broadcast(output, src=island.leader, group=topology.island_group)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        island, world = topology.island, topology.world_size
        work = input.clone()  # the input stays unmodified
        with phase("island-reduce"):
            dist.reduce(work, dst=island.leader, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            if self._compresses(work, ReduceOp.SUM):
                self._compressed_bridge(work, input, topology)
            else:
                with phase("bridge"):
                    staged = self._to_host(work)
                    topology.bridge.all_reduce(staged)
                    self._to_device(work, staged)
        with phase("island-scatter"):
            slices = work.view(world, -1)
            dist.scatter(output.view(-1),
                         [slices[topology.layout.index(r)] for r in island.ranks]
                         if topology.is_leader else None,
                         src=island.leader, group=topology.island_group)


class NativeOnly(CollectivePolicy):
    """One native collective on the only island: NCCL, RCCL, or Gloo on CPU."""

    name = "native-only"
    reduce_ops = frozenset({"sum", "max", "min"})

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return not layout.needs_bridge

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        dist.all_reduce(tensor, op=op, group=topology.island_group)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        dist.broadcast(tensor, src=src, group=topology.island_group)

    # The only island holds every member in ascending order, so its group ranks
    # are member positions and native output order is member order.
    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        _all_gather(output, input, topology.island_group, _native(topology))

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        _reduce_scatter(output, input, topology.island_group,
                        topology.layout.index(topology.rank), _native(topology))


class FlatGloo(CollectivePolicy):
    """Every rank copies its tensor to CPU and runs the collective on the world Gloo group.

    Slow, but it has no islands, no bridge and no leaders, so it is the
    reference other policies are tested against, and a fallback for debugging.
    """

    name = "flat-gloo"
    reduce_ops = frozenset({"sum", "max", "min"})

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return True

    def setup(self, topology: Topology) -> None:
        # On a bridge other than Gloo no Gloo group of every rank exists: asking for
        # this policy is what creates one, here at init() rather than mid-training.
        _world(topology)

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        staged = _stage_to_host(tensor)
        dist.all_reduce(staged, op=op, group=_world(topology))
        if staged is not tensor:
            tensor.copy_(staged)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        staged = _stage_to_host(tensor)
        dist.broadcast(staged, src=src, group=_world(topology))
        if staged is not tensor:
            tensor.copy_(staged)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        received = _host_like(output)
        _all_gather(received, _stage_to_host(input), _world(topology), native=False)
        if received.data_ptr() != output.data_ptr():
            output.copy_(received)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        received = _host_like(output)
        _reduce_scatter(received, _stage_to_host(input), _world(topology),
                        topology.layout.index(topology.rank), native=False)
        if received.data_ptr() != output.data_ptr():
            output.copy_(received)


class PipelinedReduceBridgeBroadcast(ReduceBridgeBroadcast):
    """Reduce-bridge-broadcast with the leaders' bridge step split into pipelined chunks.

    Each leader moves its tensor through the bridge chunk by chunk. On GPUs,
    the copy of chunk k+1 into pinned host memory, the bridge step of chunk k
    and the copy of chunk k-1 back to the GPU run at the same time: copies go
    on two side streams and are ordered with CUDA events, and the host waits
    only for the chunk the bridge needs next (see :mod:`gpubridge.staging`).
    ``setup()`` allocates the pinned buffers and streams once.

    Ideas 1 and 3 from docs/PRIOR_ART.md (Triton-distributed). The chunk size
    comes from ``GPUBRIDGE_CHUNK_BYTES`` (default 4 MiB) and must match on
    every rank; discovery checks. Island steps, all_gather_into_tensor and
    reduce_scatter_tensor are those of reduce-bridge-broadcast.
    """

    name = "pipelined-reduce-bridge-broadcast"
    #: Host slots: chunk k-1 copies back, k is on the bridge, k+1 copies out.
    SLOTS = 3
    DEFAULT_CHUNK_BYTES = 4 * 2**20

    def __init__(self, chunk_bytes: int | None = None, compression: str | None = None) -> None:
        super().__init__(compression)
        self.chunk_bytes = chunk_bytes
        self.staging: _staging.Staging | None = None
        #: The last copy back out of each host slot, kept from one collective to
        #: the next: a collective returns without waiting for its copies back
        #: (only the caller's stream waits for them), so the next one, perhaps
        #: from another stream, must not reuse a slot before they are done.
        self.copied_back: list[Any] = [None] * self.SLOTS
        #: With compression: host slots for the other islands' payload chunks
        #: (SLOTS per other island), and the last copy back out of each.
        self.recv_staging: _staging.Staging | None = None
        self.recv_back: list[Any] = []

    @classmethod
    def settings(cls) -> dict[str, Any]:
        return {"chunk_bytes": chunk_bytes_setting(cls.DEFAULT_CHUNK_BYTES)}

    def setup(self, topology: Topology) -> None:
        if self.chunk_bytes is None:
            self.chunk_bytes = int(topology.policy_settings.get("chunk_bytes",
                                                                self.DEFAULT_CHUNK_BYTES))
        self._setup_compression(topology, member=topology.is_leader)
        if topology.layout.needs_bridge and topology.is_leader:
            self.staging = _staging.make_staging(topology.device, self.chunk_bytes, self.SLOTS)
            if self.compressor is not None:
                slots = self.SLOTS * (len(topology.layout.islands) - 1)
                self.recv_staging = _staging.make_staging(topology.device, self.chunk_bytes, slots)
                self.recv_back = [None] * slots

    def close(self) -> None:
        for staging in (self.staging, self.recv_staging):
            if staging is not None:
                staging.close()
        self.staging = self.recv_staging = None
        self.copied_back = [None] * self.SLOTS
        self.recv_back = []
        self.compressor = None

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        leader = topology.island.leader
        with phase("island-reduce"):
            dist.reduce(tensor, dst=leader, op=op, group=topology.island_group)
        if topology.is_leader:
            bridge = topology.bridge
            assert bridge is not None
            if self._compresses(tensor, op):
                self._compressed_pipeline(tensor, topology)
            else:
                with phase("bridge"):
                    self._pipeline(tensor, lambda host: _bridge_all_reduce(bridge, host, op),
                                   send=True, receive=True)
        with phase("island-broadcast"):
            dist.broadcast(tensor, src=leader, group=topology.island_group)

    def _compressed_pipeline(self, tensor: torch.Tensor, topology: Topology) -> None:
        """Compress the whole island sum once, exchange the payloads chunk by chunk
        through the host slots, then decompress once.

        Payload chunks are ``chunk_bytes`` long. For chunk k, every leader
        broadcasts its own slice in leader order; the copy of this leader's
        chunk k+1 to the host and of the others' chunk k-1 back to the GPU
        overlap those broadcasts, as in :meth:`_pipeline`.
        """
        assert self.compressor is not None and topology.bridge is not None
        bridge, members = topology.bridge, topology.layout.bridge_ranks
        me = members.index(topology.rank)
        flat = tensor.view(-1)
        gathered = self.compressor.compress(flat, _compression.buffer_key(tensor), members, me)
        with phase("bridge"):
            self._exchange(gathered, me, members, bridge)
        self.compressor.decompress(gathered, flat)

    def _exchange(self, gathered: torch.Tensor, me: int, members: tuple[int, ...],
                  bridge: BridgeTransport) -> None:
        staging, recv, slots = self.staging, self.recv_staging, self.SLOTS
        assert staging is not None and recv is not None and self.chunk_bytes is not None
        width = gathered.shape[1]
        cuts = [(lo, min(lo + self.chunk_bytes, width)) for lo in range(0, width, self.chunk_bytes)]
        others = [i for i in range(len(members)) if i != me]
        mine = gathered[me]
        copied_out: list[Any] = [None] * len(cuts)
        last_back = None

        def copy_out(k: int) -> None:
            lo, hi = cuts[k]
            slot = staging.slot(k % slots, mine[lo:hi])
            # The slot may still be read by a copy back from an uncompressed collective.
            copied_out[k] = staging.to_host(slot, mine[lo:hi], after=self.copied_back[k % slots])

        staging.begin(gathered)
        recv.begin(gathered)
        if cuts:
            copy_out(0)
        for k, (lo, hi) in enumerate(cuts):
            if k + 1 < len(cuts):
                copy_out(k + 1)  # overlaps this chunk's broadcasts
            hosts: dict[int, torch.Tensor] = {}
            for j, i in enumerate(others):
                index = (k % slots) * len(others) + j
                # The bridge is about to overwrite this slot: its last copy back must be done.
                recv.wait_host(self.recv_back[index])
                hosts[i] = recv.slot(index, gathered[i, lo:hi])
            staging.wait_host(copied_out[k])
            hosts[me] = staging.slot(k % slots, mine[lo:hi])
            for i, src in enumerate(members):
                bridge.broadcast(hosts[i], src=src)
            for j, i in enumerate(others):
                index = (k % slots) * len(others) + j
                self.recv_back[index] = last_back = recv.to_device(gathered[i, lo:hi], hosts[i])
        recv.end(last_back)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        src_island = topology.layout.island_of(src)
        in_src_island = topology.island == src_island
        if in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=src, group=topology.island_group)
        if topology.is_leader:
            bridge = topology.bridge
            assert bridge is not None
            with phase("bridge"):
                self._pipeline(tensor, lambda host: bridge.broadcast(host, src=src_island.leader),
                               send=in_src_island, receive=not in_src_island)
        if not in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)

    def _pipeline(self, tensor: torch.Tensor, bridge_step, *, send: bool, receive: bool) -> None:
        """Run ``bridge_step`` on every chunk's host slot, overlapping the copies.

        ``send``: copy each chunk to the host before its bridge step.
        ``receive``: copy each slot back into its chunk after its bridge step.
        Every leader cuts the same chunks, because the chunk size and the
        tensor's size and dtype are the same on every rank.
        """
        staging, slots = self.staging, self.SLOTS
        assert staging is not None and self.chunk_bytes is not None
        flat = tensor.view(-1)
        step = max(1, self.chunk_bytes // tensor.element_size())
        chunks = [flat[lo:lo + step] for lo in range(0, flat.numel(), step)]
        copied_out: list[Any] = [None] * len(chunks)  # chunk k is in host memory
        copied_back = self.copied_back                # last copy back out of each slot
        last_back = None

        def copy_out(k: int) -> None:
            slot = staging.slot(k % slots, chunks[k])
            copied_out[k] = staging.to_host(slot, chunks[k], after=copied_back[k % slots])

        staging.begin(tensor)
        if send and chunks:
            copy_out(0)
        for k, chunk in enumerate(chunks):
            if send and k + 1 < len(chunks):
                copy_out(k + 1)  # overlaps this chunk's bridge step
            host = staging.slot(k % slots, chunk)
            # Before the bridge touches the slot: its data has arrived (send), or
            # the copy back of the chunk that used it last, in this collective or
            # an earlier one, is done (receive only).
            staging.wait_host(copied_out[k] if send else copied_back[k % slots])
            bridge_step(host)
            if receive:
                copied_back[k % slots] = last_back = staging.to_device(chunk, host)
        staging.end(last_back)


class ShardedBridge(ReduceBridgeBroadcast):
    """Split the bridge step across several links, so no single leader carries every byte.

    In reduce-bridge-broadcast one rank per island moves the whole tensor
    through host memory and over the bridge. Here k ranks per island do,
    where k is the size of the smallest island: rank j of every island (in
    island order) joins bridge link j, and link 0 is the leader bridge
    ``init()`` already made. ``setup()`` creates links 1 to k-1 on every rank,
    in the same order, through :meth:`BridgeTransport.create_groups`.

    all_reduce, on the tensor padded to k equal segments:

    1. Each island leaves its reduction of segment j on its rank j: one
       reduce_scatter if the island has exactly k ranks, else k reduces.
    2. Rank j of every island all_reduces segment j over link j. The k links
       run at the same time, each carrying 1/k of the bytes.
    3. Each island reassembles the tensor: one all_gather if it has exactly k
       ranks, else k broadcasts.

    This is the usual two-level all_reduce (reduce_scatter inside, all_reduce
    across, all_gather inside) with islands in place of nodes. Padding is
    reduced only with padding, and never copied back.

    broadcast: island broadcast from ``src``, then segment j from rank j of
    ``src``'s island over link j, then step 3 in the other islands.

    With an island of one rank, k is 1 and every step is reduce-bridge-
    broadcast's. all_gather_into_tensor and reduce_scatter_tensor are those
    of reduce-bridge-broadcast.
    """

    name = "sharded-bridge"

    def __init__(self, compression: str | None = None) -> None:
        super().__init__(compression)
        self.k = 1
        self.index = 0  # this rank's position in its island
        self.link: BridgeTransport | None = None  # link ``index``, on ranks below k
        self.link_members: tuple[int, ...] = ()  # its global ranks, ascending
        self._created: list[BridgeTransport] = []

    def setup(self, topology: Topology) -> None:
        islands = topology.layout.islands
        self.k = min(len(island.ranks) for island in islands)
        self.index = topology.island.ranks.index(topology.rank)
        groups = [sorted(island.ranks[j] for island in islands) for j in range(1, self.k)]
        links: list[BridgeTransport | None] = []
        if groups:
            transport = topology.transport
            assert transport is not None
            links = transport.create_groups(groups, timeout=topology.timeout)
            for ranks, link in zip(groups, links, strict=True):
                if (link is not None) != (topology.rank in ranks):
                    raise RuntimeError(
                        f"{transport.__name__}.create_groups() must return a transport exactly "
                        f"on each group's members; rank {topology.rank} got {link!r} for {ranks}")
        self._created = [link for link in links if link is not None]
        if self.index == 0:
            self.link = topology.bridge
        elif self.index < self.k:
            self.link = links[self.index - 1]
        if self.index < self.k:
            self.link_members = tuple(sorted(island.ranks[self.index] for island in islands))
        self._setup_compression(topology, member=self.index < self.k)

    def close(self) -> None:
        for link in self._created:
            link.close()
        self._created, self.link = [], None
        self.compressor = None

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        flat, padded = self._padded(tensor)
        segments = padded.view(self.k, -1)
        island, group = topology.island, topology.island_group
        exact = len(island.ranks) == self.k
        mine: torch.Tensor | None = None
        if exact:
            with phase("island-reduce-scatter"):
                mine = torch.empty_like(segments[0])
                _reduce_scatter(mine, padded, group, self.index, _native(topology), op)
        else:
            with phase("island-reduce"):
                for j in range(self.k):
                    dist.reduce(segments[j], dst=island.ranks[j], op=op, group=group)
            if self.index < self.k:
                mine = segments[self.index]
        if self.link is not None:
            assert mine is not None
            if self._compresses(mine, op):
                assert self.compressor is not None
                # Segment j's own residual: the same segment of the same buffer next time.
                self.compressor.all_reduce(mine, _compression.buffer_key(tensor, self.index),
                                           self.link, self.link_members, rank=topology.rank)
            else:
                with phase("bridge"):
                    staged = _stage_to_host(mine)
                    _bridge_all_reduce(self.link, staged, op)
                    if staged is not mine:
                        mine.copy_(staged)
        self._reassemble(padded, mine, topology)
        if padded is not flat:
            flat.copy_(padded[:flat.numel()])

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        src_island = topology.layout.island_of(src)
        in_src_island = topology.island == src_island
        if in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=src, group=topology.island_group)
        flat, padded = self._padded(tensor)
        segments = padded.view(self.k, -1)
        mine: torch.Tensor | None = None
        if self.index < self.k:
            # all_gather needs its input apart from its output.
            exact = len(topology.island.ranks) == self.k
            mine = (torch.empty_like(segments[0]) if exact and not in_src_island
                    else segments[self.index])
        if self.link is not None:
            assert mine is not None
            with phase("bridge"):
                staged = _stage_to_host(mine)
                self.link.broadcast(staged, src=src_island.ranks[self.index])
                if not in_src_island and staged is not mine:
                    mine.copy_(staged)
        if not in_src_island:
            self._reassemble(padded, mine, topology)
            if padded is not flat:
                flat.copy_(padded[:flat.numel()])

    def _padded(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``tensor`` flattened, and k equal segments holding it: itself if it divides evenly."""
        flat = tensor.view(-1)
        size = -(-flat.numel() // self.k) * self.k
        if size == flat.numel():
            return flat, flat
        padded = torch.zeros(size, dtype=flat.dtype, device=flat.device)
        padded[:flat.numel()].copy_(flat)
        return flat, padded

    def _reassemble(self, padded: torch.Tensor, mine: torch.Tensor | None,
                    topology: Topology) -> None:
        """Step 3: every island rank ends with all k segments, segment j from island rank j."""
        island, group = topology.island, topology.island_group
        if len(island.ranks) == self.k:
            assert mine is not None
            with phase("island-all-gather"):
                _all_gather(padded, mine, group, _native(topology))
            return
        segments = padded.view(self.k, -1)
        with phase("island-broadcast"):
            for j in range(self.k):
                dist.broadcast(segments[j], src=island.ranks[j], group=group)


class AutoTuned(CollectivePolicy):
    """Choose a policy by message size, from thresholds measured on this cluster.

    ``GPUBRIDGE_THRESHOLDS`` names a JSON file written by
    ``scripts/gpu/bench_all_reduce.py --write-thresholds``: rules, each up to a
    size, naming the policy that was fastest there, e.g. ``flat-gloo`` for
    small tensors and the pipelined policy for large ones (idea 2 in
    docs/PRIOR_ART.md). Without the variable every size gets what ``auto``
    would choose, so the default is safe. The rules are part of this policy's
    settings, so discovery refuses ranks that loaded different files.
    """

    name = "auto-tuned"
    reduce_ops = frozenset({"sum", "max", "min"})

    def __init__(self) -> None:
        self.rules: list[tuple[int | None, CollectivePolicy]] = []

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return True

    @classmethod
    def settings(cls) -> dict[str, Any]:
        path = os.environ.get(THRESHOLDS_ENV, "").strip()
        chunk = chunk_bytes_setting(PipelinedReduceBridgeBroadcast.DEFAULT_CHUNK_BYTES)
        if not path:
            return {"rules": [{"max_bytes": None, "policy": AUTO}], "chunk_bytes": chunk}
        return load_thresholds(path, chunk)

    def setup(self, topology: Topology) -> None:
        settings = topology.policy_settings
        for rule in settings["rules"]:
            try:
                cls = choose_policy(resolve_policy(rule["policy"]), topology.layout)
            except RuntimeError as exc:
                raise RuntimeError(f"{THRESHOLDS_ENV}: the rule for sizes up to "
                                   f"{rule['max_bytes']} bytes can't be used here. {exc}") from None
            policy = (cls(chunk_bytes=settings["chunk_bytes"])
                      if issubclass(cls, PipelinedReduceBridgeBroadcast) else cls())
            policy.setup(topology)
            self.rules.append((rule["max_bytes"], policy))

    def close(self) -> None:
        for _, policy in self.rules:
            policy.close()

    def select_policy(self, nbytes: int) -> CollectivePolicy:
        for max_bytes, policy in self.rules:
            if max_bytes is None or nbytes <= max_bytes:
                return policy.select_policy(nbytes)
        raise AssertionError("the last thresholds rule covers every size")

    def supported_reduce_ops(self, topology: Topology) -> frozenset[str]:
        ops = self.reduce_ops
        for _, policy in self.rules:
            ops &= policy.supported_reduce_ops(topology)
        return ops

    # collectives.py calls select_policy() and then the chosen policy directly;
    # these delegate the same way for anyone calling the policy itself.
    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        chosen = self.select_policy(tensor.nbytes)
        if op == ReduceOp.SUM:
            chosen.all_reduce(tensor, topology)
        else:
            chosen.all_reduce(tensor, topology, op=op)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        self.select_policy(tensor.nbytes).broadcast(tensor, src, topology)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        self.select_policy(output.nbytes).all_gather_into_tensor(output, input, topology)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        self.select_policy(input.nbytes).reduce_scatter_tensor(output, input, topology)


THRESHOLDS_FORMAT = 1


def load_thresholds(path: str, chunk_bytes: int) -> dict[str, Any]:
    """Read and check a thresholds file; return the auto-tuned policy's settings.

    Raises:
        ValueError: naming ``GPUBRIDGE_THRESHOLDS`` and what is wrong, so
            ``init()`` fails on every rank with one clear message.
    """
    where = f"{THRESHOLDS_ENV}={path!r}"
    try:
        with open(path) as fh:
            data = json.load(fh)
    except OSError as exc:
        raise ValueError(f"{where}: can't read it ({exc.strerror or exc})") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: not valid JSON ({exc})") from None
    if not isinstance(data, dict) or data.get("gpubridge_thresholds") != THRESHOLDS_FORMAT:
        raise ValueError(f"{where}: not a gpubridge thresholds file (format "
                         f"{THRESHOLDS_FORMAT}); write one with bench_all_reduce.py "
                         "--write-thresholds")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{where}: needs a non-empty list of rules")
    checked, previous = [], -1
    for i, rule in enumerate(rules):
        name, max_bytes = (rule.get("policy"), rule.get("max_bytes")) if isinstance(
            rule, dict) else (None, None)
        last = i == len(rules) - 1
        if name == AutoTuned.name or (name != AUTO and name not in _POLICIES):
            known = ", ".join([AUTO, *sorted(n for n in _POLICIES if n != AutoTuned.name)])
            raise ValueError(f"{where}: rule {i} names policy {name!r}; known: {known}")
        if last and max_bytes is not None:
            raise ValueError(f"{where}: the last rule must cover every size (max_bytes null)")
        if not last and (not isinstance(max_bytes, int) or max_bytes <= previous):
            raise ValueError(f"{where}: rule {i} needs an integer max_bytes larger than the "
                             "previous rule's")
        previous = max_bytes if max_bytes is not None else previous
        checked.append({"max_bytes": max_bytes, "policy": name})
    chunk = data.get("chunk_bytes", chunk_bytes)
    if not isinstance(chunk, int) or chunk < 8 or chunk % 8:
        raise ValueError(f"{where}: chunk_bytes must be a positive multiple of 8")
    return {"rules": checked, "chunk_bytes": chunk}


DEFAULT_POLICY = AUTO

_POLICIES: dict[str, type[CollectivePolicy]] = {
    cls.name: cls for cls in (ReduceBridgeBroadcast, NativeOnly, FlatGloo,
                              PipelinedReduceBridgeBroadcast, ShardedBridge, AutoTuned)
}


def register_policy(policy: type[CollectivePolicy]) -> type[CollectivePolicy]:
    """Make ``policy`` selectable by name in ``gpubridge.init(policy=...)``.

    Usable as a class decorator. Every rank must register the same policies.

    Raises:
        TypeError: if ``policy`` is not a :class:`CollectivePolicy` subclass.
        ValueError: if the name is ``auto`` or another class already uses it.
    """
    if not (isinstance(policy, type) and issubclass(policy, CollectivePolicy)):
        raise TypeError(f"expected a CollectivePolicy subclass, got {policy!r}")
    if policy.name == AUTO:
        raise ValueError(f"{AUTO!r} is reserved for automatic selection")
    existing = _POLICIES.get(policy.name)
    if existing is not None and existing is not policy:
        raise ValueError(f"a different policy is already registered as {policy.name!r}")
    _POLICIES[policy.name] = policy
    return policy


def resolve_policy(policy: str | type[CollectivePolicy]) -> str | type[CollectivePolicy]:
    """Check a requested policy: ``"auto"``, a registered name, or a policy class.

    Returns ``"auto"`` or a policy class.

    Raises:
        ValueError: for an unknown name.
        TypeError: for anything that is neither a name nor a policy class.
    """
    if policy == AUTO:
        return AUTO
    if isinstance(policy, str):
        try:
            return _POLICIES[policy]
        except KeyError:
            known = ", ".join([AUTO, *sorted(_POLICIES)])
            raise ValueError(f"unknown collective policy {policy!r}; known: {known}") from None
    if isinstance(policy, type) and issubclass(policy, CollectivePolicy):
        return policy
    raise TypeError(f"policy must be a policy name or CollectivePolicy subclass, got {policy!r}")


def implements(policy: CollectivePolicy | type[CollectivePolicy], fn: str) -> bool:
    """Whether ``policy`` overrides the optional collective ``fn``.

    ``fn`` is a method name, e.g. ``"reduce_scatter_tensor"``.
    """
    cls = policy if isinstance(policy, type) else type(policy)
    return getattr(cls, fn) is not getattr(CollectivePolicy, fn)


def policies_implementing(fn: str, layout: Layout) -> list[str]:
    """Registered policies that implement ``fn`` and apply to ``layout``, by name."""
    return sorted(name for name, cls in _POLICIES.items()
                  if cls.applies_to(layout) and implements(cls, fn))


def policy_name(policy: str | type[CollectivePolicy]) -> str:
    """The name a rank reports during discovery for a resolved policy."""
    return policy if isinstance(policy, str) else policy.name


def choose_policy(policy: str | type[CollectivePolicy], layout: Layout) -> type[CollectivePolicy]:
    """Turn a resolved policy into the class to run on ``layout``.

    ``auto`` picks ``native-only`` for one island and ``reduce-bridge-broadcast``
    otherwise. Every rank sees the same layout, so every rank chooses the same.

    Raises:
        RuntimeError: if the policy doesn't apply to this layout.
    """
    if policy == AUTO:
        return ReduceBridgeBroadcast if layout.needs_bridge else NativeOnly
    assert not isinstance(policy, str)
    if not policy.applies_to(layout):
        islands = len(layout.islands)
        usable = ", ".join([AUTO, *sorted(n for n, c in _POLICIES.items() if c.applies_to(layout))])
        raise RuntimeError(
            f"Collective policy {policy.name!r} doesn't apply to this cluster "
            f"({islands} island{'s' if islands != 1 else ''}). Policies that do: {usable}."
        )
    return policy
