"""Discovery, island and bridge groups, and leader election."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import cached_property
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from gpubridge import rendezvous as _rendezvous
from gpubridge.config import (
    ALLOW_DEBUG_GLOO_ENV,
    BRIDGE_COMPRESSION_ENV,
    CPU_BACKEND,
    CPU_ONLY_ENV,
    GPU_BACKEND,
    PINNED_STAGING_ENV,
    SOLO_BACKEND,
    SPLIT_TEST_ENV,
    VENDOR_ENV,
    Config,
    allow_debug_gloo_setting,
    bridge_compression_setting,
    gpu_identity,
    hostname,
    pinned_staging_setting,
    resolve_config,
    split_test_mode,
)
from gpubridge.detect import Probe, detect_vendor, probe
from gpubridge.policies import AUTO, CollectivePolicy, choose_policy, policy_name
from gpubridge.split_test import SplitTestWarning, banner, split_labels
from gpubridge.transport import AUTO as AUTO_BRIDGE
from gpubridge.transport import (
    TCP,
    VERBS,
    BridgeTransport,
    GlooTransport,
    asks_for_gloo,
    choose_bridge,
    resolve_transport,
    transport_name,
)
from gpubridge.transport import _scope as _transport_scope

log = logging.getLogger("gpubridge")


@dataclass(frozen=True)
class Island:
    """All ranks of one vendor. They share a process group on the native backend."""

    vendor: str
    ranks: tuple[int, ...]
    """Global ranks in ascending order."""

    @property
    def leader(self) -> int:
        """The island's member of the bridge: its lowest global rank."""
        return self.ranks[0]


@dataclass(frozen=True)
class Layout:
    """How the cluster splits into islands. Depends only on each rank's vendor."""

    vendors: tuple[str, ...]
    """Vendor of each member, in member order (of each global rank, for the world)."""
    islands: tuple[Island, ...]
    """Ordered by leader rank."""
    ranks: tuple[int, ...] = ()
    """The members' global ranks, ascending: a subgroup's (torch orders a group's ranks so).
    Empty for the world, whose members are ``range(len(vendors))``."""

    @property
    def world_size(self) -> int:
        """How many members: the rows of an all_gather, the slices of a reduce_scatter."""
        return len(self.vendors)

    @property
    def members(self) -> tuple[int, ...]:
        return self.ranks or tuple(range(len(self.vendors)))

    def index(self, rank: int) -> int:
        """A member's position (its group rank): its row in an all_gather."""
        return self.ranks.index(rank) if self.ranks else rank

    @property
    def bridge_ranks(self) -> tuple[int, ...]:
        """Global ranks of the island leaders, in ascending order."""
        return tuple(island.leader for island in self.islands)

    @property
    def needs_bridge(self) -> bool:
        """False for a single-vendor cluster, where collectives skip the bridge."""
        return len(self.islands) > 1

    def island_of(self, rank: int) -> Island:
        """Return the island containing global rank ``rank``."""
        vendor = self.vendors[self.index(rank)]
        return next(island for island in self.islands if island.vendor == vendor)


def plan_topology(vendors: Sequence[str], ranks: Sequence[int] = ()) -> Layout:
    """Group ranks into islands by vendor and elect each island's leader.

    This is a pure function of the vendor map, so every rank that sees the
    same map computes the same islands, leaders and bridge.

    Args:
        vendors: The vendor of each global rank, indexed by rank; for a subgroup,
            of each member in ``ranks``' order.
        ranks: A subgroup's members (global ranks, ascending). Empty: the world.

    Raises:
        ValueError: if ``vendors`` is empty.
    """
    if not vendors:
        raise ValueError("cannot plan a topology for zero ranks")
    if ranks and (len(ranks) != len(vendors) or list(ranks) != sorted(set(ranks))):
        raise ValueError(f"a subgroup's ranks must be distinct and ascending, got {list(ranks)}")
    members = tuple(ranks) or tuple(range(len(vendors)))
    ranks_by_vendor: dict[str, list[int]] = {}
    for rank, vendor in zip(members, vendors, strict=True):
        ranks_by_vendor.setdefault(vendor, []).append(rank)
    islands = sorted(
        (Island(vendor, tuple(rs)) for vendor, rs in ranks_by_vendor.items()),
        key=lambda island: island.leader,
    )
    return Layout(vendors=tuple(vendors), islands=tuple(islands), ranks=tuple(ranks))


@dataclass(frozen=True)
class PeerInfo:
    """What each rank reports about itself during discovery."""

    vendor: str | None
    """None if this rank could not work out its vendor; see :attr:`problems`."""
    simulated: bool
    hostname: str
    bridge: str
    """Name of the bridge transport this rank was asked to use: ``auto`` unless the job
    named one. What the job runs on is :attr:`Topology.bridge_name`."""
    probe: Probe
    """The rank's build, GPUs and backends, from :func:`gpubridge.probe`."""
    problems: tuple[str, ...] = ()
    """Why this rank cannot take part. A problem on any rank fails init() on every rank."""
    split_test: str = ""
    """This rank's ``GPUBRIDGE_SPLIT_TEST`` mode, or empty when off. TEST ONLY."""
    policy: str = AUTO
    """Name of the collective policy this rank was asked to use (``auto`` by default)."""
    policy_settings: dict[str, Any] = field(default_factory=dict)
    """That policy's :meth:`~gpubridge.CollectivePolicy.settings`, e.g. a chunk size."""
    gpu: str | None = None
    """The physical GPU this rank drives (UUID, PCI address); empty if unknown, None on CPU."""
    pinned_staging: bool = False
    """This rank's ``GPUBRIDGE_PINNED_STAGING``: stage the default bridge in pinned memory."""
    bridge_compression: str = ""
    """This rank's ``GPUBRIDGE_BRIDGE_COMPRESSION`` (``bf16`` or ``fp8``), or empty when off."""
    debug_detail: bool = False
    """Whether torch's distributed debug level was DETAIL on this rank when it described
    itself (:func:`debug_detail`): torch then wraps every native group it creates in a
    checker that runs on a Gloo helper group."""
    allow_debug_gloo: bool = False
    """This rank's ``allow_debug_gloo`` (``GPUBRIDGE_ALLOW_DEBUG_GLOO``): the job accepts
    those Gloo helper groups instead of being refused (:func:`debug_gloo_problem`)."""

    @property
    def torch_version(self) -> str:
        return self.probe.torch_version

    def to_wire(self) -> dict[str, Any]:
        """Plain builtins, for the discovery record every rank publishes as JSON."""
        return asdict(self)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> PeerInfo:
        data = dict(data)
        probe_data = dict(data.pop("probe"))
        probe_data["errors"] = tuple(probe_data.get("errors", ()))
        data["problems"] = tuple(data.get("problems", ()))
        return cls(probe=Probe(**probe_data), **data)


def debug_detail() -> bool:
    """Whether torch's distributed debug level is DETAIL right now, however it got there
    (``TORCH_DISTRIBUTED_DEBUG=DETAIL`` when torch was imported, or
    ``torch.distributed.set_debug_level``)."""
    return dist.get_debug_level() == dist.DebugLevel.DETAIL


def describe_local(
    bridge: str, policy: str | type[CollectivePolicy] = AUTO,
    allow_debug_gloo: bool | None = None,
) -> tuple[PeerInfo, Config]:
    """Work out this rank's vendor, configuration and capabilities.

    Anything that would stop this rank from joining (no detectable vendor, no
    GPU, no NCCL/RCCL backend, a bad policy setting) is recorded in
    :attr:`PeerInfo.problems` rather than raised. Discovery then reports every
    rank's problems at once, on every rank, instead of one rank exiting and the
    others hanging until a timeout.

    Raises:
        ValueError: for invalid ``GPUBRIDGE_VENDOR``, ``GPUBRIDGE_CPU_ONLY`` or
            ``GPUBRIDGE_SPLIT_TEST`` values, which are configuration mistakes rather
            than properties of the machine.
    """
    split = split_test_mode() or ""
    local_probe = probe()
    problems: list[str] = []
    try:
        pinned = pinned_staging_setting()
    except ValueError as exc:
        pinned = False
        problems.append(str(exc))
    try:
        compression = bridge_compression_setting()
    except ValueError as exc:
        compression = ""
        problems.append(str(exc))
    if allow_debug_gloo is None:  # not passed to init(): the environment's
        try:
            allow_debug_gloo = allow_debug_gloo_setting()
        except ValueError as exc:
            allow_debug_gloo = False
            problems.append(str(exc))
    vendor: str | None
    try:
        vendor = detect_vendor()
    except RuntimeError as exc:
        vendor = None
        problems.append(str(exc))
    try:
        config = resolve_config()
    except RuntimeError as exc:
        problems.append(str(exc))
        # Never used for communication: init() fails during discovery.
        config = Config(simulated=False, island_backend=GPU_BACKEND, device=torch.device("cpu"))
    if not config.simulated and not local_probe.nccl_available:
        problems.append(
            "This PyTorch build has no NCCL backend (RCCL on ROCm builds), so it cannot "
            "form a GPU island."
        )
    settings: dict[str, Any] = {}
    if not isinstance(policy, str):
        try:
            settings = policy.settings()
        except ValueError as exc:
            problems.append(f"Collective policy {policy.name!r}: {exc}")
    local = PeerInfo(
        vendor=vendor,
        simulated=config.simulated,
        hostname=hostname(config.simulated),
        bridge=bridge,
        probe=local_probe,
        problems=tuple(problems),
        split_test=split,
        policy=policy_name(policy),
        policy_settings=settings,
        gpu=gpu_identity(config.device),
        pinned_staging=pinned,
        bridge_compression=compression,
        debug_detail=debug_detail(),
        allow_debug_gloo=bool(allow_debug_gloo),
    )
    return local, config


def validate_peers(peers: Sequence[PeerInfo], *, warn: bool = True) -> None:
    """Check that every rank can join and that all ranks agree on how to run.

    Warns if PyTorch releases differ.

    Raises:
        RuntimeError: if any rank reported problems, if some ranks run on CPU and
            others on GPUs, or if ranks picked different bridge transports,
            collective policies, policy settings, split-test modes, pinned
            staging or bridge compression.
    """
    failing = [(rank, peer) for rank, peer in enumerate(peers) if peer.problems]
    if failing:
        lines = [
            f"  rank {rank} ({peer.hostname}): {problem}"
            for rank, peer in failing
            for problem in peer.problems
        ]
        raise RuntimeError(
            f"gpubridge cannot start: {len(failing)} of {len(peers)} ranks reported "
            "problems.\n" + "\n".join(lines)
        )
    on_cpu = [rank for rank, peer in enumerate(peers) if peer.simulated]
    if on_cpu and len(on_cpu) != len(peers):
        on_gpu = [rank for rank, peer in enumerate(peers) if not peer.simulated]
        raise RuntimeError(
            f"Ranks disagree on where to run: ranks {on_cpu} run on CPU ({VENDOR_ENV} or "
            f"{CPU_ONLY_ENV} is set) but ranks {on_gpu} run on GPUs. Use the same mode "
            "on every rank."
        )
    bridges: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        bridges.setdefault(peer.bridge, []).append(rank)
    if len(bridges) > 1:
        detail = "; ".join(f"{name!r} on ranks {ranks}" for name, ranks in bridges.items())
        raise RuntimeError(
            f"Ranks picked different bridge transports: {detail}. Pass the same bridge "
            "to gpubridge.init() on every rank."
        )
    policies: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        policies.setdefault(peer.policy, []).append(rank)
    if len(policies) > 1:
        detail = "; ".join(f"{name!r} on ranks {ranks}" for name, ranks in policies.items())
        raise RuntimeError(
            f"Ranks picked different collective policies: {detail}. Pass the same policy "
            "to gpubridge.init() on every rank."
        )
    settings: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        settings.setdefault(_canonical(peer.policy_settings), []).append(rank)
    if len(settings) > 1:
        detail = "; ".join(f"{text} on ranks {ranks}" for text, ranks in settings.items())
        raise RuntimeError(
            f"Ranks disagree on the settings of collective policy {peers[0].policy!r}: "
            f"{detail}. Set the same GPUBRIDGE_* variables (and thresholds file) on every "
            "rank: settings such as a chunk size decide how many messages a collective sends."
        )
    splits: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        splits.setdefault(peer.split_test or "off", []).append(rank)
    if len(splits) > 1:
        detail = "; ".join(f"{mode} on ranks {ranks}" for mode, ranks in splits.items())
        raise RuntimeError(
            f"Ranks disagree on {SPLIT_TEST_ENV}: {detail}. Set it to the same value on "
            "every rank, or on none."
        )
    pinned = [rank for rank, peer in enumerate(peers) if peer.pinned_staging]
    if pinned and len(pinned) != len(peers):
        off = [rank for rank, peer in enumerate(peers) if not peer.pinned_staging]
        raise RuntimeError(
            f"Ranks disagree on {PINNED_STAGING_ENV}: on for ranks {pinned}, off for ranks "
            f"{off}. Set it to the same value on every rank, so a run is one "
            "configuration or the other."
        )
    compression: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        compression.setdefault(peer.bridge_compression or "off", []).append(rank)
    if len(compression) > 1:
        detail = "; ".join(f"{mode} on ranks {ranks}" for mode, ranks in compression.items())
        raise RuntimeError(
            f"Ranks disagree on {BRIDGE_COMPRESSION_ENV}: {detail}. Set it to the same value on "
            "every rank, or on none: it decides what the bridge sends."
        )
    allowed = [rank for rank, peer in enumerate(peers) if peer.allow_debug_gloo]
    if allowed and len(allowed) != len(peers):
        off = [rank for rank, peer in enumerate(peers) if not peer.allow_debug_gloo]
        raise RuntimeError(
            f"Ranks disagree on {ALLOW_DEBUG_GLOO_ENV} (allow_debug_gloo): on for ranks "
            f"{allowed}, off for ranks {off}. Set it to the same value on every rank: it "
            "decides whether the job may have torch's Gloo helper groups."
        )
    # CUDA and ROCm builds of one release differ only in the local version
    # suffix, e.g. 2.5.1+cu124 and 2.5.1+rocm6.2.
    releases = sorted({peer.torch_version.split("+")[0] for peer in peers}, key=_release_key)
    if warn and len(releases) > 1:
        warnings.warn(
            f"Ranks run different PyTorch releases ({', '.join(releases)}). Mixed releases "
            "passed gpubridge's CPU-only tests (2.9.1 to 2.14.1) but are untested on GPUs; "
            "prefer the same release on every rank.",
            stacklevel=2,
        )


def _canonical(settings: dict[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True)


def _release_key(release: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that orders 2.9.1 before 2.14.1."""
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in release.split("."))


@dataclass(frozen=True)
class Topology:
    """This rank's view of the cluster, built by :func:`gpubridge.init`."""

    rank: int
    layout: Layout
    peers: tuple[PeerInfo, ...]
    """What every rank reported during discovery, indexed by global rank."""
    config: Config
    island_group: ProcessGroup
    """This rank's island, on the native backend (Gloo when running on CPU)."""
    bridge: BridgeTransport | None
    """The transport linking island leaders. None on non-leaders and in single-vendor clusters."""
    policy: CollectivePolicy
    """The collective policy every rank runs (``auto`` already resolved)."""
    transport: type[BridgeTransport] | None = None
    """The bridge transport class the job runs on, on every rank (leaders or not):
    what the job named, or what ``bridge="auto"`` resolved to at ``init()``."""
    selection: dict[str, Any] | None = None
    """How ``init()`` came to :attr:`transport`, the same on every rank:
    ``{"requested", "chosen", "reason"}``, and under ``auto`` ``"leaders"``, each island
    leader's report of whether the verbs bridge could run there
    (:func:`gpubridge.verbs.availability`). Fixed for the job's life."""
    timeout: timedelta | None = None
    """The timeout ``init()`` gave its groups, for groups a policy creates in ``setup()``."""
    world_group: ProcessGroup | None = None
    """A Gloo group of every member, where the job created one at setup: under the
    torch.distributed backend with ``bridge="gloo"`` (for :func:`gpubridge.barrier`,
    ``flat-gloo`` and point-to-point). None otherwise: on any other bridge no Gloo group
    of every rank exists (``barrier()`` then counts through the c10d store, and
    ``flat-gloo`` makes its own group when it is first used: :func:`gloo_world`); and
    after :func:`gpubridge.init` with ``bridge="gloo"``, whose default group is the Gloo
    one, looked up when used rather than held, so it goes when the job destroys it."""
    scope: str = "world0"
    """Names this topology in the c10d store, the same on every member and unique in the
    job: ``world<n>`` for the n-th setup in this process, a subgroup's own name."""
    state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """What this topology makes after setup: its barrier count, a Gloo group that
    ``flat-gloo`` asked for. Not part of its identity."""

    @property
    def policy_name(self) -> str:
        """Name of the active collective policy, e.g. ``"reduce-bridge-broadcast"``."""
        return self.policy.name

    @property
    def policy_settings(self) -> dict[str, Any]:
        """The policy's settings, identical on every rank (discovery checked)."""
        return self.peers[self.rank].policy_settings

    @property
    def pinned_staging(self) -> bool:
        """Whether ``GPUBRIDGE_PINNED_STAGING`` is on: the same on every rank (discovery checks)."""
        return self.peers[self.rank].pinned_staging

    @property
    def bridge_compression(self) -> str:
        """``GPUBRIDGE_BRIDGE_COMPRESSION`` (``bf16``, ``fp8``, or empty when off): the same on
        every rank (discovery checks)."""
        return self.peers[self.rank].bridge_compression

    @property
    def bridge_group(self) -> ProcessGroup | None:
        """The bridge's process group, if its transport has one (``bridge="gloo"`` does)."""
        return getattr(self.bridge, "group", None)

    @property
    def bridge_name(self) -> str:
        """Name of the bridge transport the job runs on (``"tcp"``, ``"verbs"`` ...), on
        every rank, also where no bridge is created (one island)."""
        return cast(type[BridgeTransport], self.transport).name

    @property
    def gloo_free(self) -> bool:
        """Whether this topology was set up without asking for Gloo: any bridge transport
        that isn't Gloo's. Its islands are then native groups, or in CPU simulation a
        :class:`~gpubridge.torch_backend.SoloGroup` for a one-rank island (a multi-rank
        island there is still a Gloo group: simulation is a test device). It says what
        the job asked for, not that the process holds no Gloo group: see
        :attr:`debug_gloo`, and the ``flat-gloo`` policy makes its own."""
        return self.transport is None or not issubclass(self.transport, GlooTransport)

    @property
    def debug_gloo(self) -> bool:
        """Whether the job started at torch's debug level DETAIL with
        ``allow_debug_gloo``: torch then gave each native island group a Gloo helper
        group, on a job that otherwise has none (:func:`debug_gloo_problem`)."""
        local = self.peers[self.rank]
        return self.gloo_free and local.debug_detail and local.allow_debug_gloo

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def vendor(self) -> str:
        """This rank's island label. Differs from :attr:`detected_vendor` only in a split test."""
        return self.layout.vendors[self.layout.index(self.rank)]

    @property
    def detected_vendor(self) -> str:
        """The vendor this rank detected (or was told by ``GPUBRIDGE_VENDOR``)."""
        return cast(str, self.peers[self.rank].vendor)

    @property
    def split_test(self) -> str | None:
        """The ``GPUBRIDGE_SPLIT_TEST`` mode of this run, or None. TEST ONLY."""
        return self.peers[self.rank].split_test or None

    @property
    def run_kind(self) -> str:
        """What kind of run this is, for results and logs.

        ``"split-test"`` (islands faked from one vendor), ``"cpu"`` (simulation or
        CPU-only mode), ``"gpu-mixed"`` (real GPUs of both vendors) or
        ``"gpu-single-vendor"``.
        """
        if self.split_test:
            return "split-test"
        if self.config.simulated:
            return "cpu"
        return "gpu-mixed" if self.layout.needs_bridge else "gpu-single-vendor"

    @cached_property
    def island(self) -> Island:
        """This rank's island. Looked up once: the layout and the rank never change."""
        return self.layout.island_of(self.rank)

    @cached_property
    def is_leader(self) -> bool:
        return self.island.leader == self.rank

    @property
    def simulated(self) -> bool:
        return self.config.simulated

    @property
    def device(self) -> torch.device:
        """Where this rank's tensors must live: its GPU, or CPU in simulation and CPU-only mode."""
        return self.config.device


def gpu_clashes(
    layout: Layout, hostnames: Sequence[str], gpus: Sequence[str | None]
) -> list[str]:
    """Ranks of one island that would drive the same GPU, one line per shared GPU.

    An island is one NCCL/RCCL communicator, and both refuse two of its ranks
    on one GPU. Ranks in different islands may share a GPU (split-test mode
    does that on purpose). CPU ranks (None) are not checked, and neither is an
    island with a rank whose GPU is unknown (``""``); see :func:`unchecked_islands`.
    """
    lines = []
    for island in layout.islands:
        if any(gpus[rank] == "" for rank in island.ranks):
            continue
        by_gpu: dict[tuple[str, str], list[int]] = {}
        for rank in island.ranks:
            gpu = gpus[rank]
            if gpu is not None:
                by_gpu.setdefault((hostnames[rank], gpu), []).append(rank)
        lines += [f"  island {island.vendor!r}: ranks {ranks} on {host}, GPU {gpu}"
                  for (host, gpu), ranks in by_gpu.items() if len(ranks) > 1]
    return lines


def unchecked_islands(layout: Layout, gpus: Sequence[str | None]) -> list[str]:
    """Islands that :func:`gpu_clashes` skips, one line each with the ranks whose GPU is unknown."""
    lines = []
    for island in layout.islands:
        unknown = [rank for rank in island.ranks if gpus[rank] == ""]
        if unknown:
            lines.append(f"  island {island.vendor!r}: ranks {unknown}")
    return lines


def _unchecked_warning(lines: Sequence[str]) -> str:
    return ("gpubridge could not check that ranks of these islands drive different GPUs: "
            "the ranks named report neither a GPU UUID nor a PCI address (some ROCm builds "
            "of PyTorch don't).\n" + "\n".join(lines) + "\nIf two ranks of one island "
            "share a GPU, NCCL or RCCL will refuse it when the island first communicates; "
            "give each rank its own GPU.")


def _gpu_clash_error(lines: Sequence[str], split: str) -> str:
    if split:
        hint = (f"In split-test mode on one node, {SPLIT_TEST_ENV}=half with at most twice as "
                "many ranks as GPUs gives every rank of an island its own GPU; otherwise "
                "start no more ranks per node than it has GPUs.")
    else:
        hint = ("Start no more ranks per node than it has GPUs, and check that each rank "
                "gets its own LOCAL_RANK and sees the GPUs you expect (CUDA_VISIBLE_DEVICES, "
                "HIP_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES).")
    return ("gpubridge cannot start: ranks of one island would share a GPU, and NCCL and "
            "RCCL refuse two ranks of one communicator on the same GPU.\n"
            + "\n".join(lines) + "\n" + hint)


def _debug_refusal(where: str) -> str:
    return (
        f"gpubridge cannot start: torch's distributed debug level is DETAIL {where} "
        "(TORCH_DISTRIBUTED_DEBUG=DETAIL, or torch.distributed.set_debug_level). At that "
        "level torch wraps every NCCL, RCCL or Gloo process group it creates in a checker "
        "that runs on a Gloo helper group of the same ranks. This job didn't ask for Gloo, "
        "and its island groups would each bring one: Gloo groups and their worker threads "
        "(pt_gloo_runloop), which gpubridge no longer creates unless asked, because such a "
        "thread can abort a rank at interpreter exit (Python 3.12 with torch 2.10 to "
        "2.15). Either lower the debug level on every rank (TORCH_DISTRIBUTED_DEBUG=INFO "
        "or OFF), or accept the Gloo helper groups and that exposure for this job: pass "
        f"allow_debug_gloo=True to gpubridge.init(), or set {ALLOW_DEBUG_GLOO_ENV}=1, on "
        "every rank. gpubridge doesn't change the debug level itself."
    )


def debug_gloo_problem(peers: Sequence[PeerInfo], layout: Layout, config: Config,
                       transport: str | type[BridgeTransport]) -> str | None:
    """Why this job may not create its island groups at torch's debug level DETAIL, or
    None if it may.

    At DETAIL torch's ``new_group`` wraps a group of a C++ backend (NCCL, RCCL, Gloo) in
    ``_ProcessGroupWrapper``, whose helper is a ``ProcessGroupGloo`` of the same ranks
    (torch 2.8.0 and 2.15.0, ``_new_process_group_helper``): a job that named no Gloo
    would have Gloo groups and threads. A group defined in Python (gpubridge's own, a
    one-rank simulated island) is returned before that branch and is never wrapped.

    Every rank computes this from the same ``peers`` and ``layout``, so every rank
    refuses alike, before any island group exists. A job on ``bridge="gloo"`` is not
    looked at (it has Gloo groups by request), nor is one whose islands are all Python
    groups. A debug level raised after ``init()`` is not seen here.
    """
    detail = [rank for rank, peer in enumerate(peers) if peer.debug_detail]
    if not detail or asks_for_gloo(transport):
        return None
    if all(island_backend(config, island, transport) == SOLO_BACKEND
           for island in layout.islands):
        return None
    if not peers[0].allow_debug_gloo:  # the same on every rank (validate_peers)
        return _debug_refusal(f"on ranks {detail}")
    if len(detail) != len(peers):
        off = [rank for rank, peer in enumerate(peers) if not peer.debug_detail]
        return (
            f"Ranks disagree on torch's distributed debug level: DETAIL on ranks {detail}, "
            f"not on ranks {off}. At DETAIL torch gives each island group a Gloo helper "
            "group that every member must create; set TORCH_DISTRIBUTED_DEBUG to the same "
            "value on every rank."
        )
    return None


def failure_text(exc: BaseException) -> str:
    """What a rank that failed before it could describe itself publishes as its
    discovery record's error: the message of a bad name or setting as it is, anything
    else (a failed import, say) with its type."""
    if isinstance(exc, ValueError | TypeError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def discover(store, scope: str, rank: int, world_size: int, local: PeerInfo | None,
             timeout: timedelta | None, *, problem: str | None = None) -> tuple[PeerInfo, ...]:
    """Exchange every rank's :class:`PeerInfo` through the c10d store.

    Collective over the job: every rank publishes its record under ``scope`` and reads
    all of them, so no process group has to exist yet. A rank that could not even
    describe itself passes ``problem`` instead of ``local``: every rank then raises
    with it, instead of that rank exiting alone and the others waiting.

    A rank that is about to raise because of the records (here, or after
    :func:`validate_peers` or :func:`check_plan`) first makes sure the others can read
    them too: :func:`hold_for_readers`.

    Raises:
        RuntimeError: if any rank published a problem, or a record of another version.
        TimeoutError: if a rank's record doesn't arrive in ``timeout`` (30 minutes if None).
    """
    deadline = time.monotonic() + (timeout or _rendezvous.DEFAULT_TIMEOUT).total_seconds()
    record: dict[str, Any] = {"v": _rendezvous.RECORD_VERSION}
    if problem is None:
        assert local is not None
        record["peer"] = local.to_wire()
    else:
        record["error"] = problem
    records = _rendezvous.exchange(store, f"gpubridge/{scope}/peer", rank, world_size, record,
                                   deadline, "every rank's discovery record")
    failing = [(r, rec.get("error") or f"its gpubridge writes discovery record version "
                f"{rec.get('v')!r}, this one version {_rendezvous.RECORD_VERSION}")
               for r, rec in enumerate(records)
               if rec.get("error") or rec.get("v") != _rendezvous.RECORD_VERSION]
    if failing:
        hold_for_readers(store, scope, rank, world_size, timeout)
        raise RuntimeError(
            f"gpubridge cannot start: {len(failing)} of {world_size} ranks reported "
            "problems.\n" + "\n".join(f"  rank {r}: {p}" for r, p in failing))
    return tuple(PeerInfo.from_wire(rec["peer"]) for rec in records)


def hold_for_readers(store, scope: str, rank: int, world_size: int,
                     timeout: timedelta | None) -> None:
    """Before this rank raises because of what discovery found: if it may host the
    job's store, stay until every other rank has read the discovery records, so each
    of them raises the same reason and none a store error
    (:func:`gpubridge.rendezvous.await_readers`).

    Only rank 0 waits: torch's rendezvous starts a TCP store's server there, and
    nowhere else. Where a launcher's agent or a file holds the store the wait is over
    as soon as the others have read, which they do at once. Called on the failing
    path only: a job that starts never waits here. Never raises.
    """
    if rank != 0:
        return
    deadline = time.monotonic() + (timeout or _rendezvous.DEFAULT_TIMEOUT).total_seconds()
    unread = _rendezvous.await_readers(store, f"gpubridge/{scope}/peer", rank, world_size,
                                       deadline)
    if unread:
        log.warning("gpubridge: ranks %s published a discovery record but had not read the "
                    "others' when the timeout passed; they may not learn why the job "
                    "failed", unread)


def check_plan(peers: Sequence[PeerInfo], config: Config,
               policy: str | type[CollectivePolicy],
               transport: str | type[BridgeTransport], *, rank: int | None = None
               ) -> tuple[Layout, type[CollectivePolicy]]:
    """The layout and the policy class the job would run with, or the reason it may not
    start: every refusal that follows from the ranks' reports alone.

    A pure function of ``peers`` (which :func:`validate_peers` has accepted) and of
    settings that are the same on every rank, so every rank gets the same answer. It
    creates nothing and talks to no store: ``init()`` and the backend's setup call it
    before any group exists, and :func:`build_topology` again when it builds. With
    ``rank`` it also gives that rank's warnings (the split-test banner, islands whose
    GPUs go unchecked on rank 0); without, it is silent.

    Raises:
        RuntimeError: if two ranks of an island would share a GPU, if torch's debug
            level is DETAIL on a job that neither asked for Gloo nor accepted its
            helper groups (:func:`debug_gloo_problem`), or if the policy doesn't apply
            to the layout.
    """
    split = peers[0].split_test  # the same on every rank (validate_peers)
    # validate_peers has ruled out ranks without a vendor.
    vendors = [cast(str, peer.vendor) for peer in peers]
    labels = vendors
    if split:
        labels = split_labels(vendors, [peer.hostname for peer in peers], split)
        if rank is not None:
            warnings.warn(banner(split, vendors, labels), SplitTestWarning, stacklevel=4)
    layout = plan_topology(labels)
    gpus = [peer.gpu for peer in peers]
    unchecked = unchecked_islands(layout, gpus)
    if unchecked and rank == 0:
        warnings.warn(_unchecked_warning(unchecked), stacklevel=4)
    clashes = gpu_clashes(layout, [peer.hostname for peer in peers], gpus)
    if clashes:
        raise RuntimeError(_gpu_clash_error(clashes, split))
    refused = debug_gloo_problem(peers, layout, config, transport)
    if refused:
        raise RuntimeError(refused)
    return layout, choose_policy(policy, layout)


def select_bridge(requested: str | type[BridgeTransport], layout: Layout, rank: int,
                  scope: str, timeout: timedelta | None
                  ) -> tuple[type[BridgeTransport], dict[str, Any]]:
    """The bridge transport the job runs on, and how it was chosen.

    A named bridge is itself. ``"auto"`` is resolved here, once: each island leader
    publishes whether the verbs bridge could run on its RDMA device
    (:func:`gpubridge.verbs.availability`) in the c10d store, every rank reads all of
    the leaders' reports, and :func:`gpubridge.transport.choose_bridge` turns them into
    ``"verbs"`` or ``"tcp"``: the same on every rank, since every rank reads the same
    reports. With one island no bridge is created and nothing is probed.

    Collective over every rank, once the default group exists.

    Raises:
        TimeoutError: if a leader's report doesn't arrive in ``timeout``.
        ValueError: if the chosen bridge's module can't be imported.
    """
    name = transport_name(requested)
    if requested != AUTO_BRIDGE:
        assert not isinstance(requested, str)
        return requested, {"requested": name, "chosen": name, "reason": "asked for by name"}
    selection: dict[str, Any] = {"requested": AUTO_BRIDGE}
    if not layout.needs_bridge:
        chosen, reason = TCP, "one island: no bridge is created, and no device was checked"
    else:
        from gpubridge import verbs  # a small module: no library is loaded by importing it

        leaders = layout.bridge_ranks
        store = _rendezvous.default_store()
        prefix = f"gpubridge/{scope}/verbs"
        if rank in leaders:
            try:
                report = verbs.availability()
            except Exception as exc:  # noqa: BLE001 - a probe must never take init() down
                report = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}
            store.set(f"{prefix}/{rank}", json.dumps(report))
        deadline = time.monotonic() + (timeout or _rendezvous.DEFAULT_TIMEOUT).total_seconds()
        values = _rendezvous.wait_for(
            store, [f"{prefix}/{leader}" for leader in leaders], deadline,
            "every island leader's report on its RDMA device (bridge=\"auto\")")
        reports = {leader: json.loads(value) for leader, value in zip(leaders, values,
                                                                      strict=True)}
        chosen, reason = choose_bridge(leaders, reports)
        selection["leaders"] = {str(leader): report for leader, report in reports.items()}
        surprised = [leader for leader in leaders
                     if reports[leader].get("devices") and not reports[leader].get("ok")]
        if chosen != VERBS and surprised and rank == leaders[0]:
            log.warning("gpubridge: bridge=\"auto\" runs on the TCP bridge although island "
                        "leaders %s have an RDMA device: %s", surprised, reason)
    selection.update(chosen=chosen, reason=reason)
    transport = resolve_transport(chosen)
    assert not isinstance(transport, str)
    return transport, selection


def build_topology(
    peers: Sequence[PeerInfo],
    config: Config,
    *,
    transport: str | type[BridgeTransport],
    policy: str | type[CollectivePolicy] = AUTO,
    timeout: timedelta | None = None,
    world_group: ProcessGroup | None = None,
    scope: str = "world0",
) -> Topology:
    """Create the island groups and the bridge from what every rank reported.

    Collective: every rank calls this, with the same ``peers``, once a default
    process group exists (it is what ``dist.new_group`` and the bridges' store come
    from). Outside ``bridge="gloo"``, inside :func:`setup_names`.

    Args:
        peers: Every rank's :class:`PeerInfo` by global rank, from :func:`discover`,
            which :func:`validate_peers` has accepted.
        config: This rank's resolved configuration.
        transport: The bridge every rank asked for: ``"auto"`` (resolved here,
            :func:`select_bridge`) or a transport class.
        policy: ``"auto"`` or the collective policy class; its name is ``peers``'.
        timeout: Timeout for the new groups' collectives. None keeps PyTorch's default.
        world_group: A Gloo group of every rank the caller made for ``barrier()``,
            ``flat-gloo`` and point-to-point, if it made one (see
            :attr:`Topology.world_group`).
        scope: The topology's name in the c10d store (:attr:`Topology.scope`).

    Raises:
        RuntimeError: if two ranks of an island would share a GPU, or the policy
            doesn't apply to the layout, on every rank alike.
    """
    rank = dist.get_rank()
    peers = tuple(peers)
    if not config.simulated:
        # Only now that every rank's device index has passed discovery. Called
        # earlier, an index with no GPU behind it crashed one rank on its own.
        torch.cuda.set_device(config.device)
    # Every rank sees the same layout and peers, so every rank fails these
    # checks alike (or picks the same policy), before any group is created. init()
    # and the backend's setup have run them once already, before the default group
    # or anything else existed (check_plan): here they give the layout, and warn.
    layout, policy_cls = check_plan(peers, config, policy, transport, rank=rank)
    transport, selection = select_bridge(transport, layout, rank, scope, timeout)
    log.info("gpubridge: rank %d of %d, bridge %r (%s)", rank, len(peers), transport.name,
             selection["reason"])

    # Every rank visits the groups in the same order. Inside setup_names only a
    # group's members create it and synchronize; bridge="gloo" through
    # gpubridge.init() keeps torch's default whole-world synchronization.
    island_group: ProcessGroup | None = None
    for island in layout.islands:
        group = dist.new_group(list(island.ranks), timeout=timeout,
                               backend=island_backend(config, island, transport))
        if rank in island.ranks:
            island_group = group
    bridge: BridgeTransport | None = None
    if layout.needs_bridge:
        bridge = transport.create(layout.bridge_ranks, timeout=timeout)
        if (bridge is not None) != (rank in layout.bridge_ranks):
            raise RuntimeError(
                f"{transport.__name__}.create() must return a transport exactly on the "
                f"bridge ranks {list(layout.bridge_ranks)}; rank {rank} got {bridge!r}"
            )
        if hasattr(bridge, "selection"):
            bridge.selection = selection  # for the bridge's telemetry()
    assert island_group is not None

    topology = Topology(
        rank=rank,
        layout=layout,
        peers=peers,
        config=config,
        island_group=island_group,
        bridge=bridge,
        policy=policy_cls(),
        transport=transport,
        selection=selection,
        timeout=timeout,
        world_group=world_group,
        scope=scope,
    )
    topology.policy.setup(topology)
    return topology


def island_backend(config: Config, island: Island,
                   transport: str | type[BridgeTransport]) -> str:
    """The backend of ``island``'s process group.

    NCCL (RCCL on ROCm builds) on GPUs. In CPU simulation and CPU-only mode, a test
    device: Gloo for an island of several ranks, and for a one-rank island
    :data:`~gpubridge.config.SOLO_BACKEND`, a Python group with no backend library, no
    connection and no thread (:class:`gpubridge.torch_backend.SoloGroup`), so a job of
    one-rank islands on a bridge other than Gloo has no Gloo group at all. With
    ``bridge="gloo"`` a one-rank island stays the Gloo group it always was.
    """
    if config.simulated and len(island.ranks) == 1 and not asks_for_gloo(transport):
        return SOLO_BACKEND
    return config.island_backend


def _default_is_gloo() -> bool:
    """Whether the job's default group is a Gloo one (``gpubridge.init(bridge="gloo")``)."""
    try:
        return dist.get_backend() == CPU_BACKEND
    except (RuntimeError, ValueError):
        return False


def gloo_world(topology: Topology) -> ProcessGroup:
    """A Gloo group of every member of ``topology``: what ``flat-gloo`` runs on.

    The one the job set up with ``bridge="gloo"`` where there is one. On any other
    bridge none exists, and this creates one the first time it is asked, and keeps it:
    naming ``flat-gloo`` (the policy, an ``auto-tuned`` rule, or the reference in a
    test) is then what asks for Gloo. Collective over the topology's members, like the
    collective that needs it: every member reaches its first ``flat-gloo`` call at the
    same point.
    """
    if topology.world_group is not None:
        return topology.world_group
    if not topology.layout.ranks and _default_is_gloo():
        return dist.group.WORLD
    group = topology.state.get("gloo_world")
    if group is None:
        with setup_names(f"gpubridge:{topology.scope}.gloo"):
            group = dist.new_group(list(topology.layout.members), timeout=topology.timeout,
                                   backend=CPU_BACKEND)
        topology.state["gloo_world"] = group
    return group


def barrier(topology: Topology) -> None:
    """Block until every member of ``topology`` reaches this call.

    A Gloo barrier where the job set up a Gloo group of every member
    (``bridge="gloo"``); otherwise a counter in the c10d store that every member adds
    to and watches (:func:`gpubridge.rendezvous.barrier`), with the deadline
    ``init()`` gave its groups (30 minutes without one).

    Raises:
        TimeoutError: if a member doesn't arrive in time (the store barrier).
    """
    if topology.world_group is not None:
        dist.barrier(group=topology.world_group)
        return
    if not topology.layout.ranks and _default_is_gloo():
        dist.barrier()
        return
    count = topology.state.get("barriers", 0)
    topology.state["barriers"] = count + 1
    prefix = f"gpubridge/{topology.scope}/barrier"
    first = topology.rank == topology.layout.members[0]
    deadline = time.monotonic() + (topology.timeout
                                   or _rendezvous.DEFAULT_TIMEOUT).total_seconds()
    _rendezvous.barrier(_rendezvous.default_store(), f"{prefix}/{count}",
                        len(topology.layout.members), deadline,
                        retire=f"{prefix}/{count - 1}" if first and count else None)


def _subgroup_naming_problem() -> str | None:
    """What, if anything, differs from the torch :func:`_members_only` patches:
    ``distributed_c10d._process_group_name(ranks, use_hashed_name)``, which
    ``dist.new_group`` (or a ``distributed_c10d`` function it calls) names its
    groups with, and ``dist.new_group``'s ``use_local_synchronization``."""
    c10d = dist.distributed_c10d
    name = getattr(c10d, "_process_group_name", None)
    if not callable(name):
        return "torch.distributed.distributed_c10d has no _process_group_name"
    try:
        signature = inspect.signature(name)
    except (TypeError, ValueError):
        return "its _process_group_name has no signature to check"
    params = list(signature.parameters.values())
    if ([p.name for p in params] != ["ranks", "use_hashed_name"]
            or any(p.kind is not p.POSITIONAL_OR_KEYWORD for p in params)):
        return f"its _process_group_name takes {signature}, not (ranks, use_hashed_name)"
    if "use_local_synchronization" not in inspect.signature(dist.new_group).parameters:
        return "its dist.new_group has no use_local_synchronization argument"

    def names(fn: Any) -> tuple[str, ...]:  # the globals a function's code looks up
        return getattr(getattr(inspect.unwrap(fn), "__code__", None), "co_names", ())

    called = names(dist.new_group)
    inner = [getattr(c10d, n) for n in called
             if callable(getattr(c10d, n, None))
             and getattr(getattr(c10d, n), "__module__", None) == c10d.__name__]
    if "_process_group_name" not in called + tuple(n for fn in inner for n in names(fn)):
        return "its dist.new_group doesn't name groups with _process_group_name"
    return None


def check_subgroup_naming() -> None:
    """Refuse subgroups, clearly, on a torch whose private group naming isn't what
    :func:`_members_only` patches: the patch would miss or break it, and members
    could wait on different store keys until the timeout instead.

    Raises:
        RuntimeError: naming the torch version and what differs.
    """
    problem = _subgroup_naming_problem()
    if problem is not None:
        raise RuntimeError(
            f"gpubridge can't create subgroups on torch {torch.__version__}: {problem}. While a "
            "subgroup sets up, gpubridge replaces torch's private "
            "distributed_c10d._process_group_name(ranks, use_hashed_name) so that every member "
            "names the subgroup's groups alike; on this torch that could leave members waiting "
            "on each other until the timeout, so subgroups are refused instead. Groups of every "
            "rank (DDP, FSDP2 on a 1-D mesh) are unaffected. Checked on torch 2.8.0 and 2.14.1 "
            "(docs/TORCH_BACKEND.md, \"Subgroups and HSDP\").")


@contextlib.contextmanager
def _members_only(prefix: str) -> Iterator[None]:
    """Every ``dist.new_group`` inside synchronizes only the new group's ranks
    (``use_local_synchronization=True``), so a subgroup's members build its groups
    without the rest of the job (transports and policies call ``dist.new_group``).

    And each group is named ``{prefix}.{ranks}.{k}``: its ranks, and how many
    groups of the same ranks were named here before it. torch names such groups
    from state that differs between ranks of asymmetric groups (2.8: how many
    groups this rank has; 2.14: a counter that only creating ranks advance), so
    members could compute different names and wait on different store keys, and
    later groups of the whole job too. Only a group's ranks name it (torch
    returns early on the others), and they create the same groups in the same
    order. ``prefix`` is the subgroup's own name, the same on every member.
    Transports name their links similarly, with a separate counter and a ``link``
    namespace (:func:`gpubridge.transport.scoped_name`). Unlike torch's group
    naming, transport setup may request a link name on non-members too.

    The patch is process-wide: no other thread may create groups meanwhile
    (the backend holds the subgroup's setup lock; the training thread is in
    the subgroup's first collective)."""
    check_subgroup_naming()
    c10d = dist.distributed_c10d
    plain_new, plain_name = dist.new_group, c10d._process_group_name
    seen: dict[tuple[int, ...], int] = {}
    links: dict[tuple[int, ...], int] = {}

    def local(*args: Any, **kwargs: Any) -> Any:
        kwargs["use_local_synchronization"] = True
        return plain_new(*args, **kwargs)

    def name(ranks: Any, use_hashed_name: bool = False) -> str:
        key = tuple(ranks)
        seen[key] = seen.get(key, -1) + 1
        return f"{prefix}.{'_'.join(map(str, key))}.{seen[key]}"

    def link_name(ranks: Sequence[int]) -> str:
        key = tuple(ranks)
        links[key] = links.get(key, -1) + 1
        return f"{prefix}.link.{'_'.join(map(str, key))}.{links[key]}"

    dist.new_group, c10d._process_group_name = local, name  # type: ignore[assignment]
    token = _transport_scope.set(link_name)
    try:
        yield
    finally:
        _transport_scope.reset(token)
        dist.new_group, c10d._process_group_name = plain_new, plain_name  # type: ignore[assignment]


def setup_names(prefix: str) -> contextlib.AbstractContextManager[None]:
    """Name the groups created inside like a subgroup's (``{prefix}.{ranks}.{k}``,
    :func:`_members_only`), not from torch's count of the groups each process has
    created, and have only each group's own ranks create it.

    The ranks can reach a setup having created different numbers of groups: torch
    2.9.1's ``DeviceMesh`` creates one for a (2, 1) mesh's replicate dimension where
    2.14.1's reuses the default group, groups of the gpubridge backend are created
    without any exchange, and a one-rank island's group is created by that rank alone.
    torch's names would then differ between the ranks, and they would wait on different
    store keys until the timeout. On a torch whose private naming isn't what the patch
    expects, torch's names are used as before, with a warning."""
    problem = _subgroup_naming_problem()
    if problem is None:
        return _members_only(prefix)
    warnings.warn(
        f"gpubridge names its setup groups the way torch {torch.__version__} does ({problem}), "
        "so every rank must have created the same number of groups before gpubridge sets "
        "up, or setup waits until the timeout",
        RuntimeWarning, stacklevel=3)
    return contextlib.nullcontext()


def build_subtopology(world: Topology, ranks: Sequence[int], name: str,
                      timeout: timedelta | None = None
                      ) -> tuple[Topology, ProcessGroup | None]:
    """A subgroup's own topology, over its members only; and with ``bridge="gloo"`` a
    Gloo group of them for CPU tensors (None on any other bridge, where the backend
    carries those through the topology itself).

    Collective over the members (``ranks``, global, ascending), which must all
    call it, in the same order relative to other subgroups' setups; the rest of
    the job doesn't take part. ``name`` is the subgroup's (torch's group name),
    from which its own groups are named alike on every member. Islands come from the world's labels (no new
    discovery). Single-vendor members get ``native-only``; mixed members get
    the policy the job asked for, resolved for their layout, and the world's
    bridge transport over their own leaders. Groups are created in the same
    order on every member: with ``bridge="gloo"`` a Gloo group of the members (the
    subgroup's ``world_group``) and a second one for CPU tensors; then the islands
    in leader order, the bridge, then whatever the policy's ``setup()`` creates.
    """
    ranks = tuple(ranks)
    layout = plan_topology([world.layout.vendors[world.layout.index(r)] for r in ranks], ranks)
    asked = world.peers[world.rank].policy
    if not layout.needs_bridge:
        policy_cls: type[CollectivePolicy] = choose_policy(AUTO, layout)
    else:
        policy_cls = choose_policy(AUTO if asked == AUTO else type(world.policy), layout)
    rank = world.rank
    assert world.transport is not None
    # The debug level as it is now: init() refused DETAIL (debug_gloo_problem), so this
    # is a level raised since. Seen on this member only; the others see their own.
    if (world.gloo_free and debug_detail() and not world.peers[rank].allow_debug_gloo
            and any(island_backend(world.config, island, world.transport) != SOLO_BACKEND
                    for island in layout.islands)):
        raise RuntimeError(_debug_refusal(
            f"on rank {rank}, raised after init(), when subgroup {name!r} of ranks "
            f"{list(ranks)} was set up"))
    with _members_only(f"gpubridge:{name}"):
        group = cpu = None
        if not world.gloo_free:
            group = dist.new_group(list(ranks), timeout=timeout, backend=CPU_BACKEND)
            cpu = dist.new_group(list(ranks), timeout=timeout, backend=CPU_BACKEND)
        island_group: ProcessGroup | None = None
        for island in layout.islands:
            created = dist.new_group(
                list(island.ranks), timeout=timeout,
                backend=island_backend(world.config, island, world.transport))
            if rank in island.ranks:
                island_group = created
        bridge: BridgeTransport | None = None
        if layout.needs_bridge:
            bridge = world.transport.create(layout.bridge_ranks, timeout=timeout)
            if hasattr(bridge, "selection"):
                bridge.selection = world.selection
        assert island_group is not None
        topology = Topology(
            rank=rank, layout=layout, peers=world.peers, config=world.config,
            island_group=island_group, bridge=bridge, policy=policy_cls(),
            transport=world.transport, selection=world.selection, timeout=timeout,
            world_group=group, scope=f"{world.scope}.{name}")
        topology.policy.setup(topology)
    return topology, cpu


_current: Topology | None = None
_owner_pid: int | None = None
"""The process that built :data:`_current`. A forked child inherits the topology object
but none of the threads behind its bridge, and shares its sockets and device handles
with the parent: it must never close them (:func:`owned`)."""


def owned() -> bool:
    """Whether this process built the current topology (False in a forked child, and
    when there is none): only then may it close the bridge's native resources."""
    return _current is not None and _owner_pid == os.getpid()


def get_topology() -> Topology:
    """Return this rank's topology.

    Raises:
        RuntimeError: if :func:`gpubridge.init` has not been called.
    """
    if _current is None:
        raise RuntimeError("gpubridge is not initialized; call gpubridge.init() first")
    return _current


def is_initialized() -> bool:
    """Return True between :func:`gpubridge.init` and :func:`gpubridge.destroy`."""
    return _current is not None


def _set_topology(topology: Topology | None) -> None:
    global _current, _owner_pid
    _current = topology
    _owner_pid = os.getpid() if topology is not None else None
