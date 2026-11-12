"""Bridge transports: how island leaders exchange data across vendors.

By default the bridge moves CPU tensors between island leaders, so anything
that can sum and broadcast a CPU tensor among a fixed set of processes can
carry it (a transport with ``device_tensors`` also takes GPU tensors).

The default is ``bridge="auto"``, which ``init()`` resolves once, the same on
every rank, to one of gpubridge's two Rust bridges: ``"verbs"``
(:mod:`gpubridge.verbs.transport`, RDMA) when every island leader passes that
bridge's own check for a usable RDMA device (:func:`gpubridge.verbs.availability`)
and they are on one link layer, otherwise ``"tcp"``
(:mod:`gpubridge.multitcp.transport`; ``"multitcp"`` is its older name).
:func:`choose_bridge` is the rule. It is never Gloo, and a job never changes
bridge once ``init()`` has returned.

``bridge="gloo"`` (a Gloo process group of the leaders, the default through 0.4.0-alpha)
stays for one release, opt-in: asking for it is the only way a job gets its
bridge, its default group and its barriers on Gloo. ``bridge="ucx"``
(:mod:`gpubridge.ucx.transport`) is the other built-in one. Each built-in
bridge's module is imported the first time its name is asked for, so a job
loads only the one it runs on. Another transport (MPI, another RDMA path)
plugs in by subclassing :class:`BridgeTransport` and registering it, without
touching the reduce-bridge-broadcast logic in ``collectives.py``.
"""

from __future__ import annotations

import abc
import contextvars
import importlib
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from gpubridge.config import CPU_BACKEND

_scope: contextvars.ContextVar[Callable[[Sequence[int]], str] | None] = contextvars.ContextVar(
    "gpubridge_subgroup_scope", default=None)
"""Set while a subgroup's members build its topology (topology._members_only)."""


def scoped_name(ranks: Sequence[int]) -> str | None:
    """Inside a subgroup's setup, a name for a new group or link over ``ranks``: the same
    on every member, unique in the job. None otherwise.

    A transport whose setup keys anything by a per-process count (store keys, say)
    must use this when it isn't None: a subgroup's setup runs on its members only,
    so such counts would differ between ranks."""
    namer = _scope.get()
    return namer(ranks) if namer is not None else None


class BridgeTransport(abc.ABC):
    """Moves CPU tensors between island leaders. One instance per leader.

    Subclasses set :attr:`name`, which every rank reports during discovery so
    that ``init()`` can refuse a job whose ranks picked different transports.
    """

    name: ClassVar[str]

    #: Reductions :meth:`all_reduce` can do: always ``"sum"``, plus ``"max"`` and
    #: ``"min"`` if listed. Only a transport that lists one is called with it, as
    #: ``all_reduce(tensor, op=ReduceOp.MAX)``; SUM always comes without ``op``.
    reduce_ops: ClassVar[frozenset[str]] = frozenset({"sum"})

    #: True if :meth:`all_reduce` and :meth:`broadcast` also take tensors in GPU
    #: memory and order them against the caller's stream themselves. The default
    #: policy then hands GPU tensors straight to the transport instead of copying
    #: them to host memory first. Transports without it only ever see host tensors.
    device_tensors: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> BridgeTransport | None:
        """Set up the bridge between ``ranks`` (the island leaders).

        Collective over every rank that builds the topology, in the same order
        relative to other group creation, including ranks not in ``ranks``: the
        whole job for the world's topology; for a subgroup's (torch.distributed
        backend), its members only, with :func:`scoped_name` set and
        ``dist.new_group`` synchronizing only each new group's ranks.

        Returns:
            A transport on members of ``ranks``, None everywhere else.
        """

    @classmethod
    def create_groups(
        cls, groups: Sequence[Sequence[int]], *, timeout: timedelta | None = None
    ) -> list[BridgeTransport | None]:
        """Set up one bridge per entry of ``groups``, for policies that use several links.

        Collective like :meth:`create`: every rank that builds the topology
        calls it with the same ``groups`` in the same order. The default calls
        :meth:`create` once per group, in order, so every transport supports
        it; one that can set up several links more cheaply at once may
        override it. The leader bridge ``init()`` creates is unaffected.

        Returns:
            One entry per group: a transport on that group's members, None elsewhere.
        """
        return [cls.create(ranks, timeout=timeout) for ranks in groups]

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Sum a contiguous CPU tensor across all bridge members, in place.

        Transports that list ``"max"`` or ``"min"`` in :attr:`reduce_ops` also
        take ``op=ReduceOp.MAX`` / ``ReduceOp.MIN``.
        """

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        """Copy a contiguous CPU tensor from bridge member ``src`` (a global rank), in place."""

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release resources. Called by :func:`gpubridge.destroy`; the default does nothing."""

    @classmethod
    def create_p2p(cls, group: ProcessGroup, *, store=None, timeout=None):
        """An optional all-member point-to-point link for the torch backend.

        Unlike :meth:`create`, this hook runs locally after topology setup. It
        must not create a process group or wait for another rank. The supplied
        Gloo group contains every member, including non-leaders. The default
        refuses P2P without changing existing collective-only transports.
        """
        raise NotImplementedError(
            f"bridge transport {cls.name!r} has no point-to-point link: point-to-point "
            "across islands (and of CPU tensors on a GPU job) needs bridge=\"tcp\", "
            "\"ucx\" or \"gloo\" (GPUBRIDGE_BRIDGE under the torch.distributed backend)")


class GlooTransport(BridgeTransport):
    """``bridge="gloo"``: a Gloo process group of the island leaders. Opt-in: the
    default through 0.4.0-alpha, kept for one release as the reference the other bridges are
    tested against. A job on it (or on a subclass) is the only kind whose default
    group, discovery barrier and bridge are Gloo groups."""

    name = "gloo"
    reduce_ops = frozenset({"sum", "max", "min"})

    def __init__(self, group: ProcessGroup) -> None:
        self.group = group

    @classmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> GlooTransport | None:
        group = dist.new_group(list(ranks), timeout=timeout, backend=CPU_BACKEND)
        return cls(group) if dist.get_rank() in ranks else None

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        dist.all_reduce(tensor, op=op, group=self.group)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        dist.broadcast(tensor, src=src, group=self.group)

    @classmethod
    def create_p2p(cls, group: ProcessGroup, *, store=None, timeout=None):
        from gpubridge.p2p import GlooLink

        return GlooLink(group)


#: ``bridge="auto"``: let ``init()`` choose between the Rust bridges (:func:`choose_bridge`).
AUTO = "auto"
DEFAULT_TRANSPORT = AUTO
#: What ``auto`` falls to without RDMA on every leader, and what it prefers with it.
TCP, VERBS = "tcp", "verbs"

_TRANSPORTS: dict[str, type[BridgeTransport]] = {GlooTransport.name: GlooTransport}

#: The other transports gpubridge ships, by name, and the module whose import registers
#: each. A name's module is imported at its first use (resolve_transport,
#: register_transport): ``import gpubridge`` doesn't load them, and a job loads only the
#: one it runs on (verbs still imports the GPU flush helpers from ucx.gdr).
_BUILT_IN = {"ucx": "gpubridge.ucx.transport", TCP: "gpubridge.multitcp.transport",
             VERBS: "gpubridge.verbs.transport"}
#: Older names of built-in transports: ``bridge="multitcp"`` is ``bridge="tcp"``.
_ALIASES = {"multitcp": TCP}


def choose_bridge(leaders: Sequence[int], reports: dict[int, dict]) -> tuple[str, str]:
    """What ``bridge="auto"`` resolves to, and why: a pure function of what every
    island leader reported, so every rank that reads the same reports chooses the same.

    ``"verbs"`` when every leader's report (:func:`gpubridge.verbs.availability`) says
    its RDMA device passed the verbs bridge's own check and all of them are on one link
    layer (an InfiniBand port and a RoCE port can't connect); otherwise ``"tcp"``.

    Args:
        leaders: The island leaders' global ranks.
        reports: Each leader's availability report, by rank.

    Returns:
        ``(name, reason)``.
    """
    missing = [(rank, str(reports[rank].get("reason") or "no reason given"))
               for rank in leaders if not reports[rank].get("ok")]
    if missing:
        detail = "; ".join(f"rank {rank}: {why}" for rank, why in missing)
        return TCP, f"verbs isn't available on every island leader ({detail})"
    layers = sorted({str(reports[rank].get("link_layer")) for rank in leaders})
    if len(layers) > 1:
        detail = ", ".join(f"rank {rank}: {reports[rank].get('link_layer')}" for rank in leaders)
        return TCP, ("the island leaders' RDMA ports are on different link layers, which "
                     f"verbs can't connect ({detail})")
    return VERBS, f"every island leader has a usable RDMA port ({layers[0]})"


def _load_built_in(name: str) -> None:
    """Register the built-in transport called ``name`` by importing its module, if it is
    one and isn't registered yet.

    Raises:
        ValueError: if the module can't be imported, as for an unknown name, so that
            the torch.distributed backend reports it from every rank.
    """
    module = _BUILT_IN.get(name)
    if module is None or name in _TRANSPORTS:
        return
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(f"bridge transport {name!r} can't be loaded: {exc}") from exc


def register_transport(transport: type[BridgeTransport]) -> type[BridgeTransport]:
    """Make ``transport`` selectable by name in ``gpubridge.init(bridge=...)``.

    Usable as a class decorator. Every rank must register the same transports.

    Raises:
        TypeError: if ``transport`` is not a :class:`BridgeTransport` subclass.
        ValueError: if a different class is already registered under its name.
    """
    if not (isinstance(transport, type) and issubclass(transport, BridgeTransport)):
        raise TypeError(f"expected a BridgeTransport subclass, got {transport!r}")
    if transport.name == AUTO:
        raise ValueError(f"{AUTO!r} is reserved for automatic selection")
    if transport.name in _ALIASES:
        raise ValueError(f"{transport.name!r} is another name of the built-in "
                         f"{_ALIASES[transport.name]!r} bridge")
    if _BUILT_IN.get(transport.name) not in (None, transport.__module__):
        _load_built_in(transport.name)  # a built-in's name stays the built-in's
    existing = _TRANSPORTS.get(transport.name)
    if existing is not None and existing is not transport:
        raise ValueError(f"a different transport is already registered as {transport.name!r}")
    _TRANSPORTS[transport.name] = transport
    return transport


def resolve_transport(bridge: str | type[BridgeTransport]) -> str | type[BridgeTransport]:
    """Check a requested bridge: ``"auto"``, a transport name, or a transport class.

    Returns ``"auto"`` (``init()`` chooses, :func:`choose_bridge`) or a transport class.
    ``"multitcp"`` gives the ``"tcp"`` transport.

    Raises:
        ValueError: for an unknown name, or a built-in one whose module can't be imported.
        TypeError: for anything that is neither a name nor a transport class.
    """
    if bridge == AUTO:
        return AUTO
    if isinstance(bridge, str):
        name = _ALIASES.get(bridge, bridge)
        _load_built_in(name)
        try:
            return _TRANSPORTS[name]
        except KeyError:
            known = ", ".join([AUTO, *sorted({*_TRANSPORTS, *_BUILT_IN})])
            raise ValueError(f"unknown bridge transport {bridge!r}; known: {known}") from None
    if isinstance(bridge, type) and issubclass(bridge, BridgeTransport):
        return bridge
    raise TypeError(f"bridge must be a transport name or BridgeTransport subclass, got {bridge!r}")


def transport_name(transport: str | type[BridgeTransport]) -> str:
    """The name a rank reports during discovery for a resolved bridge."""
    return transport if isinstance(transport, str) else transport.name


def asks_for_gloo(transport: str | type[BridgeTransport]) -> bool:
    """Whether a resolved bridge is Gloo's (or a subclass of it): the one request that
    gives a job Gloo groups. ``"auto"`` never is."""
    return isinstance(transport, type) and issubclass(transport, GlooTransport)
