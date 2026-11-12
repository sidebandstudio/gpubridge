"""Cluster-wide collectives: validate the call, then hand it to the active collective policy.

Every check here runs before any communication and depends only on the call's
arguments and on state that is identical on every rank (the policy, the
bridge transport, the world size), so a call that is rejected is rejected on
every rank alike and nobody is left waiting.

Every collective takes ``async_op``. With ``async_op=True`` it returns a
:class:`~gpubridge.work.Work` and runs on this rank's worker thread; see
:mod:`gpubridge.work` for ordering and stream semantics.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import torch
from torch.distributed import ReduceOp

from gpubridge import observe
from gpubridge import work as _work
from gpubridge.policies import CollectivePolicy, implements, policies_implementing
from gpubridge.topology import Topology, get_topology
from gpubridge.topology import barrier as _barrier
from gpubridge.work import Work

#: Dtypes every built-in path supports: Gloo (world group, bridge, CPU islands)
#: and NCCL/RCCL (GPU islands). Gloo all_reduce and broadcast were checked for
#: each one on torch 2.3.1, 2.6.0, 2.9.1 and 2.14.1, including the CUDA and
#: ROCm builds of 2.14.1 (rerun with scripts/gloo_dtypes.py). Not included:
#: - bool, because a SUM of bools is a logical OR on both NCCL and Gloo;
#: - int16, uint16, uint32 and float8, which Gloo rejects;
#: - complex64, whose Gloo broadcast fails before torch 2.9.
SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
)

#: Reductions gpubridge supports, by name. AVG is a SUM followed by a division
#: by the world size, done identically on every rank, because Gloo (the bridge)
#: has no AVG. PRODUCT is left out until NCCL/RCCL and Gloo are shown to agree
#: on every supported dtype (HARDWARE_VALIDATION.md item 14).
REDUCE_OPS = {ReduceOp.SUM: "sum", ReduceOp.AVG: "avg", ReduceOp.MAX: "max",
              ReduceOp.MIN: "min"}
_BY_NAME = {name: op for op, name in REDUCE_OPS.items()}
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def all_reduce(tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM, *,
               async_op: bool = False, _topology: Topology | None = None) -> Work | None:
    """Reduce ``tensor`` across every rank in the cluster, in place.

    Mirrors ``torch.distributed.all_reduce`` on the world group. Every rank must
    call it with a tensor of the same shape and dtype. The active collective
    policy decides how the data travels; in a mixed cluster the default runs
    reduce-bridge-broadcast:

    1. reduce within each island onto its leader, on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    Args:
        tensor: Input and output. Must be contiguous, on this rank's device
            (CPU in simulation and CPU-only mode), and of a dtype in
            :data:`SUPPORTED_DTYPES`.
        op: ``ReduceOp.SUM``, ``AVG``, ``MAX`` or ``MIN``. AVG needs a
            floating-point dtype: it is a SUM followed by a division by the
            world size, and an integer division would need a rounding rule.
        async_op: Return a :class:`~gpubridge.work.Work` instead of blocking.

    Returns:
        A :class:`~gpubridge.work.Work` if ``async_op``, else None.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: for an unsupported op or dtype, AVG on an integer dtype, or
            a tensor on the wrong device or not contiguous.
        NotImplementedError: if the active policy or bridge transport can't
            carry this op.
    """
    topology = _topology or get_topology()
    name = _reduce_op_name(op, "all_reduce")
    _check_tensor(tensor, topology)
    _check_avg(name, tensor.dtype, "all_reduce")
    policy = topology.policy.select_policy(tensor.nbytes)
    base = "sum" if name == "avg" else name
    _check_reduce_op(policy, base, topology, "all_reduce")

    def run() -> None:
        with observe.collective(topology, "all_reduce", tensor, policy=policy.name,
                                reduce_op=name):
            if base == "sum":
                policy.all_reduce(tensor, topology)  # no op: keeps older subclasses working
            else:
                policy.all_reduce(tensor, topology, op=_BY_NAME[base])
            if name == "avg":
                tensor.div_(topology.world_size)

    return _start(topology, "all_reduce", run, [tensor], tensor, async_op)


def broadcast(tensor: torch.Tensor, src: int, *, async_op: bool = False,
              _topology: Topology | None = None) -> Work | None:
    """Copy ``tensor`` from global rank ``src`` to every rank, in place.

    Mirrors ``torch.distributed.broadcast`` on the world group. The active
    collective policy decides how the data travels; in a mixed cluster the
    default takes three hops:

    1. From ``src`` to the rest of its island on the native backend.
    2. From that island's leader to the other leaders over the bridge.
    3. From each of those leaders to the rest of its island.

    Args:
        tensor: Data to send on ``src``, overwritten everywhere else. Must be
            contiguous, on this rank's device (CPU in simulation mode), and of a
            dtype in :data:`SUPPORTED_DTYPES`.
        src: Global rank that holds the data.
        async_op: Return a :class:`~gpubridge.work.Work` instead of blocking.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: if ``src`` is not a rank, or for an unsupported dtype, or a
            tensor on the wrong device or not contiguous.
    """
    topology = _topology or get_topology()
    if src not in topology.layout.members:
        if not topology.layout.ranks:
            raise ValueError(f"src must be a rank in [0, {topology.world_size}), got {src}")
        raise ValueError(f"src must be one of the group's ranks "
                         f"{list(topology.layout.members)}, got {src}")
    _check_tensor(tensor, topology)
    policy = topology.policy.select_policy(tensor.nbytes)

    def run() -> None:
        with observe.collective(topology, "broadcast", tensor, src=src, policy=policy.name):
            policy.broadcast(tensor, src, topology)

    return _start(topology, "broadcast", run, [tensor], tensor, async_op)


def all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor, *,  # noqa: A002
                           async_op: bool = False,
                           _topology: Topology | None = None) -> Work | None:
    """Gather every rank's ``input`` into ``output``, ordered by global rank.

    Mirrors ``torch.distributed.all_gather_into_tensor`` on the world group:
    ``output`` holds ``world_size`` copies of ``input``'s shape back to back,
    rank 0's first, e.g. ``(world_size * n, ...)`` for an input of ``(n, ...)``.

    Args:
        output: Receives the result. ``world_size`` times as many elements as
            ``input``, same dtype and device, contiguous.
        input: This rank's contribution. Not modified.
        async_op: Return a :class:`~gpubridge.work.Work` instead of blocking.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: for mismatched sizes or dtypes, or tensors that fail the
            usual dtype, device and contiguity checks.
        NotImplementedError: if the active policy doesn't implement it.
    """
    topology = _topology or get_topology()
    _check_pair(output, input, topology, "all_gather_into_tensor", big=output, small=input)
    policy = topology.policy.select_policy(output.nbytes)
    _check_implements(policy, "all_gather_into_tensor", topology)

    def run() -> None:
        with observe.collective(topology, "all_gather_into_tensor", input,
                                policy=policy.name):
            policy.all_gather_into_tensor(output, input, topology)

    return _start(topology, "all_gather_into_tensor", run, [output, input], output, async_op)


def reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                          op: ReduceOp.RedOpType = ReduceOp.SUM, *,
                          async_op: bool = False,
                          _topology: Topology | None = None) -> Work | None:
    """Reduce every rank's ``input`` and give rank ``r`` the ``r``-th slice of the result.

    Mirrors ``torch.distributed.reduce_scatter_tensor`` on the world group:
    ``input`` holds ``world_size`` slices of ``output``'s shape back to back.

    Args:
        output: Receives this rank's slice of the reduced input.
        input: ``world_size`` times as many elements as ``output``, same dtype
            and device, contiguous. Not modified.
        op: ``ReduceOp.SUM`` or ``AVG`` (floating-point dtypes only).
        async_op: Return a :class:`~gpubridge.work.Work` instead of blocking.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: for an op other than SUM or AVG, AVG on an integer dtype,
            mismatched sizes or dtypes, or tensors that fail the usual checks.
        NotImplementedError: if the active policy doesn't implement it.
    """
    topology = _topology or get_topology()
    name = _reduce_op_name(op, "reduce_scatter_tensor")
    if name not in ("sum", "avg"):
        raise ValueError(f"gpubridge.reduce_scatter_tensor supports SUM and AVG, got {op}")
    _check_pair(output, input, topology, "reduce_scatter_tensor", big=input, small=output)
    _check_avg(name, input.dtype, "reduce_scatter_tensor")
    policy = topology.policy.select_policy(input.nbytes)
    _check_implements(policy, "reduce_scatter_tensor", topology)

    def run() -> None:
        with observe.collective(topology, "reduce_scatter_tensor", input, policy=policy.name,
                                reduce_op=name):
            policy.reduce_scatter_tensor(output, input, topology)
            if name == "avg":
                output.div_(topology.world_size)

    return _start(topology, "reduce_scatter_tensor", run, [output, input], output, async_op)


def barrier(*, async_op: bool = False, _topology: Topology | None = None) -> Work | None:
    """Block until every rank in the cluster reaches this call.

    On GPUs this first waits for the rank's queued device work, so collectives
    issued before the barrier have finished locally when it returns. The barrier
    itself is a counter in the job's c10d store that every rank adds to and watches
    (with ``bridge="gloo"``, a Gloo barrier).

    Raises:
        RuntimeError: if gpubridge is not initialized.
        TimeoutError: if a rank doesn't reach it within the timeout ``init()`` gave
            its groups (30 minutes without one).
    """
    topology = _topology or get_topology()

    def run() -> None:
        with observe.collective(topology, "barrier"):
            if not topology.simulated:
                torch.cuda.synchronize(topology.device)
            _barrier(topology)

    return _start(topology, "barrier", run, [], None, async_op)


def _start(topology: Topology, op: str, run: Callable[[], None],
           tensors: list[torch.Tensor], result: object, async_op: bool) -> Work | None:
    """Run a validated collective now, or queue it on the worker thread."""
    observe.refuse_in_callback(op)
    if async_op:
        return _work.submit(topology.device, op, run, tensors, result)
    _work.drain()  # queued async collectives go first, in the order they were queued
    run()
    return None


def _reduce_op_name(op: ReduceOp.RedOpType, fn: str) -> str:
    try:
        # ReduceOp.SUM and the others: one lookup. torch.distributed.ReduceOp objects, which
        # the torch backend passes, don't match a key and take the loop below.
        name = REDUCE_OPS.get(op)
    except TypeError:  # an unhashable op object: the loop compares it
        name = None
    if name is not None:
        return name
    for candidate, name in REDUCE_OPS.items():
        try:
            if op == candidate:  # also matches dist.ReduceOp instances
                return name
        except TypeError:
            break
    raise ValueError(
        f"gpubridge.{fn} supports ReduceOp.SUM, AVG, MAX and MIN, got {op}. PRODUCT is "
        "not supported yet: NCCL/RCCL and Gloo aren't shown to agree on every dtype."
    )


def _check_avg(name: str, dtype: torch.dtype, fn: str) -> None:
    if name == "avg" and dtype not in _FLOATS:
        raise ValueError(
            f"gpubridge.{fn} with ReduceOp.AVG needs a floating-point tensor, got {dtype}: "
            "AVG divides the sum by the world size, and an integer division would need a "
            "rounding rule. Use SUM and divide the way you want, or cast to a float dtype."
        )


def _check_reduce_op(policy: CollectivePolicy, name: str, topology: Topology, fn: str) -> None:
    supported = _supported_reduce_ops(policy, topology)
    if name not in supported:
        raise NotImplementedError(
            f"gpubridge.{fn} with {name.upper()}: collective policy {policy.name!r} with bridge "
            f"transport {topology.bridge_name!r} supports only "
            f"{', '.join(sorted(n.upper() for n in supported))}."
        )


def _supported_reduce_ops(policy: CollectivePolicy, topology: Topology) -> frozenset[str]:
    """``policy.supported_reduce_ops(topology)``, asked once per policy and topology and
    remembered on the policy: the policy, its settings and the bridge transport are fixed
    from init() on, and the same on every rank. Every call still checks its op against it."""
    cached = getattr(policy, "_gpubridge_reduce_ops", None)
    if cached is not None and cached[0]() is topology:
        return cached[1]
    supported = policy.supported_reduce_ops(topology)
    try:  # weakly: no cycle keeps a destroyed topology (and its groups) alive
        policy._gpubridge_reduce_ops = (weakref.ref(topology), supported)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # __slots__, or a topology without weakrefs: ask each call
        pass
    return supported


def _check_implements(policy: CollectivePolicy, fn: str, topology: Topology) -> None:
    if not implements(policy, fn):
        usable = ", ".join(policies_implementing(fn, topology.layout))
        raise NotImplementedError(
            f"collective policy {policy.name!r} doesn't implement {fn}. Policies that do "
            f"and fit this cluster: {usable or 'none'}."
        )


def _check_pair(output: torch.Tensor, input: torch.Tensor, topology: Topology,  # noqa: A002
                fn: str, *, big: torch.Tensor, small: torch.Tensor) -> None:
    _check_tensor(output, topology)
    _check_tensor(input, topology)
    if output.dtype != input.dtype:
        raise ValueError(f"gpubridge.{fn}: output is {output.dtype} but input is {input.dtype}")
    world = topology.world_size
    if big.numel() != world * small.numel():
        raise ValueError(
            f"gpubridge.{fn}: {'output' if big is output else 'input'} must have "
            f"world_size ({world}) times as many elements as the "
            f"{'input' if big is output else 'output'} ({small.numel()}), got {big.numel()}"
        )


def _check_tensor(tensor: torch.Tensor, topology: Topology) -> None:
    """Reject bad calls before any communication, identically on every rank."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        names = ", ".join(str(dtype).removeprefix("torch.") for dtype in SUPPORTED_DTYPES)
        raise ValueError(
            f"gpubridge doesn't support {tensor.dtype}: a dtype must work on both the "
            f"island backend and Gloo. Supported: {names}."
        )
    if tensor.device != topology.device:
        hint = ""
        if topology.simulated:
            hint = " (tensors stay on CPU in simulation and CPU-only mode)"
        raise ValueError(
            f"tensor is on {tensor.device} but this rank communicates on {topology.device}{hint}"
        )
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
