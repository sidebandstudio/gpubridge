"""Collective policies: every policy that applies to a layout gives exactly FlatGloo's answer."""

from functools import partial

import pytest
import torch
from _harness import expected_sum, rank_tensor, run_failing_init, run_ranks
from _policies import SMALL, SizeSelecting

import gpubridge
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import SPLIT_TEST_ENV, VENDOR_ENV
from gpubridge.policies import (
    _POLICIES,
    AUTO,
    DEFAULT_POLICY,
    CollectivePolicy,
    FlatGloo,
    NativeOnly,
    ReduceBridgeBroadcast,
    choose_policy,
    register_policy,
    resolve_policy,
)
from gpubridge.split_test import split_labels
from gpubridge.topology import plan_topology

#: Cluster layouts: vendors per rank, and the GPUBRIDGE_SPLIT_TEST mode (or None).
LAYOUTS = {
    "mixed-2+2": (["nvidia", "nvidia", "amd", "amd"], None),
    "uneven-3+1": (["nvidia", "nvidia", "nvidia", "amd"], None),
    "interleaved": (["amd", "nvidia", "amd", "nvidia"], None),
    "single-vendor": (["amd", "amd", "amd"], None),
    "single-rank": (["nvidia"], None),
    "split-test": (["nvidia"] * 4, "half"),
}
BUILTIN = {name: cls for name, cls in _POLICIES.items() if cls.__module__ == "gpubridge.policies"}


def layout_of(vendors, split):
    labels = split_labels(vendors, ["host"] * len(vendors), split) if split else vendors
    return plan_topology(labels)


def policy_cases():
    """Every (layout, policy) pair where the policy applies, plus auto everywhere."""
    for layout_id, (vendors, split) in LAYOUTS.items():
        layout = layout_of(vendors, split)
        names = [AUTO, *sorted(n for n, cls in BUILTIN.items() if cls.applies_to(layout))]
        for name in names:
            expected = choose_policy(resolve_policy(name), layout).name
            yield pytest.param(vendors, split, name, expected, id=f"{layout_id}-{name}")


def _check_matches_flat_gloo(topology, expected_policy):
    # On CPU, staging returns the tensor itself, so the copy back that GPUs need
    # would never run. Force a real copy so it does.
    gpubridge.policies._stage_to_host = torch.Tensor.clone
    assert topology.policy_name == expected_policy
    reference = FlatGloo()
    rank, world = topology.rank, topology.world_size
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        tensor = rank_tensor((4, 5), dtype, rank)
        want = tensor.clone()
        gpubridge.all_reduce(tensor)
        reference.all_reduce(want, topology)
        assert torch.equal(tensor, want), f"all_reduce {dtype}"
        assert torch.equal(tensor, expected_sum((4, 5), dtype, world))
        for src in range(world):
            tensor = rank_tensor((7,), dtype, rank)
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            reference.broadcast(want, src, topology)
            assert torch.equal(tensor, want), f"broadcast {dtype} from {src}"


@pytest.mark.parametrize(("vendors", "split", "policy", "expected"), list(policy_cases()))
def test_every_policy_matches_flat_gloo(vendors, split, policy, expected, tmp_path, monkeypatch):
    if split:
        monkeypatch.setenv(SPLIT_TEST_ENV, split)
    run_ranks(vendors, partial(_check_matches_flat_gloo, expected_policy=expected), tmp_path,
              init_kwargs={"policy": policy})


# ---- what reduce-bridge-broadcast sends ------------------------------------------------

def _count_island_ops(topology):
    """Counts every island-group call one all_reduce makes, by torch.distributed function."""
    import torch.distributed as dist

    calls = []
    names = ("all_reduce", "reduce", "broadcast")
    reals = {name: getattr(dist, name) for name in names}
    for name in names:
        def counting(tensor, *args, _name=name, _real=reals[name], **kwargs):
            if kwargs.get("group") is topology.island_group:
                calls.append((_name, kwargs.get("dst", kwargs.get("src"))))
            return _real(tensor, *args, **kwargs)

        setattr(dist, name, counting)
    tensor = rank_tensor((6,), torch.float32, topology.rank)
    try:
        gpubridge.all_reduce(tensor)
    finally:
        # Put torch's functions back. Left on the module, the wrappers keep the topology,
        # and with it a simulated island's Gloo group and its threads, alive until the
        # interpreter finalizes, where a Gloo thread that has yet to release its last
        # collective aborts the rank (torch's exit abort: 2 of 80 runs of [1+3] on
        # Python 3.12 with torch 2.15). Released when the rank's function returns, the
        # group joins its threads first.
        for name in names:
            setattr(dist, name, reals[name])
    assert torch.equal(tensor, expected_sum((6,), torch.float32, topology.world_size))
    leader = topology.island.leader
    # One reduce onto the leader, one broadcast from it; no island all_reduce.
    assert calls == [("reduce", leader), ("broadcast", leader)]


@pytest.mark.parametrize("vendors", [["nvidia", "nvidia", "amd", "amd"],
                                     ["amd", "nvidia", "nvidia", "nvidia"]],
                         ids=["2+2", "1+3"])
def test_reduce_bridge_broadcast_reduces_onto_the_leader(vendors, tmp_path):
    run_ranks(vendors, _count_island_ops, tmp_path,
              init_kwargs={"policy": "reduce-bridge-broadcast"})


# ---- choosing and registering policies ---------------------------------------------

def test_auto_is_the_default_and_keeps_todays_behavior():
    assert DEFAULT_POLICY == AUTO
    assert choose_policy(AUTO, layout_of(["nvidia", "amd"], None)) is ReduceBridgeBroadcast
    assert choose_policy(AUTO, layout_of(["amd", "amd"], None)) is NativeOnly
    assert choose_policy(AUTO, layout_of(["nvidia"] * 4, "half")) is ReduceBridgeBroadcast


@pytest.mark.parametrize(
    ("policy", "vendors"),
    [(NativeOnly, ["nvidia", "amd"]), (ReduceBridgeBroadcast, ["amd", "amd"]),
     (ReduceBridgeBroadcast, ["nvidia"])],
)
def test_a_policy_that_does_not_fit_the_layout_is_refused(policy, vendors):
    with pytest.raises(RuntimeError, match=f"{policy.name!r} doesn't apply"):
        choose_policy(policy, layout_of(vendors, None))


def test_flat_gloo_fits_every_layout():
    assert all(FlatGloo.applies_to(layout_of(v, s)) for v, s in LAYOUTS.values())


def test_resolve_accepts_auto_names_and_classes():
    assert resolve_policy(AUTO) == AUTO
    assert resolve_policy("flat-gloo") is FlatGloo
    assert resolve_policy(NativeOnly) is NativeOnly
    with pytest.raises(ValueError, match="known: auto, auto-tuned, flat-gloo"):
        resolve_policy("chunked")
    with pytest.raises(TypeError):
        resolve_policy(3)


def test_register_policy_rules():
    with pytest.raises(TypeError):
        register_policy(dict)

    class Auto(FlatGloo):
        name = "auto"

    with pytest.raises(ValueError, match="reserved"):
        register_policy(Auto)

    class Impostor(FlatGloo):
        name = "native-only"

    with pytest.raises(ValueError, match="already registered"):
        register_policy(Impostor)
    assert register_policy(SizeSelecting) is SizeSelecting  # re-registering is fine


def test_collective_policy_is_abstract():
    with pytest.raises(TypeError):
        CollectivePolicy()


def test_init_rejects_an_unknown_policy_before_joining():
    with pytest.raises(ValueError, match="unknown collective policy"):
        gpubridge.init(policy="chunked")
    assert not gpubridge.is_initialized()


def test_ranks_with_different_policies_fail_init_on_every_rank(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia"}, "init_kwargs": {"policy": "auto"}},
         {"env": {VENDOR_ENV: "amd"}, "init_kwargs": {"policy": "flat-gloo"}}],
        tmp_path,
        [r"different collective policies", r"'auto' on ranks \[0\]",
         r"'flat-gloo' on ranks \[1\]"],
    )


@pytest.mark.parametrize(
    ("vendors", "policy"),
    [(["nvidia", "amd"], "native-only"), (["amd", "amd"], "reduce-bridge-broadcast")],
)
def test_a_policy_that_does_not_fit_fails_init_on_every_rank(vendors, policy, tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: v}, "init_kwargs": {"policy": policy}} for v in vendors],
        tmp_path,
        [rf"'{policy}' doesn't apply to this cluster",
         r"Policies that do: auto, auto-tuned, flat-gloo"],
    )


# ---- hooks ----------------------------------------------------------------------------

def _check_hooks(topology):
    policy = topology.policy
    assert isinstance(policy, SizeSelecting)
    assert policy.calls["setup"] == 1
    small = torch.ones(SMALL // 4 - 1)          # fewer than SMALL bytes of float32
    large = torch.ones(SMALL // 4)
    gpubridge.all_reduce(small)
    gpubridge.all_reduce(large)
    gpubridge.broadcast(small, 0)
    world = topology.world_size
    assert torch.equal(small, torch.full_like(small, world))
    assert torch.equal(large, torch.full_like(large, world))
    assert policy.calls["small"] == 2 and policy.calls["large"] == 1
    gpubridge.destroy()
    assert policy.closed


def test_select_policy_setup_and_close_hooks(tmp_path):
    run_ranks(["nvidia", "amd", "amd"], _check_hooks, tmp_path,
              init_kwargs={"policy": SizeSelecting})


# ---- dtype check --------------------------------------------------------------------

UNSUPPORTED = [torch.bool, torch.int16, torch.complex64]
UNSUPPORTED += [getattr(torch, n) for n in ("float8_e4m3fn", "uint16") if hasattr(torch, n)]


def test_supported_dtypes_table():
    assert set(SUPPORTED_DTYPES) == {torch.float16, torch.bfloat16, torch.float32,
                                     torch.float64, torch.int8, torch.uint8, torch.int32,
                                     torch.int64}


def _check_dtypes(topology):
    rank, world = topology.rank, topology.world_size
    for dtype in SUPPORTED_DTYPES:
        tensor = rank_tensor((3,), dtype, rank)  # small integers: no 8-bit overflow
        gpubridge.all_reduce(tensor)
        assert torch.equal(tensor, expected_sum((3,), dtype, world)), dtype
    for dtype in UNSUPPORTED:
        with pytest.raises(ValueError, match="doesn't support"):
            gpubridge.all_reduce(torch.zeros(3, dtype=dtype))
        with pytest.raises(ValueError, match="doesn't support"):
            gpubridge.broadcast(torch.zeros(3, dtype=dtype), 0)
    # Every rank rejected the calls before communicating, so the job still works.
    tensor = torch.ones(2)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.full((2,), float(world)))


@pytest.mark.parametrize("policy", ["auto", "flat-gloo"])
def test_dtype_check_is_the_same_for_every_policy(policy, tmp_path):
    run_ranks(["nvidia", "amd", "amd"], _check_dtypes, tmp_path, init_kwargs={"policy": policy})
