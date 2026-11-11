"""The bridge transport is swappable; the default is ``auto``, and Gloo is opt-in."""

import json
import subprocess
import sys

import pytest
import torch
from _harness import expected_sum, rank_tensor, run_failing_init, run_ranks
from _transports import STORE_ENV, StoreTransport

import gpubridge
from gpubridge.config import BRIDGE_ENV, VENDOR_ENV
from gpubridge.transport import (
    DEFAULT_TRANSPORT,
    BridgeTransport,
    GlooTransport,
    register_transport,
    resolve_transport,
    transport_name,
)


def test_auto_is_the_default_and_gloo_is_still_a_name():
    assert DEFAULT_TRANSPORT == "auto"
    assert resolve_transport("auto") == "auto" and transport_name("auto") == "auto"
    assert resolve_transport("gloo") is GlooTransport
    assert transport_name(GlooTransport) == "gloo"


def test_multitcp_is_another_name_of_the_tcp_bridge():
    tcp = resolve_transport("tcp")
    assert tcp.name == "tcp" and resolve_transport("multitcp") is tcp
    assert tcp.__module__ == "gpubridge.multitcp.transport"


def test_resolve_accepts_names_and_classes():
    assert resolve_transport("test-store") is StoreTransport
    assert resolve_transport(StoreTransport) is StoreTransport
    with pytest.raises(ValueError, match="known: auto, gloo"):
        resolve_transport("carrier-pigeon")
    with pytest.raises(TypeError):
        resolve_transport(42)


#: Modules of the opt-in bridges, which ``import gpubridge`` doesn't load.
OPT_IN = ("gpubridge.ucx", "gpubridge.multitcp", "gpubridge.verbs", "gpubridge.tagged")


def _fresh(code: str) -> list:
    """Run ``code`` in a new interpreter and return the JSON it prints: this one has
    imported every bridge already (pytest imports every test module)."""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_importing_gpubridge_loads_no_opt_in_bridge_and_a_name_loads_its_own():
    before, module, after, registered = _fresh(f"""
import json, sys
import gpubridge
loaded = lambda: sorted(m for m in sys.modules if m.startswith({OPT_IN!r}))
before = loaded()
module = gpubridge.resolve_transport("verbs").__module__
print(json.dumps([before, module, loaded(), sorted(gpubridge.transport._TRANSPORTS)]))
""")
    assert before == []
    assert module == "gpubridge.verbs.transport"
    assert "gpubridge.verbs.transport" in after
    assert not [m for m in after if m.startswith("gpubridge.multitcp")]
    assert "tcp" not in registered and {"gloo", "verbs"} <= set(registered)


@pytest.mark.parametrize(("name", "package"), [("ucx", "ucx"), ("tcp", "multitcp"),
                                               ("verbs", "verbs")])
def test_a_built_in_name_stays_the_built_ins_before_it_loads(name, package):
    refused, owner = _fresh(f"""
import json
from gpubridge.transport import BridgeTransport, register_transport, resolve_transport
class Mine(BridgeTransport):
    name = {name!r}
try:
    register_transport(Mine)
    refused = None
except ValueError as exc:
    refused = str(exc)
print(json.dumps([refused, resolve_transport({name!r}).__module__]))
""")
    assert "already registered" in refused
    assert owner == f"gpubridge.{package}.transport"


@pytest.mark.parametrize(("name", "why"), [("auto", "reserved for automatic selection"),
                                           ("multitcp", "another name of the built-in 'tcp'")])
def test_auto_and_an_older_name_cannot_be_registered(name, why):
    class Mine(StoreTransport):
        pass

    Mine.name = name
    with pytest.raises(ValueError, match=why):
        register_transport(Mine)


def test_unknown_names_list_the_built_ins_and_a_broken_one_is_a_value_error(monkeypatch):
    with pytest.raises(ValueError, match="known: auto, gloo, tcp, .*ucx, verbs"):
        resolve_transport("carrier-pigeon")
    monkeypatch.setitem(gpubridge.transport._BUILT_IN, "broken", "gpubridge._no_such_module")
    with pytest.raises(ValueError, match="bridge transport 'broken' can't be loaded"):
        resolve_transport("broken")


def test_register_rejects_non_transports_and_name_clashes():
    with pytest.raises(TypeError):
        register_transport(object)

    class Impostor(StoreTransport):
        name = "gloo"

    with pytest.raises(ValueError, match="already registered"):
        register_transport(Impostor)
    assert register_transport(StoreTransport) is StoreTransport  # re-registering is fine


def _check_gloo_bridge(topology):
    if topology.layout.needs_bridge and topology.is_leader:
        assert isinstance(topology.bridge, GlooTransport)
        assert topology.bridge_group is topology.bridge.group
    else:
        assert topology.bridge is None
        assert topology.bridge_group is None
    assert all(peer.bridge == "gloo" for peer in topology.peers)
    assert topology.bridge_name == "gloo" and not topology.gloo_free
    assert topology.selection == {"requested": "gloo", "chosen": "gloo",
                                  "reason": "asked for by name"}
    assert torch.distributed.get_backend() == "gloo"  # the default group, as through 0.4.0-alpha


@pytest.mark.parametrize("vendors", [["nvidia", "amd", "nvidia", "amd"], ["amd", "amd"]],
                         ids=["mixed", "single-vendor"])
@pytest.mark.parametrize("how", ["argument", "environment"])
def test_the_gloo_bridge_when_it_is_asked_for(vendors, how, tmp_path, monkeypatch):
    # By name, or for a script that passes none, by GPUBRIDGE_BRIDGE. (The default
    # bridge is tests/test_bridge_selection.py's.)
    monkeypatch.setenv(BRIDGE_ENV, "gloo" if how == "environment" else "test-store")
    run_ranks(vendors, _check_gloo_bridge, tmp_path,
              init_kwargs={"bridge": "gloo"} if how == "argument" else {})


def _check_store_bridge(topology):
    cases = [((3, 4), torch.float32), ((5,), torch.float16)]
    for shape, dtype in cases:
        tensor = rank_tensor(shape, dtype, topology.rank)
        gpubridge.all_reduce(tensor)
        assert torch.equal(tensor, expected_sum(shape, dtype, topology.world_size))
    for src in range(topology.world_size):
        tensor = rank_tensor((3, 4), torch.float32, topology.rank)
        gpubridge.broadcast(tensor, src)
        assert torch.equal(tensor, rank_tensor((3, 4), torch.float32, src))

    if topology.is_leader:
        # Every bridge hop went through the store transport, none through Gloo.
        assert isinstance(topology.bridge, StoreTransport)
        assert topology.bridge_group is None
        assert topology.bridge.calls == {"all_reduce": len(cases),
                                         "broadcast": topology.world_size}
    else:
        assert topology.bridge is None


@pytest.mark.parametrize(
    ("vendors", "bridge"),
    [
        (["nvidia", "nvidia", "amd", "amd"], StoreTransport),
        (["nvidia", "nvidia", "nvidia", "amd"], StoreTransport),
        (["amd", "nvidia", "amd"], "test-store"),
    ],
    ids=["2+2-class", "3+1-class", "interleaved-by-name"],
)
def test_collectives_run_over_a_swapped_in_transport(vendors, bridge, tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    run_ranks(vendors, _check_store_bridge, tmp_path, init_kwargs={"bridge": bridge})


def test_ranks_with_different_transports_fail_init_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    run_failing_init(
        [
            {"env": {VENDOR_ENV: "nvidia"}, "init_kwargs": {"bridge": "gloo"}},
            {"env": {VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": StoreTransport}},
        ],
        tmp_path,
        [r"different bridge transports", r"'gloo' on ranks \[0\]",
         r"'test-store' on ranks \[1\]"],
    )


def test_bridge_transport_is_abstract():
    with pytest.raises(TypeError):
        BridgeTransport()


def test_init_rejects_an_unknown_bridge_before_joining():
    with pytest.raises(ValueError, match="unknown bridge transport"):
        gpubridge.init(bridge="carrier-pigeon")
    assert not gpubridge.is_initialized()
