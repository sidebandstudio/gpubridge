"""The default bridge: ``bridge="auto"`` resolves once, at ``init()``, to a Rust bridge.

``"verbs"`` when every island leader passes the verbs bridge's own check for a usable
RDMA device and they are on one link layer, otherwise ``"tcp"``; never Gloo. The rule
is a pure function of the leaders' reports (:func:`gpubridge.transport.choose_bridge`),
so every rank chooses alike. The jobs here replace the device check with a fixed report
per rank (no test machine has an RDMA device worth depending on), and run the verbs
bridge on its MOCK test fabric where the rule chooses it.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import tempfile
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _harness import assert_no_gloo, expected_sum, rank_tensor, run_failing_init, run_ranks

import gpubridge
from gpubridge import multitcp, verbs
from gpubridge.config import BRIDGE_ENV, VENDOR_ENV
from gpubridge.multitcp.transport import IFNAME_ENV, STREAMS_ENV, TcpTransport
from gpubridge.transport import GlooTransport, choose_bridge

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"the Rust TCP bridge is unavailable: {UNAVAILABLE}")
try:
    verbs.build(test_fabric=True)
    NO_FABRIC = None
except verbs.Unavailable as exc:
    NO_FABRIC = str(exc).splitlines()[0]
needs_fabric = pytest.mark.skipif(NO_FABRIC is not None,
                                  reason=f"the verbs test fabric is unavailable: {NO_FABRIC}")

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
LOCAL = {IFNAME_ENV: LOOPBACK, STREAMS_ENV: "2"}
NO_DEVICE = {"ok": False, "devices": [], "reason": "no RDMA device (test)"}
INFINIBAND = {"ok": True, "reason": "", "devices": ["mlx5_0"], "device": "mlx5_0", "port": 1,
              "link_layer": "InfiniBand"}
ROCE = {**INFINIBAND, "link_layer": "Ethernet"}


@pytest.fixture(autouse=True)
def _the_default(monkeypatch):
    """These tests are about a job that names no bridge, whatever the suite runs with."""
    monkeypatch.delenv(BRIDGE_ENV, raising=False)


# ---- the rule ---------------------------------------------------------------------------

def test_verbs_when_every_leader_has_a_usable_device_on_one_link_layer():
    name, reason = choose_bridge([0, 2], {0: INFINIBAND, 2: INFINIBAND})
    assert name == "verbs" and "InfiniBand" in reason


def test_tcp_when_any_leader_has_no_usable_device():
    name, reason = choose_bridge([0, 2, 5], {0: INFINIBAND, 2: NO_DEVICE, 5: INFINIBAND})
    assert name == "tcp"
    assert "rank 2: no RDMA device (test)" in reason and "rank 0" not in reason


def test_tcp_when_the_leaders_are_on_different_link_layers():
    # Explorer's shape: InfiniBand on the V100 nodes, RoCE on the MI50 node.
    name, reason = choose_bridge([0, 4], {0: INFINIBAND, 4: ROCE})
    assert name == "tcp"
    assert "different link layers" in reason and "rank 4: Ethernet" in reason


def test_the_rule_reads_only_the_leaders_reports():
    reports = {0: INFINIBAND, 1: NO_DEVICE, 2: INFINIBAND}  # rank 1 is no leader
    assert choose_bridge([0, 2], reports)[0] == "verbs"
    assert choose_bridge((0, 2), dict(reversed(reports.items()))) == choose_bridge([0, 2], reports)


def test_a_machine_without_an_rdma_device_reports_so_without_loading_anything(monkeypatch):
    monkeypatch.delenv(verbs.TEST_FABRIC_ENV, raising=False)
    monkeypatch.setattr(verbs, "rdma_devices", lambda: [])
    monkeypatch.setattr(verbs, "load", lambda **_: pytest.fail("the library was loaded"))
    report = verbs.availability()
    assert report["ok"] is False and "no RDMA device" in report["reason"]


def test_the_mock_test_fabric_is_never_an_rdma_device(monkeypatch, tmp_path):
    monkeypatch.setenv(verbs.TEST_FABRIC_ENV, str(tmp_path))
    monkeypatch.setattr(verbs, "rdma_devices", lambda: ["mlx5_0"])
    report = verbs.availability()
    assert report["ok"] is False and "MOCK test fabric" in report["reason"]


def test_a_device_the_bridge_cannot_open_is_not_available(monkeypatch):
    monkeypatch.delenv(verbs.TEST_FABRIC_ENV, raising=False)
    monkeypatch.setattr(verbs, "rdma_devices", lambda: ["mlx5_0"])

    def refuse(*args, **kwargs):
        raise verbs.TransportError("no active port on mlx5_0\nmore detail")

    monkeypatch.setattr(verbs, "Endpoint", refuse)
    report = verbs.availability()
    assert report == {"ok": False, "devices": ["mlx5_0"], "reason": "no active port on mlx5_0"}


# ---- jobs -------------------------------------------------------------------------------

class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[tuple[int, str]] = []

    def emit(self, record):
        self.lines.append((record.levelno, record.getMessage()))


def _before(reports, env, rank):
    """Before init: the settings, a fixed device report for this rank in place of the
    probe, and a handler that keeps what gpubridge logs."""
    os.environ.update(env)
    verbs.availability = lambda: reports[rank]
    handler = _Records()
    logger = logging.getLogger("gpubridge")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    gpubridge._test_log = handler


def _local(rank):
    os.environ.update(LOCAL)


def _nothing_may_load(rank):
    verbs.availability = lambda: pytest.fail("a device was checked")
    multitcp.load = lambda **_: pytest.fail("the TCP library was loaded")


def _collectives(topology):
    rank, world = topology.rank, topology.world_size
    tensor = rank_tensor((257,), torch.float32, rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((257,), torch.float32, world))
    tensor = rank_tensor((5,), torch.int64, rank)
    gpubridge.broadcast(tensor, src=world - 1)
    assert torch.equal(tensor, rank_tensor((5,), torch.int64, world - 1))


def _check_choice(topology, chosen, why, leaders):
    selection = topology.selection
    assert selection["requested"] == "auto" and selection["chosen"] == chosen
    assert why in selection["reason"]
    assert sorted(selection["leaders"]) == [str(r) for r in leaders]
    assert topology.bridge_name == chosen and topology.transport.name == chosen
    assert all(peer.bridge == "auto" for peer in topology.peers)  # what each rank asked for
    assert not issubclass(topology.transport, GlooTransport) and topology.gloo_free
    if topology.is_leader:
        assert topology.bridge.name == chosen
        assert topology.bridge.telemetry()["selection"] == selection
    else:
        assert topology.bridge is None
    # Logged at startup, on every rank.
    lines = [text for _, text in gpubridge._test_log.lines]
    assert any(f"bridge {chosen!r}" in line and why in line for line in lines), lines
    # The same on every rank: the choice, its reason and the reports it came from.
    everyone: list = [None] * topology.world_size
    dist.all_gather_object(everyone, selection)
    assert all(other == selection for other in everyone)
    bridge, transport = topology.bridge, topology.transport
    _collectives(topology)
    assert topology.selection == selection and topology.transport is transport
    assert topology.bridge is bridge  # never changed mid-job


@needs_library
@pytest.mark.parametrize(("vendors", "reports", "why"), [
    (["nvidia", "amd"], [NO_DEVICE, NO_DEVICE], "rank 0: no RDMA device (test); rank 1: no"),
    (["nvidia", "amd"], [INFINIBAND, NO_DEVICE], "(rank 1: no RDMA device (test))"),
    (["nvidia", "amd"], [INFINIBAND, ROCE], "different link layers"),
    # Rank 1 is not a leader: its device, or lack of one, decides nothing.
    (["nvidia", "nvidia", "amd"], [NO_DEVICE, INFINIBAND, NO_DEVICE], "rank 0: no RDMA device"),
], ids=["no-device", "one-leader-without", "ib-and-roce", "non-leader-ignored"])
def test_auto_is_the_tcp_bridge_unless_every_leader_has_rdma(vendors, reports, why, tmp_path):
    leaders = [vendors.index("nvidia"), vendors.index("amd")]
    run_ranks(vendors, partial(_check_choice, chosen="tcp", why=why, leaders=leaders), tmp_path,
              before_init=partial(_before, reports, LOCAL))


def _check_tcp_and_no_gloo(topology):
    assert isinstance(topology.bridge, TcpTransport)
    told = topology.bridge.telemetry()
    assert told["bridge"] == "tcp" and told["state"] == "ok" and told["streams"] == 2
    assert told["members"] == [0, 1] and told["address"].startswith("127.0.0.1:")
    _collectives(topology)
    assert_no_gloo()


@needs_library
def test_a_default_job_runs_on_tcp_and_has_no_gloo_group_or_thread(tmp_path):
    # No bridge named anywhere, and the real device check: nothing here has an RDMA
    # device the verbs bridge can open, so the default is the TCP bridge.
    run_ranks(["nvidia", "amd"], _check_tcp_and_no_gloo, tmp_path, before_init=_local)


@needs_fabric
def test_auto_is_verbs_when_every_leader_has_a_usable_device(tmp_path):
    # Short: Unix socket paths are limited to about 100 bytes.
    fabric = tempfile.mkdtemp(prefix="gbvf-", dir="/tmp")
    env = {verbs.TEST_FABRIC_ENV: fabric}  # the verbs bridge itself runs on the MOCK
    try:
        run_ranks(["nvidia", "amd", "amd"],
                  partial(_check_choice, chosen="verbs", why="usable RDMA port (InfiniBand)",
                          leaders=[0, 1]),
                  tmp_path, before_init=partial(_before, [INFINIBAND, INFINIBAND, NO_DEVICE], env))
    finally:
        shutil.rmtree(fabric, ignore_errors=True)


def _check_one_island(topology):
    assert topology.bridge is None and topology.bridge_name == "tcp"
    assert topology.selection == {
        "requested": "auto", "chosen": "tcp",
        "reason": "one island: no bridge is created, and no device was checked"}
    tensor = rank_tensor((4,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((4,), torch.float32, topology.world_size))


def test_one_island_checks_no_device_and_needs_no_library(tmp_path):
    run_ranks(["amd", "amd"], _check_one_island, tmp_path, before_init=_nothing_may_load)


def _check_asked_by_name(topology, name):
    assert topology.selection == {"requested": name, "chosen": name,
                                  "reason": "asked for by name"}
    _collectives(topology)


@needs_library
def test_a_bridge_asked_for_by_name_is_never_probed_or_replaced(tmp_path):
    reports = [INFINIBAND, INFINIBAND]  # verbs "would" be available: tcp was asked for
    run_ranks(["nvidia", "amd"], partial(_check_asked_by_name, name="tcp"), tmp_path,
              before_init=partial(_before, reports, LOCAL), init_kwargs={"bridge": "tcp"})


def test_verbs_asked_for_by_name_fails_init_where_it_cannot_run_and_never_drops_to_tcp(
        tmp_path, monkeypatch):
    monkeypatch.delenv(verbs.TEST_FABRIC_ENV, raising=False)
    if verbs.rdma_devices():
        pytest.skip("this machine has an RDMA device: the refusal is for machines without")
    env = {VENDOR_ENV: "nvidia", **LOCAL}
    run_failing_init(
        [{"env": env, "init_kwargs": {"bridge": "verbs"}},
         {"env": {**env, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "verbs"}},
         {"env": env, "init_kwargs": {"bridge": "verbs"}}],
        tmp_path, [r"The verbs bridge between ranks \[0, 1\] could not be set up",
                   r"verbs is never replaced by another transport once it is chosen"])


@needs_library
def test_ranks_must_ask_alike_auto_and_a_name_differ(tmp_path):
    env = {VENDOR_ENV: "nvidia", **LOCAL}
    run_failing_init(
        [{"env": env}, {"env": {**env, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "tcp"}}],
        tmp_path, [r"Ranks picked different bridge transports: 'auto' on ranks \[0\]; "
                   r"'tcp' on ranks \[1\]"])


def _check_env_names_the_bridge(topology):
    assert topology.selection["requested"] == "tcp" and topology.bridge_name == "tcp"


@needs_library
def test_the_environment_names_the_bridge_for_a_script_that_passes_none(tmp_path, monkeypatch):
    monkeypatch.setenv(BRIDGE_ENV, "multitcp")  # the older name: the same bridge
    run_ranks(["nvidia", "amd"], _check_env_names_the_bridge, tmp_path, before_init=_local)
