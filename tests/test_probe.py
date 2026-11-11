"""probe() never raises, and every rank's probe and problems are shared during discovery."""

import json
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _harness import default_bridge, run_failing_init, run_ranks

from gpubridge.config import CPU_ONLY_ENV, VENDOR_ENV
from gpubridge.detect import probe
from gpubridge.topology import PeerInfo, validate_peers


def test_probe_reports_this_process():
    result = probe()
    assert result.torch_version == torch.__version__
    assert result.gloo_available == dist.is_gloo_available()
    assert result.nccl_available == dist.is_nccl_available()
    assert result.gpu_count == torch.cuda.device_count()
    assert result.errors == ()


@pytest.mark.parametrize(
    ("hip", "cuda", "vendor"),
    [("7.2.53211", None, "amd"), (None, "13.0", "nvidia"), (None, None, None)],
)
def test_probe_reads_the_build(monkeypatch, hip, cuda, vendor):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    result = probe()
    assert result.build_vendor == vendor
    assert (result.hip_version, result.cuda_version) == (hip, cuda)


def test_probe_never_raises(monkeypatch):
    class BrokenVersion:  # stands in for torch.version
        cuda = "13.0"

        @property
        def hip(self):
            raise RuntimeError("version table corrupted")

    def boom():
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch, "version", BrokenVersion())
    monkeypatch.setattr(torch.cuda, "device_count", boom)
    monkeypatch.setattr(dist, "is_nccl_available", boom)
    result = probe()
    assert result.build_vendor == "nvidia"
    assert result.gpu_count == 0
    assert result.nccl_available is False
    assert [error.split(":")[0] for error in result.errors] == [
        "torch.version.hip",
        "torch.cuda.device_count",
        "dist.is_nccl_available",
    ]
    assert "driver exploded" in result.errors[1]


def _peer(rank_problems=(), hostname="node0", bridge="gloo"):
    return PeerInfo("nvidia", True, hostname, bridge, probe(), tuple(rank_problems))


def test_peer_info_wire_format_is_plain_data():
    peer = _peer(["no GPU"])
    # JSON turns tuples into lists, so this also checks from_wire restores them.
    assert PeerInfo.from_wire(json.loads(json.dumps(peer.to_wire()))) == peer


def test_problems_on_any_rank_fail_validation_with_every_problem_listed():
    peers = [_peer(), _peer(["no GPU", "no NCCL"], "node1"), _peer(["no vendor"], "node2")]
    with pytest.raises(RuntimeError) as info:
        validate_peers(peers)
    message = str(info.value)
    assert "2 of 3 ranks" in message
    assert "rank 1 (node1): no GPU" in message
    assert "rank 1 (node1): no NCCL" in message
    assert "rank 2 (node2): no vendor" in message


def _check_probes_shared(topology, vendors):
    assert len(topology.peers) == len(vendors)
    for peer, vendor in zip(topology.peers, vendors, strict=True):
        assert peer.problems == ()
        assert peer.bridge == default_bridge()  # what each rank asked for
        assert peer.probe.torch_version == torch.__version__
        assert peer.probe.gloo_available
        assert peer.probe.build_vendor == vendor  # each rank's own (faked) build


def test_every_rank_sees_every_probe(tmp_path):
    vendors = ["nvidia", "amd", "nvidia"]
    run_ranks(vendors, partial(_check_probes_shared, vendors=vendors), tmp_path,
              fake_builds=True)


def test_a_rank_without_a_vendor_fails_init_on_every_rank(tmp_path):
    run_failing_init(
        [
            {"env": {VENDOR_ENV: "nvidia"}},
            {"env": {CPU_ONLY_ENV: "1"}, "cuda": None, "hip": None},  # CPU-only build
            {"env": {VENDOR_ENV: "amd"}},
        ],
        tmp_path,
        [r"1 of 3 ranks reported problems", r"rank 1 \(", CPU_ONLY_ENV],
    )


def test_an_out_of_range_local_rank_fails_init_on_every_rank(tmp_path):
    # Rank 1 looks like a CUDA build on a node with 1 visible GPU, but torchrun
    # gave it LOCAL_RANK=3. torch.cuda.set_device(3) used to crash it alone
    # (on this CPU build of torch it would raise), leaving ranks 0 and 2 stuck
    # in rendezvous. Now discovery reports it, and every rank fails together.
    one_gpu = {"env": {"LOCAL_RANK": "3"}, "cuda": "13.0", "hip": None, "gpu": True, "gpus": 1}
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia"}}, one_gpu, {"env": {VENDOR_ENV: "amd"}}],
        tmp_path,
        [r"rank 1 \([^)]*\): LOCAL_RANK=3 asks for GPU 3, but this process sees only 1 GPU\.",
         r"CUDA_VISIBLE_DEVICES", r"HIP_VISIBLE_DEVICES", r"--gpus-per-task"],
    )


def test_ranks_without_gpus_fail_init_together(tmp_path):
    no_gpu_cuda_build = {"cuda": "13.0", "hip": None, "gpu": False}
    run_failing_init(
        [no_gpu_cuda_build, no_gpu_cuda_build],
        tmp_path,
        [r"2 of 2 ranks", r"rank 0 \([^)]*\): No GPU is visible",
         r"rank 1 \([^)]*\): No GPU is visible"],
    )
