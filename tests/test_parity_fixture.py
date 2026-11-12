"""Every bridge reproduces the Gloo-generated parity hashes, in a job with no Gloo group.

``tests/data/bridge_parity.json`` (written once by ``python tests/_parity.py generate``
from ``bridge="gloo"`` jobs; ``tests/_parity.py`` describes the cases) holds the SHA-256
of each case's result. Each job here is that many one-rank islands on one bridge; every
leader must get every case's hash, and after its last collective the job must hold no
Gloo group and no Gloo thread. The verbs bridge runs on its MOCK test fabric: the
collectives, chunking, rendezvous and reductions are the bridge's own, the wire is not
RDMA.
"""

from __future__ import annotations

import shutil
import socket
import tempfile

import _parity
import pytest
import torch

from gpubridge import multitcp, ucx, verbs
from gpubridge.multitcp.transport import IFNAME_ENV

LOOPBACK = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))


def _tcp():
    multitcp.build()
    return {IFNAME_ENV: LOOPBACK}, None


def _verbs():
    verbs.build(test_fabric=True)
    fabric = tempfile.mkdtemp(prefix="gbvf-", dir="/tmp")  # short: a Unix socket path
    return {verbs.TEST_FABRIC_ENV: fabric}, fabric


def _ucx():
    ucx.build()
    return {"UCX_TLS": "tcp,self"}, None


BRIDGES = {"tcp": _tcp, "verbs": _verbs, "ucx": _ucx}


def test_the_fixture_is_the_one_this_code_reads_and_was_generated_by_gloo():
    data = _parity.load()
    assert data["generated"]["bridge"] == "gloo"
    keys = [key for leaders in _parity.LEADERS for key in _parity.cases(leaders)]
    assert sorted(data["cases"]) == sorted(keys) and len(set(keys)) == len(keys)
    assert sorted(data["ring"]) == sorted(k for k in keys if k.startswith("ring/"))
    assert data["limits"] == {"rndv_bytes": _parity.RNDV_BYTES, "chunk_bytes": _parity.CHUNK_BYTES}
    # Every dtype, every op it has, every leader count, and a size on each side of
    # both limits.
    for name, dtype in _parity.DTYPES.items():
        counts = _parity.sizes(dtype.itemsize)
        for limit in (_parity.RNDV_BYTES, _parity.CHUNK_BYTES):
            at = limit // dtype.itemsize
            assert {at - 1, at, at + 1} <= set(counts)
        for leaders in _parity.LEADERS:
            for op in ("sum", "max", "min") + (("avg",) if name in _parity.FLOATS else ()):
                assert all(f"all_reduce/{name}/{op}/{leaders}/{n}" in data["cases"]
                           for n in counts)


def test_inputs_are_the_same_bits_everywhere():
    # Pinned: the inputs come from integer arithmetic alone, so these never change with
    # the platform or the torch release. A change here means the fixture no longer
    # describes what the jobs run.
    assert _parity.digest(_parity.make_input("all_reduce/float32/sum/2/7", 0)) == \
        _parity.load()["inputs"]["all_reduce/float32/sum/2/7@0"]
    for key, want in _parity.load()["inputs"].items():
        case, leader = key.rsplit("@", 1)
        assert _parity.digest(_parity.make_input(case, int(leader))) == want, key
    for name in _parity.FLOATS:  # finite, and of the magnitudes the docs promise
        values = _parity.make_input(f"all_reduce/{name}/max/3/4099", 1).double().abs()
        assert torch.isfinite(values).all() and values.min() >= 2**-4 and values.max() < 2**4


def test_a_ring_reference_is_the_sum_in_ring_order():
    key = "ring/float32/sum/3/7"
    inputs = [_parity.make_input(key, leader) for leader in range(3)]
    got = _parity.ring_reference(key)
    # Segments of 3, 2 and 2 elements; segment s starts at leader s.
    assert torch.equal(got[:3], (inputs[0][:3] + inputs[1][:3]) + inputs[2][:3])
    assert torch.equal(got[3:5], (inputs[1][3:5] + inputs[2][3:5]) + inputs[0][3:5])
    assert torch.equal(got[5:], (inputs[2][5:] + inputs[0][5:]) + inputs[1][5:])


@pytest.mark.parametrize("leaders", _parity.LEADERS)
@pytest.mark.parametrize("bridge", list(BRIDGES))
def test_a_bridge_reproduces_every_hash_with_no_gloo_group_in_the_job(bridge, leaders, tmp_path):
    try:
        env, scratch = BRIDGES[bridge]()
    except (multitcp.Unavailable, verbs.Unavailable, ucx.Unavailable) as exc:
        pytest.skip(f"the {bridge} bridge is unavailable: {str(exc).splitlines()[0]}")
    try:
        got = _parity.run_job(leaders, bridge, tmp_path / "job", env, no_gloo=True)
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    wrong = _parity.compare(got, leaders, ring=True)
    assert wrong == [], f"{len(wrong)} of {len(_parity.cases(leaders))} cases differ: {wrong[:10]}"


@pytest.mark.parametrize("leaders", _parity.LEADERS)
def test_the_gloo_bridge_still_gives_the_hashes_it_generated(leaders, tmp_path):
    got = _parity.run_job(leaders, "gloo", tmp_path / "job")
    wrong = _parity.compare(got, leaders, ring=False)
    assert wrong == [], f"{len(wrong)} cases differ from the fixture: {wrong[:10]}"
