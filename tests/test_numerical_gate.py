"""Software-only four-process gates and deliberately incorrect numerical controls."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from _harness import run_ranks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/gpu"))
import numerical_gate as gate  # noqa: E402


def _check(topology):
    for optimizer in ("sgd", "adamw"):
        positive = gate.run_gate(topology, arm="gloo", software_only=True,
                                 opt_name=optimizer, sizes=(17,))
        assert positive["passed"], positive
        for control, prefix in (("sum", "avg/"), ("scale", "gradient/"), ("data", "gradient/")):
            negative = gate.run_gate(topology, arm="gloo", software_only=True,
                                     opt_name=optimizer, control=control, sizes=(17,))
            assert not negative["passed"], negative
            assert any(c["violations"] for c in negative["checks"] if c["name"].startswith(prefix))
            if control == "data" and topology.rank == 0:
                assert all(x["actual"] != x["expected"] for x in negative["inputs"])


#: The gate's software arm is the Gloo bridge, by name: it checks the job ran on it.
GLOO = {"bridge": "gloo"}


def test_independent_updates_and_reduced_negative_controls(tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check, tmp_path, init_kwargs=GLOO)


def _save(topology, directory):
    result = gate.run_gate(topology, arm="gloo", software_only=True, sizes=(17,))
    assert result["passed"], result
    (Path(directory) / f"rank{topology.rank}.json").write_text(json.dumps(result))


def test_comparison_requires_complete_matching_provenance_and_numeric_values(tmp_path):
    from functools import partial

    run_ranks(["nvidia", "nvidia", "amd", "amd"], partial(_save, directory=str(tmp_path)), tmp_path,
              init_kwargs=GLOO)
    # Copies test the report comparator only; these are not three independent arms.
    base = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    records = []
    for arm in gate.ARMS:
        for rank in range(4):
            record = copy.deepcopy(base[rank])
            record["arm"] = arm
            record["transport"] = arm
            if arm == "native":
                record["policy"] = "native"
                record["layout"] = [[0, 1, 2, 3]]
            records.append(record)
    assert gate.compare(records)["passed"]
    assert not gate.compare(records[:-1])["passed"]
    for key, value in (("run_id", "stale"), ("seed", 9), ("initial_sha256", "different")):
        bad = copy.deepcopy(records)
        bad[-1][key] = value
        assert not gate.compare(bad)["passed"]
    bad = copy.deepcopy(records)
    bad[-1]["snapshots"][0]["gradients"][0] += 0.1
    assert not gate.compare(bad)["passed"]
    bad = copy.deepcopy(records)
    bad[-1]["checks"] = []
    assert not gate.compare(bad)["passed"]

    hardware_leader = copy.deepcopy(base[0])
    hardware_leader.update(arm="verbs", transport="verbs", software_only=False,
                           device="cuda:0", gpu_identity="GPU-example", island_backend="nccl",
                           probe_sizes=list(gate.SIZES))
    hardware_leader["source"]["dirty"] = False
    assert "missing actual verbs endpoint/binary evidence" in gate.record_errors(hardware_leader)


def test_provider_evidence_cannot_accept_mock_or_missing_completion_counts():
    after = {"mock": False, "provider": "libibverbs (/usr/lib/libibverbs.so)",
             "state": "ok", "completion_errors": 0, "completions": 3,
             "bytes_sent": 400, "bytes_received": 400, "gpu_path_setting": "staged",
             "paths": {"staged": 1}, "rndv_bytes": 256 * 1024, "chunk_bytes": 4 * 1024**2}
    assert gate.provider_errors({}, after) == []
    for key, value in (("mock", True), ("provider", "MOCK fabric"),
                       ("completion_errors", 1), ("completions", 0), ("paths", {})):
        assert gate.provider_errors({}, {**after, key: value})


def test_exact_and_update_bounds_detect_wrong_values_and_missed_updates():
    assert gate.metric([1.0], [1.0], exact=True)["violations"] == 0
    assert gate.metric([1.0], [1.0 + 1e-8], exact=True)["violations"] == 1
    assert gate.metric([0.0], [1e-5], update=True)["violations"] == 1
    assert gate.metric([float("nan")], [1.0])["violations"] == 1


def test_hardware_entrypoint_refuses_a_mock_before_initialization(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBRIDGE_VERBS_TEST_FABRIC", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit) as error:
        gate.main(["--arm", "verbs", "--run-id", "test", "--out", str(tmp_path / "out")])
    assert error.value.code == 2
