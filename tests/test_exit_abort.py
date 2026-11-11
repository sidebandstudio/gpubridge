"""Ranks that return straight into interpreter exit: scripts/exit_abort.py, in short.

The full run is 300 repetitions of each arm on Linux with Python 3.12 and torch 2.15,
where a Gloo job of this shape aborts a few times in a hundred
(``python scripts/exit_abort.py --reps 300 --out DIR``; the CHANGELOG has its numbers).
Here the arm under test runs a few times on whatever this is, and must exit 0 every
time, with and without ``destroy()``. ``GPUBRIDGE_EXIT_ABORT_REPS`` raises the count.
The Gloo control arm is not run here: its aborts are expected, not a failure of ours.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpubridge import multitcp

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "exit_abort.py"
REPS = int(os.environ.get("GPUBRIDGE_EXIT_ABORT_REPS", "4"))

try:
    multitcp.build()
    UNAVAILABLE = None
except multitcp.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
pytestmark = pytest.mark.skipif(UNAVAILABLE is not None,
                                reason=f"the Rust TCP bridge is unavailable: {UNAVAILABLE}")


@pytest.mark.parametrize("destroy", [False, True], ids=["no-destroy", "destroy"])
def test_ranks_on_the_tcp_bridge_exit_zero_every_time(destroy, tmp_path):
    command = [sys.executable, str(SCRIPT), "--reps", str(REPS), "--arms", "tcp", "--control",
               "none", "--out", str(tmp_path / "out")] + (["--destroy"] if destroy else [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=120 * REPS + 300)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    arm = summary["arms_under_test"]["tcp"]
    assert arm["reps"] == REPS and summary["job"]["destroy_called"] is destroy
    assert arm["bad"] == 0 and arm["codes"] == {"0": 4 * REPS}, (arm, result.stderr[-2000:])
    assert result.returncode == 0 and summary["passed"] is True
    assert summary["control"] is None
    assert len((tmp_path / "out" / "reps.jsonl").read_text().splitlines()) == REPS


def test_the_script_fails_when_an_arm_under_test_has_a_bad_repetition(tmp_path):
    # A bridge name no rank can resolve: every rank exits 1, and the script says so.
    command = [sys.executable, str(SCRIPT), "--reps", "1", "--arms", "no-such-bridge",
               "--control", "none", "--ranks", "2", "--out", str(tmp_path / "out")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert result.returncode == 1 and summary["passed"] is False
    arm = summary["arms_under_test"]["no-such-bridge"]
    assert arm["bad"] == 1 and arm["bad_reps"] == [0] and arm["codes"] == {"1": 2}
    stderr = (tmp_path / "out" / "stderr" / "no-such-bridge-rep0-rank0.txt").read_text()
    assert "unknown bridge transport 'no-such-bridge'" in stderr
