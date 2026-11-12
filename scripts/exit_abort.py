"""Does a rank exit cleanly when it returns straight into interpreter exit?

The same small training job, many times: four ranks, each a one-rank island (so every
rank is a bridge leader and the job has no island of several ranks), DDP on the job's
default process group for a few steps, and then each rank returns from its main
function with nothing in between: no pause, no result file, and unless ``--destroy``
no ``gpubridge.destroy()``. Every rank's exit code is recorded. A repetition is bad if
any rank's is not 0.

Why: on Python 3.12 with torch 2.10 to 2.15 a Gloo worker thread can abort the process
at interpreter exit (signal 6, "terminate called without an active exception") when
the job's last Gloo collective ends within a few milliseconds of it. It reproduces with
torch alone, in about 2 to 3 jobs in 100 of exactly this shape. A job on gpubridge's
Rust bridges has no Gloo group and no Gloo thread, so it cannot.

Arms run side by side (one repetition of each in turn, so they see the same machine):

- the arms under test (``--arms``, default ``tcp``): must end with no bad repetition;
- the control (``--control``, default ``gloo``; ``none`` to leave it out): the same
  job on ``bridge="gloo"``. It is expected to abort sometimes on the affected Python
  and torch, which shows this test can see the bug. Its count is reported, never
  judged.

    python scripts/exit_abort.py --reps 300 --out OUT_DIR

writes OUT_DIR/reps.jsonl (one line per repetition and arm: every rank's exit code),
OUT_DIR/stderr/ (what the ranks of a bad repetition wrote) and OUT_DIR/summary.json,
prints the summary, and exits 0 only if no arm under test had a bad repetition.

Islands of one rank are a test arrangement: simulation knows two vendors, so the ranks
are labelled into four islands the way gpubridge's own tests do.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

LABELS = ("nvidia", "amd", "third", "fourth")
ROOT = Path(__file__).resolve().parents[1]


def rank_main(rank: int, world: int, bridge: str, init_file: str, steps: int, destroy: bool,
              stderr: str) -> None:
    """One rank: join, train ``steps`` DDP steps, return into interpreter exit."""
    fd = os.open(stderr, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)  # also what a C++ terminate handler writes
    os.close(fd)
    os.environ["GPUBRIDGE_VENDOR"] = "nvidia"
    os.environ.setdefault("GPUBRIDGE_MULTITCP_IFNAME", next(
        name for _, name in socket.if_nameindex() if name in ("lo", "lo0")))
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel

    import gpubridge

    torch.set_num_threads(1)
    labels = list(LABELS[:world])
    original = gpubridge.topology.plan_topology
    gpubridge.topology.plan_topology = lambda vendors, ranks=(): original(
        [labels[r] for r in ranks] if ranks else labels, ranks)
    topology = gpubridge.init(init_method=f"file://{init_file}", rank=rank, world_size=world,
                              timeout=timedelta(seconds=120), bridge=bridge)
    assert topology.bridge_name == bridge and len(topology.layout.islands) == world
    torch.manual_seed(0)
    model = DistributedDataParallel(nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                                                  nn.Linear(32, 4)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(rank + 1)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(torch.randn(8, 16)).square().mean()
        loss.backward()  # the gradients' all_reduce: the job's last collective
        optimizer.step()
    if destroy:
        gpubridge.destroy()
    # And nothing else: the rank returns straight into interpreter exit.


def run_once(arm: str, world: int, steps: int, destroy: bool, work: Path, limit: float
             ) -> tuple[list[int | str], list[str]]:
    """One job; every rank's exit code (``"hung"`` if it had to be killed) and stderr."""
    ctx = multiprocessing.get_context("spawn")
    work.mkdir(parents=True)
    errs = [work / f"rank{r}.stderr" for r in range(world)]
    ranks = [ctx.Process(target=rank_main, args=(r, world, arm, str(work / "rendezvous"), steps,
                                                 destroy, str(errs[r])))
             for r in range(world)]
    for process in ranks:
        process.start()
    deadline = time.monotonic() + limit
    codes: list[int | str] = []
    for process in ranks:
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join()
            codes.append("hung")
        else:
            codes.append(process.exitcode)
    return codes, [err.read_text(errors="replace") if err.exists() else "" for err in errs]


def commit() -> str:
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                         "src"], text=True, stderr=subprocess.DEVNULL).strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--arms", nargs="+", default=["tcp"], help="bridges under test")
    parser.add_argument("--control", default="gloo",
                        help="the bridge of the control arm, or none (default gloo)")
    parser.add_argument("--ranks", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--destroy", action="store_true",
                        help="each rank calls gpubridge.destroy() before it returns")
    parser.add_argument("--limit", type=float, default=180.0, help="seconds per job")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", help="the gpubridge commit to record (default: git "
                                         "rev-parse HEAD of this checkout)")
    args = parser.parse_args(argv)

    import torch

    import gpubridge

    arms = list(args.arms) + ([] if args.control == "none" else [args.control])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stderr").mkdir(exist_ok=True)
    tally = {arm: {"reps": 0, "bad": 0, "aborts": 0, "hung": 0,
                   "codes": collections.Counter(), "bad_reps": []} for arm in arms}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gpubridge-exit-") as tmp, \
            (args.out / "reps.jsonl").open("w") as log:
        for rep in range(args.reps):
            for arm in arms:  # side by side: one repetition of each arm in turn
                work = Path(tmp) / f"{arm}-{rep}"
                t0 = time.monotonic()
                codes, errs = run_once(arm, args.ranks, args.steps, args.destroy, work,
                                       args.limit)
                bad = any(code != 0 for code in codes)
                entry = tally[arm]
                entry["reps"] += 1
                entry["bad"] += bad
                entry["aborts"] += any(code == -signal.SIGABRT for code in codes)
                entry["hung"] += "hung" in codes
                entry["codes"].update(str(code) for code in codes)
                if bad:
                    entry["bad_reps"].append(rep)
                    for rank, text in enumerate(errs):
                        (args.out / "stderr" / f"{arm}-rep{rep}-rank{rank}.txt").write_text(text)
                log.write(json.dumps({"rep": rep, "arm": arm, "codes": codes,
                                      "seconds": round(time.monotonic() - t0, 2)}) + "\n")
                log.flush()
                shutil.rmtree(work, ignore_errors=True)
            if (rep + 1) % 25 == 0 or rep + 1 == args.reps:
                print(f"{rep + 1}/{args.reps}: " + ", ".join(
                    f"{arm} {tally[arm]['bad']} bad" for arm in arms), flush=True)
    for entry in tally.values():
        entry["codes"] = dict(entry["codes"])
    judged = {arm: tally[arm] for arm in args.arms}
    summary = {
        "what": "exit codes of every rank of the same DDP job, repeated (scripts/exit_abort.py)",
        "passed": all(entry["bad"] == 0 for entry in judged.values()),
        "arms_under_test": judged,
        "control": None if args.control == "none" else {args.control: tally[args.control]},
        "job": {"ranks": args.ranks, "islands": args.ranks, "steps": args.steps,
                "destroy_called": args.destroy, "reps_per_arm": args.reps},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "gpubridge": gpubridge.__version__,
            "gpubridge_commit": args.commit or commit(),
            "system": platform.system(), "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": round(time.monotonic() - started),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
