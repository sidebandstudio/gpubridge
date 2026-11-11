"""The HARDWARE_VALIDATION.md tables, from the kit's results.

    python scripts/gpu/validation_tables.py results/explorer [results/amd/<target> ...]

Every directory under the given paths that has a steps.jsonl is one job. From
the JSON records its steps wrote, this writes validation_tables.md and
validation_tables.json into the first path (or --out):

- Runs: per job, the steps that passed and failed, and what the job ran on:
  hosts, GPU models, driver, torch, CUDA or ROCm, NCCL or RCCL, the gpubridge
  commit and the Gloo interface.
- Items: for each HARDWARE_VALIDATION.md item, the steps and check.py stages
  that test it, and how they went.
- Network: the preflight's chosen interfaces, bandwidth and round-trip time.
- Benchmarks: gpubridge and flat-gloo against the run's baseline at a few sizes.
- Predicted vs measured: the bridge time the preflight's bandwidth predicts for
  each size, next to the bridge phase the benchmark measured
  (docs/GPU_VALIDATION_RUNBOOK.md, "Predicted vs measured").

Results from CPU simulation are marked as simulation wherever they appear, and
never count towards an item: they say nothing about GPUs.

The tables are safe to publish: hosts are numbered in order of appearance and
no path is shown. The raw results keep the real names.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


#: Island layouts that run on GPUs; simulation is told apart by device instead.
GPU_KINDS = {"split-test", "gpu-mixed", "gpu-single-vendor"}
TWO_ISLANDS = {"split-test", "gpu-mixed"}
MIXED = {"gpu-mixed"}
ISLAND_BASELINE = "island-native (fewer ranks, best case)"
#: Benchmark sizes shown, besides the largest one measured.
BENCH_SIZES = (2**20, 64 * 2**20)


@dataclass(frozen=True)
class Step:
    """A job step (steps.jsonl) whose name matches ``pattern`` in full."""

    pattern: str

    def label(self) -> str:
        return f"step `{self.pattern}`"


@dataclass(frozen=True)
class Stage:
    """A check.py stage, in runs whose kind is one of ``kinds``."""

    name: str
    kinds: frozenset[str]

    def label(self) -> str:
        return f"check.py `{self.name}` ({', '.join(sorted(self.kinds))})"


@dataclass(frozen=True)
class Manual:
    """A separate strict gate whose result the legacy kit does not ingest."""

    report: str

    def label(self) -> str:
        return f"separate `{self.report}` report (manual acceptance)"


def stage(name: str, kinds: set[str]) -> Stage:
    return Stage(name, frozenset(kinds))


#: HARDWARE_VALIDATION.md item number: (short title, what tests it).
ITEMS: dict[int, tuple[str, list[Step | Stage | Manual]]] = {
    1: ("CUDA and ROCm processes rendezvous in one job", [stage("init", MIXED)]),
    2: ("Version matching between builds", [stage("init", MIXED)]),
    3: ('`new_group(backend="nccl")` on ranks outside the group',
        [stage("init", TWO_ISLANDS)]),
    4: ('`"nccl"` on ROCm really runs RCCL', [Step("rccl_version")]),
    5: ("Device selection", [Step("device_map")]),
    6: ("Stream ordering in the bridge step", [stage("busy_gpu", TWO_ISLANDS), Step("gpu_tests")]),
    7: ("float16 and other dtypes over Gloo", [stage("dtypes", TWO_ISLANDS)]),
    8: ("Gloo networking", [Step("preflight"), Step("check_split_by_node")]),
    9: ("Timeouts", [stage("busy_gpu", TWO_ISLANDS)]),
    10: ("`barrier()` on GPUs", [stage("barrier", GPU_KINDS)]),
    11: ("Bridge throughput and latency against native",
         [Step("bench"), Step("bench_split_by_node")]),
    12: ("A rank whose `LOCAL_RANK` has no GPU fails `init()`", [Step("local_rank_check")]),
    13: ("The island step is a reduce onto the leader",
         [Step("bench"), Step("bench_split_by_node")]),
    14: ("Where NCCL/RCCL and Gloo agree on reductions",
         [stage("reduction_agreement", GPU_KINDS)]),
    15: ("MAX, MIN, AVG, all_gather and reduce_scatter on GPUs",
         [stage("reductions", GPU_KINDS), stage("all_gather", GPU_KINDS),
          stage("reduce_scatter", GPU_KINDS), Step("gpu_tests")]),
    16: ("Async collectives keep their stream semantics",
         [stage("async", GPU_KINDS), Step("gpu_tests")]),
    17: ("The pipelined bridge is correct on GPUs",
         [Step("check_pipelined-reduce-bridge-broadcast"), Step("gpu_tests")]),
    18: ("The overlap pays off",
         [Step("bench_phases_pipelined-reduce-bridge-broadcast"), Step("bench_candidates")]),
    19: ("Thresholds measured on GPUs drive `auto-tuned`",
         [Step("bench_candidates"), Step("check_auto_tuned")]),
    20: ("The sharded bridge is correct on GPUs", [Step("check_sharded-bridge")]),
    21: ("More bridge links mean more cross-vendor bandwidth",
         [Step("bench_phases_sharded-bridge"), Step("bench_candidates")]),
    22: ("DDP with `gpubridge.ddp_comm_hook` trains correctly",
         [Step(r"train_ddp(_split|_\d+)?"), Step(r"train_n\d+")]),  # _<i>: TRAIN_DDP_REPEATS
    23: ("What `gpubridge.in_flight()` shows on GPUs", [stage("in_flight", GPU_KINDS)]),
    24: ("No island puts two ranks on one GPU", [Step("shared_gpu_check")]),
    25: ("Pinned staging for the default bridge", [Step("check_pinned"), Step("bench_pinned")]),
    26: ("The UCX bridge is correct on GPUs", [Step("ucx_build"), Step("check_ucx")]),
    27: ("The UCX bridge against the Gloo bridge", [Step("bench_ucx")]),
    28: ("Bridge compression kernels give the reference's bits on GPUs",
         [Step("gpu_tests"), Step("bench_compression_kernels")]),
    29: ("Where bridge compression pays off", [Step("bench_compressed")]),
    30: ("gpubridge as a torch.distributed backend: DDP and FSDP2 on GPUs",
         [Step("verify_fsdp2_backend"), Step("train_fsdp2_backend"), Step("gpu_tests")]),
    31: ("HSDP through the backend: shard within islands, replicate across vendors",
         [Step("verify_hsdp_backend"), Step("train_hsdp_backend")]),
    32: ("Point-to-point across islands through the backend on GPUs",
         [Step("check_p2p")]),
    33: ("Data-parallel training across vendors tracks one GPU, compressed or not",
         [Step(r"dp_(ddp|fsdp2)(-.+)?"), Step("dp_single")]),
    34: ("Pipeline training across vendors tracks one GPU",
         [Step(r"pp_(gpipe|1f1b)"), Step("dp_single")]),
    35: ("The multitcp bridge against the Gloo bridge",
         [Step("multitcp_build"), Step(r"matrix_D-multitcp(-pinned|-1stream)?")]),
    36: ("Automatic stage balancing against even and hand-tuned splits",
         [Step(r"pp_(gpipe|1f1b)_(even|auto|[0-9]+(-[0-9]+)+)")]),
    37: ("Megatron-Core's pipeline across vendors tracks one GPU",
         [Step("megatron_pp2"), Step("megatron_pp1")]),
    38: ("The verbs bridge between hosts on RDMA hardware",
         [Step("verbs_build"), Step("verbs_endpoint"), Step("verbs_host"), Step("check_verbs"),
          Step("bench_verbs")]),
    39: ("GPU tensors staged through the verbs bridge", [Step("verbs_gpu_staged")]),
    40: ("NVIDIA GPU memory direct through the verbs bridge", [Step("verbs_gpu_direct")]),
    41: ("AMD GPU memory direct through the verbs bridge", [Step("verbs_gpu_direct")]),
    42: ("Training over the verbs bridge against the Gloo bridge",
         [Step("train_verbs"), Step("train_gloo")]),
    43: ("Deterministic numerical comparison before performance",
         [Manual("numerical_gate.py --compare")]),
    # 44 to 50: a job without Gloo, and bridge="auto" as the default (2026-11-11). No
    # kit step judges them yet: each is a check to run by hand and record.
    44: ("A job with no Gloo group starts and runs on GPUs",
         [Manual("check.py --bridge auto, and no Gloo thread on any rank")]),
    45: ("`barrier()` through the c10d store",
         [Manual("check.py barrier --bridge auto, timed against --bridge gloo")]),
    46: ("CPU tensors through the backend on a GPU job, without Gloo",
         [Manual("object collectives on backend=gpubridge without GPUBRIDGE_BRIDGE=gloo")]),
    47: ("The TCP bridge as the default: collectives and point-to-point between machines",
         [Manual("check.py, check_p2p and step 08 with the tcp bridge")]),
    48: ('`bridge="auto"` on real RDMA devices',
         [Manual("check.py --bridge auto on an RDMA pair and on InfiniBand + RoCE")]),
    49: ("A rank that exits without `destroy()` on GPUs",
         [Manual("a training script without destroy(), repeated; gloo as the control")]),
    50: ("The Rust bridges give Gloo's results on GPUs",
         [Manual("check.py --bridge tcp; a tcp arm of numerical_gate.py")]),
}


@dataclass
class Evidence:
    where: str
    ok: bool
    simulated: bool


@dataclass
class Job:
    path: Path
    name: str
    steps: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(default_factory=list)
    """Every JSON record under the job that carries the kit's ``environment``."""
    checks: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    """check.py output directories (relative name) -> rank -> record."""
    benches: dict[str, dict[str, Any]] = field(default_factory=dict)
    preflight: list[dict[str, Any]] = field(default_factory=list)

    @property
    def simulated(self) -> bool:
        """CPU simulation: a run on CPU, or the kit's switches for it set."""
        for record in self.records:
            device = (record.get("run") or {}).get("device") or record.get("device", "")
            if device == "cpu":
                return True
            for env in _environments(record):
                switches = env.get("env", {})
                cpu_only = switches.get("GPUBRIDGE_CPU_ONLY", "").strip().lower()
                if switches.get("GPUBRIDGE_VENDOR") or cpu_only in ("1", "true", "yes", "on"):
                    return True
        return False


class Hosts:
    """Hostnames as ``host 1``, ``host 2``, ... in order of first appearance."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def __call__(self, name: str | None) -> str | None:
        return self.names.setdefault(name, f"host {len(self.names) + 1}") if name else None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _steps(path: Path) -> list[dict[str, Any]]:
    """steps.jsonl's entries; a line cut short (the job was killed mid-write) is skipped."""
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            steps.append(entry)
    return steps


def load_job(path: Path, root: Path) -> Job:
    steps = _steps(path / "steps.jsonl")
    name = path.relative_to(root).as_posix() if path != root else path.name
    job = Job(path=path, name=name, steps=steps)
    for file in sorted(path.rglob("*.json")):
        data = _load(file)
        if data is None:
            continue
        if "environment" in data:
            job.records.append(data)
        relative = file.parent.relative_to(path).as_posix()
        if re.fullmatch(r"rank\d+\.json", file.name) and data.get("script", "").endswith(
                "check.py"):
            job.checks.setdefault(relative, {})[int(file.stem[4:])] = data
        elif file.name == "bench.json":
            job.benches[relative] = data
        elif file.name.startswith("preflight-") and "bandwidth" in data:
            job.preflight.append(data)
    return job


def find_jobs(paths: list[Path]) -> list[Job]:
    jobs = []
    for root in paths:
        for steps in sorted(root.rglob("steps.jsonl")):
            jobs.append(load_job(steps.parent, root))
    return jobs


# ---- runs ---------------------------------------------------------------------------

def _environments(record: dict[str, Any]) -> list[dict[str, Any]]:
    """A record's environments: every rank's when it has them (train_synthetic.py)."""
    found = record.get("environments") or [record.get("environment")]
    return [env for env in found if isinstance(env, dict)]


def _values(records: list[dict[str, Any]], get) -> str:
    found = []
    for env in (env for record in records for env in _environments(record)):
        value = get(env)
        for item in value if isinstance(value, list) else [value]:
            if item not in (None, "") and item not in found:
                found.append(item)
    return ", ".join(map(str, found)) or "-"


def driver_text(raw: str | None) -> str | None:
    """A driver version as recorded, or the version in ``rocm-smi --showdriverversion
    --csv`` output, which older records hold whole. rocm-smi writes either ``name,
    value`` rows (``"Driver version", "5.14.0-611.5.1.el9_7.x86_64"``) or a ``device,
    Driver version`` header over a row per card."""
    if not raw or "driver version" not in raw.lower():
        return raw
    rows = [[cell.strip() for cell in row]
            for row in csv.reader(raw.strip().splitlines(), skipinitialspace=True) if row]
    for row in rows:  # a "Driver version" row
        if len(row) >= 2 and row[0].lower() == "driver version":
            return row[1] or None
    header = [cell.lower() for cell in rows[0]] if rows else []
    if "driver version" in header and len(rows) > 1:  # a "Driver version" column
        column = header.index("driver version")
        values = {row[column] for row in rows[1:] if len(row) > column and row[column]}
        return ", ".join(sorted(values)) or None
    return raw


def _commit(env: dict[str, Any]) -> str | None:
    commit = env.get("git_commit")
    return f"{commit}+dirty" if commit and env.get("git_dirty") else commit


def _toolkit(env: dict[str, Any]) -> str | None:
    if env.get("torch_hip"):
        return f"ROCm {env.get('rocm_version') or '?'} (HIP {env['torch_hip']})"
    return f"CUDA {env['torch_cuda']}" if env.get("torch_cuda") else None


def _collectives(env: dict[str, Any]) -> str | None:
    if not env.get("nccl_version"):
        return None
    library = env.get("collective_library") or ("rccl" if env.get("torch_hip") else "nccl")
    return f"{library.upper()} {env['nccl_version']}"


def _gloo_ifname(env: dict[str, Any]) -> str:
    named = env.get("interfaces", {}).get("GLOO_SOCKET_IFNAME")
    return named or env.get("env", {}).get("GLOO_SOCKET_IFNAME") or "auto"


def run_row(job: Job, hosts: Hosts) -> dict[str, Any]:
    failed = [s.get("step") for s in job.steps if s.get("exit_code", 1) != 0]
    negative = {s.get("step"): s["result"] for s in job.steps if "result" in s}
    kinds = sorted({r["run"]["kind"] for ranks in job.checks.values() for r in ranks.values()
                    if r.get("run")})
    records = job.records
    return {
        "job": job.name,
        "simulated": job.simulated,
        "steps_passed": len(job.steps) - len(failed),
        "steps": len(job.steps),
        "failed": failed,
        "negative_steps": negative,
        "run_kinds": kinds,
        "hosts": _values(records, lambda e: hosts(e.get("hostname"))),
        "gpus": _values(records, lambda e: [d.get("name") for d in e.get("devices") or []
                                           if isinstance(d, dict)]),
        "driver": _values(records, lambda e: driver_text(e.get("driver_version"))),
        "torch": _values(records, lambda e: e.get("torch")),
        "toolkit": _values(records, _toolkit),
        "collectives": _values(records, _collectives),
        "gpubridge": _values(records, _commit),
        "gloo_ifname": _values(records, _gloo_ifname),
    }


# ---- items --------------------------------------------------------------------------

def evidence_for(test: Step | Stage | Manual, jobs: list[Job]) -> list[Evidence]:
    if isinstance(test, Manual):
        # A successful legacy training step is not the new cross-arm numerical gate.
        return []
    found = []
    for job in jobs:
        if isinstance(test, Step):
            for s in job.steps:
                if re.fullmatch(test.pattern, s["step"]):
                    found.append(Evidence(f"{job.name}/{s['step']}", s["exit_code"] == 0,
                                          job.simulated))
            continue
        for directory, ranks in job.checks.items():
            kinds = {r["run"]["kind"] for r in ranks.values() if r.get("run")}
            if not kinds & test.kinds:
                continue
            results = [next((s["ok"] for s in r["stages"] if s["name"] == test.name), "absent")
                       for r in ranks.values()]
            if all(result == "absent" for result in results):
                continue
            simulated = job.simulated or any(
                (r.get("run") or {}).get("device") == "cpu" for r in ranks.values())
            found.append(Evidence(f"{job.name}/{directory}:{test.name}",
                                  all(result is True for result in results), simulated))
    return found


def verdict(evidence: list[Evidence]) -> str:
    if not evidence:
        return "not run"
    real = [e for e in evidence if not e.simulated]
    if not real:
        return "simulation only"
    failed = [e.where for e in real if not e.ok]
    return f"FAIL ({len(failed)} of {len(real)})" if failed else f"pass ({len(real)})"


def item_rows(jobs: list[Job]) -> list[dict[str, Any]]:
    rows = []
    for number, (title, tests) in ITEMS.items():
        evidence = [e for test in tests for e in evidence_for(test, jobs)]
        rows.append({"item": number, "title": title, "tested_by": [t.label() for t in tests],
                     "verdict": verdict(evidence),
                     "evidence": [{"where": e.where, "ok": e.ok, "simulated": e.simulated}
                                  for e in evidence]})
    return rows


# ---- network, benchmarks, predicted vs measured ---------------------------------------

def network_rows(jobs: list[Job], hosts: Hosts) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        by_role = {record.get("role"): record for record in job.preflight}
        master, worker = by_role.get("master"), by_role.get("worker")
        if master is not None:
            bandwidth = master["bandwidth"]
            m2w, w2m = bandwidth["MBps_out"], bandwidth["MBps_in"]
        elif worker is not None:  # only one machine's results were copied here
            bandwidth = worker["bandwidth"]
            m2w, w2m = bandwidth["MBps_in"], bandwidth["MBps_out"]
        else:
            continue

        def side(own: dict[str, Any] | None, other: dict[str, Any] | None) -> str:
            if own is not None:
                return f"{hosts(own.get('hostname')) or '?'} {own.get('chosen_ifname')}"
            return f"? {(other or {}).get('peer_ifname')}"

        rows.append({"job": job.name, "simulated": job.simulated,
                     "master": side(master, worker), "worker": side(worker, master),
                     "MBps_master_to_worker": m2w, "MBps_worker_to_master": w2m,
                     "rtt_ms": bandwidth["rtt_ms"]})
    return rows


def baseline_us(by_op: dict[str, dict[str, Any]], baseline: str) -> float | None:
    if baseline == "native":
        return by_op.get("native", {}).get("median_us")
    islands = [r["median_us"] for op, r in by_op.items() if op.startswith("island:")]
    return max(islands) if islands else None


def bench_rows(jobs: list[Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        for directory, bench in sorted(job.benches.items()):
            by_size: dict[int, dict[str, dict[str, Any]]] = {}
            for r in bench.get("rows", []):
                by_size.setdefault(r["bytes"], {})[r["op"]] = r
            if not by_size:
                continue
            ops = {op for by_op in by_size.values() for op in by_op}
            baseline = bench.get("baseline") or (
                "native" if "native" in ops else ISLAND_BASELINE
                if any(op.startswith("island:") for op in ops) else "none")
            shown = sorted({size for size in BENCH_SIZES if size in by_size} | {max(by_size)})
            for size in shown:
                by_op = by_size[size]
                base = baseline_us(by_op, baseline)
                row = {"job": job.name, "bench": directory, "simulated": job.simulated,
                       "run_kind": bench.get("run", {}).get("kind"),
                       "bridge": _bridge(bench), "staging": _staging(bench),
                       "timer": next(iter(by_op.values())).get("timer", "host"),
                       "baseline": baseline, "bytes": size, "baseline_us": base}
                for op in ("gpubridge", "flat-gloo"):
                    us = by_op.get(op, {}).get("median_us")
                    row[f"{op}_us"] = us
                    row[f"{op}_over_baseline"] = round(us / base, 2) if us and base else None
                rows.append(row)
    return rows


def _staging(bench: dict[str, Any]) -> str:
    """How the gpubridge op's bridge reached host memory: pinned, pageable, or (UCX)
    the path its transport took."""
    run = bench.get("run", {})
    if run.get("bridge") == "ucx" and run.get("bridge_path"):
        return f"ucx {run['bridge_path']}"  # e.g. "ucx direct", or "ucx direct+staged"
    return "pinned" if run.get("pinned_staging") else "pageable"


def _bridge(bench: dict[str, Any]) -> str:
    return bench.get("run", {}).get("bridge") or "gloo"


def predicted_bridge_ms(nbytes: int, islands: int, mbps: float, rtt_ms: float) -> float:
    """What the preflight's numbers predict for the bridge's Gloo all_reduce.

    The leaders' ring all_reduce sends 2(p-1)/p of the tensor each way, with p
    leaders (one per island): the whole tensor for two islands. Its 2(p-1)
    steps take about half a round trip each: (p-1) round trips, one for two
    islands. ``mbps`` is the slower direction, in MB/s.
    """
    factor = 2 * (islands - 1) / islands
    return (islands - 1) * rtt_ms + factor * nbytes / (mbps * 1e6) * 1e3


def prediction_rows(jobs: list[Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        if not job.preflight:
            continue
        bandwidth = job.preflight[0]["bandwidth"]
        mbps = min(bandwidth["MBps_in"], bandwidth["MBps_out"])
        for directory, bench in sorted(job.benches.items()):
            for r in bench.get("rows", []):
                phases = r.get("phases_ms") or {}
                if r["op"] != "gpubridge" or "bridge" not in phases or r["islands"] < 2:
                    continue
                predicted = predicted_bridge_ms(r["bytes"], r["islands"], mbps,
                                                bandwidth["rtt_ms"])
                rows.append({"job": job.name, "bench": directory, "simulated": job.simulated,
                             "policy": r.get("policy"), "bridge": _bridge(bench),
                             "staging": _staging(bench),
                             "bytes": r["bytes"],
                             "MBps": mbps, "rtt_ms": bandwidth["rtt_ms"],
                             "predicted_ms": round(predicted, 3),
                             "measured_ms": phases["bridge"],
                             "measured_over_predicted": round(phases["bridge"] / predicted, 2)})
    return rows


# ---- markdown -------------------------------------------------------------------------

def human(nbytes: int) -> str:
    for unit, size in (("G", 2**30), ("M", 2**20), ("K", 2**10)):
        if nbytes >= size and nbytes % size == 0:
            return f"{nbytes // size}{unit}"
    return f"{nbytes}B"


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return "-" if value is None else str(value).replace("|", "\\|")

    return (["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
            + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def sim(row: dict[str, Any], text: Any) -> str:
    return f"{text} (simulation)" if row["simulated"] else str(text)


def markdown(tables: dict[str, Any]) -> str:
    lines = ["# GPU validation tables", "",
             f"Made by `scripts/gpu/validation_tables.py` from {len(tables['runs'])} job(s). "
             "Hosts are numbered in order of appearance and no paths are shown; the raw "
             "results keep the real names.", ""]
    if any(run["simulated"] for run in tables["runs"]):
        lines += ["> Rows marked **(simulation)** ran on CPU in simulation mode. They check "
                  "the scripts, not GPUs, and never count towards an item.", ""]
    lines += ["## Runs", ""]
    lines += table(
        ["Job", "Steps passed", "Failed", "Hosts", "GPUs", "Driver", "torch", "CUDA / ROCm",
         "NCCL / RCCL", "gpubridge", "Gloo interface"],
        [[sim(r, r["job"]), f"{r['steps_passed']} of {r['steps']}",
          ", ".join(r["failed"]) or "-", r["hosts"], r["gpus"], r["driver"], r["torch"],
          r["toolkit"], r["collectives"], r["gpubridge"], r["gloo_ifname"]]
         for r in tables["runs"]])
    negative = [(r, name, result) for r in tables["runs"]
                for name, result in r["negative_steps"].items()]
    if negative:
        lines += ["", "Deliberate failing runs: "
                  + "; ".join(f"{r['job']}/{name}: {result}" for r, name, result in negative)
                  + "."]
    lines += ["", "## Items", "",
              "Each item of HARDWARE_VALIDATION.md, with the steps and check.py stages that "
              "test it. `pass (n)` counts GPU runs only.", ""]
    lines += table(
        ["Item", "Tested by", "Verdict", "Runs"],
        [[f"{r['item']}. {r['title']}", "; ".join(r["tested_by"]), r["verdict"],
          "; ".join(f"{e['where']} {'ok' if e['ok'] else 'FAIL'}"
                    f"{' (simulation)' if e['simulated'] else ''}" for e in r["evidence"])
          or "-"] for r in tables["items"]])
    lines += ["", "## Network", "",
              "From the preflight: one TCP stream driven from Python, so a lower bound on "
              "fast links.", ""]
    lines += table(
        ["Job", "Master (host, interface)", "Worker (host, interface)",
         "MB/s master to worker", "MB/s worker to master", "Round trip (ms)"],
        [[sim(r, r["job"]), r["master"], r["worker"], r["MBps_master_to_worker"],
          r["MBps_worker_to_master"], r["rtt_ms"]] for r in tables["network"]])
    lines += ["", "## Benchmarks", "",
              "Median latency of the slowest rank. `island-native (fewer ranks, best case)` "
              "is each island's own all_reduce: a lower bound, not like for like.", ""]
    lines += table(
        ["Job / bench", "Run kind", "Bridge", "Staging", "Timer", "Size", "Baseline",
         "Baseline (us)", "gpubridge (us)", "flat-gloo (us)", "gpubridge / baseline",
         "flat-gloo / baseline"],
        [[sim(r, f"{r['job']}/{r['bench']}"), r["run_kind"], r["bridge"], r["staging"],
          r["timer"],
          human(r["bytes"]), r["baseline"], r["baseline_us"], r["gpubridge_us"], r["flat-gloo_us"],
          r["gpubridge_over_baseline"], r["flat-gloo_over_baseline"]]
         for r in tables["benchmarks"]])
    lines += ["", "## Predicted vs measured bridge time", "",
              "Predicted: (p-1) round trips plus 2(p-1)/p of the tensor over the preflight's "
              "slower direction, p = number of islands. Measured: the benchmark's `bridge` "
              "phase on the slowest leader, which also includes the copies to and from the "
              "host. The preflight measures TCP from Python, so for the UCX bridge over "
              "RDMA a ratio well below 1 is the point, not an error. See the runbook's "
              "\"Predicted vs measured\".", ""]
    lines += table(
        ["Job / bench", "Policy", "Bridge", "Staging", "Size", "MB/s (slower way)",
         "Round trip (ms)", "Predicted (ms)", "Measured (ms)", "Measured / predicted"],
        [[sim(r, f"{r['job']}/{r['bench']}"), r["policy"], r["bridge"], r["staging"],
          human(r["bytes"]),
          r["MBps"], r["rtt_ms"], r["predicted_ms"], r["measured_ms"],
          r["measured_over_predicted"]]
         for r in tables["predicted"]])
    from gpukit import redact  # here, not at the top: reading results needs no torch

    return redact("\n".join(lines) + "\n")


def build(paths: list[Path]) -> dict[str, Any]:
    jobs = find_jobs(paths)
    hosts = Hosts()
    return {"runs": [run_row(job, hosts) for job in jobs], "items": item_rows(jobs),
            "network": network_rows(jobs, hosts), "benchmarks": bench_rows(jobs),
            "predicted": prediction_rows(jobs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", type=Path, nargs="+", help="results directories")
    parser.add_argument("--out", type=Path, help="where to write (default: the first path)")
    args = parser.parse_args(argv)
    tables = build(args.paths)
    if not tables["runs"]:
        print(f"no job results (steps.jsonl) under {', '.join(map(str, args.paths))}",
              file=sys.stderr)
        return 1
    out = args.out or args.paths[0]
    text = markdown(tables)
    from gpukit import write_json

    write_json(out / "validation_tables.json", tables)
    (out / "validation_tables.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
