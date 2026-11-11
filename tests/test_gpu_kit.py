"""The GPU validation kit (scripts/gpu, examples/train_synthetic.py), tested without GPUs.

Shell scripts: shellcheck and --dry-run output. Python tools: run end to end
under torchrun on CPU in simulation mode, including split-test runs.
"""

import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "scripts" / "gpu"
#: The bridge a kit run that names none runs on here: what bridge="auto" resolves to on
#: a machine without a usable RDMA device. (clean_env keeps every GPUBRIDGE_* setting
#: of the developer's out of the kit's runs, a GPUBRIDGE_BRIDGE=gloo for the suite too.)
RAN_ON = "tcp"
sys.path.insert(0, str(KIT))

import bench_all_reduce  # noqa: E402
import bench_matrix  # noqa: E402
import dp_report  # noqa: E402
import llama  # noqa: E402
import preflight  # noqa: E402
import summarize  # noqa: E402
import train_bench  # noqa: E402
import train_ddp_report  # noqa: E402
import train_dp  # noqa: E402
import megatron_pp  # noqa: E402
import train_pipeline  # noqa: E402
import validation_tables  # noqa: E402

SHELL_SCRIPTS = sorted(
    [*KIT.glob("*.sh"), *KIT.glob("*/*.sh"), *KIT.glob("explorer/*.sbatch")]
)
#: Stand-ins for sbatch, srun, scontrol, module and uv (tests/test_gpu_jobs.py).
FAKE_CLUSTER_SCRIPTS = sorted((REPO / "tests" / "fake_cluster").iterdir())
FAKE_PROJECT = "/projects/example-lab"
OWN_SETTINGS = ("PROJECT_DIR", "WORK_DIR", "VENV", "VENV_ROCM", "ROCM_INDEX", "ROCM_INDEX_URL",
                "ROCM_TORCH_VERSION", "TORCH_VERSION", "TORCH_INDEX_URL", "STEPS_07",
                "TRAIN_DDP_REPEATS")


def clean_env(**extra: str) -> dict[str, str]:
    # A developer's own settings, including an explorer/local.env, must not leak in.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GPUBRIDGE_") and k not in OWN_SETTINGS}
    env.update({"OMP_NUM_THREADS": "1", "EXPLORER_LOCAL_ENV": os.devnull, **extra})
    return env


def torchrun(nproc: int, script: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    # Not --standalone: it hangs on hosts whose own name doesn't resolve (macOS).
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
           "--master-addr=127.0.0.1", f"--master-port={_free_port()}",
           f"--nproc-per-node={nproc}", str(script), *args]
    return subprocess.run(cmd, env=clean_env(**env), capture_output=True, text=True,
                          timeout=300)


def dry_run(script: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    env = {"GPUBRIDGE_REPO": str(REPO), "PROJECT_DIR": FAKE_PROJECT, **env}
    return subprocess.run(["bash", str(script), "--dry-run", *args], env=clean_env(**env),
                          capture_output=True, text=True, timeout=60)


# ---- shell scripts ---------------------------------------------------------------

@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_is_clean():
    result = subprocess.run(["shellcheck", *map(str, SHELL_SCRIPTS + FAKE_CLUSTER_SCRIPTS)],
                            cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_every_shell_script_is_covered():
    names = {p.name for p in SHELL_SCRIPTS}
    assert {"common.sh", "setup_env.sh", "submit.sh", "run_all.sh", "node.sh",
            "00_partitions.sh", "setup.sbatch", "01_probe.sbatch", "06_scaling.sbatch",
            "07_mixed_hetjob.sbatch"} <= names


@pytest.mark.parametrize("flavor", ["cuda", "rocm", "system"])
def test_setup_dry_run(flavor, tmp_path):
    result = dry_run(KIT / "setup_env.sh", "--flavor", flavor, "--venv", str(tmp_path / "v"))
    assert result.returncode == 0, result.stderr
    assert "uv pip install" in result.stdout
    if flavor == "cuda":
        assert "torch==2.14.1+cu126" in result.stdout
    if flavor == "rocm":
        assert "torch==2.14.1+rocm7.2" in result.stdout
    # The ROCm wheel's GPU-name table, for its libdrm; nothing of the kind for CUDA.
    assert ("AMDGPU_ASIC_ID_TABLE_PATHS" in result.stdout) is (flavor != "cuda")
    if flavor == "system":
        assert "--system-site-packages" in result.stdout
    assert not (tmp_path / "v").exists(), "dry-run must not create anything"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("01", ["--partition=gpu", "--gres=gpu:1", "01_probe.sbatch"]),
        ("02", ["--partition=gpu", "--gres=gpu:1", "02_split_shared_gpu.sbatch"]),
        ("03", ["--partition=multigpu", "--gres=gpu:2", "03_single_island.sbatch"]),
        ("04", ["--partition=multigpu", "--gres=gpu:4", "04_split_4gpu.sbatch"]),
        ("05", ["--partition=multigpu", "--nodes=2", "05_multinode.sbatch"]),
        ("06", ["--partition=multigpu", "--gres=gpu:4", "06_scaling.sbatch"]),
    ],
)
def test_submit_dry_run(step, expected):
    result = dry_run(KIT / "explorer" / "submit.sh", step)
    assert result.returncode == 0, result.stderr
    for text in expected:
        assert text in result.stdout, (text, result.stdout)
    assert "--account" not in result.stdout


@pytest.mark.parametrize(
    ("step", "env", "cap"),
    [
        ("01", {}, "480s (its Slurm limit, 00:10:00, less 120s), not 1200s"),
        ("02", {}, None),                                    # 25 minutes: 1200 s fits
        ("03", {}, "1080s (its Slurm limit, 00:20:00, less 120s), not 1200s"),
        ("05", {"STEP_TIMEOUT": "0", "TIME_05": "1-00"}, "86280s"),  # 0 = no limit: capped
        ("06", {"TIME_06": "01:30"}, "60s"),                 # never below a minute
        ("04", {"STEP_TIMEOUT": "400"}, None),               # already short enough
        ("04", {"TIME_04": "0"}, None),                      # Slurm's "no limit"
        ("04", {"TIME_04": "UNLIMITED"}, None),
        ("04", {"TIME_04": "infinite"}, None),
    ],
)
def test_each_jobs_step_timeout_fits_inside_its_slurm_limit(step, env, cap):
    result = dry_run(KIT / "explorer" / "submit.sh", step, **env)
    assert result.returncode == 0, result.stderr
    if cap is None:
        assert "STEP_TIMEOUT for this job" not in result.stderr
    else:
        assert f"STEP_TIMEOUT for this job: {cap}" in result.stderr
    # Only a cap at or below KIT_TIMEOUT (300 s) warns.
    assert ("is not above KIT_TIMEOUT" in result.stderr) is (cap == "60s")


def test_job_events_are_redacted_and_record_the_exit_code(tmp_path):
    # job.jsonl, as job_setup starts it: a die message loses the project, home and user,
    # and a command failing under set -e still leaves its exit code.
    script = f"""
source {KIT}/common.sh
RESULTS={tmp_path}
PROJECT_DIR=/projects/example-lab USER=someuser HOME=/home/someuser
job_started
(die "no venv at /projects/example-lab/someuser/venv; see /home/someuser/x")
false
"""
    result = subprocess.run(["bash", "-e", "-c", script], env=clean_env(), capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 1
    lines = (tmp_path / "job.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    # A subshell has no EXIT trap, so only the script's own exit (false's) is recorded.
    assert [e["event"] for e in events] == ["started", "error", "exit"]
    assert events[1]["message"] == "no venv at $PROJECT_DIR/$USER/venv; see ~/x"
    assert events[2]["code"] == 1
    assert "someuser" not in "".join(lines) and "example-lab" not in "".join(lines)


def test_submit_rejects_a_time_limit_it_cant_read():
    result = dry_run(KIT / "explorer" / "submit.sh", "01", TIME_01="ten minutes")
    assert result.returncode == 1 and "can't read the time limit 'ten minutes'" in result.stderr


@pytest.mark.parametrize(("env", "python"), [({}, "3.12"),
                                             ({"PYTHON_VERSION": "3.12.15"}, "3.12"),
                                             ({"VENV_PYTHON": "3.11"}, "3.11")])
def test_setup_venv_python_ignores_python_version(env, python, tmp_path):
    # Python's Docker images set PYTHON_VERSION to their own full version.
    result = dry_run(KIT / "setup_env.sh", "--flavor", "cuda", "--venv", str(tmp_path / "v"),
                     **env)
    assert result.returncode == 0, result.stderr
    assert f"uv venv --python {python} " in result.stdout


def fake_command(directory: Path, name: str, body: str) -> str:
    """Put an executable NAME on a PATH that starts with DIRECTORY; returns that PATH."""
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return f"{directory}{os.pathsep}{os.environ['PATH']}"


def test_setup_refuses_to_run_on_a_slurm_login_node(tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "exit 0\n")
    result = subprocess.run(["bash", str(KIT / "setup_env.sh"), "--flavor", "cuda", "--venv",
                             str(tmp_path / "v")], env=clean_env(PATH=path),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "login node" in result.stderr and "submit.sh setup" in result.stderr
    assert not (tmp_path / "v").exists()


def test_submit_setup_is_a_short_cpu_job_not_a_login_node_install():
    result = dry_run(KIT / "explorer" / "submit.sh", "setup", USER="alice")
    assert result.returncode == 0, result.stderr
    sbatch = [line for line in result.stdout.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, result.stdout
    for text in ["--partition=short", "--time=01:00:00", "--cpus-per-task=4", "--mem=16G",
                 "explorer/setup.sbatch"]:
        assert text in sbatch[0], text
    assert "--gres" not in sbatch[0]
    assert "uv pip install" not in result.stdout, "nothing may be installed on the login node"


@pytest.mark.parametrize(("flavor", "venv", "wheel"), [
    ("cuda", "/gpubridge-gpu/venv ", "torch==2.14.1+cu126"),
    ("rocm", "/gpubridge-gpu/venv-rocm ", "torch==2.14.1+rocm7.2"),
])
def test_setup_job_loads_the_explorer_module_before_installing(flavor, venv, wheel):
    result = dry_run(KIT / "explorer" / "setup.sbatch", USER="alice", TORCH_FLAVOR=flavor)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.index("+ module load explorer") < out.index("+ uv venv") < out.index(wheel)
    assert f"{FAKE_PROJECT}/alice{venv}" in out


def test_setup_rocm_uses_the_configured_build_and_checks_the_gpu():
    result = dry_run(KIT / "explorer" / "setup.sbatch", TORCH_FLAVOR="rocm",
                     ROCM_TORCH_VERSION="2.9.1", ROCM_INDEX="rocm6.4")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1+rocm6.4" in out
    assert out.index("torch==2.9.1+rocm6.4") < out.index("probe.py")
    assert "/setup_rocm" in out
    url = "https://mirror.example/whl/rocm6.4"
    out = dry_run(KIT / "explorer" / "setup.sbatch", TORCH_FLAVOR="rocm", ROCM_INDEX="rocm6.4",
                  ROCM_INDEX_URL=url).stdout
    assert f"--index-url {url} torch==2.14.1+rocm6.4" in out
    assert "probe.py" not in dry_run(KIT / "explorer" / "setup.sbatch").stdout  # CUDA: CPU job


def test_setup_env_takes_an_index_url(tmp_path):
    out = dry_run(KIT / "setup_env.sh", "--flavor", "cuda", "--venv", str(tmp_path / "v"),
                  TORCH_INDEX_URL="https://mirror.example/whl/cu126").stdout
    assert "--index-url https://mirror.example/whl/cu126 torch==2.14.1+cu126" in out


def test_setup_job_needs_a_slurm_job_and_can_skip_the_module():
    result = subprocess.run(["bash", str(KIT / "explorer" / "setup.sbatch")],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PROJECT_DIR=FAKE_PROJECT),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "not in a Slurm job" in result.stderr
    assert "module load" not in dry_run(KIT / "explorer" / "setup.sbatch",
                                        SETUP_MODULES="").stdout


def test_work_dir_comes_from_project_dir():
    work = f"{FAKE_PROJECT}/alice/gpubridge-gpu"
    out = dry_run(KIT / "explorer" / "submit.sh", "01", USER="alice").stdout
    assert f"--output={work}/logs/%x-%j.out" in out
    out = dry_run(KIT / "explorer" / "01_probe.sbatch", USER="alice").stdout
    assert f"source {work}/venv/bin/activate" in out
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="",
                  WORK_DIR="/scratch/w").stdout
    assert "--output=/scratch/w/logs/%x-%j.out" in out and "/projects/" not in out


@pytest.mark.parametrize(("script", "args"), [("submit.sh", ["01"]), ("submit.sh", ["setup"]),
                                              ("01_probe.sbatch", []), ("setup.sbatch", [])])
def test_explorer_scripts_need_a_project_dir(script, args):
    result = dry_run(KIT / "explorer" / script, *args, PROJECT_DIR="")
    assert result.returncode == 1
    assert "set PROJECT_DIR" in result.stderr and "local.env" in result.stderr
    assert "/projects/<your-project>" in result.stderr
    assert "+ sbatch" not in result.stdout


def test_project_dir_from_local_env_and_the_environment_wins(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f'# my settings\nPROJECT_DIR="/projects/from-file"\n'
                     f"touch {tmp_path}/ran\nACCOUNT=$(touch {tmp_path}/ran2)\n")
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="", USER="alice",
                  EXPLORER_LOCAL_ENV=str(local)).stdout
    assert "--output=/projects/from-file/alice/gpubridge-gpu/logs/" in out
    assert not (tmp_path / "ran").exists() and not (tmp_path / "ran2").exists(), "parsed, not run"
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="/projects/from-env",
                  USER="alice", EXPLORER_LOCAL_ENV=str(local)).stdout
    assert "--output=/projects/from-env/alice/gpubridge-gpu/logs/" in out


@pytest.mark.parametrize("path", ["scripts/gpu/explorer/local.env",
                                  "results/explorer/01_probe/probe.log",
                                  "results/explorer/02_split_shared_gpu.tgz"])
def test_local_settings_and_raw_explorer_results_are_gitignored(path):
    # Raw results hold unredacted logs: a `git add -A` on the cluster must not take them.
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    assert result.returncode == 0


def test_bench_compression_fp_fusion_switch_without_triton_kernels(tmp_path):
    # On CPU the benchmark times the PyTorch kernels, which have no such option.
    result = subprocess.run(
        [sys.executable, str(KIT / "bench_compression.py"), "--out", str(tmp_path),
         "--min-bytes", "4K", "--max-bytes", "4K", "--modes", "fp8", "--dtypes", "float32",
         "--warmup", "0", "--trials", "1", "--fp-fusion"],
        env=clean_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--fp-fusion: these are the PyTorch kernels" in result.stderr
    record = json.loads((tmp_path / "compression.json").read_text())
    assert record["fp_fusion"] is None and record["impl"] == "torch (CPU)"


FAKE_SINFO = r"""
part="" nodes=0 fmt=""
while [ $# -gt 0 ]; do
    case $1 in
    -p) part=$2; shift 2 ;;
    -o) fmt=$2; shift 2 ;;
    -N) nodes=1; shift ;;
    *) shift ;;
    esac
done
if [ "$nodes" = 0 ]; then echo "format=[$fmt] partition=$part"; exit 0; fi
case $part in
sharing) printf '%s\n' "${SHARING_GRES[@]}" ;;
gpu) printf 'd1001 gpu:v100-sxm2:4(S:0-1)\nd1002 gpu:v100-pcie:2\nd1003 gpu:2\n' ;;
*) echo "sinfo: error: invalid partition specified: $part" >&2; exit 1 ;;
esac
"""


def partitions_step(tmp_path, sharing: list[str]) -> tuple[subprocess.CompletedProcess, Path]:
    gres = " ".join(f"'{line}'" for line in sharing)
    path = fake_command(tmp_path / "bin", "sinfo", f"SHARING_GRES=({gres})\n" + FAKE_SINFO)
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "00"],
                            env=clean_env(PATH=path, RESULTS_ROOT=str(tmp_path / "results")),
                            capture_output=True, text=True, timeout=60)
    return result, tmp_path / "results" / "00_partitions"


def test_step_00_records_sinfo_and_finds_amd_gpus_in_sharing(tmp_path):
    result, out = partitions_step(tmp_path, [
        "c0101 gpu:mi100:4(S:0-1)", "c0102 gpu:mi100:4(S:0,1)", "c0103 gpu:a100:2,gpu:t4:1",
        "c0104 (null)",
    ])
    assert result.returncode == 0, result.stderr
    fmt = "format=[%20N %10c %10m %25f %10G %10t]"
    assert (out / "sinfo-sharing.txt").read_text() == f"{fmt} partition=sharing\n"
    assert (out / "sinfo-gpu.txt").read_text() == f"{fmt} partition=gpu\n"
    rows = [line.split() for line in (out / "gpu-types.txt").read_text().splitlines()]
    assert ["sharing", "mi100", "amd", "2", "8"] in rows
    assert ["sharing", "a100", "nvidia", "1", "2"] in rows
    assert ["sharing", "t4", "nvidia", "1", "1"] in rows
    assert ["gpu", "v100-sxm2", "nvidia", "1", "4"] in rows
    assert ["gpu", "(untyped)", "unknown", "1", "2"] in rows
    assert "AMD GPUs in sharing: mi100." in result.stdout


def test_step_00_says_when_sharing_has_no_amd_gpus(tmp_path):
    result, out = partitions_step(tmp_path, ["c0103 gpu:a100:2"])
    assert result.returncode == 0, result.stderr
    assert "No AMD GPU type found in sharing" in (out / "gpu-types.txt").read_text()


def test_step_00_dry_run_prints_the_sinfo_commands():
    out = dry_run(KIT / "explorer" / "submit.sh", "00").stdout
    for partition in ("sharing", "gpu", "multigpu"):
        assert f"+ sinfo -p {partition} -o %20N\\ %10c" in out, out


@pytest.mark.parametrize("step", ["setup-rocm", "amd", "07"])
def test_explorer_mixed_path_is_dry_run_only_until_amd_gpus_are_confirmed(step, tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo SUBMITTED\n")
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), step],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          GPU_TYPE_AMD="mi100", WORK_DIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "dry-run only until step 00" in result.stderr
    assert "SUBMITTED" not in result.stdout


def test_amd_nodes_confirmed_comes_from_local_env(tmp_path):
    # A local setting, so turning the AMD path on doesn't make the clone git_dirty.
    local_env = tmp_path / "local.env"
    local_env.write_text("AMD_NODES_CONFIRMED=1\nGPU_TYPE_AMD=mi50\n")
    path = fake_command(tmp_path / "bin", "sbatch", "echo 4242\n")
    overlay = tmp_path / "venv-rocm63-triton38" / "bin"  # the default ROCm venv
    overlay.mkdir(parents=True)
    (overlay / "python").write_text("#!/bin/sh\n")
    (overlay / "python").chmod(0o755)
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "amd"],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          WORK_DIR=str(tmp_path),
                                          EXPLORER_LOCAL_ENV=str(local_env)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "submitted gpubridge-amd_node: job 4242" in result.stderr


def test_the_amd_gate_says_where_to_turn_it_on(tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo 4242\n")
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "amd"],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          GPU_TYPE_AMD="mi50", WORK_DIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "add AMD_NODES_CONFIRMED=1 to" in result.stderr and "explorer/local.env" in result.stderr


@pytest.mark.parametrize(
    ("step", "env", "warns"),
    [
        # The gpu partition's QOS caps a job at 1 GPU in total, across its nodes.
        ("05", {"PARTITION_MULTI": "gpu", "GPU_TYPE_MULTI": "v100-sxm2",
                "GPUS_PER_NODE_MULTINODE": "1"}, "2 GPUs (2 node(s) x 1)"),
        ("03", {"PARTITION_MULTI": "gpu"}, "2 GPUs (1 node(s) x 2)"),
        ("01", {}, None),
        ("02", {}, None),
        ("03", {}, None),  # multigpu
        ("05", {}, None),
    ],
)
def test_submit_warns_before_asking_the_gpu_partition_for_more_than_one_gpu(step, env, warns):
    result = dry_run(KIT / "explorer" / "submit.sh", step, **env)
    assert result.returncode == 0, result.stderr
    if warns is None:
        assert "QOSMaxGRESPerJob" not in result.stderr
    else:
        assert f"asks the gpu partition for {warns}" in result.stderr
        assert "QOSMaxGRESPerJob" in result.stderr and "PARTITION_MULTI=multigpu" in result.stderr


def _asic_ids(tmp_path, *, ships: bool, preset: str | None = None) -> str:
    """rocm_asic_ids against a stand-in torch package; prints the variable after it."""
    package = tmp_path / "site" / "torch"
    (package / "share" / "libdrm").mkdir(parents=True)
    (package / "__init__.py").write_text("raise SystemExit('torch must not be imported')\n")
    if ships:
        (package / "share" / "libdrm" / "amdgpu.ids").write_text("# 1.0.0\n")
    script = f'source {KIT}/common.sh; rocm_asic_ids {sys.executable}; ' \
             'echo "AMDGPU_ASIC_ID_TABLE_PATHS=${AMDGPU_ASIC_ID_TABLE_PATHS:-}"'
    env = {"PYTHONPATH": str(tmp_path / "site"),
           **({"AMDGPU_ASIC_ID_TABLE_PATHS": preset} if preset else {})}
    result = subprocess.run(["bash", "-c", script], env=clean_env(**env), capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_rocm_asic_ids_points_libdrm_at_the_table_the_wheel_ships(tmp_path):
    # The ROCm wheels' libdrm otherwise prints "(null): No such file or directory" for
    # every GPU each process opens.
    expected = tmp_path / "site" / "torch" / "share" / "libdrm"
    assert _asic_ids(tmp_path, ships=True) == f"AMDGPU_ASIC_ID_TABLE_PATHS={expected}"


def test_rocm_asic_ids_leaves_cuda_builds_and_a_set_variable_alone(tmp_path):
    assert _asic_ids(tmp_path / "a", ships=False) == "AMDGPU_ASIC_ID_TABLE_PATHS="
    assert _asic_ids(tmp_path / "b", ships=True, preset="/mine") == (
        "AMDGPU_ASIC_ID_TABLE_PATHS=/mine")


def test_submit_amd_is_one_job_on_an_amd_node():
    out = dry_run(KIT / "explorer" / "submit.sh", "amd", GPU_TYPE_AMD="mi100").stdout
    sbatch = [line for line in out.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, out
    assert "--partition=sharing " in sbatch[0] and "--gres=gpu:mi100:1 " in sbatch[0]
    assert "--time=00:30:00" in sbatch[0] and sbatch[0].rstrip().endswith("amd_node.sbatch")


def test_submit_07_is_one_heterogeneous_job_on_gpu_and_sharing():
    out = dry_run(KIT / "explorer" / "submit.sh", "07", GPU_TYPE_AMD="mi100").stdout
    sbatch = [line for line in out.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, out
    nvidia, amd = sbatch[0].split(" : ")
    assert "--partition=gpu " in nvidia and "--gres=gpu:1 " in nvidia
    assert "--job-name=gpubridge-07_mixed_hetjob" in nvidia
    assert "--partition=sharing " in amd and "--gres=gpu:mi100:1 " in amd
    assert amd.rstrip().endswith("07_mixed_hetjob.sbatch")


def test_submit_07_can_put_the_nvidia_side_on_multigpu():
    out = dry_run(KIT / "explorer" / "submit.sh", "07", PARTITION_MIXED_NVIDIA="multigpu",
                  GPUS_MIXED_NVIDIA="2", GPUS_MIXED_AMD="2").stdout
    nvidia, amd = next(line for line in out.splitlines() if line.startswith("+ sbatch ")).split(
        " : ")
    assert "--partition=multigpu " in nvidia and "--gres=gpu:2 " in nvidia
    assert "--partition=sharing " in amd and ":2 " in amd


def test_mixed_hetjob_skips_hsdp_unless_both_sides_have_as_many_gpus():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w",
                     GPUS_MIXED_NVIDIA="2", GPUS_MIXED_AMD="1")
    assert result.returncode == 0, result.stderr
    assert "--hsdp" not in result.stdout
    assert "skipping the HSDP steps: they need as many NVIDIA GPUs as AMD (2 vs 1)" in result.stderr


def passed_steps(log: str) -> list[str]:
    """The steps a dry run's last "passed:" line names, in order."""
    line = [line for line in log.splitlines() if "] passed: " in line][-1]
    return line.split("] passed: ", 1)[1].split()


TRAINING_07 = ["train_fsdp2_backend", "verify_fsdp2_backend", "train_hsdp_backend",
               "verify_hsdp_backend"]


@pytest.mark.parametrize(("env", "expected"), [
    ({"STEPS_07": "bench"},
     ["probe", "preflight", "train_bench_gpt2-small", "train_bench_resnet50"]),
    ({"STEPS_07": "bench", "TRAIN_BENCH_MODELS": "resnet50", "TRAIN_BENCH_BUCKETS": "10 25"},
     ["probe", "preflight", "train_bench_resnet50_bucket10", "train_bench_resnet50_bucket25"]),
    ({"STEPS_07": "training", "TRAIN_BENCH_BUCKETS": "12.5"},
     ["probe", "preflight", "train_ddp", "train_bench_gpt2-small_bucket12.5",
      "train_bench_resnet50_bucket12.5", *TRAINING_07]),
    ({"STEPS_07": "ddp"}, ["probe", "preflight", "train_ddp"]),
    ({"STEPS_07": "ddp", "TRAIN_DDP_REPEATS": "3"},
     ["probe", "preflight", "train_ddp_1", "train_ddp_2", "train_ddp_3"]),
    ({"STEPS_07": "training", "TRAIN_DDP_REPEATS": "2", "TRAIN_BENCH_MODELS": "resnet50"},
     ["probe", "preflight", "train_ddp_1", "train_ddp_2", "train_bench_resnet50",
      *TRAINING_07]),
])
def test_mixed_hetjob_step_selection(env, expected):
    env = {"TRAIN_BENCH_MODELS": "gpt2-small resnet50", "TRAIN_BENCH_ARGS": "--steps 5",
           "TRAIN_BENCH_BUCKETS": "", **env}
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w", **env)
    assert result.returncode == 0, result.stderr
    assert passed_steps(result.stderr) == expected
    runs = [line for line in result.stdout.splitlines() if "train_bench.py" in line]
    assert len(runs) == len([name for name in expected if name.startswith("train_bench_")])
    for name, line in zip([n for n in expected if n.startswith("train_bench_")], runs,
                          strict=True):
        # Each run on both sides, under its own JSON name and rendezvous, after the
        # extra arguments, with its bucket size when it has one.
        assert line.count(f"/{name}.json --steps 5") == 2, line
        assert line.count(f"--rdzv-id=local-{name.replace('_', '-')} ") == 2, line
        mb = name.partition("_bucket")[2]
        assert line.count("--bucket-mb") == (2 if mb else 0), line
        assert not mb or line.count(f"--steps 5 --bucket-mb {mb} ") == 2, line


@pytest.mark.parametrize("repeats, names", [
    ("1", ["train_ddp"]), ("3", ["train_ddp_1", "train_ddp_2", "train_ddp_3"])])
def test_mixed_hetjob_runs_each_train_ddp_repeat_under_its_own_name(repeats, names):
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w",
                     STEPS_07="ddp", TRAIN_DDP_REPEATS=repeats)
    assert result.returncode == 0, result.stderr
    runs = [line for line in result.stdout.splitlines() if "train_synthetic.py" in line]
    # Each on both sides, with its own JSON and rendezvous; one run keeps the old names.
    for name, line in zip(names, runs, strict=True):
        assert line.count(f"/07_mixed_hetjob/{name}.json") == 2, line
        tag = name.replace("train_ddp", "train").replace("_", "-")
        assert line.count(f"--rdzv-id=local-{tag} ") == 2, line


@pytest.mark.parametrize(("env", "message"), [
    ({"STEPS_07": "hsdp"}, "STEPS_07 must be all, training, ddp, bench, dp, pipeline or megatron"),
    ({"TRAIN_DDP_REPEATS": "0"}, "a whole number from 1 to 999, not '0'"),
    ({"TRAIN_DDP_REPEATS": "2.5"}, "a whole number from 1 to 999, not '2.5'"),
    ({"TRAIN_DDP_REPEATS": "many"}, "a whole number from 1 to 999, not 'many'"),
    ({"TRAIN_DDP_REPEATS": "1000"}, "a whole number from 1 to 999, not '1000'"),
    ({"TRAIN_BENCH_BUCKETS": "10 big"}, "TRAIN_BENCH_BUCKETS: 'big' is not a size in MB"),
    ({"TRAIN_BENCH_BUCKETS": "10", "TRAIN_BENCH_ARGS": "--steps 5 --bucket-mb 5"},
     "in TRAIN_BENCH_BUCKETS or in TRAIN_BENCH_ARGS, not both"),
])
def test_mixed_hetjob_refuses_unknown_step_settings(env, message):
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w", **env)
    assert result.returncode != 0 and message in result.stderr, result.stderr
    assert "srun" not in result.stdout


def test_mixed_hetjob_sweeps_explicit_dp_shares_under_their_own_names():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w", STEPS_07="dp",
                     DP_RUNS="ddp ddp-shares-5-3 ddp-shares-6-2 ddp-shares-5-3-r2")
    assert result.returncode == 0, result.stderr
    names = ["dp_ddp", "dp_ddp-shares-5-3", "dp_ddp-shares-6-2", "dp_ddp-shares-5-3-r2"]
    assert passed_steps(result.stderr) == ["probe", "preflight", *names]
    runs = [line for line in result.stdout.splitlines() if "train_dp.py" in line]
    shares = [None, "5\\,3", "6\\,2", "5\\,3"]  # the dry run prints the comma escaped
    for name, split, line in zip(names, shares, runs, strict=True):
        # Each on both sides, with its own JSON and rendezvous, and its split if it has one.
        assert line.count(f"/07_mixed_hetjob_dp/{name}.json") == 2, line
        assert line.count(f"--rdzv-id=local-{name.replace('_', '-')} ") == 2, line
        assert line.count("--shares") == (2 if split else 0), line
        assert not split or line.count(f".json --shares {split} ") == 2, line


@pytest.mark.parametrize("run", ["ddp-shares-5", "ddp-shares-5-0", "ddp-shares-5,3",
                                 "ddp-shares-a-b", "ddp-shares-5-3-r", "ddp-shares-5-3-r0"])
def test_mixed_hetjob_refuses_a_malformed_dp_split(run):
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w", STEPS_07="dp",
                     DP_RUNS=run)
    assert result.returncode != 0
    assert f"DP_RUNS: '{run}' needs whole rows per rank joined by dashes" in result.stderr
    assert "train_dp.py" not in result.stdout


def test_mixed_hetjob_says_when_train_ddp_repeats_go_unused():
    used = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w", STEPS_07="dp",
                   TRAIN_DDP_REPEATS="3")
    assert "TRAIN_DDP_REPEATS does nothing with STEPS_07=dp" in used.stderr
    assert "train_ddp" not in used.stdout


def test_mixed_hetjob_runs_each_side_in_its_own_venv():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    launches = [line for line in out.splitlines() if "srun --het-group=0" in line]
    # Every launch runs under coreutils timeout (STEP_TIMEOUT, common.sh).
    assert all(line.startswith("+ timeout --kill-after=30 1200 srun ") for line in launches)
    sruns = [line[line.index("srun "):] for line in launches]
    # probe, preflight, check, check_p2p, bench_p2p, bench, a check per opt-in policy,
    # the candidates bench,
    # DDP training, then FSDP2 on the torch.distributed backend (trained, then verified),
    # then HSDP the same way (one GPU per side by default: as many on each side)
    # then train_bench.py on each of its models
    assert len(sruns) >= 13
    assert any("--policy pipelined-reduce-bridge-broadcast" in line for line in sruns[4:])
    assert any("--policy sharded-bridge" in line for line in sruns[4:])
    assert "--write-thresholds" in sruns[-8]
    assert "examples/train_synthetic.py" in sruns[-7] and "train_ddp.json" in sruns[-7]
    assert "train_bench.py --model gpt2-small" in sruns[-6]
    assert "train_bench.py --model resnet50" in sruns[-5]
    assert "examples/fsdp2_torch_backend.py" in sruns[-4] and "--verify" in sruns[-3]
    assert "--hsdp 2 --json" in sruns[-2] and "--verify --hsdp 2" in sruns[-1]
    for line in sruns:
        nvidia, amd = line.split(" : ")
        assert "/w/venv/bin/" in nvidia and "/w/venv-rocm" not in nvidia
        # The MI50's default: the Triton overlay of venv-rocm63 (submit.sh).
        assert amd.startswith("--het-group=1 ") and "/w/venv-rocm63-triton38/bin/" in amd
    assert "--role master" in sruns[1] and "--role worker" in sruns[1]
    assert "probe_nvidia" in sruns[0] and "probe_amd" in sruns[0]
    assert out.index("preflight.py") < out.index("torchrun")
    assert "check.py" in sruns[2] and "--p2p-only" in sruns[3]  # check_p2p
    assert "bench_p2p.py" in sruns[4] and "bench_all_reduce.py" in sruns[5]
    nvidia_bench, amd_bench = sruns[5].replace("\\", "").split(" : ")  # undo printf %q
    assert "--ops island,gpubridge" in nvidia_bench and "--ops island,gpubridge" in amd_bench
    assert "srun --het-group=1 --ntasks=1 rocm-smi" in out


def test_submit_uses_site_settings_from_the_environment():
    result = dry_run(KIT / "explorer" / "submit.sh", "05", ACCOUNT="lab",
                     PARTITION_MULTI="bigq", GPU_TYPE_MULTI="h100", TIME_05="00:05:00")
    for text in ["--account=lab", "--partition=bigq", "--gres=gpu:h100:2", "--time=00:05:00"]:
        assert text in result.stdout


def test_submit_free_and_multi_submit_every_step():
    def submitted(group):
        out = dry_run(KIT / "explorer" / "submit.sh", group).stdout
        return [line.split()[-1].rsplit("/", 1)[-1] for line in out.splitlines()
                if line.startswith("+ sbatch ")]
    assert submitted("free") == ["01_probe.sbatch", "02_split_shared_gpu.sbatch"]
    assert submitted("multi") == ["03_single_island.sbatch", "04_split_4gpu.sbatch",
                                  "05_multinode.sbatch", "06_scaling.sbatch"]


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ("01_probe.sbatch", ["probe.py", "--nproc-per-node=2",
                             "--expect-init-error LOCAL_RANK=1\\ asks\\ for\\ GPU\\ 1"]),
        ("02_split_shared_gpu.sbatch", ["--nproc-per-node=2", "check.py",
                                        "GPUBRIDGE_SPLIT_TEST=alternate", "--nproc-per-node=4",
                                        "--expect-init-error would\\ share\\ a\\ GPU",
                                        # its own limit, not STEP_TIMEOUT
                                        "timeout --kill-after=30 60 env "
                                        "GPUBRIDGE_SPLIT_TEST=alternate",
                                        # the slowest tests end gpu_tests' log
                                        "pytest -m gpu", "--durations=15"]),
        ("03_single_island.sbatch", ["device_map.py", "check.py", "bench_all_reduce.py"]),
        ("04_split_4gpu.sbatch", ["check_half", "check_alternate", "bench_all_reduce.py",
                                  "--durations=15",
                                  "--policy pipelined-reduce-bridge-broadcast",
                                  "--policy sharded-bridge", "--write-thresholds",
                                  "--policy auto-tuned", "train_synthetic.py",
                                  "train_ddp.json", "[check_pinned]",
                                  "env GPUBRIDGE_PINNED_STAGING=1 torchrun",
                                  "bench_pinned --max-bytes"]),
        ("05_multinode.sbatch", ["--policy pipelined-reduce-bridge-broadcast",
                                 "--policy sharded-bridge", "bench_phases_sharded-bridge",
                                 "--write-thresholds"]),
        ("05_multinode.sbatch", ["srun", "--rdzv-backend=c10d", "check_split_by_node",
                                 "--kill-on-bad-exit=1 bash KIT/node_ifname.sh torchrun"]),
        ("05_multinode.sbatch", ["preflight.py --role auto", "--bandwidth-seconds 2"]),
        ("05_multinode.sbatch", ["[check_pinned]", "[bench_pinned]", "check-pinned"]),
        ("05_multinode.sbatch", ["UCX_BRIDGE=1", "[ucx_build]", "python -m gpubridge.ucx build",
                                 "--bridge ucx", "[bench_ucx]"]),
        ("05_multinode.sbatch", ["VERBS_BRIDGE=1", "[verbs_build]", "python -m gpubridge.verbs build",
                                 "verbs_kit.py endpoint", "--nproc-per-node=1",
                                 "--memory gpu --expect-path direct", "--require-no-tcp",
                                 "--bridge verbs", "[bench_verbs]", "[train_verbs]",
                                 "[train_gloo]", "--bridge verbs --json",
                                 "--bridge gloo --json", "rdma_report.sh"]),
        ("07_mixed_hetjob.sbatch", ["VERBS_BRIDGE=1", "[verbs_build]", "[verbs_gpu_staged]",
                                    "--expect-path staged", "[check_verbs]", "[train_verbs]",
                                    "--bridge verbs --json", "--bridge gloo --json"]),
        ("07_mixed_hetjob.sbatch", ["[check_pinned]", "[bench_pinned]"]),
        ("06_scaling.sbatch", ["--nproc-per-node=4", "train_synthetic.py", "summarize.py"]),
        ("07_mixed_hetjob.sbatch", ["--het-group=1", "--rdzv-backend=c10d", "summarize.py",
                                    "export AMDGPU_ASIC_ID_TABLE_PATHS="]),
    ],
)
def test_explorer_jobs_dry_run(job, expected):
    flags = [text for text in expected if text in ("UCX_BRIDGE=1", "VERBS_BRIDGE=1")]
    env = {text.split("=")[0]: "1" for text in flags}
    expected = [text for text in expected if text not in flags]
    result = dry_run(KIT / "explorer" / job, **env)
    assert result.returncode == 0, result.stderr
    for text in expected:
        text = text.replace("KIT/", f"{KIT}/")
        assert text in result.stdout, (text, result.stdout)


def test_amd_dry_run_with_several_gpus():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="8").stdout
    # common.sh exports NCCL_DEBUG=INFO; RCCL's version comes from the step's NCCL logs.
    for text in ["rocm-smi", "probe.py", "--nproc-per-node=9", "LOCAL_RANK=8\\ asks",
                 "nccl/check_1gpu/", "--nproc-per-node=8",
                 "GPUBRIDGE_SPLIT_TEST=half", "bench_all_reduce.py", "train_n4", "tar -czf",
                 "--policy pipelined-reduce-bridge-broadcast", "--policy sharded-bridge",
                 "train_ddp_split.json", "GPUBRIDGE_PINNED_STAGING=1", "bench_pinned",
                 "--durations=15", "export AMDGPU_ASIC_ID_TABLE_PATHS=",
                 "train_bench_gpt2-small_1gpu.json", "train_bench_resnet50_island.json",
                 "train_bench_gpt2-small_split.json"]:
        assert text in out, text
    assert "train_n5" not in out


def test_amd_dry_run_with_one_gpu_shares_it():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="1").stdout
    assert "check_split_shared_gpu" in out and "--nproc-per-node=2" in out
    assert "bench_all_reduce.py" not in out
    # Training throughput on the one GPU alone; nothing to split it with.
    assert "train_bench_gpt2-small_1gpu.json" in out and "_split.json" not in out


def test_amd_dry_run_passes_train_bench_args_to_every_run():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="2", TRAIN_BENCH_MODELS="gpt2-small",
                  TRAIN_BENCH_ARGS="--batch-size 16").stdout
    runs = [line for line in out.splitlines()
            if line.startswith("+ timeout") and "train_bench.py" in line]
    assert len(runs) == 3, out
    assert all("--batch-size 16" in line for line in runs)
    assert "resnet50" not in out


def _overlay_venvs(tmp_path: Path, triton_minor: int | None = None) -> tuple[Path, Path]:
    """A base venv whose python runs this interpreter with its own site-packages (an old
    triton) after PYTHONPATH, and a venv holding a newer triton and its dist-info."""
    minor = f"python3.{sys.version_info.minor}"
    base, newer = tmp_path / "venv-old", tmp_path / "venv-new"
    base_site = base / "lib" / minor / "site-packages"
    (base_site / "triton").mkdir(parents=True)
    (base_site / "triton" / "__init__.py").write_text('__version__ = "3.5.1"\n')
    (base / "bin").mkdir()
    for tool in ("python", "python3", "torchrun"):
        (base / "bin" / tool).write_text(
            "#!/usr/bin/env bash\n"
            f'export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}{base_site}"\n'
            f'exec {sys.executable} "$@"\n')
        (base / "bin" / tool).chmod(0o755)
    newer_site = newer / "lib" / f"python3.{triton_minor or sys.version_info.minor}" / "site-packages"
    (newer_site / "triton").mkdir(parents=True)
    (newer_site / "triton" / "__init__.py").write_text('__version__ = "3.8.0"\n')
    (newer_site / "triton_rocm-3.8.0.dist-info").mkdir()
    return base, newer


@pytest.mark.parametrize("directory", ["plain", "with spaces"])
def test_rocm_triton_overlay_puts_the_newer_triton_first(tmp_path, directory):
    tmp_path = tmp_path / directory
    base, newer = _overlay_venvs(tmp_path)
    out = tmp_path / "venv-old-triton38"
    result = subprocess.run([str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                             str(out)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "triton_rocm-3.8.0" in result.stdout
    overlay = out / "overlay"
    assert sorted(p.name for p in overlay.iterdir()) == ["triton", "triton_rocm-3.8.0.dist-info"]
    assert all(p.is_symlink() for p in overlay.iterdir())  # links, not a 1 GB copy
    for tool in ("python", "python3", "torchrun"):
        run = subprocess.run([str(out / "bin" / tool), "-c",
                              "import triton; print(triton.__version__)"],
                             capture_output=True, text=True, timeout=60)
        assert run.stdout.strip() == "3.8.0", (tool, run.stderr)
    # Without the overlay the base venv's own triton is the one imported.
    run = subprocess.run([str(base / "bin" / "python"), "-c",
                          "import triton; print(triton.__version__)"],
                         capture_output=True, text=True, timeout=60)
    assert run.stdout.strip() == "3.5.1"
    # Rerunning replaces the links instead of failing on them.
    again = subprocess.run(["bash", str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                            str(out)], capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stderr


@pytest.mark.parametrize("destination", ["base", "triton", "alias", "venv"])
def test_rocm_triton_overlay_refuses_source_or_existing_venv(tmp_path, destination):
    base, newer = _overlay_venvs(tmp_path)
    interpreter = base / "bin" / "python"
    original = interpreter.read_bytes()
    if destination == "base":
        out = base
    elif destination == "triton":
        out = newer
    elif destination == "alias":
        out = tmp_path / "alias"
        out.symlink_to(base, target_is_directory=True)
    else:
        out = tmp_path / "existing-venv"
        out.mkdir()
        (out / "pyvenv.cfg").write_text("home = example\n")
    result = subprocess.run([str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                             str(out)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stderr
    assert "OUT_DIR" in result.stderr
    assert interpreter.read_bytes() == original
    assert not (out / "overlay").exists()


@pytest.mark.parametrize("directory", ["bin", "overlay"])
def test_rocm_triton_overlay_refuses_linked_output_directories(tmp_path, directory):
    base, newer = _overlay_venvs(tmp_path)
    original = (base / "bin" / "python").read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    (out / directory).symlink_to(base / "bin", target_is_directory=True)
    result = subprocess.run([str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                             str(out)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "symbolic link" in result.stderr
    assert (base / "bin" / "python").read_bytes() == original


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_rocm_triton_overlay_replaces_wrapper_links_without_overwriting_target(tmp_path, link_type):
    base, newer = _overlay_venvs(tmp_path)
    target = tmp_path / "shared-interpreter"
    target.write_text("preserve the shared interpreter\n")
    out = tmp_path / "out"
    (out / "bin").mkdir(parents=True)
    for tool in ("python", "python3", "torchrun"):
        link = out / "bin" / tool
        if link_type == "symlink":
            link.symlink_to(target)
        else:
            link.hardlink_to(target)
    result = subprocess.run([str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                             str(out)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert target.read_text() == "preserve the shared interpreter\n"
    for tool in ("python", "python3", "torchrun"):
        run = subprocess.run([str(out / "bin" / tool), "-c",
                              "import triton; print(triton.__version__)"],
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0 and run.stdout.strip() == "3.8.0", run.stderr


def test_rocm_triton_overlay_refuses_another_python_version(tmp_path):
    base, newer = _overlay_venvs(tmp_path, triton_minor=sys.version_info.minor + 1)
    result = subprocess.run(["bash", str(KIT / "rocm_triton_overlay.sh"), str(base), str(newer),
                             str(tmp_path / "out")], capture_output=True, text=True, timeout=60)
    minor = sys.version_info.minor
    assert result.returncode == 1, result.stderr
    assert f"{newer} is python3.{minor + 1}, {base} is python3.{minor}" in result.stderr


def test_rocm_triton_overlay_needs_three_arguments():
    result = subprocess.run(["bash", str(KIT / "rocm_triton_overlay.sh")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "BASE_VENV TRITON_VENV OUT_DIR" in result.stderr

@pytest.mark.parametrize(("role", "is_host"), [("master", "1"), ("worker", "0")])
def test_mixed_node_dry_run(role, is_host):
    result = dry_run(KIT / "mixed" / "node.sh", ROLE=role, MASTER_ADDR="100.64.0.1",
                     GPU_COUNT="2")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.index("preflight.py") < out.index("torchrun"), "preflight must come first"
    assert f"--rdzv-conf=is_host={is_host}" in out
    assert "--rdzv-endpoint=100.64.0.1:29500" in out
    assert "--local-addr=" in out


def test_mixed_node_needs_a_role_and_master():
    assert "ROLE=master or ROLE=worker" in dry_run(KIT / "mixed" / "node.sh").stderr
    assert "MASTER_ADDR" in dry_run(KIT / "mixed" / "node.sh", ROLE="worker").stderr


# ---- Python tools end to end on CPU ------------------------------------------------

def read_ranks(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("rank*.json"))]


def test_check_in_a_split_test_is_flagged_everywhere(tmp_path):
    out = tmp_path / "check"
    result = torchrun(4, KIT / "check.py", "--out", str(out),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "NOT a mixed-vendor run" in result.stderr
    records = read_ranks(out)
    assert len(records) == 4
    for record in records:
        assert record["ok"]
        assert record["run"]["kind"] == "split-test"
        assert record["run"]["split_test"] == "half"
        assert record["run"]["real_mixed_vendor"] is False
        assert "NOT a mixed-vendor result" in record["run"]["warning"]
        assert [s["name"] for s in record["stages"]] == [
            "init", "device", "dtypes", "busy_gpu", "broadcast", "reductions", "all_gather",
            "reduce_scatter", "async", "reduction_agreement", "in_flight", "barrier",
            "destroy"]
        agreement = next(s for s in record["stages"] if s["name"] == "reduction_agreement")
        assert agreement["all_product_agree"] is True  # Gloo against Gloo on CPU
        assert agreement["backend"] == "gloo"
    # Islands {0, 1} and {2, 3}; rank 3 is late, so its leader waits in island-reduce.
    phases = {}
    for record in records:
        stage = next(s for s in record["stages"] if s["name"] == "in_flight")
        assert stage["late_rank"] == 3
        flights = stage["in_flight"]
        phases[record["run"]["rank"]] = flights[0]["phase"] if flights else None
        assert len({f["seq"] for f in flights}) <= 1
    assert phases == {0: "bridge", 1: "island-broadcast", 2: "island-reduce", 3: None}
    assert summarize.main([str(out)]) == 0
    assert "SPLIT TEST - not a mixed-vendor result" in (out / "summary.md").read_text()
    assert json.loads((out / "summary.json").read_text())["split_test"] == "half"


@pytest.mark.parametrize("delayed", [0, 2])  # island A's leader, island B's leader
def test_in_flight_stage_does_not_need_ranks_to_arrive_together(delayed, tmp_path):
    # One rank reaches the in_flight stage 1.5 s after the others, as after a slow
    # previous stage on a loaded machine. Every rank must still record its view
    # while the late rank is missing.
    wrapper = tmp_path / "check_skewed.py"
    wrapper.write_text(f"""import sys, time
sys.path.insert(0, {str(KIT)!r})
sys.argv[0] = {str(KIT / "check.py")!r}
import check
stage = check.in_flight_while_late

def skewed(rank, *args, **kwargs):
    if rank == {delayed}:
        time.sleep(1.5)
    return stage(rank, *args, **kwargs)

check.in_flight_while_late = skewed
sys.exit(check.main())
""")
    out = tmp_path / "check"
    result = torchrun(4, wrapper, "--out", str(out), GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    phases = {}
    for record in read_ranks(out):
        flights = next(s for s in record["stages"] if s["name"] == "in_flight")["in_flight"]
        phases[record["run"]["rank"]] = flights[0]["phase"] if flights else None
    assert phases == {0: "bridge", 1: "island-broadcast", 2: "island-reduce", 3: None}


@pytest.mark.parametrize(
    ("pattern", "ok"),
    [("doesn't apply to this cluster", True), ("LOCAL_RANK", False)],
)
def test_check_can_expect_init_to_fail_on_every_rank(pattern, ok, tmp_path):
    # native-only on a split test's two islands fails init() on every rank.
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), "--policy", "native-only",
                      "--expect-init-error", pattern,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert (result.returncode == 0) == ok, result.stderr[-3000:]
    records = read_ranks(out)
    assert len(records) == 2
    for record in records:
        assert record["ok"] is ok
        (stage,) = record["stages"]
        assert stage["name"] == "expected_init_error"
        assert "doesn't apply" in stage["message"]
    for rank in range(2):  # in the log too, where negative_step looks (common.sh)
        assert f"rank {rank}: init() failed: " in result.stderr
    assert result.stderr.count("doesn't apply to this cluster") >= 2
    assert summarize.main([str(out)]) == (0 if ok else 1)


def test_check_expecting_an_init_error_fails_when_init_succeeds(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), "--expect-init-error", "LOCAL_RANK",
                      GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 1
    for record in read_ranks(out):
        assert "init() succeeded" in record["stages"][0]["error"]
    assert "rank 1: init() succeeded; this run expected it to fail" in result.stderr
    assert "LOCAL_RANK" not in result.stderr  # nothing a negative_step could mistake for it


def test_check_without_split_is_not_flagged(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 0, result.stderr[-3000:]
    for record in read_ranks(out):
        assert record["run"]["split_test"] is None
        assert "warning" not in record["run"]
    # One island: rank 0 waits inside the all_reduce, the late rank 1 shows nothing.
    flights = [next(s for s in record["stages"] if s["name"] == "in_flight")["in_flight"]
               for record in read_ranks(out)]
    assert [f["op"] for f in flights[0]] == ["all_reduce"] and flights[1] == []
    summarize.main([str(out)])
    assert "SPLIT TEST" not in (out / "summary.md").read_text()


def test_bench_measures_native_and_bridged_in_one_split_run(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "16K",
                      "--warmup", "1", "--trials", "3",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    # By default: the native baseline, gpubridge and flat-gloo, side by side.
    assert {r["op"] for r in rows} == {"native", "gpubridge", "flat-gloo"}
    assert {r["policy"] for r in rows if r["op"] == "gpubridge"} == {"reduce-bridge-broadcast"}
    assert {r["policy"] for r in rows if r["op"] == "flat-gloo"} == {"flat-gloo"}
    assert {r["bytes"] for r in rows} == {1024, 4096, 16384}
    assert all(r["split_test"] == "half" and r["islands"] == 2 for r in rows)
    assert all(r["timer"] == "host" and r["host_median_us"] > 0 for r in rows)  # CPU
    assert all(r["baseline"] == "native" for r in rows)
    assert (out / "bench.csv").read_text().startswith("op,policy,baseline,run_kind,split_test,")
    summary = (out / "summary.md").read_text()
    assert "SPLIT TEST" in summary and "timed with the host timer" in summary
    assert "| gpubridge / native | flat-gloo / native |" in summary
    assert "fewer ranks" not in summary


def test_bench_times_each_island_natively_in_the_same_job(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "4K",
                      "--warmup", "1", "--trials", "3", "--ops", "island,gpubridge",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    assert {(r["op"], r["world_size"]) for r in rows} == {
        ("island:nvidia", 2), ("island:amd", 2), ("gpubridge", 4)}
    assert {r["bytes"] for r in rows} == {1024, 4096}
    assert {r["baseline"] for r in rows} == {"island-native (fewer ranks, best case)"}
    summary = (out / "summary.md").read_text()
    assert "island:amd median (us)" in summary and "island:nvidia median (us)" in summary
    assert "| gpubridge / island-native (fewer ranks, best case) |" in summary
    assert "gpubridge / native" not in summary
    assert "not a like-for-like comparison" in summary
    assert "flat-gloo" not in summary  # only what --ops asked for


@pytest.mark.parametrize(("ops", "baseline"), [
    (["native", "gpubridge", "flat-gloo"], "native"),
    (["island", "gpubridge", "flat-gloo"], "island-native (fewer ranks, best case)"),  # fallback
    (["gpubridge"], "none"),
])
def test_every_bench_row_names_its_baseline(ops, baseline):
    assert bench_all_reduce.baseline_of(ops) == baseline


def test_bench_reports_the_phases_of_the_gpubridge_op(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "4K",
                      "--warmup", "1", "--trials", "3", "--ops", "island,gpubridge", "--phases",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    bridged = [r for r in rows if r["op"] == "gpubridge"]
    assert len(bridged) == 2
    for r in bridged:
        assert list(r["phases_ms"]) == ["island-reduce", "bridge", "island-broadcast"]
        assert all(ms > 0 for ms in r["phases_ms"].values())
    assert all("phases_ms" not in r for r in rows if r["op"] != "gpubridge")
    assert "Phases of the gpubridge op (reduce-bridge-broadcast)" in (
        out / "summary.md").read_text()
    assert "phases_ms" not in (out / "bench.csv").read_text()


def test_bench_size_helpers():
    assert bench_all_reduce.parse_size("1K") == 1024
    assert bench_all_reduce.parse_size("1GB") == 2**30
    steps = bench_all_reduce.sizes(1024, 2**30, 4)
    assert steps[0] == 1024 and steps[-1] == 2**30 and len(steps) == 11
    info = {"kind": "gpu-mixed", "split_test": None, "islands": [1, 2], "world_size": 4,
            "policy": "reduce-bridge-broadcast"}
    r = bench_all_reduce.row("gpubridge", info, 2**20, "float32", [0.001, 0.002, 0.003])
    assert r["median_us"] == 2000.0
    assert r["policy"] == "reduce-bridge-broadcast"
    assert r["busbw_GBps"] == pytest.approx(r["algbw_GBps"] * 1.5, rel=1e-3)
    island = bench_all_reduce.row("island:amd", info, 2**20, "float32", [0.001], world=2)
    assert island["world_size"] == 2 and island["busbw_GBps"] == island["algbw_GBps"]
    trials = [[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.1, 0.2]]
    assert bench_all_reduce.slowest(trials) == [0.4, 0.5]
    assert bench_all_reduce.slowest(trials, (2, 3)) == [0.3, 0.2]
    timed = bench_all_reduce.row("native", info, 2**20, "float32", [0.001, 0.002, 0.003],
                                 host=[0.004, 0.005, 0.006], timer="cuda-events")
    assert timed["timer"] == "cuda-events" and timed["median_us"] == 2000.0
    assert timed["host_median_us"] == 5000.0 and timed["policy"] == ""
    assert bench_all_reduce.row("flat-gloo", info, 2**20, "float32", [0.001])["policy"] == (
        "flat-gloo")


def test_bench_times_gpu_trials_with_cuda_events(tmp_path, monkeypatch):
    # The GPU branch of time_op, on one CPU rank with fake events: the timed
    # latency is the events' elapsed time, the host one the wall clock.
    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self, stream=None):
            pass

        def elapsed_time(self, end):
            return 2.5  # ms

    import torch.distributed as dist
    monkeypatch.setattr(bench_all_reduce.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(bench_all_reduce.torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(bench_all_reduce, "sync", lambda device: None)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        calls = []
        timed, host = bench_all_reduce.time_op(calls.append, "tensor", torch.device("cuda", 0),
                                               warmup=2, trials=3, max_seconds=60)
    finally:
        dist.destroy_process_group()
    assert len(calls) == 5  # 2 warmup, 3 timed
    assert timed == [[0.0025]] * 3
    assert len(host) == 3 and all(0 < t[0] < 1 for t in host)


def _bench_topology(peers):
    return SimpleNamespace(peers=[SimpleNamespace(vendor=v, hostname=h, gpu=g) for v, h, g in peers])


def test_bench_native_baseline_needs_one_vendor_and_a_gpu_per_rank():
    two_gpus = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n1", "pci 0000:02:00")]
    assert bench_all_reduce.native_possible(_bench_topology(two_gpus)) is None
    shared = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n1", "pci 0000:01:00")]
    assert "share a GPU" in bench_all_reduce.native_possible(_bench_topology(shared))
    other_host = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n2", "pci 0000:01:00")]
    assert bench_all_reduce.native_possible(_bench_topology(other_host)) is None
    mixed = [("nvidia", "n1", "pci 0000:01:00"), ("amd", "n2", "pci 0000:c1:00")]
    assert "different real vendors" in bench_all_reduce.native_possible(_bench_topology(mixed))
    cpu = [("nvidia", "n1", ""), ("nvidia", "n1", "")]
    assert bench_all_reduce.native_possible(_bench_topology(cpu)) is None


def test_training_demo_keeps_ranks_in_sync_and_learns(tmp_path):
    record = tmp_path / "train-n2.json"
    result = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "30",
                      "--dim", "128", "--batch-size", "32", "--json", str(record),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(record.read_text())
    assert data["params_in_sync"] is True
    assert data["loss_last"] < data["loss_first"]
    assert data["run"]["split_test"] == "half" and "warning" in data
    assert "ddp_comm_hook" in data["grad_sync"] and data["policy"] == "reduce-bridge-broadcast"
    # Every rank's versions and settings, as the kit records them.
    assert len({env["pid"] for env in data["environments"]}) == 2
    assert data["environment"] == data["environments"][0]
    for env in data["environments"]:
        assert env["torch"] == torch.__version__ and env["git_commit"]
        assert env["env"]["GPUBRIDGE_VENDOR"] == "nvidia"
    assert data["timeout_s"] > 0 and data["warmup_steps"] == 5


def _train_synthetic():
    spec = importlib.util.spec_from_file_location("train_synthetic",
                                                  REPO / "examples" / "train_synthetic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fingerprints_are_bit_exact_and_independent_of_summation_order():
    ts = _train_synthetic()
    x = torch.randn(1000, generator=torch.Generator().manual_seed(3))
    assert ts.fingerprint(x) == ts.fingerprint(x.clone())
    nudged = x.clone()
    nudged[500] = torch.nextafter(nudged[500], nudged[500] + 1)  # one ulp
    assert ts.fingerprint(nudged) != ts.fingerprint(x)
    # The same bits summed in another order (as another device might) agree.
    bits = x.view(torch.int32).to(torch.int64)
    assert int(bits.flip(0).sum()) == ts.fingerprint(x)[0]
    # -0.0 and 0.0 are equal as floats but not as bits: the check counts them as different.
    assert ts.fingerprint(torch.tensor([0.0])) != ts.fingerprint(torch.tensor([-0.0]))
    assert ts.fingerprint(torch.ones(4, dtype=torch.bfloat16)) != ts.fingerprint(
        torch.full((4,), 2.0, dtype=torch.bfloat16))


def test_mismatch_and_largest_difference_name_the_tensor_and_element():
    ts = _train_synthetic()
    assert ts.first_mismatch([[(1, 2), (3, 4)], [(1, 2), (3, 4)]]) == []
    assert ts.first_mismatch([[(1, 2), (3, 4)], [(1, 2), (3, 5)]]) == [1]
    a, b = torch.zeros(3), torch.tensor([0.0, 0.5, -2.0])
    worst = ts.largest_difference(["w"], [[a], [b]])
    assert worst == {"max_abs_diff": 2.0, "tensor": "w", "index": 2, "rank": 1,
                     "elements_differing": 2}


def test_training_demo_records_no_mismatch_when_ranks_agree(tmp_path):
    record = tmp_path / "train.json"
    result = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "4",
                      "--warmup-steps", "1", "--dim", "32", "--batch-size", "8",
                      "--json", str(record), GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(record.read_text())
    assert data["params_in_sync"] is True
    assert data["first_grad_mismatch"] is None and data["first_param_mismatch"] is None
    assert data["params_checksums_equal"] is True


def test_training_demo_finds_the_step_a_parameter_diverges_and_blames_the_optimizer(tmp_path):
    # A one-ulp change to one parameter on the last rank after step 2: the
    # gradients (synced) still agree, the parameters differ from step 2 on.
    record = tmp_path / "train.json"
    result = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "4",
                      "--warmup-steps", "1", "--dim", "32", "--batch-size", "8",
                      "--perturb-step", "2", "--json", str(record),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode != 0 and "parameters diverged" in result.stderr
    data = json.loads(record.read_text())
    assert data["params_in_sync"] is False and data["first_grad_mismatch"] is None
    first = data["first_param_mismatch"]
    assert first["step"] == 2 and first["tensor"] == "0.weight" and first["index"] == 0
    assert first["tensors"] == ["0.weight"] and first["elements_differing"] == 1
    assert 0 < first["max_abs_diff"] < 1e-6
    assert "params first differed at step 2" in result.stdout


def _ddp_record(in_sync=True, grads=None, params=None, checksums=True, envs=({}, {}),
                kind="gpu-mixed", split=None):
    """A train_synthetic.py record with only what train_ddp_report.py reads."""
    return {"script": "examples/train_synthetic.py", "world_size": 2, "params_in_sync": in_sync,
            "run": {"kind": kind, "split_test": split},
            "first_grad_mismatch": grads, "first_param_mismatch": params,
            "params_checksums_equal": checksums,
            "environment": {"env": dict(envs[0])}, "environments": [{"env": dict(e)} for e in envs]}


def _mismatch(step, diff, tensor):
    return {"step": step, "tensors": [tensor], "max_abs_diff": diff, "tensor": tensor,
            "index": 3, "rank": 1, "elements_differing": 1}


def test_train_ddp_report_gives_each_run_a_verdict_and_counts_divergences(tmp_path):
    a = tmp_path / "a" / "07_mixed_hetjob"
    a.mkdir(parents=True)
    # The job from scontrol's record: a heterogeneous job's first component.
    (a / "slurm-job.txt").write_text("JobId=111 HetJobId=111 HetJobOffset=0 JobName=x\n"
                                     "JobId=112 HetJobId=111 HetJobOffset=1 JobName=x\n")
    (a / "steps.jsonl").write_text("".join(json.dumps({"step": name, "exit_code": code}) + "\n"
                                           for name, code in [
        ("probe", 1), ("train_ddp_1", 0), ("train_ddp_2", 1), ("train_ddp_3", 1),
        ("train_ddp_10", 124)]))
    (a / "train_ddp_1.json").write_text(json.dumps(_ddp_record()))
    (a / "train_ddp_2.json").write_text(json.dumps(_ddp_record(
        in_sync=False, grads=_mismatch(7, 2e-8, "0.weight"),
        params=_mismatch(7, 3.5e-8, "2.weight"), checksums=False)))
    (a / "train_ddp_3.json").write_text(json.dumps(_ddp_record(
        in_sync=False, params=_mismatch(40, 1.2e-7, "4.bias"))))
    b = tmp_path / "b" / "07_mixed_hetjob"
    b.mkdir(parents=True)
    fp8 = {"GPUBRIDGE_BRIDGE_COMPRESSION": "fp8", "SLURM_JOB_ID": "222"}
    (b / "train_ddp.json").write_text(json.dumps(_ddp_record(
        checksums=False, envs=(fp8, {**fp8, "GPUBRIDGE_COMPRESSION_KERNELS": "torch"}))))
    # A record from before the bit-for-bit check: the checksums alone, which differed.
    c = tmp_path / "c"
    c.mkdir()
    (c / "job.jsonl").write_text('{"event": "started", "job_id": "333"}\n')
    old = _ddp_record(in_sync=False)
    for key in ("first_grad_mismatch", "first_param_mismatch", "params_checksums_equal"):
        del old[key]
    old["run"] = {"kind": "split-test", "split_test": "alternate"}
    (c / "train_ddp_split.json").write_text(json.dumps(old))
    (c / "train_ddp_bench.json").write_text(json.dumps(_ddp_record()))  # not train_ddp's

    found = train_ddp_report.rows([tmp_path / "a", tmp_path / "b", c])
    assert [(r["folder"], r["job"], r["repeat"], r["verdict"]) for r in found] == [
        ("a/07_mixed_hetjob", "111", "1", "in sync"),
        ("a/07_mixed_hetjob", "111", "2", "gradients differ after all_reduce"),
        ("a/07_mixed_hetjob", "111", "3", "parameters differ, gradients equal"),
        ("a/07_mixed_hetjob", "111", "10", "no record (step exit 124)"),
        ("b/07_mixed_hetjob", "222", "-", "old checksum only"),
        ("c", "333", "split", "no per-step check")]
    table = train_ddp_report.table(found).splitlines()
    assert table[3] == ("| a/07_mixed_hetjob | 111 | 2 | gpu-mixed | none | auto | yes | 7 | 7 | "
                        "no | 3.5e-08 in 2.weight | gradients differ after all_reduce |")
    assert table[4].endswith("| yes | - | 40 | yes | 1.2e-07 in 4.bias | "
                             "parameters differ, gradients equal |")
    assert table[5].endswith("| - | - | - | - | - | - | - | - | no record (step exit 124) |")
    assert ("| b/07_mixed_hetjob | 222 | - | gpu-mixed | fp8 | auto/torch | no | - | - | no | - |"
            in table[6])
    assert "| c | 333 | split | split-test, split test | none | auto | yes | - | - |" in table[7]
    # One total per setting and kind of run; the split test's old record apart.
    assert train_ddp_report.total(found).splitlines() == [
        "bridge none, kernels auto, gpu-mixed: 2 of 3 diverged",
        "bridge fp8, kernels auto/torch, gpu-mixed: 0 of 1 diverged",
        "bridge none, kernels auto, split-test, split test (float64 checksums only, before the "
        "per-step check): 1 of 1 diverged",
        "1 train_ddp run(s) left no record"]
    # Named twice (relative and absolute), a folder counts once.
    twice = train_ddp_report.rows([b, Path(os.path.relpath(b))])
    assert len(twice) == 1
    assert train_ddp_report.main([str(a / "train_ddp_1.json")]) == 0
    assert train_ddp_report.main([str(tmp_path / "nothing")]) == 1


def test_train_ddp_report_reads_odd_records_without_crashing(tmp_path):
    d = tmp_path / "07_mixed_hetjob"
    d.mkdir()
    # The job planned 6 repeats; its time ran out during the 5th.
    (d / "job.jsonl").write_text('{"event": "started", "job_id": "9"}\n'
                                 '{"event": "train_ddp_repeats", "job_id": "9", "planned": 6}\n')
    (d / "steps.jsonl").write_text('{"step": "train_ddp_4", "exit_code": null}\n')
    (d / "train_ddp_1.json").write_text("[1, 2]")  # not a record
    (d / "train_ddp_2.json").write_text(json.dumps(_ddp_record(  # no max_abs_diff, no step
        in_sync=False, params={"tensor": "0.weight"})))
    (d / "train_ddp_3.json").write_text(json.dumps(_ddp_record(  # params first, grads later
        in_sync=False, grads=_mismatch(30, 1e-8, "1.weight"),
        params=_mismatch(3, 0.0, None))))
    (d / "train_ddp_5.json").write_text(json.dumps(_ddp_record(  # grads only: rounded away
        grads=_mismatch(9, 1e-12, "0.bias"))))
    (d / "train_ddp_7.json").write_text(json.dumps(_ddp_record(in_sync=None)))
    found = train_ddp_report.rows([tmp_path])
    assert [(r["repeat"], r["verdict"]) for r in found] == [
        ("1", "no record (unreadable)"),
        ("2", "parameters differ, gradients equal"),
        ("3", "parameters differ, gradients equal"),
        ("4", "no record (step exit ?)"),
        ("5", "gradients differed, parameters stayed in sync"),
        ("6", "no record (never ran)"),
        ("7", "in sync")]
    assert found[1]["largest"] == "?" and found[1]["first_param_mismatch"] == "?"
    assert found[2]["largest"] == "0 (signed zeros or NaNs)"
    assert train_ddp_report.total(found).splitlines() == [
        "bridge none, kernels auto, gpu-mixed: 2 of 3 diverged",
        "1 record(s) without params_in_sync", "3 train_ddp run(s) left no record"]
    print(train_ddp_report.table(found))  # renders every row


def test_scaling_summary_computes_efficiency(tmp_path):
    for n, sps in [(1, 100.0), (2, 180.0), (4, 300.0)]:
        (tmp_path / f"train-n{n}.json").write_text(json.dumps({
            "world_size": n, "samples_per_sec": sps, "step_ms_median": 1.0,
            "params_in_sync": True, "run": {"kind": "gpu-single-vendor", "split_test": None},
        }))
    assert summarize.main([str(tmp_path)]) == 0
    rows = json.loads((tmp_path / "summary.json").read_text())["rows"]
    assert [round(r["efficiency"], 2) for r in rows] == [1.0, 0.9, 0.75]


def _fake_rank(rank: int, world: int, bus: str, split=None) -> dict:
    return {
        "ok": True, "environment": {"hostname": "node0"},
        "run": {"kind": "split-test" if split else "gpu-single-vendor", "split_test": split,
                "world_size": world, "islands": []},
        "stages": [{"name": "device", "ok": True, "pci_bus_id": bus}],
    }


def test_summary_fails_on_missing_ranks_unless_partial(tmp_path):
    (tmp_path / "rank0.json").write_text(json.dumps(_fake_rank(0, 2, "0000:01:00.0")))
    assert summarize.main([str(tmp_path)]) == 1
    assert summarize.main(["--partial", str(tmp_path)]) == 0


def test_summary_flags_a_shared_gpu_outside_a_split_test(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(_fake_rank(rank, 2, "0000:01:00.0")))
    assert summarize.main([str(tmp_path)]) == 1
    for rank in (0, 1):
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(_fake_rank(rank, 2, "0000:01:00.0", split="half")))
    assert summarize.main([str(tmp_path)]) == 0


def test_environment_names_versions_driver_and_interfaces(monkeypatch):
    import gpukit

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ib0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    env = gpukit.environment()
    assert env["interfaces"] == {"GLOO_SOCKET_IFNAME": "ib0", "NCCL_SOCKET_IFNAME": "auto"}
    assert env["git_commit"] and env["git_dirty"] in (True, False)
    for key in ("torch", "torch_cuda", "torch_hip", "rocm_version", "collective_library",
                "nccl_version", "driver_version", "devices", "hostname"):
        assert key in env
    if not torch.version.cuda and not torch.version.hip:  # a CPU build of PyTorch
        assert env["collective_library"] is None and env["driver_version"] is None


#: The arch list of PyTorch's cu126 wheels.
CU126 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def fake_device(monkeypatch, *, name: str, capability: tuple[int, int], arch_list: list[str],
                gcn: str | None = None, hip: str | None = None) -> dict:
    """gpukit.device_info for one made-up GPU. CUDA builds of PyTorch fill gcnArchName
    with the product name, so that is the default."""
    import gpukit

    props = SimpleNamespace(name=name, major=capability[0], minor=capability[1],
                            gcnArchName=name if gcn is None else gcn, total_memory=32 * 2**30)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))
    return gpukit.device_info(0)


@pytest.mark.parametrize(
    ("gpu", "target", "match"),
    [
        # A CUDA device whose gcnArchName is its product name: the target comes from
        # the compute capability instead.
        ({"name": "NVIDIA A100-SXM4-40GB", "capability": (8, 0), "arch_list": CU126},
         "sm_80", "exact"),
        # A V100 on the cu126 build: its gcnArchName, "Tesla V100-SXM2-32GB", is not
        # its target.
        ({"name": "Tesla V100-SXM2-32GB", "capability": (7, 0), "arch_list": CU126},
         "sm_70", "exact"),
        # ROCm: gcnArchName without its feature flags.
        ({"name": "AMD Instinct MI50", "capability": (9, 0), "gcn": "gfx906:sramecc+:xnack-",
          "hip": "6.4.43482", "arch_list": ["gfx900", "gfx906", "gfx908", "gfx90a"]},
         "gfx906", "exact"),
        # Only PTX for an earlier capability: the driver compiles it at load time.
        ({"name": "NVIDIA H100 80GB HBM3", "capability": (9, 0),
          "arch_list": ["sm_70", "sm_80", "compute_80"]}, "sm_90", "ptx"),
        # A cubin for an earlier minor of the same major runs unchanged.
        ({"name": "NVIDIA L40S", "capability": (8, 9), "arch_list": CU126},
         "sm_89", "compatible"),
        # A genuine miss: no sm_12x cubin and no PTX, so the probe must still fail.
        ({"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "capability": (12, 0),
          "arch_list": CU126}, "sm_120", None),
    ],
)
def test_device_target_and_the_rule_that_covers_it(monkeypatch, gpu, target, match):
    info = fake_device(monkeypatch, **gpu)
    assert (info["target"], info["target_match"]) == (target, match)
    assert info["target_in_build"] is (match is not None)


@pytest.mark.parametrize(
    ("target", "capability", "arch_list", "match"),
    [
        ("sm_80", (8, 0), ["sm_86"], None),            # a later minor's cubin doesn't run
        ("sm_80", (8, 0), ["sm_70"], None),            # nor an earlier major's
        ("sm_80", (8, 0), ["compute_90"], None),       # PTX above the capability doesn't
        ("sm_90", (9, 0), ["sm_90a", "sm_100f"], None),  # arch- and family-specific cubins
        ("sm_75", (7, 5), ["sm_70", "compute_70"], "compatible"),  # the cubin is preferred
        ("sm_120", (12, 0), ["compute_90"], "ptx"),
        ("gfx90a", None, ["gfx90a:xnack-"], "exact"),  # ROCm list entries with flags
        ("gfx942", None, ["gfx90a", "sm_90"], None),
        ("sm_70", (7, 0), [], None),
    ],
)
def test_target_match_rules(target, capability, arch_list, match):
    import gpukit

    assert gpukit.target_match(target, capability, arch_list) == match


def run_probe(monkeypatch, tmp_path, devices: list, *, nccl: bool = True,
              blas: dict | None = None, triton: dict | None = None) -> int:
    """probe.py's main() on a made-up environment (BLAS available unless ``blas``; no
    Triton unless ``triton``; gpubridge imported from this checkout, as it is here)."""
    import gpukit
    import probe

    env = {"device_count": len(devices), "devices": devices, "arch_list": CU126,
           "probe": {"nccl_available": nccl, "gloo_available": True}, "torch": "2.14.1+cu126",
           "torch_cuda": "12.6", "torch_hip": None, "nccl_version": "2.21.5",
           "import": gpukit.kit_import()}
    monkeypatch.setattr(gpukit, "environment", lambda: env)
    monkeypatch.setattr(probe, "blas_check", lambda index: blas or {"blas_available": True})
    monkeypatch.setattr(probe, "triton_check",
                        lambda index: triton or {"triton": None, "compiles": None})
    monkeypatch.setattr(sys, "argv", ["probe.py", "--out", str(tmp_path)])
    return probe.main()


ROCBLAS_ABORT = ("rocBLAS error: Cannot read /venv-rocm/lib/python3.12/site-packages/torch/"
                 "lib/rocblas/library/TensileLibrary.dat: No such file or directory")


def test_probe_records_blas_without_failing_on_it(monkeypatch, tmp_path, capsys):
    # gfx906 on the ROCm 7.2 wheel: rocBLAS has no kernels for it. gpubridge needs none.
    mi50 = fake_device(monkeypatch, name="AMD Instinct MI50", capability=(9, 0),
                       gcn="gfx906:sramecc+:xnack-", hip="7.2.53211", arch_list=["gfx906"])
    blas = {"blas_available": False, "exit_code": -6, "error": ROCBLAS_ABORT}
    assert run_probe(monkeypatch, tmp_path, [mi50], blas=blas) == 0
    assert f"note: no BLAS on device 0 (gpubridge doesn't need it): {ROCBLAS_ABORT}" in (
        capsys.readouterr().err)
    stage = next(s for s in json.loads((tmp_path / "probe.json").read_text())["stages"]
                 if s["name"] == "blas")
    assert stage["ok"] and stage["blas_available"] is False and stage["exit_code"] == -6


def test_blas_check_runs_the_matmul_in_a_child_and_keeps_its_blas_error(monkeypatch):
    import gpukit

    # rocBLAS aborts the process that asked (exit -6) rather than raise.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, -6, stdout="",
                                           stderr=f"some warning\n{ROCBLAS_ABORT}\n")

    monkeypatch.setattr(gpukit.subprocess, "run", fake_run)
    assert gpukit.blas_check(0) == {"blas_available": False, "exit_code": -6,
                                    "error": ROCBLAS_ABORT}
    assert seen["cmd"][0] == sys.executable and "cuda:0" in seen["cmd"][-1]
    monkeypatch.setattr(gpukit.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="64.0\n", stderr=""))
    assert gpukit.blas_check(0) == {"blas_available": True}


def _gpu_tests_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_tests_mutations",
                                                  REPO / "tests" / "test_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_stream_mutation_undoes_its_fix_and_names_its_test():
    from gpubridge import policies, staging, work

    tests = _gpu_tests_module()
    saved = [(work._Worker, "_run_one"), (staging.CudaStaging, "begin"), (work.Work, "wait"),
             (policies.PipelinedReduceBridgeBroadcast, "_pipeline")]
    originals = [getattr(cls, name) for cls, name in saved]
    try:
        for name, (_, test) in tests.MUTATIONS.items():
            before = [getattr(cls, attr) for cls, attr in saved]
            tests.apply_mutation(name)
            after = [getattr(cls, attr) for cls, attr in saved]
            assert sum(a is not b for a, b in zip(before, after, strict=True)) == 1, name
            assert hasattr(tests, test), test  # the guarding test exists
    finally:
        for (cls, attr), original in zip(saved, originals, strict=True):
            setattr(cls, attr, original)
    tests.apply_mutation(None)
    tests.apply_mutation("")
    with pytest.raises(ValueError, match="GPUBRIDGE_TEST_MUTATION='nope'"):
        tests.apply_mutation("nope")


def test_stream_mutations_plans_one_run_per_mutation_and_one_without():
    result = subprocess.run([sys.executable, str(KIT / "stream_mutations.py"), "--dry-run"],
                            env=clean_env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 5
    for name in ("worker-ignores-caller", "staging-ignores-caller", "wait-skips-stream",
                 "slots-forget-copies-back"):
        assert any(f"GPUBRIDGE_TEST_MUTATION={name} " in line for line in lines), name
    assert "GPUBRIDGE_TEST_MUTATION" not in lines[-1] and " or " in lines[-1]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the refusal without a GPU")
def test_stream_mutations_says_when_the_gpu_tests_cannot_run(tmp_path):
    result = subprocess.run([sys.executable, str(KIT / "stream_mutations.py"), "--out",
                             str(tmp_path), "--mutation", "slots-forget-copies-back"],
                            env=clean_env(), capture_output=True, text=True, timeout=600)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "needs a GPU" in result.stderr
    rows = json.loads((tmp_path / "mutations.json").read_text())
    assert [row["verdict"] for row in rows] == ["did not run (no GPU?)"] * 2


def test_time_triton_cache_finds_each_directorys_filesystem(tmp_path):
    sys.path.insert(0, str(KIT))
    import time_triton_cache

    # Mount points under a resolved directory of our own: filesystem() resolves the path
    # first, and on some hosts /home or /tmp is a symlink (macOS: /tmp -> /private/tmp).
    root = tmp_path.resolve()
    mounts = (f"sysfs /sys sysfs rw 0 0\n/dev/sda1 / xfs rw 0 0\n"
              f"tmpfs {root}/tmp tmpfs rw 0 0\nnas:/home {root}/home nfs4 rw 0 0\n"
              f"vast:/projects {root}/projects nfs rw 0 0\n")
    assert time_triton_cache.filesystem(root / "home/someuser/.triton", mounts) == "nfs4"
    assert time_triton_cache.filesystem(root / "tmp/x", mounts) == "tmpfs"
    assert time_triton_cache.filesystem(root / "projects/lab/user", mounts) == "nfs"
    assert time_triton_cache.filesystem(root / "opt/x", mounts) == "xfs"
    assert len(time_triton_cache.VARIANTS) == 12  # bf16 mode skips 16-bit inputs


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the refusal without a GPU")
def test_time_triton_cache_needs_a_gpu():
    result = subprocess.run([sys.executable, str(KIT / "time_triton_cache.py")],
                            env=clean_env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "needs a GPU" in result.stderr


def _triton_cache(tmp_path, *, store: str | None, preset: str | None = None,
                  job: str = "77") -> str:
    """triton_cache_begin, a new kernel built into the cache, triton_cache_end."""
    script = f"""
source {KIT}/common.sh
RESULTS={tmp_path}
triton_cache_begin
echo "dir=${{TRITON_CACHE_DIR:-}}"
new="${{TRITON_CACHE_DIR:-}}/newkernel"
[ -z "${{TRITON_CACHE_DIR:-}}" ] || {{ mkdir -p "$new" && touch "$new/__grp__k.json"; }}
finish >/dev/null 2>&1 || true
left=no
[ -z "${{TRITON_CACHE_DIR:-}}" ] || [ ! -d "$TRITON_CACHE_DIR" ] || left=yes
echo "local_left=$left"
"""
    env = {"TRITON_CACHE_LOCAL": str(tmp_path / "node" / "triton"), "SLURM_JOB_ID": job}
    if store is not None:
        env["TRITON_CACHE_STORE"] = store
    if preset:
        env["TRITON_CACHE_DIR"] = preset
    (tmp_path / "node").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["bash", "-c", script], env=clean_env(**env), capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _gnu_mv() -> bool:
    """Whether mv has GNU's -T, as on the Linux nodes the kit runs on (BSD mv hasn't)."""
    result = subprocess.run(["mv", "--version"], capture_output=True, text=True)
    return result.returncode == 0 and "GNU" in result.stdout


def test_triton_cache_begin_fills_node_local_cache_portably_from_store(tmp_path):
    store = tmp_path / "store"
    (store / "oldkernel").mkdir(parents=True)
    (store / "oldkernel" / "k.json").write_text("{}")
    local = tmp_path / "node" / "triton"
    script = f"""
source {KIT}/common.sh
RESULTS={tmp_path}
TRITON_CACHE_STORE={store}
TRITON_CACHE_LOCAL={local}
triton_cache_begin
echo "dir=${{TRITON_CACHE_DIR:-}}"
"""
    result = subprocess.run(["bash", "-e", "-c", script], env=clean_env(), capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert f"dir={local}" in result.stdout
    assert (local / "oldkernel" / "k.json").exists()


@pytest.mark.skipif(not _gnu_mv(), reason="triton_cache_end renames with GNU mv -T")
def test_a_job_builds_triton_kernels_in_a_node_local_cache_kept_in_the_store(tmp_path):
    store = tmp_path / "store"
    (store / "oldkernel").mkdir(parents=True)
    (store / "oldkernel" / "k.json").write_text("{}")
    out = _triton_cache(tmp_path, store=str(store))
    local = tmp_path / "node" / "triton"
    assert f"dir={local}" in out  # filled from the store, then used for the job
    assert (local / "oldkernel" / "k.json").exists()
    assert (store / "newkernel" / "__grp__k.json").exists()  # the job's new kernels kept
    assert not list(store.glob(".incoming-*"))  # added by rename, nothing half-copied
    # The same path for the next job: Triton's group files name kernels by absolute
    # path, so the store's kernels only load where they were built.
    assert f"dir={local}" in _triton_cache(tmp_path, store=str(store), job="78")
    assert "local_left=yes" in out  # kept: other jobs on the node may be using it


def test_a_triton_cache_directory_not_our_own_is_not_used(tmp_path):
    # /tmp is shared, and Triton imports compiled code from its cache: a symlink (or a
    # directory someone else made) is refused, and Triton keeps its default.
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "node" / "triton"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    out = _triton_cache(tmp_path, store=str(tmp_path / "store"))
    assert "dir=\n" in out


def test_only_finished_kernels_reach_the_store(tmp_path):
    store = tmp_path / "store"
    local = tmp_path / "node" / "triton"
    (local / "building").mkdir(parents=True)  # no __grp__ file yet: another job's build
    (local / "building" / "k.cubin").write_text("half")
    _triton_cache(tmp_path, store=str(store))
    assert not (store / "building").exists()
    assert not list(store.glob(".incoming-*"))


def test_a_set_triton_cache_dir_or_an_empty_store_leaves_triton_alone(tmp_path):
    assert "dir=/mine" in _triton_cache(tmp_path / "a", store=str(tmp_path / "s"),
                                        preset="/mine")
    assert "dir=\n" in _triton_cache(tmp_path / "b", store="")
    assert "dir=\n" in _triton_cache(tmp_path / "c", store=None)  # no store: off


def test_queue_gpu_delay_does_nothing_on_cpu():
    import gpukit

    gpukit.queue_gpu_delay(torch.device("cpu"), 1.0)  # returns at once, no error
    gpukit.queue_gpu_delay(torch.device("cpu"), 0)


def test_no_busy_work_uses_blas():
    # Busy work stays BLAS-free: the ROCm wheels' rocBLAS has no kernels for gfx906.
    for path in (KIT / "check.py", REPO / "tests" / "test_gpu.py"):
        text = path.read_text()
        assert not re.search(r"\w @ \w|\.mm\(|\bmatmul\(|\bbmm\(|\blinear\(", text), path.name


ROCM_SMI_DRIVER = 'name, value\n"Driver version", "5.14.0-611.5.1.el9_7.x86_64"\n'


@pytest.mark.parametrize(("raw", "want"), [
    (ROCM_SMI_DRIVER, "5.14.0-611.5.1.el9_7.x86_64"),       # rocm-smi 6.x and later
    ("device,Driver version\ncard0,6.7.0\ncard1,6.7.0\n", "6.7.0"),  # older, per card
    ("550.54.15", "550.54.15"),                               # nvidia-smi: as is
    (None, None),
])
def test_driver_text_reads_rocm_smi_csv(raw, want):
    assert validation_tables.driver_text(raw) == want


def test_amd_driver_version_is_recorded_as_the_value(monkeypatch):
    import gpukit

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.2.53211")
    monkeypatch.setattr(gpukit, "_read", lambda path: None)  # an in-tree amdgpu
    monkeypatch.setattr(gpukit, "_command", lambda *args: ROCM_SMI_DRIVER.strip())
    assert gpukit.driver_version() == "5.14.0-611.5.1.el9_7.x86_64"


def test_probe_passes_a_gpu_its_build_covers(monkeypatch, tmp_path, capsys):
    v100 = fake_device(monkeypatch, name="Tesla V100-SXM2-32GB", capability=(7, 0),
                       arch_list=CU126)
    assert run_probe(monkeypatch, tmp_path, [v100]) == 0
    out = capsys.readouterr()
    assert "GPUs: Tesla V100-SXM2-32GB (sm_70, exact)" in out.out and "FAIL" not in out.err
    record = json.loads((tmp_path / "probe.json").read_text())
    assert record["ok"] and record["environment"]["devices"][0]["target_match"] == "exact"


def test_probe_names_each_failed_check_and_its_values(monkeypatch, tmp_path, capsys):
    blackwell = fake_device(monkeypatch, name="NVIDIA RTX PRO 6000", capability=(12, 0),
                            arch_list=CU126)
    unreadable = "error: RuntimeError: CUDA error: unknown error"
    assert run_probe(monkeypatch, tmp_path, [blackwell, unreadable], nccl=False) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"FAIL device_info: device 1: {unreadable}",
        f"FAIL target_in_build: device 0 target sm_120 not in arch_list {CU126} (nor an sm_12x "
        "cubin at or below 12.0, nor PTX at or below 12.0)",
        "FAIL backends: nccl_available False, gloo_available True",
    ]


def test_probe_without_a_gpu_says_so(monkeypatch, tmp_path, capsys):
    assert run_probe(monkeypatch, tmp_path, []) == 1
    assert capsys.readouterr().err.splitlines() == [
        "FAIL gpus_visible: device_count 0: no GPU visible to PyTorch (check the job's GPU "
        "request and driver)"]


#: Must never reach a record: credentials, who ran it, where the project lives.
SECRETS = {
    "GPUBRIDGE_API_TOKEN": "tok-0a1b2c3d", "NCCL_SECRET_KEY": "key-4e5f6a7b",
    "GLOO_PASSWORD": "pw-8c9d0e1f", "AWS_SECRET_ACCESS_KEY": "aws-2a3b4c5d",
    "SLURM_JOB_USER": "someuser", "SLURM_JOB_ACCOUNT": "example-lab-account",
    "TORCH_INDEX_URL": "https://bob:hunter2@pypi.example.org/simple",
    "GPUBRIDGE_REPO_DIR": f"{FAKE_PROJECT}/gpubridge",
}


def test_records_hold_only_allowlisted_env_vars_and_no_secrets(tmp_path):
    env = {**SECRETS, "GPUBRIDGE_VENDOR": "nvidia", "SLURM_JOB_ID": "4242"}
    probe = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(tmp_path)],
                           env=clean_env(**env), capture_output=True, text=True, timeout=120)
    train = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "2",
                     "--warmup-steps", "1", "--dim", "16", "--batch-size", "4",
                     "--json", str(tmp_path / "train.json"), **env)
    assert train.returncode == 0, train.stderr[-3000:]
    for name in ("probe.json", "train.json"):
        text = (tmp_path / name).read_text()
        for secret in ("tok-0a1b2c3d", "key-4e5f6a7b", "pw-8c9d0e1f", "aws-2a3b4c5d",
                       "someuser", "example-lab", "hunter2", FAKE_PROJECT, str(Path.home())):
            assert secret not in text, (name, secret, probe.stderr[-2000:])
        for env_record in [json.loads(text)["environment"]]:
            recorded = env_record["env"]
            assert recorded["GPUBRIDGE_VENDOR"] == "nvidia" and recorded["SLURM_JOB_ID"] == "4242"
            assert recorded["TORCH_INDEX_URL"] == "https://***@pypi.example.org/simple"
            assert recorded["GPUBRIDGE_REPO_DIR"] == "$PROJECT_DIR/gpubridge"
            assert not set(recorded) & {"GPUBRIDGE_API_TOKEN", "NCCL_SECRET_KEY",
                                        "GLOO_PASSWORD", "AWS_SECRET_ACCESS_KEY",
                                        "SLURM_JOB_USER", "SLURM_JOB_ACCOUNT", "HOME", "USER",
                                        "PATH"}


def test_redact_replaces_paths_users_and_url_credentials(monkeypatch):
    import gpukit

    monkeypatch.setenv("HOME", "/home/someuser")
    monkeypatch.setattr(gpukit, "_user", lambda: "someuser")
    assert gpukit.redact("/projects/example-lab/gpubridge/results/x.json") == (
        "$PROJECT_DIR/gpubridge/results/x.json")
    assert gpukit.redact("/home/someuser/venv/bin/python") == "~/venv/bin/python"
    assert gpukit.redact("/home/someusers/x") == "/home/someusers/x"
    assert gpukit.redact("/scratch/someuser/out /scratch/someusers") == (
        "/scratch/$USER/out /scratch/someusers")
    assert gpukit.redact("git+https://u:t@host/r@abc") == "git+https://***@host/r@abc"
    assert gpukit.scrub({"argv": ["--out", "/projects/x/r"], "n": 1}) == {
        "argv": ["--out", "$PROJECT_DIR/r"], "n": 1}


# ---- validation_tables.py ------------------------------------------------------------

def test_predicted_bridge_time_is_a_round_trip_plus_the_ring_traffic():
    # Two islands: the whole tensor each way. 100 MB at 100 MB/s is a second.
    assert validation_tables.predicted_bridge_ms(10**8, 2, 100.0, 0.5) == pytest.approx(1000.5)
    # Four islands send 2(p-1)/p = 1.5 times the tensor, and wait p-1 = 3 round trips.
    assert validation_tables.predicted_bridge_ms(10**8, 4, 100.0, 0.0) == pytest.approx(1500)
    assert validation_tables.predicted_bridge_ms(10**8, 4, 100.0, 0.5) == pytest.approx(1501.5)


def test_verdict_counts_only_gpu_runs():
    evidence = validation_tables.Evidence
    assert validation_tables.verdict([]) == "not run"
    assert validation_tables.verdict([evidence("a", True, True)]) == "simulation only"
    assert validation_tables.verdict(
        [evidence("a", True, False), evidence("b", True, False), evidence("c", False, True)]
    ) == "pass (2)"
    assert validation_tables.verdict(
        [evidence("a", True, False), evidence("b", False, False)]) == "FAIL (1 of 2)"


@pytest.mark.docs
def test_tables_cover_every_hardware_validation_item():
    text = (REPO / "HARDWARE_VALIDATION.md").read_text().split("## Mixed build results")[0]
    numbers = {int(m) for m in re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)}
    assert numbers == set(validation_tables.ITEMS)


def test_numerical_gate_is_not_inferred_from_successful_legacy_training(tmp_path):
    job = _job(tmp_path, "legacy-training", [("train_verbs", 0), ("train_gloo", 0)],
               GPU_ENV, "cuda")
    rows = validation_tables.item_rows([validation_tables.load_job(job, tmp_path)])
    numerical = next(row for row in rows if row["item"] == 43)
    assert numerical["verdict"] == "not run" and numerical["evidence"] == []
    assert "manual acceptance" in numerical["tested_by"][0]


def test_every_step_the_tables_look_for_is_one_a_job_runs():
    runs = [dry_run(KIT / "explorer" / job, UCX_BRIDGE="1", MULTITCP_BRIDGE="1",
                    VERBS_BRIDGE="1") for job in sorted(
        p.name for p in (KIT / "explorer").glob("0*.sbatch"))]
    runs += [dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", STEPS_07="dp"),
             dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", STEPS_07="pipeline"),
             dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", STEPS_07="pipeline",
                     PP_SPLITS="even 8,4 auto"),
             dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", STEPS_07="megatron",
                     MEGATRON_PATH="/opt/megatron-core"),
             dry_run(KIT / "explorer" / "dp_baseline.sbatch"),
             dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="4"),
             dry_run(KIT / "mixed" / "node.sh", ROLE="master", MASTER_ADDR="10.0.0.1",
                     GPU_COUNT="2")]
    names = set(re.findall(r"^\+ \[([^\]]+)\]", "".join(r.stdout for r in runs),
                           flags=re.MULTILINE))
    for _, tests in validation_tables.ITEMS.values():
        for test in tests:
            if isinstance(test, validation_tables.Step):
                assert any(re.fullmatch(test.pattern, n) for n in names), test.pattern


GPU_ENV = {"hostname": "gpu-a", "devices": [{"name": "NVIDIA H200"}], "driver_version": "570.86",
           "torch": "2.8.0+cu128", "torch_cuda": "12.8", "torch_hip": None,
           "collective_library": "nccl", "nccl_version": "2.26.2", "git_commit": "abc1234",
           "git_dirty": False, "interfaces": {"GLOO_SOCKET_IFNAME": "ib0"}, "env": {}}


def _job(root: Path, name: str, steps: list[tuple[str, int]], env: dict, device: str) -> Path:
    """A job's results as the kit leaves them: steps.jsonl and the steps' records."""
    job = root / name
    job.mkdir(parents=True)
    (job / "steps.jsonl").write_text("".join(
        json.dumps({"step": step, "exit_code": code,
                    **({"result": "expected failure confirmed"} if step == "shared_gpu_check"
                       else {})}) + "\n" for step, code in steps))
    for rank, host in enumerate(("gpu-a", "gpu-b")):
        (job / "check_split_by_node").mkdir(exist_ok=True)
        (job / "check_split_by_node" / f"rank{rank}.json").write_text(json.dumps({
            "script": "scripts/gpu/check.py", "environment": {**env, "hostname": host},
            "run": {"kind": "split-test", "device": device},
            "stages": [{"name": "init", "ok": True}, {"name": "barrier", "ok": True}]}))
    for role, host, ifname, out, back in (("master", "gpu-a", "ib0", 1000.0, 800.0),
                                          ("worker", "gpu-b", "ib1", 800.0, 1000.0)):
        (job / "preflight").mkdir(exist_ok=True)
        (job / "preflight" / f"preflight-{role}.json").write_text(json.dumps({
            "role": role, "hostname": host, "chosen_ifname": ifname, "environment": env,
            "bandwidth": {"MBps_out": out, "MBps_in": back, "rtt_ms": 0.5}}))
    rows = []
    for size in (2**20, 64 * 2**20, 256 * 2**20):
        ms = size / 2**20  # 1 ms per MiB, so ratios are easy to check
        rows += [{"op": "native", "bytes": size, "median_us": ms * 100, "islands": 2},
                 {"op": "gpubridge", "bytes": size, "median_us": ms * 1500, "islands": 2,
                  "policy": "reduce-bridge-broadcast", "phases_ms": {"bridge": ms}},
                 {"op": "flat-gloo", "bytes": size, "median_us": ms * 3000, "islands": 2}]
    (job / "bench_phases_x").mkdir()
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps({
        "environment": env, "run": {"kind": "split-test", "device": device},
        "baseline": "native", "rows": rows}))
    return job


@pytest.mark.parametrize("historical_exit", [0, 1])
def test_historical_gpu_tests_are_not_p2p_evidence(tmp_path, historical_exit):
    _job(tmp_path, "old", [("gpu_tests", historical_exit)], GPU_ENV, "cuda:0")
    rows = validation_tables.build([tmp_path])["items"]
    row = next(row for row in rows if row["item"] == 32)
    assert row["verdict"] == "not run" and row["evidence"] == []


@pytest.mark.parametrize("exit_code,expected", [(0, "pass (1)"), (1, "FAIL (1 of 1)")])
def test_p2p_step_is_p2p_evidence(tmp_path, exit_code, expected):
    _job(tmp_path, "new", [("check_p2p", exit_code)], GPU_ENV, "cuda:0")
    row = next(row for row in validation_tables.build([tmp_path])["items"] if row["item"] == 32)
    assert row["verdict"] == expected


def test_train_ddp_repeats_are_each_item_22_evidence(tmp_path):
    _job(tmp_path, "07_mixed_hetjob", [("train_ddp_1", 0), ("train_ddp_2", 1), ("train_ddp_3", 0),
                                       ("train_ddp_bench", 0)], GPU_ENV, "cuda:0")
    _job(tmp_path, "amd_node", [("train_ddp_split", 0)], GPU_ENV, "cuda:0")
    row = next(row for row in validation_tables.build([tmp_path])["items"] if row["item"] == 22)
    assert row["verdict"] == "FAIL (1 of 4)"
    assert [e["where"] for e in row["evidence"]] == [
        "07_mixed_hetjob/train_ddp_1", "07_mixed_hetjob/train_ddp_2",
        "07_mixed_hetjob/train_ddp_3", "amd_node/train_ddp_split"]


def test_tables_from_a_results_tree(tmp_path):
    passed = [("preflight", 0), ("check_split_by_node", 0), ("bench_phases_x", 0),
              ("shared_gpu_check", 0)]
    _job(tmp_path, "explorer/05_multinode", passed, GPU_ENV, "cuda:0")
    # A simulation run whose steps all failed must not change any verdict.
    _job(tmp_path, "sim/05_multinode", [(name, 1) for name, _ in passed],
         {**GPU_ENV, "env": {"GPUBRIDGE_VENDOR": "nvidia"}}, "cpu")
    assert validation_tables.main([str(tmp_path)]) == 0
    tables = json.loads((tmp_path / "validation_tables.json").read_text())
    md = (tmp_path / "validation_tables.md").read_text()

    real, sim = tables["runs"]
    assert not real["simulated"] and sim["simulated"]
    assert real["hosts"] == "host 1, host 2" and real["gpus"] == "NVIDIA H200"
    assert (real["driver"], real["toolkit"], real["collectives"], real["gpubridge"],
            real["gloo_ifname"]) == ("570.86", "CUDA 12.8", "NCCL 2.26.2", "abc1234", "ib0")
    assert real["negative_steps"] == {"shared_gpu_check": "expected failure confirmed"}

    verdicts = {row["item"]: row["verdict"] for row in tables["items"]}
    assert verdicts[8] == "pass (2)"  # the preflight and check_split_by_node
    assert verdicts[10] == "pass (1)" and verdicts[24] == "pass (1)"
    assert verdicts[1] == "not run"  # needs a real mixed run, not a split test
    assert verdicts[22] == "not run"

    network = tables["network"][0]
    assert (network["master"], network["worker"]) == ("host 1 ib0", "host 2 ib1")
    assert (network["MBps_master_to_worker"], network["MBps_worker_to_master"]) == (1000, 800)

    bench = [r for r in tables["benchmarks"] if not r["simulated"]]
    assert [r["bytes"] for r in bench] == [2**20, 64 * 2**20, 256 * 2**20]
    assert {(r["baseline"], r["gpubridge_over_baseline"], r["flat-gloo_over_baseline"])
            for r in bench} == {("native", 15.0, 30.0)}
    assert {(r["bridge"], r["staging"]) for r in tables["benchmarks"]} == {("gloo", "pageable")}

    predicted = next(r for r in tables["predicted"]
                     if r["bytes"] == 64 * 2**20 and not r["simulated"])
    assert predicted["MBps"] == 800  # the slower direction
    assert predicted["predicted_ms"] == pytest.approx(0.5 + 64 * 2**20 / 800e6 * 1e3, abs=1e-3)
    assert predicted["measured_ms"] == 64

    assert "never count towards an item" in md
    assert "| sim/05_multinode (simulation) |" in md
    assert "explorer/05_multinode/shared_gpu_check: expected failure confirmed" in md
    assert "## Predicted vs measured bridge time" in md
    # Safe to publish: no hostname, no path.
    out = md + (tmp_path / "validation_tables.json").read_text()
    assert "gpu-a" not in out and "gpu-b" not in out and str(tmp_path) not in out


def test_tables_name_the_island_baseline(tmp_path):
    job = _job(tmp_path, "amd/target", [("bench", 0)], GPU_ENV, "cuda:0")
    bench = json.loads((job / "bench_phases_x" / "bench.json").read_text())
    del bench["baseline"]  # a bench.json from before it named its baseline
    bench["rows"] = [{**r, "op": "island:nvidia"} if r["op"] == "native" else r
                     for r in bench["rows"]]
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps(bench))
    rows = validation_tables.build([tmp_path])["benchmarks"]
    assert {r["baseline"] for r in rows} == {"island-native (fewer ranks, best case)"}


def test_tables_need_job_results(tmp_path, capsys):
    assert validation_tables.main([str(tmp_path)]) == 1
    assert "no job results" in capsys.readouterr().err


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_preflight_pair(tmp_path: Path, *extra: str, env=None) -> dict[str, dict]:
    """Both sides of a preflight on this machine; their records by role."""
    cmd = [sys.executable, str(KIT / "preflight.py"), "--master-addr", "127.0.0.1",
           "--port", str(_free_port()), "--timeout", "30", "--bandwidth-seconds", "0.3",
           "--out", str(tmp_path), *extra]
    master = subprocess.Popen([*cmd, "--role", "master"], env=env or clean_env(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    worker = subprocess.run([*cmd, "--role", "worker"], env=env or clean_env(),
                            capture_output=True, text=True, timeout=120)
    master_err = master.communicate(timeout=120)[1]
    assert worker.returncode == 0, worker.stderr
    assert master.returncode == 0, master_err
    return {role: json.loads((tmp_path / f"preflight-{role}.json").read_text())
            for role in ("master", "worker")}


def test_preflight_passes_on_loopback(tmp_path):
    records = run_preflight_pair(tmp_path)
    worker, master = records["worker"], records["master"]
    assert [s["name"] for s in worker["stages"]] == [
        "resolve", "interface", "tcp", "store", "interfaces", "reachability", "choose", "gloo"]
    assert "tcp" not in [s["name"] for s in master["stages"]]
    # Both sides agree on one pair, and Gloo ran on it.
    assert master["chosen_ifname"] == worker["peer_ifname"]
    assert worker["chosen_ifname"] == master["peer_ifname"]
    assert master["pairs"] == worker["pairs"] and master["pairs"]
    gloo = worker["stages"][-1]["result"]
    assert gloo["GLOO_SOCKET_IFNAME"] == worker["chosen_ifname"]
    for record in records.values():
        bandwidth = record["bandwidth"]
        assert bandwidth["MBps_in"] > 0 and bandwidth["MBps_out"] > 0
        assert bandwidth["rtt_ms"] > 0 and bandwidth["streams"] == 1
        assert "one stream driven from Python; may understate fast links" in bandwidth["note"]
    # The other side's out is this side's in.
    assert master["bandwidth"]["MBps_out"] == worker["bandwidth"]["MBps_in"]
    for record in records.values():
        assert record["environment"]["hostname"] and "--out" in record["argv"]


def test_preflight_measures_only_the_interface_it_is_given(tmp_path):
    names = {name for _, name in socket.if_nameindex()}
    lo = next(name for name in ("lo", "lo0") if name in names)
    records = run_preflight_pair(tmp_path, env=clean_env(GLOO_SOCKET_IFNAME=lo))
    for record in records.values():
        assert record["chosen_ifname"] == lo
        assert [(p["master"], p["worker"]) for p in record["pairs"]] == [(lo, lo)]


def test_preflight_without_a_worker_fails_cleanly(tmp_path):
    # The master alone: the store's wait for the worker times out, and the
    # preflight reports it and writes its record instead of a traceback.
    result = subprocess.run([sys.executable, str(KIT / "preflight.py"), "--role", "master",
                             "--master-addr", "127.0.0.1", "--port", str(_free_port()),
                             "--timeout", "3", "--out", str(tmp_path)],
                            env=clean_env(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "FAILED: reachability:" in result.stderr and "Traceback" not in result.stderr
    record = json.loads((tmp_path / "preflight-master.json").read_text())
    assert record["ok"] is False
    assert record["stages"][-1]["name"] == "reachability" and not record["stages"][-1]["ok"]


def test_preflight_candidates_skip_loopback_and_container_bridges():
    found = {"lo": "127.0.0.1", "eno1": "10.1.0.5", "ib0": "10.2.0.5",
             "docker0": "172.17.0.1", "veth12": "10.3.0.1", "usb0": "169.254.3.1"}
    assert preflight.candidates(found, None, loopback=False) == [
        ("eno1", "10.1.0.5"), ("ib0", "10.2.0.5")]
    assert ("lo", "127.0.0.1") in preflight.candidates(found, None, loopback=True)
    assert preflight.candidates(found, "ib0", loopback=False) == [("ib0", "10.2.0.5")]
    with pytest.raises(preflight.PreflightError, match="no IPv4 address"):
        preflight.candidates(found, "wg0", loopback=False)
    with pytest.raises(preflight.PreflightError, match="no candidate interface"):
        preflight.candidates({"lo": "127.0.0.1"}, None, loopback=False)


def _pair(master, worker, m2w=None, w2m=None, error=None):
    pair = {"master": master, "master_address": "10.0.0.1", "worker": worker,
            "worker_address": "10.0.0.2", "connects": error is None}
    if error is None:
        pair.update(MBps_master_to_worker=m2w, MBps_worker_to_master=w2m, rtt_ms=0.1)
    else:
        pair["error"] = error
    return pair


def test_preflight_chooses_the_pair_whose_slower_direction_is_fastest():
    pairs = [_pair("eno1", "eno1", 117, 118), _pair("ib0", "ib0", 3000, 900),
             _pair("ib1", "ib1", 1200, 1100), _pair("ib2", "ib2", error="timed out")]
    assert preflight.choose(pairs)["master"] == "ib1"
    with pytest.raises(preflight.PreflightError, match="ib2 10.0.0.1 <-> worker ib2"):
        preflight.choose([_pair("ib2", "ib2", error="timed out")])


def test_preflight_pairs_follow_the_routes():
    # On one machine every local address routes from itself.
    mine = [("lo0", "127.0.0.1")]
    assert preflight.routed_pairs(mine, [("lo", "127.0.0.1")], "master") == [("lo0", "lo")]
    assert preflight.routed_pairs(mine, [("lo", "127.0.0.1")], "worker") == [("lo", "lo0")]
    assert preflight.routed_pairs([("eno1", "10.9.9.9")], [("lo", "127.0.0.1")], "master") == []


def test_preflight_role_auto_follows_srun(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert preflight.resolve_role("auto") == "master"
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert preflight.resolve_role("auto") == "worker"
    assert preflight.resolve_role("master") == "master"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--master-addr", "127.0.0.1", "--port", "PORT", "--timeout", "2"],
         "could not reach 127.0.0.1"),
        (["--master-addr", "no-such-host.invalid", "--port", "1"], "cannot resolve"),
        (["--master-addr", "127.0.0.1", "--port", "1", "--ifname", "no-such-if0"],
         "does not exist here"),
    ],
)
def test_preflight_fails_fast_with_a_clear_message(args, message):
    args = [str(_free_port()) if a == "PORT" else a for a in args]
    result = subprocess.run([sys.executable, str(KIT / "preflight.py"), "--role", "worker",
                             *args], env=clean_env(), capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 1
    assert message in result.stderr


def test_probe_reports_a_missing_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(tmp_path)],
                            env=clean_env(), capture_output=True, text=True, timeout=120)
    record = json.loads((tmp_path / "probe.json").read_text())
    if record["environment"]["device_count"] == 0:
        assert result.returncode == 1
        assert record["stages"][0]["name"] == "gpus_visible"
        assert record["stages"][0]["ok"] is False


# ---- whole scripts, for real, in simulation mode -------------------------------------

def run_script(script: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=600,
                          env=clean_env(GPUBRIDGE_REPO=str(REPO), VENV=str(Path(sys.prefix)),
                                        PROJECT_DIR=FAKE_PROJECT, **env))


def steps(results: Path) -> dict[str, int]:
    lines = (results / "steps.jsonl").read_text().splitlines()
    return {json.loads(line)["step"]: json.loads(line)["exit_code"] for line in lines}


def test_explorer_split_job_runs_end_to_end_in_simulation(tmp_path):
    result = run_script(KIT / "explorer" / "04_split_4gpu.sbatch", GPUBRIDGE_VENDOR="nvidia",
                        RESULTS_ROOT=str(tmp_path), BENCH_MAX_BYTES="16K")
    assert result.returncode == 0, result.stderr[-3000:]
    out = tmp_path / "04_split_4gpu"
    assert set(steps(out).values()) == {0}
    for mode in ("half", "alternate"):
        summary = json.loads((out / f"check_{mode}" / "summary.json").read_text())
        assert summary["ok"] and summary["split_test"] == mode
    assert (out / "bench" / "bench.csv").exists()


def test_amd_script_keeps_going_after_failures_and_packages_results(tmp_path):
    # No GPU here: probe, device_map and rccl_version must fail, everything else must run.
    result = run_script(KIT / "amd" / "run_all.sh", GPUBRIDGE_VENDOR="amd", SKIP_SETUP="1",
                        GPU_COUNT="2", SCALING_MAX_GPUS="2", BENCH_MAX_BYTES="16K",
                        BENCH_TRIALS="3", RESULTS_ROOT=str(tmp_path), TARGET="sim",
                        TRAIN_BENCH_MODELS="tiny-gpt")
    assert result.returncode == 1  # some steps failed, so the run reports failure
    codes = steps(tmp_path / "sim")
    assert {name for name, code in codes.items() if code} == {"probe", "device_map",
                                                               "rccl_version"}
    assert {"check_1gpu", "check_island", "check_split", "bench", "train_n2",
            "train_bench_tiny-gpt_1gpu", "train_bench_tiny-gpt_island",
            "train_bench_tiny-gpt_split", "summarize"} <= {
                name for name, code in codes.items() if code == 0}
    split = json.loads((tmp_path / "sim" / "train_bench_tiny-gpt_split.json").read_text())
    assert split["world_size"] == 2 and split["run"]["split_test"] == "half"
    assert (tmp_path / "sim.tgz").exists()
    assert "Download it before stopping this machine" in result.stderr


def test_check_and_bench_take_a_policy(tmp_path):
    check = torchrun(4, KIT / "check.py", "--out", str(tmp_path / "check"), "--policy",
                     "flat-gloo", GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert check.returncode == 0, check.stderr[-3000:]
    for record in read_ranks(tmp_path / "check"):
        assert record["ok"] and record["run"]["policy"] == "flat-gloo"
        dtypes = next(s for s in record["stages"] if s["name"] == "dtypes")
        assert [c["dtype"] for c in dtypes["cases"]] == [
            "float16", "bfloat16", "float32", "float64", "int8", "uint8", "int32", "int64"]
    summarize.main([str(tmp_path / "check")])
    assert "policy: `flat-gloo`" in (tmp_path / "check" / "summary.md").read_text()

    bench = torchrun(2, KIT / "bench_all_reduce.py", "--out", str(tmp_path / "bench"),
                     "--max-bytes", "4K", "--warmup", "1", "--trials", "2", "--policy",
                     "flat-gloo", GPUBRIDGE_VENDOR="amd")
    assert bench.returncode == 0, bench.stderr[-3000:]
    rows = json.loads((tmp_path / "bench" / "bench.json").read_text())["rows"]
    assert {r["policy"] for r in rows if r["op"] == "gpubridge"} == {"flat-gloo"}



@pytest.mark.parametrize(("ranks", "routes"), [(2, {"bridge"}), (4, {"bridge", "native"})])
def test_bench_p2p_times_each_route_and_checks_the_payload(ranks, routes, tmp_path):
    out = tmp_path / "p2p"
    result = torchrun(ranks, KIT / "bench_p2p.py", "--out", str(out), "--max-bytes", "4K",
                      "--trials", "2", "--warmup", "1", GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads((out / "bench_p2p.json").read_text())
    assert set(record["routes"]) == routes
    assert {(r["route"], r["bytes"]) for r in record["rows"]} == {
        (route, n) for route in routes for n in (1024, 4096)}
    assert all(r["payload_ok"] and r["trials"] == 2 and r["bandwidth_GBps"] > 0
               for r in record["rows"])
    # The transport the job ran on: without RDMA here, the default is the TCP bridge.
    assert record["staging"] == {"pinned_staging": False, "p2p_chunk_bytes": 0,
                                 "transport": RAN_ON}
    assert "| bridge |" in (out / "summary.md").read_text()


def test_bench_p2p_records_the_staging_it_ran_with(tmp_path):
    out = tmp_path / "p2p"
    result = torchrun(2, KIT / "bench_p2p.py", "--out", str(out), "--max-bytes", "1K",
                      "--trials", "1", "--warmup", "0", GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half", GPUBRIDGE_P2P_CHUNK_BYTES="512")
    assert result.returncode == 0, result.stderr[-3000:]
    staging = json.loads((out / "bench_p2p.json").read_text())["staging"]
    assert staging["p2p_chunk_bytes"] == 512

def test_ucx_steps_are_off_unless_asked_for():
    out = dry_run(KIT / "explorer" / "05_multinode.sbatch").stdout
    assert "--bridge ucx" not in out and "ucx_build" not in out


def test_tables_name_the_bridge_and_its_path(tmp_path):
    job = _job(tmp_path, "explorer/05_multinode", [("bench_ucx", 0)], GPU_ENV, "cuda:0")
    bench = json.loads((job / "bench_phases_x" / "bench.json").read_text())
    bench["run"].update({"bridge": "ucx", "bridge_path": "direct"})
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps(bench))
    tables = validation_tables.build([tmp_path])
    assert {(r["bridge"], r["staging"]) for r in tables["benchmarks"]} == {("ucx", "ucx direct")}
    assert {(r["bridge"], r["staging"]) for r in tables["predicted"]} == {("ucx", "ucx direct")}


# ---- the comparison matrix (step 08) and training throughput -------------------------

def test_submit_08_is_one_heterogeneous_job_of_a_v100_and_an_mi50():
    out = dry_run(KIT / "explorer" / "submit.sh", "08", GPU_TYPE_1GPU="v100-sxm2",
                  GPU_TYPE_AMD="mi50").stdout
    sbatch = [line for line in out.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, out
    nvidia, amd = sbatch[0].split(" : ")
    assert "--partition=gpu " in nvidia and "--gres=gpu:v100-sxm2:1 " in nvidia
    assert "--job-name=gpubridge-08_matrix_nvidia-amd" in nvidia
    assert "--time=00:40:00" in nvidia and "--time=00:40:00" in amd
    assert "--partition=sharing " in amd and "--gres=gpu:mi50:1 " in amd
    assert amd.rstrip().endswith("08_bench_matrix.sbatch")


@pytest.mark.parametrize(("target", "partition", "gres"),
                         [("08-nvidia-pair", "gpu", "gpu:v100-sxm2:1"),
                          ("08-amd-pair", "sharing", "gpu:mi50:1")])
def test_submit_same_vendor_pairs_are_ready_but_dry_run_only(target, partition, gres):
    result = dry_run(KIT / "explorer" / "submit.sh", target, GPU_TYPE_1GPU="v100-sxm2",
                     GPU_TYPE_AMD="mi50")
    assert "same-partition GPU pairs wait for Research Computing" in result.stderr
    sbatch = [line for line in result.stdout.splitlines() if line.startswith("+ sbatch ")]
    first, second = sbatch[0].split(" : ")
    for side in (first, second):
        assert f"--partition={partition} " in side and f"--gres={gres} " in side


def _matrix_launches(pair: str, **env: str) -> dict[str, str]:
    result = dry_run(KIT / "explorer" / "08_bench_matrix.sbatch", WORK_DIR="/w",
                     MATRIX_PAIR=pair, **env)
    assert result.returncode == 0, result.stderr
    found = {}
    for line in result.stdout.replace("\\", "").splitlines():
        if (m := re.search(r"--out /\S+/matrix/(\S+) ", line)) and "srun" in line:
            found[m[1]] = line
    return found


def test_matrix_job_runs_every_policy_on_the_mixed_pair():
    runs = _matrix_launches("nvidia-amd")
    assert list(runs) == ["D-default", "D-pinned", "D-pipelined-1M", "D-pipelined-4M",
                          "D-pipelined-16M", "D-pipelined-64M", "D-sharded", "D-bf16", "D-fp8",
                          "E", "D-default-repeat"]
    assert "GPUBRIDGE_PINNED_STAGING=1" in runs["D-pinned"]
    assert "GPUBRIDGE_CHUNK_BYTES=16M" in runs["D-pipelined-16M"]
    assert "--policy pipelined-reduce-bridge-broadcast" in runs["D-pipelined-16M"]
    assert "--policy sharded-bridge" in runs["D-sharded"]
    assert "GPUBRIDGE_BRIDGE_COMPRESSION=fp8" in runs["D-fp8"]
    assert "--ops flat-gloo" in runs["E"]
    for line in runs.values():
        nvidia, amd = line.split(" : ")
        assert "/w/venv/bin/torchrun" in nvidia and "/w/venv-rocm63-triton38/bin/torchrun" in amd
        assert "--phases" in line or "flat-gloo" in line
        assert "--max-bytes 1G" in line


@pytest.mark.parametrize(("pair", "names", "venv"),
                         [("nvidia-nvidia", ["A1", "A2", "C", "E-nvidia"], "/w/venv/"),
                          ("amd-amd", ["B1", "B2", "C-amd", "E-amd"],
                           "/w/venv-rocm63-triton38/")])
def test_matrix_job_runs_the_native_baselines_on_same_vendor_pairs(pair, names, venv):
    runs = _matrix_launches(pair)
    assert list(runs) == names
    native, tcp, bridged, _ = (runs[name] for name in names)
    assert "--ops native" in native and "NCCL_IB_DISABLE" not in native
    assert "--ops native" in tcp and "NCCL_IB_DISABLE=1" in tcp
    assert "GPUBRIDGE_SPLIT_TEST=node" in bridged and "--ops gpubridge --phases" in bridged
    for line in runs.values():
        for side in line.split(" : "):
            assert f"{venv}bin/torchrun" in side


def _bench(op: str, us: dict[int, float], *, env: dict | None = None, phases=None,
           islands=None, **run: Any) -> dict:
    islands = islands or [{"vendor": "nvidia", "ranks": [0]}, {"vendor": "amd", "ranks": [1]}]
    return {"run": {"kind": "gpu-mixed", "split_test": None, "world_size": 2,
                    "islands": islands, "policy": "reduce-bridge-broadcast", **run},
            "environment": {"env": env or {}, "collective_library": "nccl",
                            "hostname": "gpunode17.cluster.example.edu"},
            "rows": [{"op": op, "bytes": n, "median_us": t, "busbw_GBps": n / t / 1e3,
                      "islands": len(islands), "timer": "cuda-events",
                      **({"phases_ms": phases(n)} if phases else {})}
                     for n, t in us.items()]}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _matrix_tree(root: Path) -> None:
    sizes = (2**10, 2**20, 2**30)
    pair = root / "08_matrix_nvidia-nvidia"
    nvidia = [{"vendor": "nvidia", "ranks": [0, 1]}]
    _write(pair / "matrix" / "A1" / "bench.json",
           _bench("native", {n: 10 + n / 10e3 for n in sizes}, islands=nvidia))
    _write(pair / "matrix" / "A2" / "bench.json",
           _bench("native", {n: 20 + n / 1e3 for n in sizes}, islands=nvidia,
                  env={"NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "eth1"}))
    mixed = root / "08_matrix_nvidia-amd"
    _write(mixed / "preflight" / "preflight-master.json",
           {"role": "master", "hostname": "gpunode17", "chosen_ifname": "eth1",
            "bandwidth": {"MBps_in": 1000.0, "MBps_out": 1200.0, "rtt_ms": 0.1}})
    _write(mixed / "preflight" / "preflight-worker.json",
           {"role": "worker", "hostname": "amdnode3", "chosen_ifname": "ib0",
            "bandwidth": {"MBps_in": 1200.0, "MBps_out": 1000.0, "rtt_ms": 0.1}})
    _write(mixed / "matrix" / "D-default" / "bench.json",
           _bench("gpubridge", {n: 40 + n / 500 for n in sizes},
                  phases=lambda n: {"island-reduce": 0.01, "bridge": n / 5e5, "island-broadcast": 0.01}))
    _write(mixed / "matrix" / "D-bf16" / "bench.json",
           _bench("gpubridge", {n: 40 + n / 900 for n in sizes},
                  env={"GPUBRIDGE_BRIDGE_COMPRESSION": "bf16"},
                  phases=lambda n: {"compress": 0.1, "bridge": n / 1e6, "decompress": 0.1}))
    _write(mixed / "matrix" / "E" / "bench.json", _bench("flat-gloo", {n: 30 + n / 450 for n in sizes}))
    side = mixed / "network" / "nvidia"
    side.mkdir(parents=True)
    (side / "sys_class_infiniband.txt").write_text("mlx5_0\n")
    (side / "ibstat.txt").write_text("CA 'mlx5_0'\n\tPort 1:\n\t\tState: Active\n\t\tRate: 100\n"
                                     "\t\tLink layer: InfiniBand\n")
    (side / "ibv_devinfo.txt").write_text("hca_id:\tmlx5_0\n\t\tport:\t1\n\t\t\tstate:\t\t"
                                          "PORT_ACTIVE (4)\n\t\t\tlink_layer:\t\tInfiniBand\n")
    (side / "ip_addr.txt").write_text("lo UNKNOWN 127.0.0.1/8\neth1 UP 10.0.12.34/16\n")
    amd = mixed / "network" / "amd"
    amd.mkdir(parents=True)
    (amd / "sys_class_infiniband.txt").write_text("")
    (amd / "ibstat.txt").write_text("ibstat: not installed on this node\n")
    (amd / "ibv_devinfo.txt").write_text("ibv_devinfo: not installed on this node\n")
    logs = mixed / "nccl" / "matrix_D-default"
    logs.mkdir(parents=True)
    (logs / "gpunode17.1.log").write_text(
        "gpunode17:1:1 [0] NCCL INFO cudaDriverVersion 12030\n"
        "gpunode17:1:1 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB eth1:10.0.12.34<0>\n"
        "gpunode17:1:1 [0] NCCL INFO Using network IB\n")
    (logs / "amdnode3.2.log").write_text(
        "amdnode3:2:2 [0] NCCL INFO ROCr version 1.18\n"
        "amdnode3:2:2 [0] NCCL INFO Bootstrap : Using ibs8:172.16.5.6<0>\n"
        "amdnode3:2:2 [0] NCCL INFO NET/Socket : Using [0]ib0:fe80::1%ib0<0>\n"
        "amdnode3:2:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/Socket/0\n"
        "amdnode3:2:2 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] [send] via NET/Socket/0\n")


def test_matrix_compares_every_run_with_both_native_baselines(tmp_path):
    _matrix_tree(tmp_path)
    runs = bench_matrix.find_runs([tmp_path])
    assert [r.name for r in runs] == ["A1", "A2", "D-default", "D-bf16", "E"]
    rows = {(r["run"], r["bytes"]): r for r in bench_matrix.matrix_rows(runs)}
    big = 2**30
    a2, a1 = rows[("A2", big)]["median_us"], rows[("A1", big)]["median_us"]
    d = rows[("D-default", big)]
    assert d["x_A2"] == pytest.approx(d["median_us"] / a2, abs=0.01)
    assert d["x_A1"] == pytest.approx(d["median_us"] / a1, abs=0.01)
    assert rows[("A2", big)]["x_A2"] == 1.0
    # Predicted from the preflight's slower direction (1000 MB/s) and round trip.
    assert d["predicted_bridge_ms"] == pytest.approx(0.1 + big / 1e9 * 1e3, abs=1e-3)
    assert d["measured_over_predicted"] == pytest.approx(d["bridge_ms"] / d["predicted_bridge_ms"],
                                                         abs=0.01)
    # Compressed, each leader broadcasts its payload in turn: bf16's half-size payload
    # crosses twice, one way at a time, so two islands predict what uncompressed does.
    assert rows[("D-bf16", big)]["predicted_bridge_ms"] == pytest.approx(
        0.1 + 2 * (big / 2) / 1e9 * 1e3, abs=1e-3)
    # D ran in another job than A1 and A2, so its multiples are marked; A2's own isn't.
    assert d["x_A2_other_job"] and not rows[("A2", big)]["x_A2_other_job"]
    assert "predicted_bridge_ms" not in rows[("E", big)]  # flat Gloo has no bridge phase
    assert bench_matrix.described(runs[1]) == "native NCCL, NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=eth1"
    assert "bf16 compression" in bench_matrix.described(runs[3])


def test_matrix_markdown_has_a_table_per_size_range_and_no_hosts_or_addresses(tmp_path):
    _matrix_tree(tmp_path)
    assert bench_matrix.main([str(tmp_path), "--pending", "B1,B2=waiting on access"]) == 0
    md = (tmp_path / "matrix.md").read_text()
    for title in ("## Small", "## Medium", "## Large", "## Phases of the gpubridge runs",
                  "## Network", "A2 against C is gpubridge's own cost"):
        assert title in md
    assert "**B1** (waiting on access); **B2** (waiting on access)" in md
    assert re.search(r"\| 1GB \| D-default \| [^|]+ \| [^|]+ \| [\d.]+x † \|", md), md
    assert re.search(r"\| 1GB \| A2 \| [^|]+ \| [^|]+ \| 1\.00x \|", md), md
    # Each job's network, from its preflight.
    assert "| 08_matrix_nvidia-amd | amd+nvidia | eth1, ib0 | 1200.0 | 1000.0 | 0.1 |" in md
    # The network: InfiniBand on the V100 node, none on the other; what each library chose.
    assert "mlx5_0:1 InfiniBand, Active, 100 Gb/s" in md
    assert "ibstat and ibv_devinfo not installed" in md
    assert "Using network IB" in md
    assert "2 channel connections via NET/Socket/0 (one rank's log)" in md
    assert "OOB eth1" in md and "Bootstrap : Using ibs8" in md
    assert "NET/Socket : Using [0]ib0" in md and "fe80" not in md
    for secret in ("gpunode17", "amdnode3", "10.0.12.", "172.16.", "example.edu"):
        assert secret not in md
    data = json.loads((tmp_path / "matrix.json").read_text())
    assert {r["run"] for r in data["runs"]} == {"A1", "A2", "D-default", "D-bf16", "E"}
    assert (tmp_path / "matrix.csv").read_text().startswith(
        "run,job,what,bytes,median_us,p90_us,busbw_GBps,x_A2,x_A2_other_job,x_A1,x_A1_other_job")


def test_matrix_without_baselines_says_so(tmp_path):
    _matrix_tree(tmp_path)
    shutil.rmtree(tmp_path / "08_matrix_nvidia-nvidia")
    assert bench_matrix.main([str(tmp_path)]) == 0
    md = (tmp_path / "matrix.md").read_text()
    assert "Without A1 and A2, the x A2 and x A1 columns are empty." in md


def test_matrix_needs_results(tmp_path, capsys):
    assert bench_matrix.main([str(tmp_path)]) == 1
    assert "no matrix runs" in capsys.readouterr().err


def test_matrix_names_runs_in_order():
    names = ["E", "D-default-repeat", "D-pipelined-16M", "D-pipelined-1M", "A2", "C-amd",
             "D-pipelined-4M", "D-default", "C", "A1"]
    assert sorted(names, key=bench_matrix.order_key) == [
        "A1", "A2", "C", "C-amd", "D-default", "D-pipelined-1M", "D-pipelined-4M",
        "D-pipelined-16M", "D-default-repeat", "E"]


def test_matrix_size_ranges_cover_every_size():
    for size in (1, 2**10, 100 * 2**10, 256 * 2**10, 32 * 2**20, 64 * 2**20 - 1, 2**30):
        assert sum(lo <= size < hi for _, lo, hi in bench_matrix.RANGES) == 1, size


@pytest.mark.parametrize(("mode", "dtype"), [("fp8", "float16"), ("FP8", "float32"),
                                             (" bf16 ", "float32")])
def test_matrix_predicts_compressed_payloads_whatever_the_case(mode, dtype):
    from gpubridge.compression import payload_nbytes

    run = bench_matrix.Run("D-x", "job", {"environment": {"env": {
        "GPUBRIDGE_BRIDGE_COMPRESSION": mode}}}, None)
    size = {"float16": 2, "float32": 4}[dtype]
    row = {"bytes": 2**20, "islands": 2, "dtype": dtype}
    payload = payload_nbytes(mode.strip().lower(), 2**20 // size)
    assert bench_matrix.predicted_ms(run, row, 100.0, 0.5) == pytest.approx(
        0.5 + 2 * payload / 1e8 * 1e3)


def test_matrix_predicts_the_ring_where_the_mode_leaves_the_dtype_alone():
    # bf16 compression doesn't touch float16, so that bridge is the plain all_reduce.
    run = bench_matrix.Run("D-x", "job", {"environment": {"env": {
        "GPUBRIDGE_BRIDGE_COMPRESSION": "bf16"}}}, None)
    row = {"bytes": 10**8, "islands": 2, "dtype": "float16"}
    assert bench_matrix.predicted_ms(run, row, 100.0, 0.5) == pytest.approx(1000.5)


def test_matrix_keeps_two_jobs_of_one_name_apart(tmp_path):
    for download in ("first", "second"):
        _matrix_tree(tmp_path / download)
    runs = bench_matrix.find_runs([tmp_path / "first", tmp_path / "second"])
    jobs = {r.job for r in runs if r.name == "D-default"}
    assert jobs == {"08_matrix_nvidia-amd", "08_matrix_nvidia-amd (2)"}
    rows = bench_matrix.matrix_rows(runs)
    second = next(r for r in rows if r["run"] == "A2" and r["job"].endswith("(2)"))
    assert second["x_A2_other_job"]  # the first download's A2 is the baseline
    # The Network rows name the same jobs as the runs.
    labels = bench_matrix.JobLabels()
    roots = [tmp_path / "first", tmp_path / "second"]
    bench_matrix.find_runs(roots, labels)
    assert {r["job"] for r in bench_matrix.network(roots, labels)} == jobs


def test_train_bench_splits_the_step_into_compute_waiting_and_exposed_communication():
    # Rank 0 computes in 1.0 s, rank 1 in 1.5 s; the ddp step takes 2.0 s; the
    # gradients alone take 0.8 s, of which 0.5 s stayed exposed.
    out = train_bench.decompose([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]],
                                [[2.0, 1.9, 2.1], [2.0, 2.0, 2.0]],
                                [[0.8, 0.8], [0.7, 0.8]], batch=8)
    assert out["step_ms"] == 2000.0 and out["samples_per_s"] == 8.0
    assert out["slowest_compute_ms"] == 1500.0 and out["exposed_comm_ms"] == 500.0
    assert out["comm_alone_ms"] == 800.0 and out["hidden_comm_ms"] == 300.0
    assert [r["waiting_ms"] for r in out["ranks"]] == [500.0, 0.0]
    assert [r["alone_samples_per_s"] for r in out["ranks"]] == [8.0, 5.333]


def test_train_bench_models_are_the_real_sizes():
    count = {name: sum(p.numel() for p in train_bench.build(name).parameters())
             for name in ("gpt2-small", "resnet50")}
    assert count == {"gpt2-small": 124_439_808, "resnet50": 25_557_032}


def test_train_bench_runs_ddp_through_gpubridge_on_cpu(tmp_path):
    out = tmp_path / "train_bench.json"
    result = torchrun(2, KIT / "train_bench.py", "--model", "tiny-resnet", "--steps", "2",
                      "--warmup-steps", "1", "--comm-trials", "1", "--json", str(out),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["world_size"] == 2 and record["run"]["split_test"] == "half"
    assert record["grad_sync"] == "DistributedDataParallel + gpubridge.ddp_comm_hook"
    assert record["buckets"] >= 1 and record["model"]["gradient_bytes"] > 0
    assert record["params_identical"] and record["params_finite"]
    assert [r["vendor"] for r in record["ranks"]] == ["nvidia", "amd"]
    assert record["samples_per_s"] > 0 and len(record["environments"]) == 2
    assert "SPLIT TEST" in record["warning"]


@pytest.mark.parametrize(("args", "split", "want"), [
    (("--batch-sizes", "1,3"), "explicit", [1, 3]),
    (("--global-batch", "7"), "even", [4, 3]),
    (("--global-batch", "6", "--balance", "--calibration-steps", "1"), "balanced", None),
])
def test_train_bench_takes_unequal_batches_and_reports_them(args, split, want, tmp_path):
    out = tmp_path / "train_bench.json"
    result = torchrun(2, KIT / "train_bench.py", "--model", "tiny-gpt", "--steps", "2",
                      "--warmup-steps", "1", "--comm-trials", "1", "--json", str(out), *args,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads(out.read_text())
    sizes = record["batch_sizes"]
    assert record["batch_split"] == split and (want is None or sizes == want)
    assert record["global_batch"] == sum(sizes) and min(sizes) >= 1
    assert record["loss_weighting"].startswith("sample-weighted")
    assert [r["batch_size"] for r in record["ranks"]] == sizes
    assert all(r["compute_ms_per_sample"] > 0 and r["waiting_ms"] >= 0 for r in record["ranks"])
    assert record["params_identical"] and record["params_finite"]
    if split == "balanced":
        assert record["global_batch"] == 6
        assert record["calibration"]["batch_size"] == 3
        assert [r["rank"] for r in record["calibration"]["ranks"]] == [0, 1]
        assert f"balanced split of 6: {sizes}" in result.stdout
    else:
        assert record["calibration"] is None


def test_train_bench_alone_runs_the_whole_global_batch_with_the_same_flags(tmp_path):
    out = tmp_path / "train_bench.json"
    result = torchrun(1, KIT / "train_bench.py", "--model", "tiny-gpt", "--steps", "1",
                      "--warmup-steps", "1", "--comm-trials", "1", "--json", str(out),
                      "--global-batch", "6", "--balance", "--accumulate", "2",
                      GPUBRIDGE_VENDOR="nvidia")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["batch_sizes"] == [6] and record["global_batch"] == 6
    assert record["batch_split"] == "balanced" and record["calibration"] is None
    assert record["micro_batch_sizes"] == [[3, 3]] and record["exchanges_per_step"] == 1.0


def test_train_bench_default_batches_are_unweighted_and_say_so(tmp_path):
    out = tmp_path / "train_bench.json"
    result = torchrun(2, KIT / "train_bench.py", "--model", "tiny-gpt", "--steps", "1",
                      "--warmup-steps", "1", "--comm-trials", "1", "--json", str(out),
                      "--batch-size", "2", GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["batch_size_per_rank"] == 2 and record["global_batch"] == 4
    assert record["batch_split"] == "per-rank"
    assert record["loss_weighting"] == "mean per rank, averaged over ranks"
    assert record["accumulate"] == 1 and record["micro_batch_sizes"] == [[2], [2]]
    assert record["exchanges_per_step"] == 1.0
    assert [r["memory"] for r in record["ranks"]] == [None, None]  # GPU only


@pytest.mark.parametrize(("args", "micro"), [
    (("--batch-size", "4", "--accumulate", "2"), [[2, 2], [2, 2]]),
    (("--batch-sizes", "2,5", "--accumulate", "2"), [[1, 1], [3, 2]]),
    (("--global-batch", "8", "--balance", "--accumulate", "2", "--calibration-steps", "1"),
     None),
])
def test_train_bench_accumulates_micro_batches_with_one_exchange_per_step(args, micro,
                                                                          tmp_path):
    out = tmp_path / "train_bench.json"
    result = torchrun(2, KIT / "train_bench.py", "--model", "tiny-gpt", "--steps", "2",
                      "--warmup-steps", "1", "--comm-trials", "1", "--json", str(out), *args,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["accumulate"] == 2 and record["exchanges_per_step"] == 1.0
    got = record["micro_batch_sizes"]
    assert micro is None or got == micro
    assert [sum(m) for m in got] == record["batch_sizes"]
    assert all(len(m) == 2 and min(m) >= 1 for m in got)
    n = record["global_batch"]
    assert record["samples_per_step"] == n == sum(record["batch_sizes"])
    assert record["samples_per_exchange"] == n and record["exchanges_per_sample"] == round(1 / n, 6)
    assert record["loss_weighting"].startswith("sample-weighted")
    assert record["params_identical"] and record["params_finite"]
    if micro is None:
        assert n == 8 and record["calibration"]["batch_size"] == 2


@pytest.mark.parametrize(("args", "message"), [
    (("--accumulate", "3", "--batch-size", "2"), "needs at least 3 samples per rank"),
    (("--accumulate", "0"), "--accumulate must be at least 1"),
    (("--balance",), "--balance needs --global-batch"),
    (("--batch-sizes", "1,2,3"), "--batch-sizes needs 2 sizes"),
    (("--global-batch", "1"), "can't give each of 2 ranks a sample"),
    (("--batch-size", "2", "--global-batch", "4"), "not allowed with"),
])
def test_train_bench_rejects_batch_settings_it_cant_honour(args, message):
    result = torchrun(2, KIT / "train_bench.py", "--model", "tiny-gpt", *args,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode != 0
    assert message in result.stderr


# ---- data-parallel training: llama.py, train_dp.py, dp_report.py -------------------------

def test_llama_models_are_the_stated_sizes():
    assert {n: llama.parameters(n) for n in llama.MODELS} == {
        "llama-125m": 134_105_856, "llama-350m": 373_867_520, "llama-1b": 1_032_931_328,
        "llama-tiny": 45_344}
    model = llama.build("llama-tiny")
    assert sum(p.numel() for p in model.parameters()) == llama.parameters("llama-tiny")


def test_llama_weights_and_batches_are_the_same_everywhere():
    a, b = llama.build("llama-tiny", seed=3), llama.build("llama-tiny", seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    assert not torch.equal(next(a.parameters()), next(llama.build("llama-tiny").parameters()))
    x, y = llama.global_batch("llama-tiny", 7, 6, 10)
    again, _ = llama.global_batch("llama-tiny", 7, 6, 10)
    assert torch.equal(x, again) and torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, (x * 7 + 13 + (y - x * 7 - 13) % 128) % 128)  # the rule, noise 0..3
    assert int(((y - x * 7 - 13) % 128).max()) <= 3


def test_train_dp_splits_rows_by_weight():
    assert train_dp.shares_for([1.0, 1.0], 8) == [4, 4]
    assert train_dp.shares_for([3.0, 1.0], 8) == [6, 2]
    assert train_dp.shares_for([1.0, 1.0, 1.0], 8) == [3, 3, 2]  # ties to the lower rank
    assert train_dp.shares_for([100.0, 1.0], 4) == [3, 1]  # everyone keeps a row
    with pytest.raises(ValueError):
        train_dp.shares_for([1.0, 1.0, 1.0], 2)
    assert train_dp.rows([3, 5], 1) == slice(3, 8)


def _train_dp(tmp_path, name: str, nproc: int, *args: str, **env: str) -> dict:
    out = tmp_path / f"dp_{name}.json"
    common = ("--model", "llama-tiny", "--steps", "12", "--global-batch", "8", "--seq-len", "16",
              "--lr", "3e-3", "--check-every", "6", "--probe-steps", "1", "--comm-trials", "1",
              "--json", str(out))
    if nproc == 1:
        result = subprocess.run([sys.executable, str(KIT / "train_dp.py"), *common, *args],
                                env=clean_env(**env), capture_output=True, text=True,
                                timeout=300)
    else:
        result = torchrun(nproc, KIT / "train_dp.py", *common, *args, GPUBRIDGE_VENDOR="nvidia",
                          GPUBRIDGE_SPLIT_TEST="half", **env)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(out.read_text())


def test_data_parallel_runs_track_one_gpu_on_cpu(tmp_path):
    single = _train_dp(tmp_path, "single", 1, "--mode", "single", "--micro", "2")
    assert single["world_size"] == 1 and single["loss_last"] < single["loss_first"]
    runs = {"ddp": _train_dp(tmp_path, "ddp", 2, "--mode", "ddp"),
            "uneven": _train_dp(tmp_path, "uneven", 2, "--mode", "ddp", "--shares", "6,2"),
            "fsdp2": _train_dp(tmp_path, "fsdp2", 2, "--mode", "fsdp2")}
    for name, run in runs.items():
        gap = max(abs(a - b) for a, b in zip(run["losses"], single["losses"], strict=True))
        assert gap < 1e-4, (name, gap)  # the same mean gradient, summed in another order
        assert [r["vendor"] for r in run["ranks"]] == ["nvidia", "amd"]
    assert [r["share"] for r in runs["uneven"]["ranks"]] == [6, 2]
    assert all(c["identical"] for c in runs["ddp"]["param_checks"])
    assert [c["step"] for c in runs["ddp"]["param_checks"]] == [6, 12]
    assert runs["fsdp2"]["param_checks"] == []  # one copy of each shard: nothing to compare
    assert runs["ddp"]["buckets"] >= 1 and 0 <= runs["ddp"]["hidden_fraction"] <= 1
    fp8 = _train_dp(tmp_path, "fp8", 2, "--mode", "ddp", "--kernels-on-amd", "torch",
                    GPUBRIDGE_BRIDGE_COMPRESSION="fp8")
    # Step 1's loss comes before any (compressed) gradient: the same as one GPU's.
    assert fp8["compression"] == "fp8" and abs(fp8["losses"][0] - single["losses"][0]) < 1e-5
    assert [r["compression_kernels"] for r in fp8["ranks"]] == ["auto", "torch"]
    assert max(abs(a - b) for a, b in zip(fp8["losses"], single["losses"], strict=True)) < 0.05
    report = dp_report.table(dp_report.load([tmp_path]), "single")
    assert "| single (baseline) | single |" in report and "| uneven | ddp | none |" in report


def test_dp_report_compares_curves_with_the_baseline(tmp_path):
    def record(mode, losses, **extra):
        return {"script": "scripts/gpu/train_dp.py", "mode": mode, "compression": "none",
                "ranks": [{"device": "Tesla V100-SXM2-32GB", "share": 8}], "tokens_per_s": 10.0,
                "median_step_ms": 5.0, "loss_last": losses[-1], "losses": losses, **extra}

    (tmp_path / "dp_single.json").write_text(json.dumps(record("single", [3.0, 2.0, 1.0])))
    (tmp_path / "dp_ddp.json").write_text(json.dumps(record(
        "ddp", [3.0, 2.5, 1.0], exposed_comm_ms=1.0, hidden_fraction=0.25,
        param_checks=[{"step": 3, "identical": False, "sum_spread": 0.5, "finite": True}])))
    runs = dp_report.load([tmp_path])
    assert dp_report.gaps(runs["ddp"]["losses"], runs["single"]["losses"]) == (0.5, 0.5 / 3)
    table = dp_report.table(runs, "single")
    assert "| ddp | ddp | none | V100 | 8 | 10 | 5.0 | 1.0 | 25% | 1.0000 | 0.5000 |" in table
    assert "no (sums up to 0.5 apart)" in table
    assert dp_report.main([str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "dp_losses.svg").read_text().count("<polyline") == 2
    assert (tmp_path / "out" / "dp_losses.csv").read_text().startswith("step,ddp,single\n1,3.0,3.0")
    for job in ("a", "b", "c"):  # three jobs' runs of one name: none dropped
        (tmp_path / "jobs" / job / "07_mixed_hetjob_dp").mkdir(parents=True)
        (tmp_path / "jobs" / job / "07_mixed_hetjob_dp" / "dp_ddp.json").write_text(
            json.dumps(record("ddp", [3.0])))
    assert sorted(dp_report.load([tmp_path / "jobs"])) == [
        f"jobs/{job}/07_mixed_hetjob_dp/dp_ddp" for job in ("a", "b", "c")]



# ---- run guards: the import, the results folder, the MI50's Triton ---------------

def _other_checkout(tmp_path: Path) -> Path:
    """A copy of this checkout's package somewhere else, as another clone would hold it."""
    other = tmp_path / "other" / "src"
    shutil.copytree(REPO / "src" / "gpubridge", other / "gpubridge",
                    ignore=shutil.ignore_patterns("__pycache__", "target"))
    return other


def test_import_check_passes_for_this_checkout_and_names_a_foreign_import(tmp_path):
    ok = subprocess.run([sys.executable, str(KIT / "import_check.py")],
                        env=clean_env(PYTHONPATH=str(REPO / "src")), capture_output=True,
                        text=True, timeout=120)
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert "gpubridge from" in ok.stdout
    other = _other_checkout(tmp_path)
    bad = subprocess.run([sys.executable, str(KIT / "import_check.py")],
                         env=clean_env(PYTHONPATH=str(other)), capture_output=True,
                         text=True, timeout=120)
    assert bad.returncode == 1
    assert "gpubridge-kit: import check failed" in bad.stderr
    import gpukit
    assert (str(other / "gpubridge") in bad.stderr or gpukit.redact(str(other / "gpubridge")) in bad.stderr) and "PYTHONPATH is" in bad.stderr


def test_probe_records_where_gpubridge_came_from_and_fails_on_a_foreign_import(tmp_path):
    out = tmp_path / "probe"
    ok = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(out),
                         "--allow-no-gpu"], env=clean_env(PYTHONPATH=str(REPO / "src")),
                        capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0, ok.stderr[-2000:]
    record = json.loads((out / "probe.json").read_text())
    imported = record["environment"]["import"]
    assert imported["gpubridge_in_kit_checkout"] is True
    assert imported["gpubridge_file"].endswith("src/gpubridge/__init__.py")
    assert record["environment"]["env"]["PYTHONPATH"]  # recorded, redacted like any path
    stages = {s["name"]: s for s in record["stages"]}
    assert stages["kit_import"]["ok"] and stages["triton_kernels"]["ok"]

    other = _other_checkout(tmp_path)
    bad = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(tmp_path / "b"),
                          "--allow-no-gpu"], env=clean_env(PYTHONPATH=str(other)),
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 1 and "gpubridge-kit: import check failed" in bad.stderr


def _bash(script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-e", "-c", script], env=clean_env(**env),
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("marker", ["gpubridge-kit: import check failed: elsewhere"])
def test_a_failed_probe_check_stops_the_job_before_the_next_step(marker, tmp_path):
    result = _bash(f"""
source {KIT}/common.sh
RESULTS={tmp_path}
step probe sh -c 'echo "{marker}"; exit 1'
step after echo should-not-run
""")
    assert result.returncode != 0
    assert "stopping before any other step" in result.stderr
    assert "should-not-run" not in result.stdout + result.stderr
    assert not (tmp_path / "after.log").exists()


def test_require_kit_import_stops_on_a_foreign_import(tmp_path):
    other = _other_checkout(tmp_path)
    venv_path = f"{Path(sys.executable).parent}:{os.environ['PATH']}"  # as an active venv
    good = _bash(f"source {KIT}/common.sh; require_kit_import; echo went-on",
                 PYTHONPATH=str(REPO / "src"), PATH=venv_path)
    assert good.returncode == 0 and "went-on" in good.stdout, good.stderr[-2000:]
    bad = _bash(f"source {KIT}/common.sh; require_kit_import; echo went-on",
                PYTHONPATH=str(other), PATH=venv_path)
    assert bad.returncode != 0 and "went-on" not in bad.stdout
    assert "isn't this checkout's" in bad.stderr


def test_a_results_folder_with_records_gets_a_job_folder_of_its_own(tmp_path):
    fresh = tmp_path / "07_mixed_hetjob"
    used = tmp_path / "amd_node"
    used.mkdir()
    (used / "steps.jsonl").write_text("{}\n")
    out = _bash(f"source {KIT}/common.sh; fresh_results {fresh}; fresh_results {used}",
                SLURM_JOB_ID="4242")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [str(fresh), f"{used}_job4242"]
    assert "already holds another run's records" in out.stderr
    for taken in (f"{used}_jobgpubridge-mixed", f"{used}_jobgpubridge-mixed_2"):
        Path(taken).mkdir()
        (Path(taken) / "job.jsonl").write_text("{}\n")
    again = _bash(f"source {KIT}/common.sh; fresh_results {used}", RUN_ID="gpubridge-mixed")
    assert again.stdout.split() == [f"{used}_jobgpubridge-mixed_3"]  # a fixed RUN_ID's 4th run


def test_job_setup_never_writes_into_another_runs_folder(tmp_path):
    # job_setup (explorer/job_env.sh), the start of every Explorer job.
    used = tmp_path / "02_split_shared_gpu"
    used.mkdir()
    (used / "job.jsonl").write_text('{"event": "started", "job_id": "1"}\n')
    out = _bash(f"""
source {KIT}/common.sh --dry-run
source {KIT}/explorer/job_env.sh
VENV={tmp_path}/no-venv
job_setup 02_split_shared_gpu
echo "RESULTS=$RESULTS"
""", SLURM_JOB_ID="77", RESULTS_ROOT=str(tmp_path), PROJECT_DIR=FAKE_PROJECT,
                GPUBRIDGE_REPO=str(REPO))
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"RESULTS={used}_job77" in out.stdout
    assert (used / "job.jsonl").read_text().count("started") == 1  # untouched
    assert "import_check.py" in out.stdout  # and checks where gpubridge comes from


@pytest.mark.parametrize("script,target,env", [
    ("amd/run_all.sh", "amd_node", {}),
    ("mixed/node.sh", "master", {"ROLE": "master", "MASTER_ADDR": "127.0.0.1",
                                 "TORCH_FLAVOR": "system"}),
])
def test_the_scripts_for_other_machines_check_the_import_and_keep_old_folders(
        script, target, env, tmp_path):
    used = tmp_path / target
    used.mkdir()
    (used / "steps.jsonl").write_text("{}\n")
    out = subprocess.run(["bash", str(KIT / script), "--dry-run"],
                         env=clean_env(RESULTS_ROOT=str(tmp_path), TARGET=target, RUN_ID="r1",
                                       SKIP_SETUP="1", VENV=str(tmp_path / "venv"),
                                       GPUBRIDGE_REPO=str(REPO), **env),
                         capture_output=True, text=True, timeout=120)
    both = out.stdout + out.stderr
    assert "import_check.py" in out.stdout, both[-2000:]
    assert f"{used}_jobr1" in both and (used / "steps.jsonl").read_text() == "{}\n"


def test_a_directory_of_wrappers_is_a_venv_the_kit_can_use(tmp_path):
    # rocm_triton_overlay.sh writes bin/python, bin/torchrun wrappers and no bin/activate.
    wrappers = tmp_path / "venv-overlay" / "bin"
    wrappers.mkdir(parents=True)
    (wrappers / "python").write_text(f"#!/usr/bin/env bash\nexec {sys.executable} \"$@\"\n")
    (wrappers / "python").chmod(0o755)
    out = _bash(f"source {KIT}/common.sh; activate_venv {wrappers.parent}; command -v python")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(wrappers / "python")
    missing = _bash(f"source {KIT}/common.sh; activate_venv {tmp_path}/nothing")
    assert missing.returncode != 0 and "run setup first" in missing.stderr


@pytest.mark.parametrize("compiles", [True, None, False])
def test_probe_records_whether_triton_compiles_without_failing_on_it(compiles, monkeypatch,
                                                                      tmp_path, capsys):
    # Only the steps that use the kernels check them (triton_check.py, common.sh).
    mi50 = fake_device(monkeypatch, name="AMD Instinct MI50", capability=(9, 0),
                       gcn="gfx906:sramecc+:xnack-", hip="6.3.42134", arch_list=["gfx906"])
    triton = {"triton": "3.5.1", "compiles": compiles,
              **({"error": "RuntimeError: PassManager::run failed"} if compiles is False else {})}
    assert run_probe(monkeypatch, tmp_path, [mi50], triton=triton) == 0
    stage = next(s for s in json.loads((tmp_path / "probe.json").read_text())["stages"]
                 if s["name"] == "triton_kernels")
    assert stage["ok"] is True and stage["compiles"] is compiles
    assert ("steps that use them won't run" in capsys.readouterr().err) is (compiles is False)


@pytest.mark.parametrize("broken", [None, ("bf16", "quantize"), ("fp8", "dequantize")])
def test_triton_check_runs_every_kernel_a_compressed_collective_uses(broken, monkeypatch):
    import gpukit

    import gpubridge
    calls = []

    def kernel(name):
        def run(mode, *args):
            calls.append((mode, name))
            if (mode, name) == broken:
                raise RuntimeError("PassManager::run failed\nmore")
        return run

    kernels = SimpleNamespace(quantize=kernel("quantize"),
                              dequantize_accumulate=kernel("dequantize"))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.5.1"))
    monkeypatch.setitem(sys.modules, "gpubridge._compression_kernels", kernels)
    monkeypatch.setattr(gpubridge, "_compression_kernels", kernels, raising=False)
    cpu = torch.device("cpu")
    monkeypatch.setattr(gpukit.torch, "device", lambda *a: cpu)
    monkeypatch.setattr(gpukit.torch.cuda, "synchronize", lambda *a: None)
    result = gpukit.triton_check(0)
    if broken is None:
        assert result == {"triton": "3.5.1", "compiles": True}
        assert calls == [("bf16", "quantize"), ("bf16", "dequantize"),
                         ("fp8", "quantize"), ("fp8", "dequantize")]
    else:
        assert result["compiles"] is False and calls[-1] == broken
        assert result["error"] == "RuntimeError: PassManager::run failed"


def _triton_check_module():
    sys.path.insert(0, str(KIT))
    import triton_check
    return triton_check


@pytest.mark.parametrize("result,forced,ok", [
    (None, "auto", True),                                        # no GPU: PyTorch ops on CPU
    ({"triton": None, "compiles": None}, "auto", True),          # no Triton: PyTorch kernels
    ({"triton": None, "compiles": None}, "triton", False),       # forced but not installed
    ({"triton": "3.8.0", "compiles": True}, "auto", True),
    ({"triton": "3.5.1", "compiles": False, "error": "x"}, "auto", False),
    ({"triton": "3.5.1", "compiles": False, "error": "x"}, "triton", False),
])
def test_triton_check_says_whether_the_kernels_can_run(result, forced, ok):
    problem = _triton_check_module().verdict(result, forced)
    assert (problem is None) is ok
    if result and result["compiles"] is False:
        assert "rocm_triton_overlay.sh" in problem and "3.5.1" in problem


def _gate(tmp_path: Path, check_passes: bool, body: str, **env: str):
    """common.sh with a stand-in triton_check.py launcher that passes or fails."""
    launcher = tmp_path / "fake-python"
    launcher.write_text(f"#!/bin/sh\necho checked >>{tmp_path}/checks\nexit "
                        f"{0 if check_passes else 1}\n")
    launcher.chmod(0o755)
    result = _bash(f"""
source {KIT}/common.sh
RESULTS={tmp_path}
TRITON_CHECK_LAUNCH={launcher}
{body}
finish || true
""", **env)
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    checks = (tmp_path / "checks").read_text().count("checked") if (
        tmp_path / "checks").exists() else 0
    return result, {s["step"]: s for s in steps}, checks


def test_uncompressed_steps_never_check_triton_and_run_even_where_it_would_fail(tmp_path):
    result, steps, checks = _gate(tmp_path, False, """
step train_ddp echo ran-train
step check echo ran-check
GPUBRIDGE_BRIDGE_COMPRESSION=none step bench echo ran-bench
""")
    assert checks == 0 and "triton_check" not in steps
    assert all(steps[name]["exit_code"] == 0 for name in ("train_ddp", "check", "bench"))
    assert "ran-train" in result.stdout and "ran-bench" in result.stdout


def test_steps_that_use_the_kernels_are_not_run_when_the_check_fails(tmp_path):
    result, steps, checks = _gate(tmp_path, False, """
mixed_torchrun() { echo "ran-$1"; }
step check echo ran-check
step gpu_tests echo ran-gpu-tests
step bench_compression_kernels echo ran-kernels
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 step dp_ddp-fp8 mixed_torchrun fp8
step train_ddp echo ran-train
echo "failed: ${FAILED_STEPS[*]}"
""")
    assert checks == 1 and steps["triton_check"]["exit_code"] != 0  # checked once
    for name in ("gpu_tests", "bench_compression_kernels", "dp_ddp-fp8"):
        assert steps[name]["exit_code"] == 1
        assert steps[name]["not_run"].startswith("triton_check failed")
        assert f"ran-{name}" not in result.stdout
    assert "ran-gpu-tests" not in result.stdout and "ran-fp8" not in result.stdout
    assert steps["check"]["exit_code"] == 0 and steps["train_ddp"]["exit_code"] == 0
    assert "ran-train" in result.stdout  # the rest of the job goes on
    failed_line = next(line for line in result.stdout.splitlines() if line.startswith("failed:"))
    assert set(failed_line.split()[1:]) == {"triton_check", "gpu_tests",
                                            "bench_compression_kernels", "dp_ddp-fp8"}


def test_runners_whose_machines_decide_alone_never_skip_a_mixed_step(tmp_path):
    # mixed/node.sh: a step skipped on one machine only would leave the two a step apart.
    result, steps, checks = _gate(tmp_path, False, """
mixed_torchrun() { echo "ran-mixed $1"; }
TRITON_GATE_MIXED=0
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 step dp_ddp-fp8 mixed_torchrun dp-ddp-fp8
step bench_compressed mixed_torchrun bench-compressed
step bench_compression_kernels echo ran-kernels
""")
    assert "ran-mixed dp-ddp-fp8" in result.stdout and "ran-mixed bench-compressed" in result.stdout
    assert steps["dp_ddp-fp8"]["exit_code"] == 0 and "not_run" not in steps["dp_ddp-fp8"]
    assert checks == 1 and "ran-kernels" not in result.stdout  # a local step still checks
    assert steps["bench_compression_kernels"]["not_run"].startswith("triton_check failed")


def test_a_negative_steps_expectation_and_time_limit_stay_out_of_the_triton_check(tmp_path):
    (tmp_path / "check-env").write_text("")
    result, steps, checks = _gate(tmp_path, True, f"""
mixed_torchrun() {{ echo "ran-$1 fired"; }}
TRITON_CHECK_LAUNCH="env-check {tmp_path}/fake-python"
env-check() {{ echo "timeout=$STEP_TIMEOUT expect=${{STEP_EXPECT:-}}" >>{tmp_path}/check-env; "$@"; }}
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 negative_step neg_fp8 7 fired mixed_torchrun fp8
""", STEP_TIMEOUT="900")
    assert checks == 1 and steps["triton_check"]["exit_code"] == 0
    assert "result" not in steps["triton_check"]  # not "check did not fire"
    assert (tmp_path / "check-env").read_text().split() == ["timeout=900", "expect="]
    assert steps["neg_fp8"]["result"] == "expected failure confirmed"


def test_steps_that_use_the_kernels_run_after_a_passing_check(tmp_path):
    result, steps, checks = _gate(tmp_path, True, """
mixed_torchrun() { echo "ran-$1"; }
GPUBRIDGE_BRIDGE_COMPRESSION=bf16 step bench_bf16 mixed_torchrun bf16
step gpu_tests echo ran-gpu-tests
""")
    assert checks == 1 and steps["triton_check"]["exit_code"] == 0
    assert "ran-gpu-tests" in result.stdout and "ran-bf16" in result.stdout


def test_forced_pytorch_kernels_need_no_triton_check_for_compressed_steps(tmp_path):
    result, steps, checks = _gate(tmp_path, False, """
mixed_torchrun() { echo "ran-$1"; }
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 step dp_ddp-fp8 mixed_torchrun fp8
""", GPUBRIDGE_COMPRESSION_KERNELS="torch")
    assert checks == 0 and "ran-fp8" in result.stdout


def test_compression_exported_for_the_whole_job_gates_only_its_collectives(tmp_path):
    # The runbook's overlay recipe exports GPUBRIDGE_BRIDGE_COMPRESSION for a whole job:
    # the probe, the preflight and one-process runs must still run, checked or not.
    result, steps, checks = _gate(tmp_path, False, """
mixed_torchrun() { echo "ran-$1"; }
step probe both @VENV@/bin/python probe.py
step preflight echo ran-preflight
step dp_single limited echo ran-single
step matrix_D both env A=1 @VENV@/bin/torchrun x
step train_ddp mixed_torchrun train
""", GPUBRIDGE_BRIDGE_COMPRESSION="fp8")
    assert checks == 1 and "ran-preflight" in result.stdout and "ran-single" in result.stdout
    assert all("not_run" not in steps[name] for name in ("probe", "preflight", "dp_single"))
    assert steps["matrix_D"]["not_run"] and steps["train_ddp"]["not_run"]
    assert "ran-train" not in result.stdout


def test_setup_rocm_builds_venv_rocm_unless_vENV_ROCM_names_another():
    # Jobs default to the overlay; setup-rocm still builds a real venv.
    default = dry_run(KIT / "explorer" / "setup.sbatch", USER="alice", TORCH_FLAVOR="rocm")
    assert default.returncode == 0, default.stderr
    assert f"{FAKE_PROJECT}/alice/gpubridge-gpu/venv-rocm " in default.stdout
    assert "triton38" not in default.stdout
    named = dry_run(KIT / "explorer" / "setup.sbatch", USER="alice", TORCH_FLAVOR="rocm",
                    VENV_ROCM="/w/venv-rocm63", ROCM_TORCH_VERSION="2.9.1",
                    ROCM_INDEX="rocm6.3")
    assert named.returncode == 0, named.stderr
    assert "/w/venv-rocm63 " in named.stdout and "torch==2.9.1+rocm6.3" in named.stdout


def test_the_amd_path_says_how_to_build_the_overlay_when_it_is_missing(tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo 4242\n")
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "amd"],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                                          WORK_DIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert f"no ROCm venv at {tmp_path}/venv-rocm63-triton38" in result.stderr
    assert "rocm_triton_overlay.sh" in result.stderr and "4242" not in result.stderr


def test_setup_rocm_builds_its_venv_when_no_rocm_venv_exists_yet(tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo 4242\n")
    for extra in ({}, {"VENV_ROCM": str(tmp_path / "venv-rocm63"), "ROCM_TORCH_VERSION": "2.9.1"}):
        result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "setup-rocm"],
                                env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                              AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                                              WORK_DIR=str(tmp_path), **extra),
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "4242" in result.stderr + result.stdout
    overlay = tmp_path / "venv-rocm63-triton38" / "bin"
    overlay.mkdir(parents=True)
    (overlay / "python").write_text("#!/bin/sh\n")
    (overlay / "python").chmod(0o755)
    refused = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "setup-rocm"],
                             env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                           AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                                           WORK_DIR=str(tmp_path),
                                           VENV_ROCM=str(overlay.parent)),
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 1 and "wrapper scripts" in refused.stderr
    assert "4242" not in refused.stderr + refused.stdout


# ---- Megatron-Core's pipeline: megatron_pp.py ---------------------------------------------

def test_megatron_tokens_follow_the_rule_and_the_seed():
    x, y = megatron_pp.batch(3, 4, 16, 1024, seed=0)
    again, _ = megatron_pp.batch(3, 4, 16, 1024, seed=0)
    assert torch.equal(x, again) and torch.equal(x[:, 1:], y[:, :-1])
    assert (((y - (x * 7 + 13)) % 1024) < 4).all()
    assert not torch.equal(x, megatron_pp.batch(4, 4, 16, 1024, seed=0)[0])


def test_megatron_stages_take_the_whole_models_weights():
    full = {"embedding.word_embeddings.weight": torch.zeros(1),
            **{f"decoder.layers.{i}.mlp.weight": torch.full((1,), float(i)) for i in range(4)},
            "decoder.final_layernorm.weight": torch.ones(1), "output_layer.weight": torch.ones(2)}
    last = {"decoder.layers.0.mlp.weight": None, "decoder.layers.1.mlp.weight": None,
            "decoder.layers.1.mlp._extra_state": "own", "decoder.final_layernorm.weight": None,
            "output_layer.weight": None}
    out = megatron_pp.stage_weights(full, last, offset=2)
    assert out["decoder.layers.0.mlp.weight"] is full["decoder.layers.2.mlp.weight"]
    assert out["decoder.layers.1.mlp.weight"] is full["decoder.layers.3.mlp.weight"]
    assert out["output_layer.weight"] is full["output_layer.weight"]
    assert out["decoder.layers.1.mlp._extra_state"] == "own" and set(out) == set(last)


def test_step_07_megatron_runs_one_stage_then_two():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", STEPS_07="megatron",
                     MEGATRON_PATH="/opt/megatron-core", MEGATRON_STEPS="50")
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("[megatron_pp1]") < result.stdout.index("[megatron_pp2]")
    lines = [line for line in result.stdout.splitlines() if "megatron_pp.py" in line]
    assert len(lines) == 2
    assert "--het-group=0" in lines[0] and "--pipeline 1" in lines[0] and "--steps 50" in lines[0]
    assert "--pipeline 2" in lines[1] and "--nnodes=2" in lines[1]


# ---- pipeline training: train_pipeline.py ------------------------------------------------

def test_pipeline_puts_nvidia_stages_first_whatever_the_ranks():
    assert train_pipeline.stage_order(["nvidia", "amd"]) == [0, 1]
    assert train_pipeline.stage_order(["amd", "nvidia"]) == [1, 0]
    assert train_pipeline.stage_order(["amd", "nvidia", "amd", "nvidia"]) == [1, 3, 0, 2]
    assert train_pipeline.stage_order(["cpu", "cpu"]) == [0, 1]


def test_pipeline_needs_one_pytorch_release_on_every_rank():
    assert train_pipeline.release_mismatch(["2.9.1+cu126", "2.9.1+rocm6.3"]) is None
    mismatch = train_pipeline.release_mismatch(["2.14.1+cu126", "2.9.1+rocm6.3"])
    assert "same PyTorch release" in mismatch and "2.14.1+cu126, 2.9.1+rocm6.3" in mismatch


def test_pipeline_handshake_sends_float32_as_torch_2_14_does():
    from types import SimpleNamespace

    import torch.distributed as dist
    sent = SimpleNamespace(op=dist.isend, tensor=torch.tensor(1))  # torch 2.9.1's int64 index
    received = SimpleNamespace(op=dist.irecv, tensor=torch.zeros(1))
    stage = SimpleNamespace(_get_init_p2p_neighbors_ops=lambda: [sent, received])
    train_pipeline.float_handshake(stage)
    ops = stage._get_init_p2p_neighbors_ops()
    assert ops[0].tensor.dtype == torch.float32 and ops[0].tensor.tolist() == [1.0]
    assert ops[1] is received and ops[1].tensor.dtype == torch.float32
    plain = SimpleNamespace()
    train_pipeline.float_handshake(plain)  # torch 2.8: no handshake, nothing to change
    assert not vars(plain)


def test_pipeline_splits_blocks_evenly():
    assert train_pipeline.even_split(12, 2) == [6, 6]
    assert train_pipeline.even_split(5, 3) == [2, 2, 1]
    with pytest.raises(ValueError):
        train_pipeline.even_split(2, 3)


def _train_pipeline(tmp_path, name: str, nproc: int, *args: str, **env: str) -> dict:
    out = tmp_path / f"pp_{name}.json"
    result = torchrun(nproc, KIT / "train_pipeline.py", "--model", "llama-tiny", "--steps", "8",
                      "--global-batch", "8", "--seq-len", "16", "--lr", "3e-3",
                      "--microbatches", "4", "--probe-trials", "1", "--json", str(out), *args,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half", **env)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(out.read_text())


def test_pipeline_auto_split_balances_on_the_measured_blocks_and_links(tmp_path):
    auto = _train_pipeline(tmp_path, "auto", 2, "--split", "auto", "--schedule", "1f1b")
    assert auto["split_rule"] == "auto" and sum(auto["split"]) == 4 and min(auto["split"]) >= 1
    plan = auto["balance"]
    assert [len(row) for row in plan["layer_ms_by_stage"]] == [4, 4]
    assert len(plan["link_bytes_per_ms"]) == 1 and plan["link_bytes_per_ms"][0] > 0
    assert plan["predicted_speedup_over_even"] >= 1 and plan["even_split"] == [2, 2]
    assert [r["blocks"] for r in auto["ranks"]] == [[0, auto["split"][0]], [auto["split"][0], 4]]
    # Profiling leaves the weights alone: the same split given by hand trains bit for bit
    # the same.
    given = _train_pipeline(tmp_path, "given", 2, "--schedule", "1f1b",
                            "--split", ",".join(map(str, auto["split"])))
    assert given["split_rule"] == "given" and given["balance"] is None
    assert given["losses"] == auto["losses"]


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pipeline_through_gpubridge_matches_plain_gloo_bit_for_bit(schedule, tmp_path):
    # Two stages, one per island: every hop crosses the bridge's point-to-point.
    bridged = _train_pipeline(tmp_path, f"bridged-{schedule}", 2, "--schedule", schedule)
    gloo = _train_pipeline(tmp_path, f"gloo-{schedule}", 2, "--schedule", schedule,
                           "--backend", "gloo")
    assert bridged["losses"] == gloo["losses"]  # full precision
    assert bridged["loss_last"] < bridged["loss_first"]
    assert [r["vendor"] for r in bridged["ranks"]] == ["nvidia", "amd"]
    assert [r["blocks"] for r in bridged["ranks"]] == [[0, 2], [2, 4]]
    assert bridged["ranks"][0]["hop_in_ms"] is None and bridged["ranks"][1]["hop_in_ms"] > 0
    assert bridged["ideal_bubble_fraction"] == round(1 / 5, 4)


def test_four_stages_across_two_islands_match_plain_gloo_and_one_process(tmp_path):
    # 2 + 2: stages 0-1 and 2-3 hop within an island, 1-2 across the bridge.
    bridged = _train_pipeline(tmp_path, "four", 4, "--split", "1,1,1,1")
    gloo = _train_pipeline(tmp_path, "four-gloo", 4, "--split", "1,1,1,1", "--backend", "gloo")
    assert bridged["losses"] == gloo["losses"]
    assert [r["vendor"] for r in bridged["ranks"]] == ["nvidia", "nvidia", "amd", "amd"]
    one = tmp_path / "dp_single.json"
    result = subprocess.run([sys.executable, str(KIT / "train_dp.py"), "--mode", "single",
                             "--model", "llama-tiny", "--steps", "8", "--global-batch", "8",
                             "--seq-len", "16", "--lr", "3e-3", "--micro", "4", "--json", str(one)],
                            env=clean_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    single = json.loads(one.read_text())["losses"]
    # The same mean gradient over the microbatches, summed in another order.
    assert max(abs(a - b) for a, b in zip(bridged["losses"], single, strict=True)) < 1e-4
    report = dp_report.table(dp_report.load([tmp_path]), "single")
    assert "| four | pipeline-gpipe |" in report


def _reversed_pipeline(rank: int, port: int, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE="2", LOCAL_RANK=str(rank))
    train_pipeline.stage_order = lambda vendors: [1, 0]  # stage 0 on rank 1
    sys.argv = ["train_pipeline.py", "--model", "llama-tiny", "--steps", "6", "--global-batch",
                "8", "--seq-len", "16", "--lr", "3e-3", "--microbatches", "4", "--probe-trials",
                "1", "--backend", "gloo", "--split", "3,1", "--json", out]
    train_pipeline.main()


def test_stages_follow_the_stage_order_not_the_ranks(tmp_path):
    import torch.multiprocessing as mp

    out = tmp_path / "pp_reversed.json"
    mp.spawn(_reversed_pipeline, args=(_free_port(), str(out)), nprocs=2)
    reversed_run = json.loads(out.read_text())
    assert [(r["stage"], r["rank"]) for r in reversed_run["ranks"]] == [(0, 1), (1, 0)]
    assert [r["blocks"] for r in reversed_run["ranks"]] == [[0, 3], [3, 4]]  # stage order
    normal = _train_pipeline(tmp_path, "normal", 2, "--steps", "6", "--backend", "gloo",
                             "--split", "3,1")
    assert reversed_run["losses"] == normal["losses"]  # the same pipeline, bit for bit
