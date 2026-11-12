# GPU validation runbook

Step-by-step order for closing the open items in
[HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md) once GPUs are available.
Everything is scripted under [`scripts/gpu/`](../scripts/gpu), so a step is one
command, and every step writes its records to `results/<target>/`. Every shell
script accepts `--dry-run`, which prints the commands without running them.
Try that first on any new machine.

**Which bridge a step runs on.** Since 2026-11-11 the library's default bridge
is `auto` (the verbs bridge where every island leader has a usable RDMA device,
else the TCP bridge), and a job creates no Gloo group unless it asks for Gloo
(with `TORCH_DISTRIBUTED_DEBUG=DETAIL` such a job is refused at `init()` unless
it sets `GPUBRIDGE_ALLOW_DEBUG_GLOO=1`: torch would add Gloo helper groups).
The kit did not move with it: `check.py` and `bench_all_reduce.py` keep
`--bridge gloo` as their default, so the steps built on them still measure the
Gloo bridge, the path every result recorded so far was measured on. Steps that
name no bridge (`train_dp.py`, `train_bench.py` without `--bridge`, the
point-to-point and pipeline scripts) follow `GPUBRIDGE_BRIDGE` and otherwise
run on `auto`; each record's `run.bridge` says what the run used, beside
`bridge_requested` and `bridge_selection`. To repeat an earlier result exactly,
set `GPUBRIDGE_BRIDGE=gloo`. What the new default itself still needs from GPUs
is HARDWARE_VALIDATION.md items 44 to 50, none of which has a kit step yet.

**Before an allocation.** `pytest tests/test_gpu_jobs.py` runs every job script
end to end on CPU, against a fake Slurm (`tests/fake_cluster`: sbatch, srun,
scontrol, module, uv) and gpubridge's simulation mode: `submit.sh` through each
Explorer job, the two-node and heterogeneous jobs on pretend nodes, and
`mixed/node.sh` as two machines. Only the steps that need a GPU (`probe`,
`device_map`) fail there. CI runs it on every push (in the Python 3.12 job),
so a script bug shows up before it can cost GPU time.

**Limits on every job.**
- `KIT_TIMEOUT` (default 300 s, set in `common.sh`): the process-group timeout
  every tool passes to `gpubridge.init`. It bounds the rendezvous and each
  collective, Gloo on the bridge and NCCL/RCCL in the islands, where
  `TORCH_NCCL_ASYNC_ERROR_HANDLING=3` ends the process once a collective
  exceeds it. A tool's `--timeout` overrides it.
- `STEP_TIMEOUT` (default 1200 s): a wall-clock limit on each `torchrun` or
  `srun` launch, through coreutils `timeout`, so one hung step can't use up
  the job. `steps.jsonl` marks such a step `"timed_out": true`.
- Slurm's `--time` on every Explorer job (`TIME_01` ... `TIME_07` in `submit.sh`).
  `submit.sh` caps each job's `STEP_TIMEOUT` at its `--time` less
  `STEP_MARGIN` (120 s), so a single hung launch is stopped and recorded
  before Slurm ends the job: step 01's 10 minutes give 480 s.

**No BLAS in the checks.** The stream-ordering checks hold their input back
behind queued GPU work so that a copy that didn't wait would read stale data.
That work is a spin kernel (`gpukit.queue_gpu_delay`, `torch.cuda._sleep`
calibrated once per device), not matmuls: some ROCm wheels have no BLAS
kernels for older GPUs (gfx906), and gpubridge needs none. To show that the
GPU tests still catch what they guard,
[`stream_mutations.py`](../scripts/gpu/stream_mutations.py) undoes each
ordering fix in turn (`GPUBRIDGE_TEST_MUTATION`) and checks that its test
fails on wrong values, and that the tests pass without one (one GPU):

```bash
# on Explorer (an AMD node: --partition=sharing, its --gres type, venv-rocm)
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:45:00 \
    "$PROJECT_DIR/$USER/gpubridge-gpu/venv/bin/python" scripts/gpu/stream_mutations.py \
    --out results/explorer/stream_mutations
# anywhere else with a GPU
python scripts/gpu/stream_mutations.py --out results/stream_mutations
```

It ends with "every mutation was caught, and the tests pass without one", or
names the run that wasn't, and writes `mutations.json`. Exit 2 means the GPU
tests didn't run.

**Triton's kernel cache.** Explorer jobs build Triton kernels in a node-local
cache at a fixed path, `/tmp/gpubridge-triton-$USER`, filled from, and added
to, `$WORK_DIR/triton-cache` (`TRITON_CACHE_STORE`; `triton_cache_begin` in
`common.sh`). The path stays the same because Triton's cache names its files
by absolute path. A `TRITON_CACHE_DIR` you set wins. `time_triton_cache.py`
times builds and loads by cache location.

**NCCL/RCCL logs.** Every step runs with `NCCL_DEBUG=INFO`, written to
`results/<target>/nccl/<step>/<host>.<pid>.log` instead of the
console. When a step fails, the last 40 lines of each of its logs are appended
to `<step>.log`, so the first thing to read already has them.

**Failures in the job log.** When a step fails, the job's own log (Slurm's
`.out` file) gets the last 20 lines of the step's output right after its
`FAILED` line, before torchrun's failure report. `probe.py` ends a failed
probe with one line per failed check, naming it and its values, such as
`FAIL target_in_build: device 0 target sm_120 not in arch_list [...]`.

**Each job's start and end.** `job_setup` (steps 01 to 07) begins `job.jsonl`
in the job's results directory: a line when the job starts, one for an error
that stops it (`die`), one when it reaches `finish`, and one with the script's
exit code. A job that stops before or during its first step has no
`steps.jsonl` line, but `triage.py` still lists it as unfinished from this
file, with its error or exit code.

## Split-test mode, in one paragraph

`GPUBRIDGE_SPLIT_TEST=half|alternate|node` labels some ranks of an all-NVIDIA
(or all-AMD) job as the other vendor. It is **test only**, and that
environment variable is the only way to turn it on. Ranks keep their real
GPUs and real NCCL/RCCL, so a split test exercises the whole GPU code path:
two native islands, the Gloo bridge, reduce-bridge-broadcast on device
tensors, and `new_group` on ranks outside the group. Only NCCL talking to RCCL
is left untested. Split runs are flagged everywhere, so they can't pass for a
real mixed-vendor result:

- every rank logs a `SplitTestWarning`;
- every JSON record has `run.kind = "split-test"`, `run.split_test`, and a
  `run.warning`;
- every `summary.md` starts with a banner.

## The ladder

Four rungs, climbed in order. Each one is a result the next builds on, so
don't spend GPU time on a rung until the one below passes: a failure there
would fail everything above it, for the same reason.

| Rung | Proves | On Explorer | Elsewhere | Passing means |
| --- | --- | --- | --- | --- |
| 1. Single NVIDIA | the CUDA build runs; NCCL in one native island; the records | `submit.sh 01` (1 GPU, free); with `multigpu`, `submit.sh 03` (2+ GPUs, one NCCL island, native benchmark) | any NVIDIA machine: `torchrun --nnodes=1 --master-addr=127.0.0.1 --nproc-per-node=<GPUs> scripts/gpu/check.py --out results/nvidia` | `01_probe/probe.json` names the GPU and `build_supports_gpus` passes; `03_single_island/check/summary.md` passes with `run.kind` `gpu-single-vendor` |
| 2. NVIDIA split test | two NCCL islands and the Gloo bridge on device tensors, every policy, the GPU-only tests | `submit.sh 02` (2 ranks share 1 GPU, free); with `multigpu`, `submit.sh 04` (2 + 2 on one 4-GPU node, both layouts), then `05` (split by node, so the bridge crosses the network) | one multi-GPU NVIDIA machine: the same command with `GPUBRIDGE_SPLIT_TEST=half` | every summary passes with `run.kind` `split-test`, including `busy_gpu` and `async`; `gpu_tests.xml` has no failures |
| 3. Single AMD | the ROCm build runs; RCCL in one native island; an AMD split test | the ROCm venvs and the MI50 overlay ([Choosing the ROCm build](#choosing-the-rocm-build)), then `submit.sh amd` (one node in `sharing`), if step 00 finds AMD GPUs there | `amd/run_all.sh` on an AMD cloud machine | `check_1gpu`, `rccl_version` and `gpu_tests` pass; with 2+ GPUs, `check_island` and `check_split` too |
| 4. Mixed | NCCL and RCCL islands joined by the bridge: the result items 1 and 2 ask for | `submit.sh 07` (one heterogeneous job on `gpu` + `sharing`), if AMD GPUs are there | `mixed/node.sh` on an NVIDIA and an AMD machine | `check/summary.md` passes with `run.kind` `gpu-mixed` |

- **Split-test mode on one NVIDIA node.** Leave `GPUBRIDGE_VENDOR` unset:
  setting it switches to CPU simulation. The vendor comes from the CUDA
  build, each rank uses `cuda:$LOCAL_RANK`, and both islands use NCCL. With
  fewer GPUs than ranks, split-test mode lets ranks share them (step 02),
  but never two ranks of one island: NCCL refuses that, so `init()` fails
  first on every rank, naming them (item 24). On one node, `half` with at
  most twice as many ranks as GPUs is always fine; every layout here passes.
  If PyTorch reports neither a UUID nor a PCI address for a GPU (some ROCm
  builds), `init()` warns that the island goes unchecked instead of failing.
  Step 02's `shared_gpu_check` runs that layout on purpose, with its own
  60 s limit (`SHARED_GPU_CHECK_SECONDS`), and passes only if the duplicate-GPU
  error appears: `steps.jsonl` records `"result": "expected failure
  confirmed"` or `"check did not fire"`.
  `tests/test_split_test.py` checks this on CPU: the CUDA-build path, and the
  device of each rank on a 4-GPU and a 2-GPU node.
- **Where it went wrong.** Rung 1 failing points at the environment, driver
  or build. Rung 2 failing after 1 passed points at gpubridge's bridge or
  policies on GPUs: fix that before spending AMD time. Rung 3 failing points
  at the ROCm build or RCCL. Rung 4 failing after 2 and 3 passed points at
  NCCL and RCCL together, or at the network between the nodes.
- Scaling (step 06) and DDP training sit beside rung 2; they need rung 2's
  bridge to work.

## Every step at a glance

| # | Rung | Where | Step | What it proves | Items closed | Time (compute) | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0a | - | Explorer login node | `submit.sh 00` | GPU types in `sharing`, `gpu` and `multigpu` (sinfo only): are there AMD GPUs in `sharing`? | - | seconds | free |
| 0b | - | Explorer, CPU job | `submit.sh setup` | venv with the CUDA build of torch 2.14.1, built on a compute node | - | ~10 min | free |
| 1 | 1 | Explorer, 1 GPU | `submit.sh 01` | the build runs on the GPU; records the hardware; a rank whose `LOCAL_RANK` has no GPU fails `init()` on every rank | prerequisite, **12** | ~2 min | free |
| 2 | 2 | Explorer, 1 GPU | `submit.sh 02` | two single-rank NCCL islands sharing one GPU, plus the bridge; a layout with two ranks of one island on the GPU is refused | **3**, 6 (partly), 10 on CUDA, 24 | ~3 min | free |
| 3 | 1 | Explorer, 2+ GPUs | `submit.sh 03` | real NCCL island; each rank drives its own GPU | **5** on NVIDIA, baseline | ~5 min | free* |
| 4 | 2 | Explorer, 4 GPUs | `submit.sh 04` | two 2-GPU NCCL islands and the bridge, both layouts; native vs bridged benchmark | **3, 6, 7**, 9 (data), **11**, 13 | ~10 min | free* |
| 5 | 2 | Explorer, 2 nodes | `submit.sh 05` | the network preflight between the nodes (interfaces, raw bandwidth); NCCL across nodes; the bridge crossing the network between nodes | **8**, 11 and 13 across nodes | ~10 min | free* |
| 6 | 2 | Explorer, 4 GPUs | `submit.sh 06` | training throughput and scaling 1 to 4 GPUs (table for the multigpu request) | - | ~10 min | free* |
| 7 | 3 | AMD cloud machine | `amd/run_all.sh` | RCCL islands, split test on AMD, benchmark, scaling | **4, 5** on AMD, 6, 7, 10 on ROCm, 11, 12, 13 | 20-30 min plus training | ~0.5 h plus training at the hourly rate |
| 7 alt | 3 | Explorer `sharing`, 1 AMD GPU | the ROCm venvs and the MI50 overlay ([Choosing the ROCm build](#choosing-the-rocm-build)), then `submit.sh amd` | rung 3 inside Explorer, **if** step 0a finds AMD GPUs in `sharing`: `amd/run_all.sh` on the node, without setup or tarball. Dry-run only until then | **4, 5** on AMD, 6, 7, 10 on ROCm, 12 | ~15 min + ~15 min ROCm setup | free |
| 8 | 4 | NVIDIA + AMD cloud | `mixed/node.sh` on both | the real mixed-vendor job | **1, 2** on GPUs, **8** across networks | 30-45 min | ~0.75 h of each machine's rate |
| 8 alt | 4 | Explorer `gpu` + `sharing`, one heterogeneous job | the ROCm venvs and the MI50 overlay ([Choosing the ROCm build](#choosing-the-rocm-build)), then `submit.sh 07` | the real mixed-vendor job inside Explorer, **if** step 0a finds AMD GPUs in `sharing`. Dry-run only until then | **1, 2** on GPUs, 3 with real RCCL ranks, 8 between nodes, 11 against per-island baselines, 13 | ~15 min + ~15 min ROCm setup | free |

\* Steps 3 to 6 need the `multigpu` partition, which needs an access request
(see [Explorer](#explorer)). Steps 1 and 2 run on the open `gpu` partition.
Step 8 alt replaces step 8, and saves the cloud machines, only if Explorer has
AMD GPUs; see [Mixed-vendor job inside Explorer](#mixed-vendor-job-inside-explorer).
Every step that runs `check.py` also covers items 14 to 16 (reductions,
all_gather, reduce_scatter, async, and where NCCL/RCCL and Gloo agree) and
item 23 (what `in_flight()` shows while one rank is late).
Steps 04, 05, 07 and the AMD script also check every opt-in policy in
`OPT_IN_POLICIES` (the pipelined and sharded policies) and write
`thresholds.json` (items 17 to 21); step 05 also times each one's phases.
Opt-in policies are not validated on GPUs until those steps pass.
The same steps then rerun the check and the gpubridge benchmark with
`GPUBRIDGE_PINNED_STAGING=1` (`check_pinned`, `bench_pinned`; item 25), so one
allocation compares the default bridge's pageable copies with pinned staging:
read `bench_pinned/summary.md` next to the pageable benchmark's gpubridge rows
and `bridge` phase. `validation_tables.py` shows each benchmark's staging.
Steps 02 and 07 also run `check.py --p2p-only` (`check_p2p`; item 32): point-to-point
through `backend="gpubridge"` across the two islands, six dtypes and odd sizes bit for
bit, with the send's input written on a busy stream and dropped before it completes, the
received tensor read on another stream right after `wait()`, and two tags matched out of
order. `gpu_tests` runs the same check in
`test_point_to_point_across_islands_keeps_stream_order_and_storage`.
Both steps also run `bench_p2p.py` (`bench_p2p`, and in step 07 `bench_p2p_pinned`
with `GPUBRIDGE_PINNED_STAGING=1`): ping-pong between rank 0 and a rank of another island
(the bridge route) and, where rank 0's island has another rank, that rank (the native
route), from 1 KB to `BENCH_MAX_BYTES`, with the one-way latency, bandwidth and a payload
check per size in `bench_p2p/summary.md`. Verification uses a separate untimed exchange
before warmups and timed trials, including with `--warmup 0`. To compare other staging,
rerun it with `GPUBRIDGE_P2P_CHUNK_BYTES` or `GPUBRIDGE_BRIDGE=ucx`; each record names its staging.
With `UCX_BRIDGE=1`, steps 05 and 07 also run the check and the gpubridge
benchmark over the UCX bridge (`check_ucx`, `bench_ucx`; items 26 and 27); the
run records say whether GPU tensors went to UCX directly or through pinned host
memory (`run.bridge_path`), and how received data was flushed into the GPU
(`run.bridge_flush`).
With `VERBS_BRIDGE=1`, steps 05 (same vendor) and 07 (NVIDIA against AMD) also
build the raw-verbs bridge on each node (`python -m gpubridge.verbs build`; rdma-core's
headers, or `GPUBRIDGE_VERBS_INCLUDE_DIR`) and run its kit
([VERBS_TRANSPORT.md](VERBS_TRANSPORT.md), items 38 to 42). In order:
`verbs_rdma_report` (`rdma_report.sh` per node: kernel dma-buf options, peer-memory
modules, `ibv_reg_dmabuf_mr`, ROCr's dma-buf export); `verbs_endpoint` (host-to-host
verbs between one rank per node: latency, bandwidth, SHA-256 of every buffer);
`verbs_host`, then `verbs_gpu_staged` and `verbs_gpu_direct` (`verbs_kit.py
collectives`: host tensors, then GPU tensors with `GPUBRIDGE_VERBS_GPU_PATH=staged` and
`=direct`, exactly against a Gloo reduction; a direct path that can't run fails the step
with its reason instead of staging); `check_verbs` and `bench_verbs` over
`bridge="verbs"`, next to the Gloo bridge's runs and UCX's (`UCX_BRIDGE=1`) as matched
references; `train_verbs` and `train_gloo` (`train_bench.py`, `VERBS_TRAIN_MODEL`).
The engineering baseline they are compared against is `train_bench.py --grad-sync
native`: plain DDP on an NCCL (RCCL on ROCm) world group, no gpubridge, same flags;
one vendor only. The 2026-10-20 split-test sweep ran this native baseline
beside Gloo and staged verbs at matching model, batch and bucket settings;
repeat it beside `train_verbs` for future comparisons. On CPU it runs on Gloo
and its record says `native_stand_in`: never an NCCL/RCCL result.
Each `verbs_kit` record holds the endpoint telemetry (device, port, GID, provider: it
must not say MOCK), every call's `path` and `path_reason`, completion errors, a window
of every NIC counter and every interface's bytes (context only: shared, and RoCE
netdevs carry non-TCP traffic), CPU time, and `tcp_evidence`; with `--require-no-tcp` a
bridge leader's step fails unless that evidence is `pass`: the bridge's own telemetry
(libibverbs, not MOCK) moved the payload with completions and no errors, and the
process's own TCP sockets (`ss -tinp`) stayed idle. Without `ss` on the node it is
`unknown` and the step fails; install or load iproute2's `ss` before the run.
Transport observations go to `verbs_observations/rank-<N>.jsonl`
(`GPUBRIDGE_VERBS_OBSERVE`). Host mode uses CPU/Gloo islands. With co-resident
leaders its TCP evidence is `not-decisive`, and `--require-no-tcp` fails closed,
unless every same-host socket is attributed by the process at its other end: then
only the leader's own island's sockets are left out, every other one counts, and the
record's `local_tcp_attribution` shows each socket (docs/VERBS_TRANSPORT.md); GPU
mode still counts same-host non-loopback TCP.

On 2026-10-20, `c92684b` passed the staged-verbs kit, default-policy check,
benchmark and training sweep on two 2-GPU NCCL islands of identical V100s.
The original host kit failed before payload; the `0cf32a3` recheck passed exact
host collectives with CPU/Gloo islands, with the TCP limitation above, and
passed GPU-staged strict regression checks. Hardware: one node with 4x Tesla
V100-SXM2-32GB, driver 545.23.08, torch 2.14.1+cu126, NCCL 2.29.3, Mellanox
ConnectX (mlx5) HDR InfiniBand. Verbs used one HCA on one host: loopback, not
cross-node, mixed-vendor or GPUDirect. UCX, 8-GPU scaling, statistical
significance, the unrestricted pipeline speed goal and the collective
`wait-skips-stream` mutation gap remain open. See
[hardware results](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20)
and the [changelog](../CHANGELOG.md#unreleased).

Steps 04 and 07, the AMD script and `mixed/node.sh` also measure opt-in bridge
compression (items 28 and 29). `bench_compression_kernels` times the Triton
kernels against the GPU's memory bandwidth and checks they give the CPU
reference's bits; in step 07 it runs on both sides, and the digests in
`bench_compression_gpu` and `bench_compression_amd` must match.
`bench_compressed` times the bridge with and without compression
(`--compression none,bf16,fp8 --phases`), and its summary.md says from which
size each mode is faster.
Steps 04, 06, 07 and the AMD script train with DDP and `gpubridge.ddp_comm_hook`
(`examples/train_synthetic.py`, item 22).
Steps 02, 04 and the AMD script also run the GPU-only tests, `pytest -m gpu
tests/test_gpu.py tests/test_compression_kernels.py` (step `gpu_tests`, results
in `gpu_tests.xml`): the compression kernels against the CPU reference, and every
bridge policy against `flat-gloo` bit for bit for every op, fp32/fp16/bf16, odd
and large sizes; non-contiguous tensors refused; sync and async
collectives called from non-default streams behind queued GPU work; the
pipelined policy's pinned slots reused by the next collective while copies
back out of them are held back; and a result read right after
`is_completed()` (items 6, 15, 16 and 17). On one GPU they run two ranks
sharing it.
After the steps, `scripts/gpu/validation_tables.py` builds HARDWARE_VALIDATION.md's
tables from every job's results, with the bridge time each preflight predicts
next to the one measured ([The validation tables](#the-validation-tables)).
Times exclude queue waits. For cost, multiply by the provider's current hourly
rate; for example, at a hypothetical $3/h, step 7 costs about $1.50.

Run them in this order. Each step is cheaper than the next and catches the
problems the next one would otherwise waste time on.

## Explorer

### Settings to verify first

All site settings live at the top of
[`scripts/gpu/explorer/submit.sh`](../scripts/gpu/explorer/submit.sh). Override
any of them with an environment variable of the same name. The defaults come
from the [RC docs](https://rc-docs.northeastern.edu/en/explorer-main/gpus/accessinggpus.html)
as of 2026-10-06; **verify each one**:

| Setting | Default | Check in the RC docs |
| --- | --- | --- |
| `PARTITION_SETUP` | `short` | a CPU partition for the setup job (RC's interactive example uses `short`) |
| `PARTITION_1GPU` | `gpu` | name of the open GPU partition (1 GPU per job) |
| `PARTITION_MULTI` | `multigpu` | name of the multi-GPU partition, and that you have access |
| `PARTITION_AMD` | `sharing` | the partition step 00 checks for AMD GPUs (step 07 only) |
| `GPU_TYPE_1GPU`, `GPU_TYPE_MULTI` | empty (any GPU) | `--gres` type strings, e.g. `v100-pcie`, `v100-sxm2`, A100/H100/H200 names |
| `GPU_TYPE_AMD` | empty, required for step 07 | the AMD `--gres` type from step 00's `gpu-types.txt` |
| `ACCOUNT` | empty | whether jobs need `--account` |
| `CPUS_PER_GPU`, `MEM_PER_GPU` | `4`, `32G` | per-GPU CPU and memory limits |
| `CPUS_SETUP`, `MEM_SETUP`, `TIME_SETUP` | `4`, `16G`, 1 h | limits of the setup partition |
| `TIME_01` ... `TIME_07` | 10-40 min (`TIME_04` is 40) | maximum walltime per step's job |
| `PROJECT_DIR` | none: **required** | your group's project space, e.g. `/projects/<your-project>`; set it in the environment or in `scripts/gpu/explorer/local.env` (see [Steps](#steps)) |
| `WORK_DIR` | `$PROJECT_DIR/$USER/gpubridge-gpu` | venvs, caches and job logs: ~7 GB, ~25 GB with the ROCm venv. Not home, which is capped at 75 GB |
| `VENV`, `VENV_ROCM` | `$WORK_DIR/venv`, `$WORK_DIR/venv-rocm63-triton38` | the CUDA venv, and the AMD side's: the MI50's Triton overlay (see [Choosing the ROCm build](#choosing-the-rocm-build)) |
| `ROCM_SETUP_VENV` | `VENV_ROCM` if you set it, else `$WORK_DIR/venv-rocm` | the venv `setup-rocm` builds |
| `ROCM_TORCH_VERSION`, `ROCM_INDEX`, `ROCM_INDEX_URL` | `2.14.1`, `rocm7.2`, the PyTorch index for `ROCM_INDEX` | the ROCm build for `setup-rocm`; see [Choosing the ROCm build](#choosing-the-rocm-build) |
| `PARTITION_MIXED_NVIDIA`, `GPUS_MIXED_NVIDIA`, `GPUS_MIXED_AMD` | `gpu`, `1`, `1` | step 07's GPUs per side |
| `SETUP_MODULES` | `explorer` | loaded before installing: it routes compute nodes through the proxy. Set it to `""` to load nothing |
| `MODULES` | empty | modules for steps 01 to 07; none needed: the PyTorch wheel bundles CUDA/ROCm and uv brings Python |
| `NCCL_SOCKET_IFNAME` | empty (auto) | name of the fast interconnect interface for steps 05 and 07, e.g. `ib0` |
| `GLOO_SOCKET_IFNAME` | empty: the preflight chooses | in steps 05 and 07, the interface Gloo uses on every node; the preflight then measures only it. Unset, each node uses the interface the preflight chose for it, even when the two nodes' names differ |
| `USE_PREFLIGHT_IFNAME` | `1` | `0` leaves `GLOO_SOCKET_IFNAME` unset after the preflight, so Gloo picks by hostname |
| `SHARED_GPU_CHECK_SECONDS` | `60` | step 02's limit for its deliberate failing run, which fails within seconds when the check works |
| `UCX_BRIDGE` | `0` | `1`: steps 05 and 07 also build the UCX bridge on each node (`python -m gpubridge.ucx build`) and run `check_ucx` and `bench_ucx` over `bridge="ucx"` (items 26 and 27; [RDMA_TRANSPORT.md](RDMA_TRANSPORT.md)). Needs UCX and a C++ compiler on the nodes, e.g. `MODULES=ucx` |
| `VERBS_BRIDGE` | `0` | `1`: steps 05 and 07 also build the verbs bridge on each node and run its kit (`verbs_endpoint`, `verbs_host`, `verbs_gpu_staged`, `verbs_gpu_direct`, `check_verbs`, `bench_verbs`, `train_verbs`, `train_gloo`; items 38 to 42; [VERBS_TRANSPORT.md](VERBS_TRANSPORT.md)). Needs an RDMA device on each node that a job may open, and rdma-core's headers where it builds |
| `VERBS_TRAIN_MODEL` | `gpt2-small` | the model `train_verbs` and `train_gloo` train |
| `PREFLIGHT_SECONDS` | `2` | seconds of raw TCP data each way per pair of interfaces in the preflight |
| `TIME_NET`, `TIME_08`, `TIME_TRAIN` | 10, 40, 30 min | walltimes of the network check, the comparison matrix and the one-GPU training benchmark |
| `SAME_PARTITION_PAIRS_OK` | `0` | `1` lets `08-nvidia-pair`, `08-amd-pair`, `07-nvidia-pair` and `07-amd-pair` submit: two GPUs of one type in one partition, once RC confirms that is within intended use. The 07 pairs run step 07's `STEPS_07=dp` or `pipeline` on one island of two (native NCCL or RCCL), the same-vendor baselines in [BENCHMARKS.md](BENCHMARKS.md#what-mixing-vendors-costs-and-whether-it-helps) |
| `PARTITION_PAIR_NVIDIA`, `PARTITION_PAIR_AMD` | `gpu`, `sharing` | where those pairs' two components go |
| `PIPELINE_CHUNKS` | `1M 4M 16M 64M` | step 08's pipelined runs, one per `GPUBRIDGE_CHUNK_BYTES` |
| `TRAIN_BENCH_MODELS`, `TRAIN_BENCH_ARGS` | `gpt2-small resnet50`, empty | `train_bench.py`'s models and extra flags, in step 07 and `train-nvidia`/`train-amd` |
| `STEPS_07` | `all` | `training`: step 07 runs only the probe, the preflight and its training steps; `ddp`: only the probe, the preflight and `train_ddp`; `bench`: only the probe, the preflight and the `train_bench_<model>` steps (no `train_ddp`, FSDP2 or HSDP) |
| `TRAIN_DDP_REPEATS` | `1` | step 07 runs `train_ddp` this many times (1 to 999), each its own step: `train_ddp_1` to `train_ddp_<N>` (step and JSON). `1`: once, as `train_ddp` (see [Repeating train_ddp](#repeating-train_ddp)) |
| `TRAIN_BENCH_BUCKETS` | empty | DDP bucket sizes in MB, e.g. `10 25 50 100`: step 07 runs each `train_bench_<model>` step once per size (`--bucket-mb`), named `train_bench_<model>_bucket<MB>` (step and JSON). Empty: once, at `train_bench.py`'s default (25) |

Also check:

- **CUDA wheel vs GPU.** `setup_env.sh` installs the `cu126` build, because
  CUDA 13 builds dropped V100 (sm_70). Step 01 fails with a clear message if
  the build has no kernels for your GPU; set `CUDA_INDEX` (e.g. `cu128`) and
  rerun setup.
- **multigpu access.** The request form asks for run times on 1, 2, 3 and 4
  GPUs with scaling efficiency above about 0.5. Step 06 produces exactly that
  table (`results/explorer/06_scaling/summary.md`), so run it as soon as RC
  grants a test window.

### Steps

For a first session, [EXPLORER_FIRST_SESSION.md](EXPLORER_FIRST_SESSION.md)
has every command from a fresh login, what each prints, and what to paste
back. In short:

```bash
PROJECT_DIR=/projects/<your-project>
mkdir -p "$PROJECT_DIR/$USER" && cd "$PROJECT_DIR/$USER"   # the project space, not home
git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
echo "PROJECT_DIR=$PROJECT_DIR" > scripts/gpu/explorer/local.env   # once
scripts/gpu/explorer/submit.sh 00               # GPU types per partition (sinfo, seconds)
scripts/gpu/explorer/submit.sh --dry-run free   # see what will be submitted
scripts/gpu/explorer/submit.sh setup            # a short CPU job, once (~10 min)
squeue -u "$USER"                               # wait for the setup job to finish
scripts/gpu/explorer/submit.sh free             # steps 01 and 02
squeue -u "$USER"                               # wait for them
cat results/explorer/02_split_shared_gpu/check/summary.md
# after multigpu access:
scripts/gpu/explorer/submit.sh multi            # steps 03 to 06
scripts/gpu/explorer/submit.sh train-multi      # training throughput: 1 GPU, 4 as one island, 2 + 2
scripts/gpu/explorer/submit.sh 07-multi         # step 07 with 4 NVIDIA GPUs + the AMD side
```

**Your settings.** Every step except 00 needs `PROJECT_DIR`, your group's
project space, and stops with an error until it's set. Put it in the
environment or in `scripts/gpu/explorer/local.env`, which git ignores. That
file is parsed, not run: one `NAME=value` per line, and any setting from
`submit.sh` works there. The environment overrides it.

**Setup runs on a compute node, never the login node.** `submit.sh setup`
submits [`setup.sbatch`](../scripts/gpu/explorer/setup.sbatch) to the
`short` partition. The job loads the `explorer` module first, because compute
nodes reach the internet only through the proxy that module sets up; then it
installs uv, Python 3.12 and torch into `$WORK_DIR/venv`. Its log is
`$WORK_DIR/logs/gpubridge-setup-cuda-<job id>.out`. `setup_env.sh` itself
refuses to run on a Slurm login node (sbatch present, no job), so a direct call
there fails instead of installing.

**Or, from an interactive job:**

```bash
srun -p short -N 1 -n 1 -c 4 --mem=16G -t 01:00:00 --pty bash   # wait for a shell on a compute node
cd gpubridge
GPUBRIDGE_REPO=$PWD bash scripts/gpu/explorer/setup.sbatch      # loads explorer, then installs
exit
```

Step 00 runs `sinfo` on the login node: it only queries Slurm, and starts
nothing. It writes to `results/explorer/00_partitions/`:

- `sinfo-<partition>.txt`: `sinfo -p <partition> -o "%20N %10c %10m %25f %10G %10t"` for `sharing`, `gpu` and `multigpu`;
- `gres-<partition>.txt`: each node's full GRES string, which that table's 10-character column truncates;
- `gpu-types.txt`: every GPU type per partition, labelled `amd`, `nvidia` or `unknown` from its name, and a closing line that says whether `sharing` has AMD GPUs.

Job logs go to `$WORK_DIR/logs/`, and results to `results/explorer/<step>/`.
A job never writes into another run's folder: if `results/explorer/<step>/`
already holds a `job.jsonl` or `steps.jsonl`, its results go to
`results/explorer/<step>_job<id>/` instead (`_2`, `_3` ... if that is taken too), and
the log says so.

A job stops before its first step if **`import gpubridge` isn't this checkout's
`src/`** (`import_check.py`, and the probe on each side of a mixed job). The venvs
install gpubridge editable from one checkout, so a job run from another checkout, or
with `PYTHONPATH` set, would otherwise run that checkout's scripts on other code. Run from the checkout the venv
installs, or put this checkout's `src` first on `PYTHONPATH`. The probe records where
gpubridge came from, its commit and `PYTHONPATH`.

Steps that use gpubridge's Triton compression kernels check first that Triton can
compile and run them (all four: quantize and dequantize, bf16 and fp8), once per job
before the first such step (step `triton_check`, `triton_check.py`): on both sides of
07 and 08, on the NVIDIA side only for 07's dp runs with `DP_KERNELS_ON_AMD=torch`, and
on the machine itself elsewhere. The steps: `gpu_tests`,
`bench_compression_kernels`, `bench_compressed`, and any torchrun launch with
`GPUBRIDGE_BRIDGE_COMPRESSION` set unless `GPUBRIDGE_COMPRESSION_KERNELS=torch`. The
probe, the preflight and one-process runs never check, even with compression
exported for the whole job. If the check fails, as `venv-rocm63`'s own Triton does on
the MI50's gfx906, those steps are recorded as failed without running (`"not_run"` in
`steps.jsonl`) and the rest of the job goes on; `gpu_tests` is skipped as a whole,
its Triton-free tests included, so use the overlay venv for it. The probe records
whether Triton compiles the kernels, without failing on it. `mixed/node.sh` checks
the import too, but never skips a mixed step: its two machines decide alone, so a
compressed step runs on both and fails on both if either's Triton can't compile.

Keep the raw results on the cluster: the step logs, stack dumps and NCCL logs
aren't redacted (see [Reading results](#reading-results)). For review, paste
the output of `"$PROJECT_DIR/$USER/gpubridge-gpu/venv/bin/python" scripts/gpu/triage.py results/explorer`
(the venv's Python: a login node's `python3` may be older than 3.10). What
gets published is the validation tables, not the results.

### Mixed-vendor job inside Explorer

**Dry-run only until step 00 shows AMD GPUs in `sharing`.** If it does, one
Slurm heterogeneous job can run the real mixed-vendor test without cloud
machines:

- component 0 is on `gpu` (`PARTITION_MIXED_NVIDIA`): `GPUS_MIXED_NVIDIA`
  NVIDIA GPUs, default 1, using the CUDA venv;
- component 1 is on `sharing`: `GPUS_MIXED_AMD` AMD GPUs of type
  `GPU_TYPE_AMD`, using a ROCm venv built by a setup job on an AMD node.

Both components sit on the cluster network, so no overlay is needed.

```bash
scripts/gpu/explorer/submit.sh 00                  # then read 00_partitions/gpu-types.txt
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh --dry-run setup-rocm
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh --dry-run amd
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh --dry-run 07
# once the hardware is confirmed and AMD_NODES_CONFIRMED=1 is in explorer/local.env:
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh setup-rocm   # venv-rocm (~16 GB), on an AMD node
VENV_ROCM=$WORK_DIR/venv-rocm63 ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.3 \
    GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh setup-rocm   # venv-rocm63, for training
scripts/gpu/rocm_triton_overlay.sh $WORK_DIR/venv-rocm63 $WORK_DIR/venv-rocm \
    $WORK_DIR/venv-rocm63-triton38                                # the jobs' default VENV_ROCM
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh amd          # rung 3: one AMD node on its own
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh 07           # rung 4, once amd passes
```

`submit.sh amd` ([`amd_node.sbatch`](../scripts/gpu/explorer/amd_node.sbatch))
runs `amd/run_all.sh` on one AMD node with the ROCm venv, skipping its setup
and its results tarball; `GPUS_AMD_NODE` (default 1) sets how many GPUs. Until
the hardware is confirmed, `setup-rocm`, `amd` and `07` stop with an error
unless `--dry-run` is given. The switch is `AMD_NODES_CONFIRMED=1` in
`scripts/gpu/explorer/local.env` (or the environment): a local setting, so
turning the path on leaves the clone at a clean commit, and the records'
`git_dirty` false.

[`07_mixed_hetjob.sbatch`](../scripts/gpu/explorer/07_mixed_hetjob.sbatch)
runs each phase as one heterogeneous job step (`srun --het-group=0 ... :
--het-group=1 ...`), each side with its own venv's `python` or `torchrun`:

1. `probe` on each side (`probe_nvidia/`, `probe_amd/`), plus `rocm-smi` and `amd-smi` from the AMD node;
2. `preflight`: the gpu node as master, the AMD node as worker (see [The network preflight](#the-network-preflight)). Gloo on each node then uses the interface the preflight chose there. The job stops here if either side lacks a usable GPU or no pair of interfaces connects both ways;
3. `check`: one torchrun per node, with the c10d rendezvous on the gpu node. Both write into `check/` on the shared filesystem, so `summarize` covers every rank without copying. `run.kind` must be `gpu-mixed`, with `split_test` null;
4. `bench` (`--ops island,gpubridge,flat-gloo`): first each island times its
   own native all_reduce, both at once (NCCL on the gpu node, RCCL on the AMD
   node; rows `island:nvidia` and `island:amd`), then the bridged all_reduce
   across both, then flat-gloo across both. A native all_reduce over every
   rank is impossible here, because NCCL and RCCL ranks can't form one group,
   so the baseline is each island's own, labeled `island-native (fewer ranks,
   best case)` in every row and in the `gpubridge / ...` and `flat-gloo / ...`
   columns of `summary.md`. It is the slowest island at each size: fewer ranks
   and no bridge, so a best case, not a like-for-like comparison.

**Per-island baselines need 2+ GPUs per island.** With one GPU, an island's
all_reduce has nothing to exchange, so its row only times the call itself.
`gpu` allows one GPU per job. For real baselines, set `GPUS_MIXED_AMD=2`, and,
with multigpu access, use `submit.sh 07-multi`: the NVIDIA side on `multigpu`
with `GPUS_MIXED_NVIDIA_MULTI` GPUs (4 by default, of `GPU_TYPE_MULTI`).

Also check with RC before the first real run:

- that heterogeneous jobs are allowed, and can span `gpu` and `sharing`;
- `sharing`'s walltime and preemption rules;
- which ROCm build supports the AMD GPUs; see below.

Interface names may differ between the two partitions' nodes, which is why
the preflight chooses one per node. Set `GLOO_SOCKET_IFNAME` only to force an
interface that exists on both.

#### The network preflight

`scripts/gpu/preflight.py` runs on the two leader nodes before any torchrun:
step 05's two nodes in `multigpu`, step 07's `gpu` and `sharing` nodes, and
`mixed/node.sh`'s two machines. In order:

1. `resolve`, `interface`, `tcp`: the master's address resolves, a given
   `GLOO_SOCKET_IFNAME` exists, and the worker reaches the master's port.
2. `interfaces`: each node's candidates, the interfaces with an IPv4 address,
   minus loopback, link-local addresses and container bridges (or only
   `GLOO_SOCKET_IFNAME`'s).
3. `reachability`: the pairs of interfaces, one per node, that the kernels
   route between. For each pair, a TCP connection each way (a Gloo pair
   connects in whichever direction its addresses decide), then over the first
   one the round-trip time and raw TCP bandwidth one way and then the other,
   over one stream, as Gloo uses one connection per pair of ranks.
4. `choose`: the pair that connects both ways and whose slower direction is
   fastest. Each node's interface in it becomes its `GLOO_SOCKET_IFNAME` for
   the rest of the job. Where the two names differ, step 05 starts each
   node's torchrun through `node_ifname.sh`, which sets that node's own name;
   step 07 sets each component's.
5. `gloo`: a real two-rank Gloo group on the chosen interfaces.

`preflight/preflight-<role>.json` records every pair (both connections,
MB/s each way, round-trip time) and, for the chosen one, `chosen_ifname`,
`peer_ifname` and `bandwidth` from that node's side. These are host network
numbers, measured before any GPU work. The bandwidth is one TCP stream driven
from Python, so it may understate a fast link (100 Gb/s Ethernet, IPoIB):
read it as a lower bound. The record's `bandwidth.note` says so too.

#### Choosing the ROCm build

Choose `setup-rocm`'s PyTorch from the AMD GPU model that step 00 finds. The
default, torch 2.14.1 for ROCm 7.2, may not support older Instinct cards. The
GRES type usually names the model:

| GPU | gfx target |
| --- | --- |
| MI50, MI60 | gfx906 |
| MI100 | gfx908 |
| MI210, MI250, MI250X | gfx90a |
| MI300A, MI300X | gfx942 |

1. Find the newest ROCm release that supports that gfx target in AMD's
   [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
2. Pick the newest PyTorch release with a wheel for that ROCm line and
   Python 3.12 on [download.pytorch.org/whl](https://download.pytorch.org/whl/).
   Index names look like `rocm6.4`. gpubridge needs 2.8 or later; if no
   wheel for that ROCm line is that new, gpubridge doesn't support the GPU.
3. Set them for `setup-rocm`, e.g.
   `ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.4`. Set `ROCM_INDEX_URL` only
   for a mirror or another index; it defaults to
   `https://download.pytorch.org/whl/$ROCM_INDEX`.

`setup-rocm` ends with a probe on the AMD node, and fails if the build has no
kernels for the GPU. `results/explorer/setup_rocm/probe.json` lists the GPU's
target and the build's targets.

**The target check is not enough for training: rocBLAS needs the GPU's kernels
too.** Every ROCm wheel from 2.8 on lists gfx906 (MI50, MI60) as a target, but
among them only the `rocm6.3` builds of 2.8.0, 2.9.0 and 2.9.1 ship rocBLAS
kernels for it ([BENCHMARKS.md](BENCHMARKS.md#which-rocm-wheels-can-train-on-an-mi50-gfx906)
has the scan). With the default wheel the kit's checks pass, but the first
matmul aborts, so the training steps fail. With 2.9.1+rocm6.3, matmuls run, but
its Triton (pytorch-triton-rocm 3.5.1) can't compile gpubridge's compression
kernels for gfx906. So on an MI50, keep `venv-rocm` and add `venv-rocm63` for
training; the jobs' default, the overlay below, joins the two. To use `venv-rocm63`
alone, for steps that train without the compression kernels:

```bash
VENV_ROCM=$WORK_DIR/venv-rocm63 ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.3 \
    scripts/gpu/explorer/submit.sh setup-rocm
VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=training scripts/gpu/explorer/submit.sh 07
```

The probe's `blas` stage (`blas_available` in `probe.json`) says whether a venv's
matmuls run on the GPU.

**Compression kernels while training on an MI50.** The default venv's Triton
(triton-rocm 3.8.0) compiles them for gfx906. `scripts/gpu/rocm_triton_overlay.sh`
makes a third, env-only venv: `bin/python`, `python3` and `torchrun` that run
`venv-rocm63`'s, with the default venv's Triton linked in front on the path.
Nothing in either venv changes, nothing is copied, and no Python runs, so it is
fine on a login node. On Explorer it is the default `VENV_ROCM`
(`$WORK_DIR/venv-rocm63-triton38`) for every job with an AMD side; submit.sh stops
with these commands if it is missing, and setup-rocm still builds `venv-rocm` unless
`VENV_ROCM` (or `ROCM_SETUP_VENV`) names another venv; it refuses to build into a
directory of wrapper scripts such as the overlay:

```bash
scripts/gpu/rocm_triton_overlay.sh $WORK_DIR/venv-rocm63 $WORK_DIR/venv-rocm \
    $WORK_DIR/venv-rocm63-triton38
VENV_ROCM=$WORK_DIR/venv-rocm63-triton38 GPUBRIDGE_BRIDGE_COMPRESSION=fp8 \
    GPUBRIDGE_COMPRESSION_KERNELS=triton STEPS_07=training scripts/gpu/explorer/submit.sh 07
```

The GPU tests and fp8 training passed with a manually constructed overlay of
the same Triton files; this script's generated wrappers have CPU tests but
have not yet run on the MI50
([BENCHMARKS.md](BENCHMARKS.md#fp8-with-triton-kernels-on-the-mi50)). Rebuilding
the default venv changes the Triton the overlay links to.

**`(null): No such file or directory`, once per GPU per process.** The
`libdrm_amdgpu` that the ROCm wheels bundle looks for its GPU-name table,
`amdgpu.ids`, in `/opt/amdgpu/share/libdrm` and then by walking the directory
tree from the Python executable. When both fail, it prints that line, with
`(null)` where the path should be, for each GPU a process opens. The wheel
ships the table itself (`torch/share/libdrm/amdgpu.ids`), so the kit exports
`AMDGPU_ASIC_ID_TABLE_PATHS` pointing there before any ROCm step
(`rocm_asic_ids` in `common.sh`), unless you set it. The line is harmless
either way: only the GPU's marketing name comes from that table.

The CUDA side stays at 2.14.1. On CPU, mixed releases from 2.9.1 to 2.14.1
worked (item 2 in HARDWARE_VALIDATION.md). An older ROCm-side release is
untested, and `gpubridge.init()` warns when releases differ. For identical
releases, set `TORCH_VERSION` (and `CUDA_INDEX` if needed) for `submit.sh setup` too.

### The comparison matrix and training throughput

What gpubridge costs against native NCCL/RCCL and flat Gloo. Every run is
2 ranks, one GPU each on two nodes, `bench_all_reduce.py` with float32 from
1 KB to 1 GB, timed with CUDA events, with `--phases` on every gpubridge run.

| Run | GPUs | What | Job |
| --- | --- | --- | --- |
| A1 | V100 + V100 | native NCCL, the transport NCCL picks | `submit.sh 08-nvidia-pair` |
| A2 | V100 + V100 | native NCCL over TCP: `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME` = the bridge's interface | `08-nvidia-pair` |
| B1, B2 | MI50 + MI50 | the same with RCCL | `submit.sh 08-amd-pair` |
| C | V100 + V100 | through gpubridge's bridge, each GPU an island (`GPUBRIDGE_SPLIT_TEST=node`) | `08-nvidia-pair` |
| C-amd | MI50 + MI50 | the same | `08-amd-pair` |
| D-* | V100 + MI50 | through gpubridge: `default`, `pinned`, `pipelined-<chunk>` (`PIPELINE_CHUNKS`, default `1M 4M 16M 64M`), `sharded`, `bf16`, `fp8`, and `default-repeat` last | `submit.sh 08` |
| E | V100 + MI50 | one flat Gloo group of both ranks (`flat-gloo`) | `08` |
| E-nvidia, E-amd | the pairs | the same | the pair jobs |

How to read it: **A1 against A2 is the network** (the transport NCCL picks
against TCP, for the same NCCL in one job), **A2 against C is gpubridge's own
cost** (the same job, GPUs and interface), **C against D is the vendor mixing**,
and D's variants against `D-default` are what each policy buys. Each pair is
its own job on its own nodes, so C against D, and every multiple of A1 or A2 on
a D or E row, also compares two node pairs' networks: `matrix.md` lists each
job's preflight and marks those multiples with †.

```bash
scripts/gpu/explorer/submit.sh net-nvidia   # one V100 node's network (~1 min)
scripts/gpu/explorer/submit.sh net-amd      # one MI50 node's
scripts/gpu/explorer/submit.sh 08           # D and E: one gpu + sharing job (~20 min)
scripts/gpu/explorer/submit.sh 08-nvidia-pair   # A1, A2, C: refused until
scripts/gpu/explorer/submit.sh 08-amd-pair      # SAME_PARTITION_PAIRS_OK=1
```

[`08_bench_matrix.sbatch`](../scripts/gpu/explorer/08_bench_matrix.sbatch) is one
heterogeneous job of two 1-GPU components (`MATRIX_PAIR`, set by `submit.sh`).
It records each node's network (`net_report.sh`: `/sys/class/infiniband`,
`ibstat`, `ibv_devinfo`, `ip -br addr`, into `network/<side>/`), probes both
GPUs, runs the preflight, then each run into `matrix/<run>/`, and ends with
`bench_matrix.py`, which writes `matrix.md`: per size range (under 256 KB,
256 KB to under 64 MB, 64 MB and up) every run's latency and bus bandwidth, its time
as a multiple of A2's and A1's, the bridge phase against the preflight's
prediction (a ring all_reduce of the tensor, both ways at once; with `bf16` or
`fp8`, each leader's compressed payload broadcast in turn, one way at a time), each
gpubridge run's phases, and what NCCL and RCCL logged about the network in
each run (`Using network IB` or `Socket`, `via NET/IB` or `NET/Socket`). Point
`bench_matrix.py` at several jobs' results to put the pairs side by side; with
no A1 and A2 the multiples stay empty. The job refuses to run if both
components land on one node. `net_check.sbatch` (`net-nvidia`, `net-amd`)
records one node's network the same way and has NCCL or RCCL initialize on one
rank, as chosen and with `NCCL_IB_DISABLE=1`, so its log names the network the
library would use between nodes. Last, `rdma_report.sh` records what RDMA from
GPU memory would need there, into `rdma/<side>/`: dma-buf and peer-to-peer
options in the kernel, `nvidia_peermem` or `nv_peer_mem`, rdma-core and
`ibv_reg_dmabuf_mr`, `rdma link`, the HSA runtime's dma-buf export and UCX. For
`ucx_info`, set `RDMA_MODULES` to a module that provides it (on Explorer,
`OpenMPI/4.1.6`); it is loaded inside that step only.

**Same-partition pairs wait for Research Computing.** A1, A2, B1, B2 and C need
two GPUs of one type, which `08-nvidia-pair` and `08-amd-pair` ask for as two
1-GPU components of one heterogeneous job in one partition
(`PARTITION_PAIR_NVIDIA`, `PARTITION_PAIR_AMD`). Whether that is within the
partitions' intended use is RC's call, so `submit.sh` refuses them (dry-run
only) until `SAME_PARTITION_PAIRS_OK=1` is in `explorer/local.env`.

**Training throughput** ([`train_bench.py`](../scripts/gpu/train_bench.py)):
GPT-2 small (124M parameters) and ResNet-50 (25.6M), built from `torch.nn`
alone, on synthetic data, trained with DDP and `gpubridge.ddp_comm_hook`. Each
step's time splits into compute (the same steps before DDP wraps the model),
waiting for the slower GPU and exposed communication; `comm_alone_ms` is the
gradient buckets all_reduced with no compute. Step 07 runs it on both sides
(`train_bench_<model>`, with the other training steps; `STEPS_07=training`
runs only the probe, the preflight and those, `STEPS_07=bench` only the probe,
the preflight and `train_bench_<model>`, and `TRAIN_BENCH_BUCKETS` repeats each
at several DDP bucket sizes), and `submit.sh train-nvidia` or
`train-amd` on one GPU alone, the baseline: adding a GPU through gpubridge
pays off only if the pair's `samples_per_s` beats the faster GPU's alone.
`TRAIN_BENCH_MODELS` and `TRAIN_BENCH_ARGS` choose the models and flags. Both
sides take the same batch per rank unless told otherwise: `--batch-sizes a,b`
sets one per rank, `--global-batch N` splits N samples evenly, and
`--global-batch N --balance` times each rank alone for a few steps first and
splits N in proportion to its speed, at least one sample each, so the faster
GPU stops waiting for the slower one. With either, each rank's loss is weighted
by its share of the samples, and the update is still the gradient of the mean
loss over all N (bit for bit on CPU with integer data,
[tests/test_train_bench.py](../tests/test_train_bench.py); CPU only so far).
One rank alone with `--global-batch N` takes all N, so the baseline for a
mixed run at global batch N is `TRAIN_BENCH_ARGS="--global-batch N"` on
`train-nvidia`. The JSON has `global_batch`, `batch_sizes`, `batch_split`, each
rank's `batch_size`, `compute_ms_per_sample` and `waiting_ms`, and with
`--balance` the calibration's per-sample times. Models with BatchNorm
(ResNet-50) normalize over each rank's own batch, so their result depends on
the split, as it does in any DDP run without SyncBatchNorm.
`--accumulate K` cuts each rank's batch (its samples per optimizer step) into K
micro-batches and syncs the gradients once per step, in the last backward
(`no_sync()` for the others): a global batch too big for a GPU's memory runs as
K smaller forward and backward passes with one exchange. It combines with the
flags above (with `--balance` the calibration runs at the micro-batch size and
each rank gets at least K samples), and the update is exact the same way (CPU
only so far), BatchNorm aside: it normalizes over each micro-batch. The JSON
adds `accumulate`, `micro_batch_sizes`, `samples_per_step` and the
`exchanges_per_step` the hook counted. On GPUs each rank's `memory` has, for the
compute, ddp and comm-alone phases, the caching allocator's peaks
(`max_allocated_mib`, `max_reserved_mib`) and the allocations it retried after
freeing its cache (`alloc_retries`, a sign it ran short), and the GPU's
`total_mib`.
With multigpu access, `submit.sh train-multi` runs each model on one GPU, on
`GPUS_TRAIN_MULTI` GPUs (4 by default) as one NCCL island, then on the same GPUs
split into two islands joined by the bridge (`train_bench_<model>_split`, a
split test): what several GPUs per island cost through the bridge, against the
same GPUs on NCCL alone. `submit.sh 07-multi` does the same across vendors
(several NVIDIA GPUs, one island, and the AMD side). With `STEPS_07=dp` or
`pipeline`, `07-multi` runs the data-parallel or pipeline training instead, a
rank per GPU; the pipeline puts one stage on each, so `PP_MICROBATCHES` must be
at least the GPUs in all.
Both models need BLAS on every GPU (the probe's `blas_available`):
with a ROCm build whose rocBLAS lacks the GPU's kernels, its rank aborts at the
first matmul, as the other training steps do; see [Choosing the ROCm build](#choosing-the-rocm-build).

### Repeating train_ddp

`train_ddp` (item 22) failed its parameter check, then the old float64
checksum, in some mixed V100 + MI50 runs and not in others. In 24 later runs
under the bit-for-bit check the parameters were identical every time and the
checksum still differed in 4, so those failures were most likely false alarms.
Repeating `train_ddp` in one job is how such a rate is measured.
`STEPS_07=ddp` makes step 07 run only the probe, the preflight and `train_ddp`, and
`TRAIN_DDP_REPEATS=N` runs `train_ddp` N times in the job, as steps `train_ddp_1` to
`train_ddp_N` (records `train_ddp_<i>.json`); one failing doesn't stop the rest.
`TRAIN_DDP_REPEATS` (1 to 999) works with `all` and `training` too; `bench`, `dp` and
`pipeline` run no `train_ddp` and say so. Bridge compression comes from
the environment at submit time: `submit.sh` submits with `--export=ALL`, so
`GPUBRIDGE_BRIDGE_COMPRESSION` and `GPUBRIDGE_COMPRESSION_KERNELS` reach every rank on
both sides, and each record's `environments` holds them. With `STEPS_07=ddp` nothing
else in the job uses them. Budget the job for a repeat that hangs: each launch is cut
off after `STEP_TIMEOUT` seconds (default 1200), so give the job time for every repeat
at that limit, or lower it. One `train_ddp` takes under a minute on a V100 + MI50, and a
stuck collective fails by itself after `KIT_TIMEOUT` (300), so `KIT_TIMEOUT=120
STEP_TIMEOUT=180` and 8 repeats fit in `TIME_07=00:30:00` even if every one hangs. A job
whose time runs out anyway lists the repeats it planned but never ran (its `job.jsonl`
records the plan). Each job into a folder of its own:

```bash
export VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=ddp TRAIN_DDP_REPEATS=8 TIME_07=00:30:00 \
    KIT_TIMEOUT=120 STEP_TIMEOUT=180
RESULTS_ROOT=$PWD/results/explorer/ddp-none scripts/gpu/explorer/submit.sh 07
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 GPUBRIDGE_COMPRESSION_KERNELS=torch \
    RESULTS_ROOT=$PWD/results/explorer/ddp-fp8 scripts/gpu/explorer/submit.sh 07
python scripts/gpu/train_ddp_report.py results/explorer/ddp-none results/explorer/ddp-fp8
```

[`train_ddp_report.py`](../scripts/gpu/train_ddp_report.py) prints a row per run:
the job, the repeat, compression and each rank's kernels, whether the parameters
diverged, the first step at which the synced gradients and the parameters differed,
whether the float64 checksums agreed, the largest difference and its tensor, and a
verdict (in sync; gradients differ after all_reduce; parameters differ, gradients
equal; gradients differed, parameters stayed in sync; old checksum only), then how many
diverged, one line per compression, kernels and kind of run, with records from before
the per-step check counted apart. A `train_ddp` that left no record, never ran, or
whose record can't be read is listed too.

### Pipeline training across vendors

`STEPS_07=pipeline` runs [`train_pipeline.py`](../scripts/gpu/train_pipeline.py):
the same model in two stages, the first on the NVIDIA GPU, the second on the AMD
one, with GPipe and 1F1B (`PP_RUNS`), against `submit.sh dp-baseline` with
`DP_MICRO` set to the microbatches. The design, the commands and what's validated
where are in [docs/PIPELINE.md](PIPELINE.md).

### Data-parallel training across vendors

[`train_dp.py`](../scripts/gpu/train_dp.py) trains [`llama.py`](../scripts/gpu/llama.py)'s
LLaMA-style decoder (RMSNorm, rotary embeddings, SwiGLU; 134M parameters as
`llama-125m`, up to 1.03B) on synthetic tokens with a learnable rule, the same
global batch at every step whatever the world size, and records every step's loss
over that whole batch. `STEPS_07=dp` makes step 07 run only the probe, the preflight
and one such run per name in `DP_RUNS`, into `07_mixed_hetjob_dp/`:

| Run | What |
| --- | --- |
| `ddp`, `ddp-bf16`, `ddp-fp8` | DDP with `gpubridge.ddp_comm_hook`, the bridge uncompressed, then compressed with error feedback |
| `fsdp2` | FSDP2 through `backend="gpubridge"`: parameters sharded over both GPUs |
| `ddp-speed` | each GPU's share of the batch in proportion to its measured speed, with sample-weighted averaging, so the update stays the whole batch's mean gradient |
| `ddp-shares-<rows>` | the rows of the global batch each rank takes, in rank order, joined by dashes: `ddp-shares-5-3` is `--shares 5,3` (whole rows adding up to `DP_GLOBAL_BATCH`), with the same sample-weighted averaging; `-r<N>` at the end (`ddp-shares-5-3-r2`) only gives a repeat its own name, so one job can sweep the split |
| `ddp-bucket-<MB>` | DDP's bucket size (25 MB in the others) |

`submit.sh dp-baseline` runs the same model, global batch, sequence length, steps and
seed on one V100 alone (`--mode single`), the curve the others are set against;
`dp-baseline-amd` does the same on the MI50. On the MI50, train with the
`venv-rocm63` venv (torch 2.9.1+rocm6.3, the build whose rocBLAS has gfx906
kernels); its Triton can't compile the compression kernels for gfx906, so the AMD
side uses gpubridge's PyTorch codec (`DP_KERNELS_ON_AMD=torch`, the default here; the
same bits, slower). The sharing partition allows an hour per job, so split the runs:

```bash
export VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=dp TIME_07=00:55:00
RESULTS_ROOT=$PWD/results/explorer/dp-a DP_RUNS="ddp ddp-bf16 ddp-fp8" \
    scripts/gpu/explorer/submit.sh 07
RESULTS_ROOT=$PWD/results/explorer/dp-b DP_RUNS="fsdp2 ddp-speed ddp-bucket-5 ddp-bucket-100" \
    scripts/gpu/explorer/submit.sh 07
RESULTS_ROOT=$PWD/results/explorer/dp-base scripts/gpu/explorer/submit.sh dp-baseline
python scripts/gpu/dp_report.py results/explorer/dp-a results/explorer/dp-b \
    results/explorer/dp-base --out results/explorer/dp-report
```

[`dp_report.py`](../scripts/gpu/dp_report.py) puts them side by side (`dp.md`): tokens/s,
DDP's exposed communication and the share of it backward hid (estimates, from the step less each GPU's compute alone), the last loss, the
largest gap from the baseline's curve and the mean gap over the last 100 steps, and
whether the ranks' parameters stayed identical (checked every 50 steps); it also
writes the curves as `dp_losses.csv` and `dp_losses.svg`. `DP_MODEL`, `DP_STEPS`,
`DP_GLOBAL_BATCH` and `DP_SEQ` change the run (llama-125m, 500 steps, 8 sequences of
512 tokens by default).

## Kaggle (free): two NVIDIA T4s

A free Kaggle notebook with **GPU T4 x2** and internet on runs the
single-machine NVIDIA part of the ladder:
- the GPU-only tests and the probe on each GPU;
- the split test on 2 GPUs, the all_reduce benchmark against native NCCL and
  flat-gloo, and the FSDP2 example with `--verify`;
- the compression kernels' bit-identical check.

T4s are compute capability 7.5, so the runner first checks CUDA events,
bfloat16 and Triton on them, and skips or adapts what they can't run, saying
why. Upload `scripts/gpu/kaggle/kaggle_gpubridge.ipynb` and follow the clicks
in [scripts/gpu/kaggle/README.md](../scripts/gpu/kaggle/README.md); the
results come back as one tarball, and `python scripts/gpu/triage.py <tarball>`
summarizes it. It is a split test, not a mixed-vendor run.

## AMD cloud machine

Pick any Linux machine or container with ROCm and AMD GPUs (e.g. MI300X). An
image that already has a ROCm build of PyTorch saves the 6+ GB download, with
`TORCH_FLAVOR=system`. Then:

```bash
git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
bash scripts/gpu/amd/run_all.sh --dry-run       # 1 second, no cost surprises
bash scripts/gpu/amd/run_all.sh                 # ~20-30 min plus training
```

The script runs:

1. setup
2. `rocm-smi` / `amd-smi` / `rocminfo` output
3. probe
4. device mapping under `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`, then
   one more process than GPUs, which must fail `init()` on every rank (item 12)
5. a 1-GPU check, then greps its NCCL log (`nccl/check_1gpu/`) for RCCL's
   version (item 4)
6. the GPU-only tests, `pytest -m gpu tests/test_gpu.py`

With 2+ GPUs it continues with:

7. an RCCL island
8. a split test
9. the native vs bridged benchmark
10. training scaling

With 1 GPU it runs a shared-GPU split test instead. Then, either way, training
throughput (`train_bench.py`, `TRAIN_BENCH_MODELS`, default GPT-2 small and
ResNet-50; `TRAIN_BENCH_ARGS` for extra flags): each model on one GPU alone and,
with 2+ GPUs, on all of them as one RCCL island and split into two islands
through the bridge.

A failing step is logged and skipped, so one problem doesn't waste the
session. At the end the script prints the exact `scp` command for the results
tarball. **Download it before stopping the machine.** To rerun after fixing
something, `SKIP_SETUP=1` skips the install.

### An MI300X on AMD Developer Cloud

AMD Developer Cloud runs on DigitalOcean's GPU Droplets, whose MI300X
Droplets have 1 or 8 GPUs. gfx942 is in the default ROCm wheel's build, with
rocBLAS and hipBLASLt files for it
([BENCHMARKS.md](BENCHMARKS.md#which-rocm-wheels-can-train-on-an-mi50-gfx906)),
so no second venv is needed. Nothing here has run on one yet.

1. Create an MI300X GPU Droplet from an image with ROCm (one with a ROCm
   PyTorch lets you set `TORCH_FLAVOR=system` and skip a 6+ GB download), and
   add your SSH key.
2. `ssh` in, then:

   ```bash
   git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
   bash scripts/gpu/amd/run_all.sh --dry-run    # checks the plan, costs nothing
   bash scripts/gpu/amd/run_all.sh              # 8 GPUs: about 20-30 min plus training
   ```

   With one GPU it runs the single-GPU steps and the shared-GPU split test.
3. Copy the results tarball off with the `scp` command it prints, then run
   `python scripts/gpu/triage.py <tarball>` locally.
4. **Destroy the Droplet**: DigitalOcean bills a powered-off Droplet until it
   is destroyed.

## Real mixed-vendor job (two cloud machines)

Explorer's compute nodes reach the internet only through an HTTP proxy, so
they can't join a job with a machine outside the cluster. Use two cloud
machines: one NVIDIA, one AMD. If step 00 finds AMD GPUs on Explorer, the
[heterogeneous job](#mixed-vendor-job-inside-explorer) does the same for free.

### Network: what gpubridge needs

Gloo, which carries discovery, the world group and the bridge, connects every
rank to every other rank on **random TCP ports, in both directions**. So the
two machines need a flat network with all TCP open between them, not just the
rendezvous port. Opening every port on public IPs isn't safe; use an overlay.

**Tailscale (default).** On both machines:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # prints a login URL the first time
tailscale ip -4              # this machine's tailnet address, e.g. 100.64.0.1
ip -4 addr show tailscale0   # the interface the scripts use (IFNAME=tailscale0)
```

- Tailscale needs a TUN device. That works on VMs, and in containers started
  with `/dev/net/tun` and `NET_ADMIN`. Check with `ls -l /dev/net/tun`.
- In an unprivileged container, which is common for GPU pods, only userspace
  mode is available. That mode is a proxy, which Gloo can't use. In that case
  pick a VM, or use the provider's private network (below).
- The default tailnet policy allows all traffic between your machines. If you
  use ACLs, allow all TCP between the two.

**WireGuard (alternative).** Also needs root and a TUN-capable kernel or
container. Generate keys on each machine
(`wg genkey | tee privatekey | wg pubkey > publickey`), then write
`/etc/wireguard/wg0.conf`:

```ini
[Interface]
PrivateKey = <this machine's private key>
Address = 10.99.0.1/24            # 10.99.0.2/24 on the other machine
ListenPort = 51820

[Peer]
PublicKey = <the other machine's public key>
AllowedIPs = 10.99.0.2/32         # 10.99.0.1/32 on the other machine
Endpoint = <other public IP>:51820  # needed on at least one side
PersistentKeepalive = 25
```

Start it with `sudo wg-quick up wg0`. UDP 51820 must be reachable on at least
one side. Use `IFNAME=wg0`, and the `10.99.0.x` addresses for `MASTER_ADDR`
and `LOCAL_ADDR`.

**Same provider, private network.** If both machines are in one provider
project or region with private networking and open internal firewall rules,
use those private IPs and that interface directly.

### Launch

Use the same PyTorch release on both machines; `setup_env.sh` installs 2.14.1
by default. Start the master first, then the worker within about 90 seconds
(the preflight timeout):

```bash
# NVIDIA machine (master):
ROLE=master MASTER_ADDR=100.64.0.1 bash scripts/gpu/mixed/node.sh
# AMD machine (worker):
ROLE=worker MASTER_ADDR=100.64.0.1 bash scripts/gpu/mixed/node.sh
```

`node.sh` sets up the venv. Then it runs `preflight.py` on the overlay
interface (`IFNAME`), which checks:

1. the address resolves
2. the interface exists
3. TCP reaches the master
4. TCP connects both ways between the two overlay interfaces, and the raw
   bandwidth over them
5. a real two-machine Gloo handshake succeeds

The preflight **stops with a specific message before torchrun starts** if any
check fails. After that, the script runs the correctness check, the bridged
benchmarks (uncompressed, then bf16 and fp8), FSDP2 through the backend, and
training throughput (`train_bench.py` on GPT-2 small and ResNet-50, DDP with
`gpubridge.ddp_comm_hook`): first each machine's GPUs alone, one island and
nothing on the bridge, then both machines together. Last it writes
`triage.txt` (this machine's half) and packs the results.

Ports: torchrun uses `MASTER_PORT` (29500) and the preflight uses
`PREFLIGHT_PORT` (29501), plus Gloo's random ports. All are covered by an
overlay network.

Each machine only holds its own ranks' records. For the full verdict, copy
both `results/mixed-*/check/rank*.json` sets into one directory and run
`python scripts/gpu/summarize.py <dir>`. That run's `run.kind` must be
`gpu-mixed`, with `split_test` null.

### Dry run first: two containers

[`mixed/containers.sh`](../scripts/gpu/mixed/containers.sh) runs `node.sh` end to
end on CPU in two Linux containers on one Docker network, a stand-in for the
two machines on an overlay: the master pretends to be the NVIDIA machine and the
worker the AMD one (`GPUBRIDGE_VENDOR`), two ranks each by default (`NPROC`).
Everything after setup runs as it would on GPUs, over a real network between
two hosts: the preflight on `eth0`, the check, the benchmarks, FSDP2, training
throughput on the tiny models, triage and both tarballs. It can't show GPUs,
NCCL/RCCL, the overlay itself or a wide-area link.

```bash
scripts/gpu/mixed/containers.sh            # results in results/mixed_containers/
NPROC=4 CPUS=4 scripts/gpu/mixed/containers.sh   # four ranks a machine
```

It builds a small image (`mixed/Dockerfile`: torch's CPU build and gpubridge)
and needs only Docker. With Colima or Docker Desktop, keep the repo under your
home directory, which the VM shares. Run it after any change to `node.sh` and
before renting machines.

### An 8+8 run, step by step

One machine with 8 NVIDIA GPUs and one with 8 AMD GPUs. Nothing here has run on
8-GPU machines yet, so the times below are the limits the kit enforces, not
measurements.

1. **Pick the pair.** Prefer one provider and region with a private network
   between the two machines (no overlay needed, and the most bandwidth); else
   any two machines and Tailscale. Both on Linux with the GPU driver installed
   and a ROCm release that supports the AMD GPU.
2. **Network.** Same provider: note each machine's private IP and interface
   (`ip -br addr`), and open all TCP between the two in the provider's
   firewall. Otherwise, Tailscale on both, as [above](#network-what-gpubridge-needs).
   Check both ways: `ping` the other machine's address from each.
3. **Code, on both, at one commit:**

   ```bash
   git clone https://github.com/DavidCzartoryski/gpubridge.git && cd gpubridge
   git checkout <commit>       # the same on both machines
   ```

4. **Look at the commands first**, on both (`--dry-run` only prints them):

   ```bash
   ROLE=master MASTER_ADDR=<master IP> IFNAME=<interface> NPROC=8 \
       bash scripts/gpu/mixed/node.sh --dry-run
   ```

5. **Run.** Master first, the worker within about 90 seconds (the preflight's
   timeout). `node.sh` sets up the venv first (PyTorch 2.14.1 for the
   machine's GPUs: about 7 GB for the CUDA build, more for ROCm's). On a
   private network of 25 Gb/s or more, `BENCH_MAX_BYTES=1G` also measures the
   large sizes:

   ```bash
   # NVIDIA machine:
   ROLE=master MASTER_ADDR=<master IP> IFNAME=<interface> NPROC=8 BENCH_MAX_BYTES=1G \
       bash scripts/gpu/mixed/node.sh
   # AMD machine:
   ROLE=worker MASTER_ADDR=<master IP> IFNAME=<interface> NPROC=8 BENCH_MAX_BYTES=1G \
       bash scripts/gpu/mixed/node.sh
   ```

   Each machine is one island of 8 (NCCL on one, RCCL on the other) and the
   bridge joins the two leaders. Every launch is stopped after `STEP_TIMEOUT`
   seconds (1200 by default) and recorded as timed out, so a hung step costs
   at most 20 minutes; 17 steps follow the preflight.

   The data-parallel and pipeline runs (`DP_RUNS`, `PP_RUNS`) default to step
   07's settings, sized for 1 + 1 GPUs: `llama-125m`, 500 steps, a global batch
   of 8. For 8 + 8, give each GPU a few sequences and the pipeline a stage per
   GPU: `DP_GLOBAL_BATCH=64`, and `DP_MODEL=llama-1b PP_MICROBATCHES=16` (20
   blocks for 16 stages), or `PP_RUNS=""` to leave the pipeline out. To compare
   loss curves, `DP_STEPS=1000`
   ([COMPRESSION.md](COMPRESSION.md#training-with-compression)), with
   `STEP_TIMEOUT` raised to fit.
6. **Collect.** Each machine prints the `scp` line for its tarball. Copy both,
   then:

   ```bash
   python scripts/gpu/triage.py mixed-master-<host>.tgz
   python scripts/gpu/triage.py mixed-worker-<host>.tgz
   ```

   The two-machine training records (`train_bench_<model>.json`) are in the
   tarball of the machine that got rank 0, which writes them, as it writes the
   benchmarks' and FSDP2's. torchrun numbers the machines by their addresses
   (`LOCAL_ADDR`), not by role, so it can be either. Each tarball holds its own
   machine's baseline (`train_bench_alone_<model>.json`). The same holds for
   the data-parallel and pipeline records (`dp_<run>.json`, `pp_<schedule>.json`)
   and their one-GPU baseline (`dp_single.json`, on each machine);
   `scripts/gpu/dp_report.py` sets them side by side. For the check's verdict,
   see above.
7. **Stop billing.** Destroy both machines once the tarballs are copied; on
   some providers a powered-off GPU machine still bills.

What it answers: whether gpubridge's bridge, compression and training paths
work on 8+8 GPUs of current generations; what the bridge costs at each size on
that network (the benchmark against the preflight's prediction); and whether
16 GPUs across vendors train faster than each machine's 8 alone, for GPT-2
small and ResNet-50 (`samples_per_s`, with the step split into compute, waiting
for the slower machine and exposed communication); and for `llama.py`'s model,
data-parallel training uncompressed and with bf16 and fp8 on the bridge, and
pipeline training, against one GPU alone.

## Reading results

- `results/<target>/<step>/rank<N>.json`: one record per rank. It holds:
  - `environment`: host, GPUs with PCI bus ids, driver-visible devices, the
    GPU driver, torch, CUDA or ROCm (and HIP), NCCL or RCCL and its version,
    the gpubridge commit (`git_dirty` if tracked files had changes), the Gloo
    and NCCL interfaces (`auto` when unset), allowlisted env vars,
    `gpubridge.probe()`;
  - `run`: kind, split mode, islands, device;
  - `stages`: pass/fail per stage, with full tracebacks on failure;
  - `rank<N>.stacks.txt`: a stack dump if the rank hung in a stage.
- `summary.md` / `summary.json`: the per-stage table, missing ranks, and GPUs
  shared outside a split test.
- `steps.jsonl` and `<step>.log`: exit code (and `timed_out`) and full output
  of every scripted step. A failed step's log ends with the last lines of its
  NCCL/RCCL logs.
- `job.jsonl` (Explorer jobs): when each job started, the error that stopped it,
  if any, and when it finished.
- `nccl/<step>/<host>.<pid>.log`: NCCL/RCCL's `NCCL_DEBUG=INFO` output, one file
  per rank.
- `preflight/preflight-<role>.json` (steps 05 and 07): every pair of interfaces
  between the two leader nodes, whether it connects each way, MB/s each way and
  the round-trip time, and the interface each node's Gloo used.
- `bench/bench.csv`, `bench.json` and `summary.md`: per size, the native
  NCCL/RCCL baseline (over every rank, or each island's), gpubridge and
  flat-gloo side by side: median, p90 and min latency of the slowest rank,
  timed with CUDA events on the current stream (`host_median_us` is the host
  timer), and algbw and busbw as nccl-tests computes them.
- `gpu_tests.xml` and `gpu_tests.log`: the GPU-only tests (JUnit XML and
  pytest's output). A failure names the policy, op, dtype, size and mode of
  each wrong case.
- `train-n<N>.json`, `train_ddp*.json`: the training run's throughput, losses
  and settings, with every rank's `environment`.

Every JSON record a GPU script writes carries the same `environment` and its
settings: `check.py`, `probe.py` and `device_map.py` records, `bench.json`,
`preflight-<role>.json` and the training records.

What a record may hold from the environment is an allowlist (`ENV_NAMES` and
`ENV_PREFIXES` in `gpukit.py`): `GPUBRIDGE_*`, `NCCL_*`, `RCCL_*`, `GLOO_*`,
`TORCH_*`, the visible-devices and rank variables, and a short list of
`SLURM_*` ones (job id, partition, nodes, task ids, GPUs). A name with
`TOKEN`, `KEY`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `AUTH`, `USER`,
`HOME` or `ACCOUNT` in it is never recorded, whatever its prefix. Every string
a kit record holds, tracebacks and command lines included, is redacted before
it is written: `/projects/<name>` and `PROJECT_DIR` become `$PROJECT_DIR`, the
home directory `~`, the user name as a path segment `$USER`, and credentials in
a URL `***`. `job.jsonl`'s error messages get the same path replacements.
The step logs (`<step>.log`), stack dumps (`rank*.stacks.txt`)
and NCCL/RCCL logs are not redacted: keep them, and the results tarballs, out
of anything public.

### A summary to paste: triage.py

`scripts/gpu/triage.py` turns a results directory or tarball, Explorer's or
Kaggle's, into a short plain-text summary, under 80 lines, to paste into a
review:

    python scripts/gpu/triage.py results/explorer
    python scripts/gpu/triage.py kaggle-t4x2-20261014-0930.tgz

- **Problems first:** an Explorer job that started but never finished (in
  `job.jsonl`), with the error that stopped it if it recorded one; each step
  that failed, timed out or never finished (a log with no `steps.jsonl` line:
  the job was killed or ran out of Slurm time), with the last error lines of
  its log. Then the skipped steps, with their reasons. A step run again counts
  as its last attempt.
- **For each job:** what it ran on (hosts numbered, GPU models, driver, torch,
  CUDA or ROCm, NCCL or RCCL, the gpubridge commit).
- **Key numbers:** gpubridge's bus bandwidth against the baseline's; the
  bridge phase against the preflight's prediction; and the compression
  kernels' check against the CPU reference, with the kernels that ran.
- **Last:** whether the compression digests agree across every compression
  record, and one verdict line.

It needs only Python 3.10 or later, no torch. It exits 0 if everything that
ran passed, 1 if something failed, timed out or never finished (or nothing was
found), and 2 if it can't read the input. Host names become `host N`, IPv4
addresses `<ip>`, and paths lose their owner (`$PROJECT_DIR`, `~`). The error
lines still come from the step logs, so read the output before pasting it
anywhere public.

### The validation tables

`scripts/gpu/validation_tables.py` turns results into the tables
HARDWARE_VALIDATION.md needs. Copy every job's results into one place (for
the mixed job, both machines' `results/mixed-*` side by side) and run:

    python scripts/gpu/validation_tables.py results/explorer results/amd/<target> results/mixed

Each directory with a `steps.jsonl` is one job. It writes
`validation_tables.md` and `validation_tables.json` into the first path (or
`--out`), with these sections:

- **Runs**: per job, steps passed and failed, the deliberate failing runs'
  results, and what it ran on: hosts, GPU models, driver, torch, CUDA or
  ROCm, NCCL or RCCL, gpubridge commit, Gloo interface.
- **Items**: each HARDWARE_VALIDATION.md item, the steps and `check.py`
  stages that test it, and a verdict: `not run`, `simulation only`,
  `pass (n)` or `FAIL (k of n)`. A stage counts only in runs of the kind its
  item needs (a split test never closes item 1).
- **Network**: the preflight's chosen interfaces, MB/s each way, round trip.
- **Benchmarks**: gpubridge and flat-gloo against the row's named baseline at
  1 MiB, 64 MiB and the largest size.
- **Predicted vs measured**: below.

Runs in CPU simulation (`GPUBRIDGE_VENDOR` or `GPUBRIDGE_CPU_ONLY` set, or a
CPU device) are marked `(simulation)` everywhere and never count towards an
item.

The tables are safe to commit publicly: hosts appear as `host 1`, `host 2`,
... in order of first appearance (the same host has the same number in every
section), and no path is shown. The raw results keep the real hostnames;
publish the tables, not the results.

`run.kind` values:

| `run.kind` | Meaning |
| --- | --- |
| `gpu-mixed` | real GPUs, both vendors: the result that closes items 1 and 2 |
| `gpu-single-vendor` | real GPUs, one vendor, no bridge |
| `split-test` | islands faked with `GPUBRIDGE_SPLIT_TEST`: never a mixed-vendor result |
| `cpu` | simulation or CPU-only mode |

## Predicted vs measured

Predict the bridge's time from the preflight before reading the benchmark. In
reduce-bridge-broadcast, the island leaders run one Gloo all_reduce among
themselves. A ring all_reduce over p leaders sends 2(p-1)/p of the tensor each
way: the whole tensor for two islands. With B the preflight's slower direction
in MB/s (1 MB = 10^6 bytes) and RTT its round trip in ms:

    predicted bridge ms = (p-1) * RTT + 2(p-1)/p * bytes / (B * 10^6) * 1000

The latency part is one RTT for two islands. With p leaders in a ring, the
all_reduce takes 2(p-1) steps, each about half a round trip, so it is roughly
(p-1) x RTT.

For example, two islands, 64 MiB, a link the preflight measured at 1000 MB/s
and 0.3 ms: 0.3 + 67.1, about 67 ms.

The benchmark's `--phases` runs (step 05's `bench` and `bench_phases_*`, step
07, `mixed/node.sh`) time the `bridge` phase on the slowest leader, and
`validation_tables.py` puts it next to the prediction as "measured /
predicted". Split tests on one node have no preflight, so no prediction. How
to read the ratio:

- **About 1**: the bridge runs at the speed the preflight measured; the
  network is the limit.
- **Well above 1 at large sizes**: something besides the wire costs time. The
  `bridge` phase also includes the copies from the GPU to the host and back,
  so compare it with a host copy of the same size, then look at CPU load on
  the leaders.
- **Well above 1 at small sizes only**: per-call overhead, not bandwidth; the
  round trip dominates the prediction there.
- **Below 1**: the preflight understated the link. It measures one TCP stream
  driven from Python, which may not fill 100 Gb/s Ethernet or IPoIB. For the
  UCX bridge over RDMA (`bench_ucx`) that is the expected outcome: the
  prediction is the TCP baseline it should beat. Policies
  that split or overlap the transfer (`sharded-bridge`,
  `pipelined-reduce-bridge-broadcast`) can also beat it, so for them the
  prediction is closer to an upper bound.

## If a step fails

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| "set PROJECT_DIR" | no project space configured | put `PROJECT_DIR=/projects/<your-project>` in `scripts/gpu/explorer/local.env`, or export it |
| setup-rocm: the probe at the end fails | the ROCm build has no kernels for this AMD GPU | pick another build: [Choosing the ROCm build](#choosing-the-rocm-build) |
| setup: downloads time out or can't connect | the job has no route out | check that the log shows `module load explorer` ran; `SETUP_MODULES` must include `explorer` |
| setup: "this looks like a Slurm login node" | `setup_env.sh` was run outside a job | use `submit.sh setup`, or run it inside `srun --pty bash` |
| 00: no AMD GPU type found in `sharing` | no AMD nodes there, or an unrecognized type name | read `gres-sharing.txt`; if the AMD GPUs are there under another name, set `GPU_TYPE_AMD` to it; otherwise use step 8 |
| 07: "not a heterogeneous job" | the script was submitted with plain `sbatch` | submit with `submit.sh 07` |
| 05 or 07: stops after `preflight` | no pair of interfaces connects the nodes both ways, or (07) a GPU the build can't run | read `pairs` in `preflight/preflight-*.json` (which connection failed, from which address) and `probe_*/probe.json`; a pair that connects one way only means a firewall between the partitions |
| 05 or 07: the preflight chose a slow interface | the fast one didn't connect both ways, or isn't routed between the nodes | compare the pairs' MB/s in `preflight-*.json`; set `GLOO_SOCKET_IFNAME` to force one |
| 01: `build_supports_gpus` fails | wheel without kernels for this GPU | the job log's `FAIL target_in_build` line names the GPU's target and the build's arch list; set `CUDA_INDEX` (cu126 for V100; cu128/cu130 for newer GPUs), rerun setup. A GPU counts as covered by its own `sm_XY`, a cubin for an earlier minor of its major, or PTX at or below its capability; `target_match` in `probe.json` says which |
| 01: no GPU visible | job didn't get a GPU, or driver problem | check `nvidia-smi.txt` and `slurm-job.txt` in the results |
| 01/AMD: `local_rank_check` fails | init() succeeded with more processes than GPUs, or failed for another reason | read `expected_init_error` in `local_rank_check/rank*.json`; if init succeeded, the visible-device count isn't what PyTorch reports (item 12) |
| 02: `shared_gpu_check` fails, `"result": "check did not fire"` | init() succeeded with two ranks of one island on the GPU, failed for another reason, or hung past `SHARED_GPU_CHECK_SECONDS` | read `shared_gpu_check.log` and `expected_init_error` in `shared_gpu_check/rank*.json`; if the log has init()'s "could not check" warning, PyTorch reported neither a UUID nor a PCI address for the GPU, so the island went unchecked (item 24) |
| `init` stuck (summary says "stuck in this stage") | new_group / NCCL init problem (item 3) | read `rank*.stacks.txt` and the NCCL/RCCL log lines at the end of `<step>.log` (all of them in `nccl/<step>/`); this is a real finding, file it |
| `steps.jsonl` says `"timed_out": true` | the launch ran past `STEP_TIMEOUT` | read `<step>.log` and `rank*.stacks.txt`; raise `STEP_TIMEOUT` only if the step was still making progress, and on Explorer the job's `TIME_0N` with it: `submit.sh` caps `STEP_TIMEOUT` at `TIME_0N` less `STEP_MARGIN` |
| `busy_gpu` wrong values | stream ordering bug in the bridge (item 6) | serious; keep the JSON and open an issue |
| `dtypes` mismatch | precision or reduction-order problem (item 7) | check `max_abs_error` in the stage |
| `reductions`, `all_gather` or `reduce_scatter` mismatch | a MAX/MIN/AVG or tensor-collective path differs on NCCL/RCCL (item 15) | the failing `cases` name the dtype and op; rerun with `--policy flat-gloo` to see whether the policy or the backend is at fault |
| `async` wrong values | the worker's stream didn't wait for the caller's, or `wait()` didn't order the caller's stream (item 16) | serious; keep the JSON and open an issue |
| `reduction_agreement` shows `product_agree` false | NCCL/RCCL and Gloo disagree on PRODUCT for that dtype (item 14) | not a failure: it's the evidence item 14 asks for; record it |
| timeouts during large all_reduces | NCCL/RCCL watchdog vs a slow bridge (item 9) | raise `KIT_TIMEOUT` (or `--timeout`); note the `all_reduce_seconds` of `busy_gpu` |
| 05: rendezvous never completes after the preflight passed | NCCL picked an interface the other node can't reach | set `NCCL_SOCKET_IFNAME` to the preflight's `chosen_ifname` (e.g. `ib0`) |
| 03/AMD: `device_map` mismatch | visible-devices variable maps to a different GPU (item 5) | record it; check scheduler GPU binding |
| AMD: `rccl_version` fails | RCCL didn't log, or this isn't a ROCm build | check `nccl/check_1gpu/` and `probe/probe.json` (`torch_hip`) |
| mixed: preflight `resolve` / `tcp` fails | wrong address, or overlay down | `tailscale status`; use `tailscale ip -4` of the master |
| mixed: preflight `gloo` fails, `tcp` passes | rendezvous port open, random ports blocked | use the overlay addresses and interface, not public IPs |
| NCCL "unhandled system error" in a container | too little shared memory | start the container with a larger `--shm-size` |
