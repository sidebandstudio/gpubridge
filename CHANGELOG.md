# Changelog

All notable changes to gpubridge. Versions follow the release tags; every
release so far is an alpha. Numbers in parentheses are pull requests.

**GPU runs so far**, on Explorer:

- **NVIDIA V100 (sm_70), commit af51b20**, torch 2.14.1+cu126, NCCL 2.29.3: the
  bridge on device tensors and the shared-GPU check passed; two fp8
  error-feedback residual tests failed, from FMA contraction plus a codec
  overflow, fixed in #50. Step 01's probe failed: it took the build target
  from `gcnArchName`, which holds the product name on CUDA builds, fixed in
  #49.
- **NVIDIA V100 (sm_70), commit 14ed60c**, step 01 rerun after #49: driver
  545.23.08, torch 2.14.1+cu126, NCCL 2.29.3. The probe passed (an exact
  target match), and so did the job's other 2 steps.
- **NVIDIA H200 (sm_90), commit 4a478f4** (#50's branch before it merged):
  driver 570.86.15, torch 2.14.1+cu126, CUDA 12.6, NCCL 2.29.3. Step 02
  passed; `gpu_tests` 26 passed, 8 skipped, in 162 s.
- **NVIDIA V100 (SXM2), commit df80abb**, step 02 rerun after #50: passed;
  `gpu_tests` 26 passed, 8 skipped, in 782 s.
- **NVIDIA A100-SXM4-80GB (sm_80), commit df80abb** with one local change
  (`AMD_NODES_CONFIRMED=1` in `submit.sh`, no library code; the records say
  df80abb+dirty): driver 570.86.15, torch 2.14.1+cu126, NCCL 2.29.3. Step 02
  passed; `gpu_tests` 26 passed, 8 skipped, in 437 s.
- **NVIDIA Tesla T4 (sm_75), commit df80abb** with the same local line:
  driver 570.86.15. Step 02 passed; `gpu_tests` 26 passed, 8 skipped, in
  205 s.
- **NVIDIA L40S (sm_89), commit df80abb** with the same local line, `sharing`
  partition: driver 545.23.08. Step 02 passed; `gpu_tests` 26 passed, 8
  skipped, in 186 s. The probe found the build covers it with sm_86 cubins
  (`target_match` "compatible").
- **AMD MI50 (gfx906), commit df80abb** (from the clone's history; the
  run's records were overwritten), `sharing` partition, torch
  2.14.1+rocm7.2, HIP 7.2.53211, RCCL 2.27.7: `setup-rocm` succeeded, and the
  probe found gfx906 in the build's arch list (an exact match). The
  single-node AMD step: `gpu_tests` 24 passed, 2 failed, 8 skipped;
  `check_1gpu` and `check_split_shared_gpu` aborted (exit -6); the
  compression kernels matched the CPU reference in 3 of 3 cases. Every
  failure was the wheel's rocBLAS, which has no kernels for gfx906, in a
  matmul used as busy work, which #52 replaces. `diag_fp8_residual.py` found
  every fp8 payload and residual matching the CPU reference, for float32 and
  float64, with fusion on and off.
- **Compression kernels at df80abb** (`bench_compression.py`; the records
  say df80abb+dirty) on a V100, an MI50 and an H200: the kernels matched the
  CPU reference in 3 of 3 cases on each. Their speed is in
  [docs/COMPRESSION.md](docs/COMPRESSION.md). The float32 outputs differ
  between vendors only in NaN bits, where the input is NaN: `0x7fffffff` on
  the NVIDIA GPUs, `0x7fc00000` on the MI50, as in the CPU reference. The
  check compares where the NaNs are, not their bits.
- **AMD MI50 (gfx906), commit 4014234** (after #52), `sharing` partition:
  driver 5.14.0-611.5.1.el9_7 (as `rocm-smi` reports it), torch
  2.14.1+rocm7.2, HIP 7.2.53211, RCCL 2.27.7. The single-node AMD step passed
  all 9 steps, in each of two runs: `gpu_tests` 26 passed, 8 skipped (in
  170 s and 178 s); `check_1gpu` and `check_split_shared_gpu` passed; the
  compression kernels matched the CPU reference in 3 of 3 cases. The wheel's
  rocBLAS still has no gfx906 kernels (the probe's `blas` stage says so);
  nothing in the checks needs it now.
- **Stream mutations at 4014234** (`stream_mutations.py`, #52), on a V100
  (sm_70; torch 2.14.1+cu126, NCCL 2.29.3) and on the MI50 (gfx906; torch
  2.14.1+rocm7.2, RCCL 2.27.7), with the same results on both:
  `worker-ignores-caller`, `staging-ignores-caller` and
  `slots-forget-copies-back` were caught (the guarding test failed on wrong
  values), and both guarding tests passed without a mutation.
  `wait-skips-stream` was **not caught**: its test passed while `Work.wait()`
  ordered no stream, so no GPU test yet shows that this ordering is needed.
  The script doesn't record the driver.
- **Triton's kernel cache at 4014234** (`time_triton_cache.py`, #52), on a
  V100, torch 2.14.1+cu126, Triton 3.8.0, the median of 12 kernel variants:
  a build took 4.14 s with the cache in the home directory (NFS), 0.32 s in
  project space (NFS) and 0.13 s in node-local `/tmp` (XFS); a load from the
  cache took 0.04 s, 0.009 s and 0.001 s. The home directory was timed first,
  and each location's first build was its slowest (9.6 s, 2.7 s and 2.3 s).
- **The first mixed-vendor job: an NVIDIA V100 (sm_70) and an AMD MI50
  (gfx906), commit 4014234**, step 07: one heterogeneous Slurm job across the
  `gpu` and `sharing` partitions, one GPU and one rank each. V100: driver
  545.23.08, torch 2.14.1+cu126, NCCL 2.29.3. MI50: driver
  5.14.0-611.5.1.el9_7, torch 2.14.1+rocm7.2, RCCL 2.27.7. The Gloo bridge
  ran over TCP between the two nodes; in both runs the preflight measured
  about 1.15 GB/s each way and a 0.05 to 0.07 ms round trip. The second run
  (`TIME_07=01:00:00`) passed 15 of 20 steps:
  - the bridge check on device tensors passed every stage on both ranks,
    with the default policy, with pinned staging, and with the pipelined and
    sharded policies;
  - a 1 GB all_reduce took 1.69 s through the bridge (0.64 GB/s bus
    bandwidth) and 1.28 s with pinned staging (0.84 GB/s), against 1.73 s
    for one flat Gloo group of both ranks;
  - bridge compression: `bf16` was faster than uncompressed from 1 MB up and
    `fp8` from 256 KB up, with every rank holding the same bits; the kernels
    matched the CPU reference on both GPUs;
  - the 5 training steps failed: DDP with `gpubridge.ddp_comm_hook`, FSDP2
    and HSDP through the backend, and the bit-for-bit checks of the last
    two. In each, the MI50's rank aborted (exit -6) in rocBLAS, which has no
    gfx906 kernels in this wheel, and the V100's rank then lost its
    connection.
  - The first run, at the same commit with a 30-minute limit, passed 10
    steps. In 4, Slurm couldn't create the job step (`srun: error: Unable to
    create step ... Unexpected message received`) while it still listed the
    previous step as running, and `summarize_pinned` then had no records to
    summarize; all 5 passed in the second run. `train_ddp` failed
    differently: the MI50's rank timed out (Gloo, 300 s) in DDP's parameter
    check while the V100's printed nothing, cause unknown. The job reached
    its limit during `train_fsdp2_backend`.
- **Network checks at 934f87c** (`submit.sh net-nvidia`, `net-amd`), one
  node of each type. The V100 node (`gpu`) has one RDMA port, `mlx5_0`:
  InfiniBand, active, 100 Gb/s; NCCL 2.29.3 initialized alone chose NET/IB on
  it, and with `NCCL_IB_DISABLE=1` NET/Socket on `internal` and
  `internal.2137`. The MI50 node (`sharing`) has `mlx5_0` (Ethernet, so RoCE;
  active, 10 Gb/s), `mlx5_1` (Ethernet, down) and `mlx5_2` (InfiniBand, down);
  RCCL 2.27.7 chose NET/IB over RoCE on `mlx5_0`, and with
  `NCCL_IB_DISABLE=1` NET/Socket on `ibs8`, an interface `ip` reports as down.
  The bridge's preflight uses TCP on `internal.2137` on both. Details in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#network).
- **ROCm wheel scan** (a CPU job; no GPU): of every torch ROCm wheel from 2.8
  on for Python 3.12, only the `rocm6.3` builds of 2.8.0, 2.9.0 and 2.9.1 ship
  rocBLAS kernels for gfx906 (155 Tensile files each); the table is in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#which-rocm-wheels-can-train-on-an-mi50-gfx906).
- **AMD MI50 (gfx906) with torch 2.9.1+rocm6.3**, in a second venv
  (`venv-rocm63`; the default ROCm venv unchanged), HIP 6.3.42134, RCCL 2.21.5,
  driver 5.14.0-611.5.1.el9_7: `setup-rocm`'s probe found BLAS working. On one
  MI50, float32 and float16 matmuls (float32 within 1.6e-6 of a float64 CPU
  result), a conv2d, causal attention and a linear layer's backward ran. The
  single-node AMD step at dbed5c9 passed 7 of 9 steps; `gpu_tests` (20
  failed, 6 passed, 8 skipped) and `bench_compression_kernels` failed, every
  failure the wheel's Triton (pytorch-triton-rocm 3.5.1) failing to compile
  the compression kernels for gfx906 (`PassManager::run failed`).
- **The comparison matrix, an NVIDIA V100 (sm_70) and an AMD MI50 (gfx906),
  commit 934f87c**, step 08: one heterogeneous job, one GPU and one rank
  each, the V100 as in step 07 (driver 545.23.08, torch 2.14.1+cu126, NCCL
  2.29.3) and the MI50 on the default venv (torch 2.14.1+rocm7.2, RCCL
  2.27.7). All 15 steps passed. The preflight measured 1173.7 and 1172.1 MB/s
  over `internal.2137` and a 0.048 ms round trip. A float32 all_reduce of 1 GB
  took 1.67 s through the default bridge (1.60 s in a repeat run), 1.64 s in
  one flat Gloo group, 1.21 s with pinned staging, 1.07 s pipelined in 16 MB
  chunks, 1.58 s sharded (one link), 1.23 s with bf16 compression and 609 ms
  with fp8. The native baselines (A1, A2, B1, B2) and the same-vendor bridge
  (C) wait on Research Computing. Tables in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Training throughput on one V100 alone, commit 934f87c**
  (`submit.sh train-nvidia`; driver 545.23.08, torch 2.14.1+cu126, NCCL
  2.29.3): `train_bench.py`, float32, DDP with `gpubridge.ddp_comm_hook` on one
  rank: GPT-2 small at batch 4 x 512 tokens, 25.3 samples/s (158 ms a step);
  ResNet-50 at batch 32, 315.8 samples/s (101 ms).
- **Training throughput on one MI50 alone, commit 934f87c**, with torch
  2.9.1+rocm6.3 (`submit.sh train-amd`, `VENV_ROCM` at `venv-rocm63`; HIP
  6.3.42134, RCCL 2.21.5): GPT-2 small 16.4 samples/s (244 ms a step);
  ResNet-50 247.3 samples/s (129 ms).
- **Training on an NVIDIA V100 (sm_70) and an AMD MI50 (gfx906) together,
  commit 934f87c**: step 07 with `STEPS_07=training`, the V100 on torch
  2.14.1+cu126 (driver 545.23.08, NCCL 2.29.3) and the MI50 on torch
  2.9.1+rocm6.3 (`venv-rocm63`; HIP 6.3.42134, RCCL 2.21.5), the default
  bridge over TCP (1173.8 and 1169.7 MB/s, a 0.046 ms round trip). With
  gfx906 rocBLAS kernels on the MI50, the training steps that failed this
  afternoon ran: DDP with `gpubridge.ddp_comm_hook` (`train_ddp`; parameters
  in sync), and FSDP2 through the `gpubridge` backend, which also matched a
  single-process replay bit for bit. HSDP through the backend printed
  nothing for 20 minutes and the step timeout stopped it; the job was then
  cancelled, before HSDP's bit-for-bit check. `train_bench.py` (float32, the
  same batches per rank as alone): GPT-2 small 8.9 samples/s (a 899 ms step,
  659 ms of it exposed communication), against 25.3 on the V100 alone;
  ResNet-50 327.7 samples/s (a 195 ms step), against 315.8. Adding the MI50
  lowered GPT-2 small's throughput and left ResNet-50's about where it was.
- **Training on a V100 and an MI50 with fp8 on the bridge, commit 4e7f04e**:
  the same job and versions as the entry above, with
  `GPUBRIDGE_BRIDGE_COMPRESSION=fp8` and `GPUBRIDGE_COMPRESSION_KERNELS=torch`
  on both ranks (`venv-rocm63`'s Triton can't compile the compression kernels
  for gfx906, so fp8's saving is understated); the bridge's link measured 1164
  to 1174 MB/s each way. Jobs 11089348, 11090051, 11090726 and 11091146, and
  the V100 alone at the same per-rank batches in 11090049, 11090725 and
  11091145. `train_bench.py`, float32, GPT-2 small at 4, 8, 16 and 32 per
  rank: 12.495, 19.686, 26.274 and 28.311 samples/s, against 25.3, 27.669,
  28.395 and 29.247 on the V100 alone (0.49x to 0.97x); ResNet-50 at 32,
  366.849 against 315.8 (1.16x). From batch 16 on, the V100 waits longer for
  the MI50 each step (409 and 728 ms) than the exposed communication (258
  and 445 ms). Communication alone was 472 to 485 ms up to batch 16 and 605
  ms at 32, not yet explained. In the same jobs `train_ddp`'s old float64
  checksum differed across ranks once (11090726; most likely a false alarm,
  item 22), FSDP2's bit-for-bit check differed from its uncompressed
  single-process replay in all four, and HSDP hung
  until its step timeout in all four. Details in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#fp8-on-the-bridge-and-larger-batches).
- **RDMA between the node types, commit 1ac6fa9** (the network check with
  `rdma_report.sh`; its other steps as at 934f87c), one V100 node and one MI50
  node: **no RDMA path between the nodes checked.** The V100 node's only RDMA device is
  `mlx5_0`, InfiniBand; the interface the bridge uses there has none. The MI50
  node's only active one is `mlx5_0`, RoCE on that interface at 10 Gb/s; its
  InfiniBand port is disabled. On the GPU side, `nvidia_peermem` (545.23.08) is
  installed for the V100 node's kernel but not loaded; the MI50 node's kernel
  has `CONFIG_HSA_AMD_P2P` and torch 2.14.1+rocm7.2's HSA runtime exports
  dma-bufs; both kernels have `CONFIG_DMABUF_MOVE_NOTIFY`, and both nodes'
  libibverbs (rdma-core 46.0 and 57.0) have `ibv_reg_dmabuf_mr`. The UCX
  module (1.16.0, in OpenMPI 4.1.6) offered TCP and shared-memory transports
  and host memory only, and installs no `ib`, `cuda` or `rocm` transport
  module; UCX copies bundled with some application modules have `ib`, none
  `cuda` or `rocm`. Details in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#rdma-between-the-node-types).
- **A multi-stream TCP bridge prototype (#66), commit 23180fe of its
  branch**, step 08 with `MULTITCP_BRIDGE=1`, two jobs on a V100 + MI50: with
  pinned staging, a float32 all_reduce took 1.099 and 1.098 s at 1 GB against
  1.250 and 1.241 s on the pinned Gloo bridge, 0.88 times its time, and 1.07 and
  1.02 times at 64 MB. Its pre-set gate (at most 0.8 times at every size from 64
  MB) isn't met, so the prototype stops there. Its first run was seven times
  slower: torch's intra-op threads competing with the transport's network threads
  on 4-CPU jobs made each reduction 10 to 25 times slower, until it reduced on one
  thread. Details in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#a-multi-stream-tcp-bridge-prototype).
- **Triton 3.8.0 with torch 2.9.1+rocm6.3 on the AMD MI50 (gfx906), commit
  4e7f04e**: Triton from the default ROCm venv in front of `venv-rocm63`'s own
  (3.5.1), which can't compile the compression kernels for gfx906. The kit's
  GPU tests (`test_gpu.py`, `test_compression_kernels.py`,
  `test_torch_backend.py`) gave 26 passed, 8 skipped, with Triton's bundled HIP
  and with torch's HIP 6.3 (job 11091760); with 3.5.1, 20 failed. Then GPT-2
  small through the bridge with fp8 and these Triton kernels on both ranks
  (V100: torch 2.14.1+cu126, NCCL 2.29.3; MI50: RCCL 2.21.5): 29.631 samples/s
  at batch 16 (job 11091811) and 32.726 at 32 (11093097), against 28.395 and
  29.247 on the V100 alone (1.04x and 1.12x) and 26.274 and 28.311 with the
  PyTorch kernels. Communication alone fell to 346 and 362 ms from 485 and
  605. HSDP hung again in both. Details in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#fp8-with-triton-kernels-on-the-mi50).
- **Point-to-point, HSDP and the kit on a V100 + MI50, commit 1c7d107** (V100:
  torch 2.14.1+cu126, NCCL 2.29.3; MI50: torch 2.9.1+rocm6.3, RCCL 2.21.5, in
  `venv-rocm63`). Point-to-point through `backend="gpubridge"` passed on GPUs:
  `check_p2p` on the split test (one V100, job 11112711) and across vendors (job
  11112713), and the GPU stream-order test in `gpu_tests` on the V100 and the
  MI50. Across vendors, a message took about 2.8 to 3.4 ms one way from 1 KB to
  1 MB, and moved 0.8 GB/s at 64 MB and up (1.0 GB/s with pinned staging),
  payload intact at every size. HSDP through the backend passed with mixed torch
  releases (`train_hsdp_backend`, `verify_hsdp_backend`; it hung before #80).
  `train_ddp`'s old float64 checksum differed, as in 1 of 7 earlier mixed runs
  (most likely a false alarm; item 22). The MI50's compression-kernel steps
  failed on `venv-rocm63`'s Triton 3.5.1, which can't compile for gfx906, and
  passed with the Triton 3.8.0 overlay (job 11113623: every step of the AMD node
  job, GPU tests 27 passed, 8 skipped).
- **Items 22 and 33 on a V100 + MI50, commits 1659f4f, fe84663 and 48840ba**
  (V100: torch 2.14.1+cu126; MI50: torch 2.9.1+rocm6.3 in `venv-rocm63`, with
  the Triton 3.8.0 overlay for item 33's compressed runs). `train_ddp` passed
  in 24 of 24 runs, 16 uncompressed and 8 fp8 with the PyTorch kernels, under
  the bit-for-bit check (#88); the old float64 checksum differed in 4 of them
  with identical parameters, so the two earlier failures were most likely its
  false alarms (item 22). `train_dp.py` at 1,000 steps against eight one-V100
  runs, by a rule fixed before the new runs (S = 0.0131, the widest gap between
  two of those runs): uncompressed DDP passes; shares by speed and FSDP2 fail
  by the rule, one run of two each outside S, both late to reach the plateau
  exit (10-step mean loss below 2.0); bf16 and fp8 fail by the rule in all four
  runs (2.51 S to 528 S), as a lossy mode is expected to against a measure of
  uncompressed noise, and whether those gaps are acceptable is open. At 2,000
  steps one of three one-V100 runs ended 25 S from the other two, so a
  redesigned test is planned (item 33).

Everything else is **validated on CPU only**:
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) lists what still needs real
hardware.

## [Unreleased]

### Changed: the default bridge, and no Gloo unless asked for (2026-11-11)

**The default bridge changed, and a job no longer creates a Gloo process group
unless it asks for one.** Read this before upgrading a job.

- **The default bridge is `bridge="auto"`, a Rust bridge, never Gloo.** `init()`
  resolves it once, the same on every rank: `"verbs"` (RDMA) when every island
  leader passes the verbs bridge's own check for a usable RDMA device and all
  leaders are on one link layer, otherwise `"tcp"`, the Rust TCP bridge
  (`"multitcp"` until now, kept as another name for it). The choice is recorded
  in `Topology.selection` and the bridge's `telemetry()`, logged by the
  `gpubridge` logger, and never changes during the job. A bridge asked for by
  name is never replaced: `bridge="verbs"` where verbs can't run still fails
  `init()` on every rank and never drops to TCP. `GPUBRIDGE_BRIDGE` now also
  names the bridge for `gpubridge.init()` when the script passes none.
- **Gloo is no longer created unless asked for.** Until now every job made its
  default process group a Gloo group of every rank, ran discovery and
  `barrier()` on it, and under the torch.distributed backend made two more.
  Now discovery is a record per rank in the job's c10d store, read before any
  process group exists; `barrier()` is a counter in the store; and the default
  process group is gpubridge's own (`GpubridgeProcessGroup`, as under
  `backend="gpubridge"`), so `torch.distributed` calls on the default group run
  through gpubridge and observers see them. Gloo groups are created only by
  `bridge="gloo"` (opt-in, kept for one release: the whole job as it was
  before), by the `flat-gloo` policy (which makes its own group), and in CPU
  simulation for an island of several ranks, a test device. A GPU job on the
  default has no thread named `pt_gloo_runloop` or `gloo_tcp_loop`, with one
  qualification, next.
- **torch's debug level DETAIL is refused unless the job opts in.** At
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` (or `torch.distributed.set_debug_level`)
  torch wraps every NCCL, RCCL or Gloo group it creates in a checker that runs
  on a Gloo helper group, which would put Gloo and its worker threads back
  into a job that never named it. `init()` (both APIs) now refuses a job that
  didn't ask for Gloo at that level: on every rank, with one message that says
  why and what to do, before any island group exists. Lower the level, or
  accept the helper groups and their exit-abort exposure with
  `gpubridge.init(allow_debug_gloo=True)` or `GPUBRIDGE_ALLOW_DEBUG_GLOO=1`
  (new; the same on every rank). gpubridge never changes the level. Not
  covered: a level raised after `init()` (checked again only when a subgroup is
  set up, and then on each member by itself). Tested on CPU with simulation's
  Gloo islands standing in for NCCL ones (`tests/test_debug_gloo.py`); that
  torch wraps an NCCL group the same way is read from torch 2.8.0 and 2.15.0,
  not run on a GPU.
- **A refusal found in discovery reaches the ranks that read the discovery
  records, and rank 0 waits for them to, up to a deadline.** A rank that cannot
  resolve its own settings (an unknown bridge or policy name, a bridge module
  that fails to import, an invalid `GPUBRIDGE_*` value) publishes the error as
  its discovery record before it raises, so the others raise with its message
  instead of waiting for it until the timeout. Each rank marks in the store
  that it has read the records, and when they make the job fail (a problem, a
  mismatch, the DETAIL refusal, two ranks on one GPU, a policy that doesn't
  fit: all checked before any island group exists; with `gpubridge.init()`
  before the default group too) rank 0 waits for the other ranks' marks before
  it raises. Under a `tcp://` or plain `env://` rendezvous the store's server
  lives in rank 0; without this a slower rank could get a network error from
  a store that had gone, not the reason (`tests/test_store_owner.py` stops a
  rank between its write and its reads to show it). The exact promise:
  refusals decided from the discovery records have the same `RuntimeError`
  text on ranks that read those records; a rank with a malformed local setting
  keeps its original exception (`ValueError`). Rank 0 waits for other ranks'
  read marks only until its original discovery deadline, the one the wait for
  the records had, not a second timeout after it. If that deadline expires
  first, it logs the unread ranks when the store permits, then raises the
  original refusal; a reader still delayed at that point can instead receive
  a store error when the owner exits, even if it is alive and connected. The
  deadline bounds polling, not synchronous store calls or total init time. A
  rank that never published is named at the timeout on every rank, as before.
  It also needs discovery keys no earlier job wrote (a fresh store per job, or
  a namespace per job or attempt): starting a new job on a persistent store
  under a prefix an earlier job used is not supported, as that job's records
  and read marks would be taken for the new one's. A job that starts pays one
  more store
  write per rank and waits for no mark. Not covered: failures after discovery
  (a bridge that cannot connect is published through the bridge's own status
  keys, as before).
- **Why.** On Python 3.12 with torch 2.10 to 2.15 a Gloo worker thread can abort
  a process at interpreter exit (signal 6, "terminate called without an active
  exception"), after training has finished; it reproduces with torch alone.
  docs/RUNTIME_INVENTORY.md, section 6, records the reversal of the 2026-10-20
  decision to keep Gloo, and what replaced each use of it.
- **What you need to do.** The island leaders of a mixed job need the bridge's
  Rust library, where the Gloo bridge needed nothing beyond torch: a built
  library, or cargo to build it on first use.
  `python -m gpubridge.multitcp build` (cargo, no network), or
  `GPUBRIDGE_MULTITCP_LIB`; without it `init()` fails on every rank and says
  so. To keep a job exactly as it was, set `GPUBRIDGE_BRIDGE=gloo` (or pass
  `bridge="gloo"`). Code that relied on the default group being Gloo
  (`dist.get_backend() == "gloo"`, `topology.bridge_group`, a monitoring tool's
  own exchange on the default group while observing) needs a look: the default
  group's collectives are gpubridge collectives now, they take sequence numbers,
  and after a failed collective they refuse to start like any other.
- **Also changed with it:**
  - Under the torch.distributed backend, CPU tensors on a GPU job are staged
    onto the rank's GPU and carried by the topology instead of a Gloo group;
    `gather` and `scatter` run as an all-gather and a broadcast of bytes.
  - Cross-island point-to-point on the TCP bridge runs over TCP lanes of its
    own (`src/gpubridge/multitcp/p2p.py`). The verbs bridge still has none.
  - A one-rank island in CPU simulation is a `SoloGroup`: a Python group with
    no backend, connection or thread.
  - A failed collective on a group of the backend now raises from the
    `torch.distributed` call that made it. Before, a synchronous
    `dist.all_reduce` returned as if it had worked (torch's Work for a Python
    backend doesn't raise from `wait()`), and only the next collective refused
    to start. DDP without a comm hook is called from C++ and still learns of a
    failure at its next collective.
  - `gpubridge.destroy()` is no longer the only way out: an exit handler closes
    a bridge the script left open (joining the bridge's threads) before the
    interpreter finalizes, does nothing after `destroy()`, and nothing in a
    forked child. It is best effort: it runs at a normal interpreter exit, not
    after `os._exit`, SIGKILL or a fatal abort; handlers registered after
    gpubridge was imported run before it; an exception from a policy's `close()`
    leaves the bridge unclosed; and it is not bounded in time. Draining and the
    TCP accepting thread have deadlines, but the TCP sending and receiving
    threads and the verbs completion thread are joined without one, since they
    can still touch the caller's memory and are never detached; a device call
    that never returned would hold the exit. Teardown faults on hardware were
    not tested. The TCP library's threads are named (`gb-tcp-accept`,
    `gb-tcp-send`, `gb-tcp-recv`); both Rust crates state, and
    `tests/test_native_threads.py` checks, that their threads never touch a
    Python object.
  - The kit's records report the bridge a run used (`run.bridge`), beside what
    it asked for and how `init()` chose. `check.py` and `bench_all_reduce.py`
    keep `--bridge gloo` as their own default: the path their recorded results
    were measured on.
- **What was validated, and what was not.**
  - On CPU (Linux aarch64 containers; Python 3.10.22 with torch 2.8.0 and
    Python 3.12.3 with torch 2.15.0), the whole suite on the new default and
    again with `GPUBRIDGE_BRIDGE=gloo`: no failure in any of the four runs.
    Python 3.10 (without `tests/test_gpu_jobs.py`, as CI's 3.10 leg): 1675
    passed and 123 skipped on each. Python 3.12 (everything): 1771 passed and
    81 skipped on each. Getting there took three corrections to tests that had
    each failed once in a whole-suite run, none of them in the bridges: a test
    that kept a simulated island's Gloo group alive into interpreter exit
    (torch's abort again: 11 of 380 runs of that test alone), a check that
    listed a thread the kernel had not yet removed after its join, and two
    ranks that could close before the other had counted its threads.
  - The same bits as Gloo: `tests/test_parity_fixture.py` holds the SHA-256 of
    1240 results, 1080 of them generated once by the Gloo bridge (every supported
    dtype, SUM, MAX, MIN and AVG, 2, 3 and 4 bridge members, sizes one element
    below, at and above 256 KiB and 4 MiB). The TCP bridge, the verbs bridge on
    its MOCK fabric and the UCX bridge over TCP reproduce every one in jobs with
    no Gloo group. With two bridge members, which is every NVIDIA + AMD job, that
    is on rounding data. With three or four, the Rust and UCX bridges add a
    floating-point SUM in ring order, which is not Gloo's: on rounding data
    Gloo gave the ring's bits in 27 of 160 cases, so those 160 are
    pinned to the ring order instead, and the Gloo-generated ones use sums that
    are exact. **So this is not bit parity with Gloo everywhere: in 133 of the
    160 order-sensitive cases with three or four leaders the default now gives
    other last bits than the Gloo bridge did.** The order is the one the opt-in
    ring transports (`multitcp`, `verbs`, `ucx`) always used, both results are
    correct sums, and an NVIDIA + AMD job has two leaders, where it cannot
    arise. The 1240 hashes are not a general numerical qualification. They do
    not cover: all-gather, reduce-scatter and gather, point-to-point, GPU and
    device paths, reductions inside an island or islands of unequal size, more
    than four leaders, zero-sized or non-contiguous tensors, a broadcast from
    any rank but the last, NaN, infinities and signed zeros, compression, or
    speed. Other tests cover some of those. The file was generated on Linux aarch64; on x86-64, which was
    only emulated (qemu), the TCP bridge, the verbs bridge on its MOCK fabric
    and Gloo itself gave the same hashes (the UCX bridge was not built there).
  - The exit abort (`scripts/exit_abort.py`: four DDP ranks returning straight
    into interpreter exit, 300 repetitions of each arm side by side, Linux,
    Python 3.12.3, torch 2.15.0): on the TCP bridge 0 of 300 repetitions had a
    rank exit non-zero (1200 exits, all 0); on the Gloo control 32 of 300 did
    (33 ranks ended by signal 6, each with "terminate called without an active
    exception"). The same with every rank calling `destroy()` first: 0 of 300
    on the TCP bridge and 0 of 300 on the control, so only the trial without
    `destroy()` shows that the test can see the abort.
  - Found on the way and not fixed, a defect of the UCX bridge that is older
    than this change (`auto` never selects UCX, so the default cannot meet it;
    the exit handler can make a forgotten UCX teardown happen earlier than it
    used to): on the UCX bridge (`bridge="ucx"`, opt-in)
    a leader that closes its bridge while another is still inside the job's
    last collective fails that one ("UCX peer failed ... Connection reset by
    remote peer"): 16 of 40 four-leader jobs that end with a large broadcast
    and an immediate `destroy()`, the same at 82eca8e. End such a job with
    `gpubridge.barrier()` before `destroy()` (0 of 40). The TCP bridge and the
    verbs bridge on its MOCK fabric did not show it (0 of 200 and 0 of 100).
  - **On hardware, nothing here was run.** The verbs bridge has run only on one
    node through one InfiniBand adapter in loopback, staged through host memory
    (2026-10-20, below). The TCP bridge ran between a V100 node and an MI50 node
    by name, beside a Gloo default group (23180fe). **Nothing in this change was
    run on a GPU**: not discovery through the store, not gpubridge's own default
    group over NCCL or RCCL islands, not the store barrier, not `auto`'s device
    check on a real RDMA device, not CPU tensors staged through a GPU, not the TCP
    point-to-point lanes, not the exit handler with NCCL groups.
    HARDWARE_VALIDATION.md items 44 to 50 say what must be run.

### Hardware validation

- **2026-10-20: one node with 4x Tesla V100-SXM2-32GB, driver 545.23.08,
  torch 2.14.1+cu126, NCCL 2.29.3, Mellanox ConnectX (mlx5) HDR InfiniBand.**
  At GPUBridge **`c92684b`**, two 2-GPU NCCL islands of identical V100s passed
  Gloo correctness in half/alternate layouts, pinned/pipelined/sharded k=2
  checks, auto-tuned thresholds, DDP and exact FSDP2/HSDP replay; GPU tests:
  29 passed, 6 skipped (2 UCX, 4 inapplicable bf16-compression cases),
  no failures. The staged-verbs kit and default-policy
  check passed over one HCA on one host (loopback), with non-mock libibverbs,
  zero completion errors and passing GPU strict TCP evidence. Exact kit
  payloads reached 512 MiB; the verbs benchmark reached 1 GiB. The original
  host kit failed before payload due to CPU/CUDA topology mismatch.
  The native/Gloo/staged-verbs GPT-2-small bucket sweep had finite,
  rank-identical parameters; all pipeline timing sweeps completed and all
  66 native/Gloo P2P benchmark payload rows passed. Item 18's unrestricted
  speed goal is not established, especially for verbs; timing passes do not
  establish significant speedups, direct overlap or cross-arm GPT equivalence.
- **Host recheck, same date/hardware/driver/torch/NCCL:** GPUBridge
  **`0cf32a3`**, whose production fix is retained in **`54f701e`** (test
  isolation only), passed exact host verbs collectives using CPU/Gloo
  islands and the GPU-staged strict regression check. Host TCP evidence was
  `not-decisive`; strict host mode failed closed as intended. `54f701e`
  itself was not the hardware checkout. Cross-node, AMD/mixed-vendor verbs,
  GPUDirect, UCX and 8-GPU scaling remain unproven by these runs; the
  collective `wait-skips-stream` mutation remains uncaught.
- **Earlier split-test runs on the same node:** `616f2f7` (2026-10-19)
  exercised the 2-rank staged-verbs kit and staged training comparisons;
  the separate tiny numerical gate (2026-10-20) at **`f4146a4`** accepted 48/48 positive
  rank records with zero violations and rejected all three bad controls.
  Same GPU class, driver 545.23.08, torch 2.14.1+cu126, NCCL 2.29.3;
  the tiny gate establishes only its bounded workload, not representative
  GPT equivalence. See the scoped
  [hardware results](HARDWARE_VALIDATION.md#split-test-results-2026-10-20),
  [verbs status](docs/VERBS_TRANSPORT.md) and
  [runbook](docs/GPU_VALIDATION_RUNBOOK.md).

### Fixed

- The verbs library's build cache has its own variable, `GPUBRIDGE_VERBS_CACHE_DIR`.
  `GPUBRIDGE_VERBS_LIB_DIR` keeps the meaning the crate's build script and
  docs/VERBS_TRANSPORT.md give it, the directory of `libibverbs.so.1`; the Python side
  also used it as the cache, so setting it as documented put the built library (and
  its temporary build directories) into the libibverbs directory. The crate is unchanged.
- The kit's Triton cache fill copies each stored kernel directory whole on BSD `cp` too
  (macOS), and two kit tests no longer depend on GNU-only behaviour of the test host:
  the mount-table test resolves its own paths, and the store test skips where `mv`
  lacks GNU's `-T`.
- The verbs kit's host collectives configure CPU/Gloo islands before init, record that mode, and refuse contradictory inherited CPU-only settings.
- Host mode selects CPU-only operation before runtime imports. The verbs kit excludes
  same-host non-loopback TCP only for host/Gloo islands, marks co-resident or unknown
  bridge peers' TCP evidence `not-decisive` (strict mode fails closed), and preserves
  GPU-mode local TCP and cross-node remote TCP checks.
- The verbs kit's host mode attributes loopback and same-host TCP sockets by the
  process owning the other end in the same `ss` snapshot (ranks gather host, boot ID,
  pid and pid namespace). Only sockets to the leader's own island are left out; the
  other leader, other ranks and the store count; unattributed bytes that would exceed
  the unchanged limit keep the evidence `not-decisive`, as do unidentified island
  membership and island traffic that isn't seen. Records list every socket's class, owners and bytes
  (`local_tcp_attribution`). Ranks snapshot sockets one at a time, all before any
  teardown. Co-resident host leaders can now get a decisive verdict; the first hardware
  run (without serialized snapshots) stayed `not-decisive`. On one host with 4 ranks as
  two 2-rank CPU/Gloo islands over one HCA (loopback; not cross-node, AMD or GPUDirect
  evidence), the rerun at `4ffcf93` was decisive: strict host collectives passed (28/28
  exact, island TCP attributed and left out, 1728 B counted), both real-TCP negatives
  recorded verdict `fail` on counted TCP over the limit, and staged-GPU strict was
  unchanged.

### Added

- `scripts/baseline.sh OUT_DIR`: the CPU baseline in one command, CI's crate checks,
  native builds, the test suite and the simulation example, every step's exit code in
  `steps.jsonl`; `scripts/baseline_identity.py` records the git trees, packages, Rust
  and C toolchains and the SHA-256 of each built library. Its pass is not GPU or RDMA
  validation (the verbs bridge runs on its MOCK fabric). `scripts/bench_import.py`
  compares `import gpubridge` between two source trees.
- docs/RUNTIME_INVENTORY.md: where Python and Gloo run in a job, and a keep, demote or
  remove decision for each part with its callers and measurements;
  docs/BENCHMARKS.md has the single-node verbs-against-Gloo timings.
- The staged verbs path's costs are measured (docs/verbs-perf-20261019.md): the
  library times its write windows' registration and release
  (`direct.write_window_register_ns`, `write_windows_released`,
  `write_window_release_ns`), the bridge splits each staged call's host time into
  copy out, bridge step (posts, library waits, the rest) and copy back
  (`staged_phases`), and `scripts/gpu/verbs_staged_ab.py` runs one arm of a one-node
  A/B, compares arms against the predeclared rules and decides the plan's copy
  overlap trigger. Measurement only; CPU/MOCK cost checked, not
  validated on GPUs or RDMA hardware.
- `GPUBRIDGE_VERBS_STAGED_OVERLAP=1` (opt-in, off by default): a staged two-leader
  verbs all_reduce of more than one chunk overlaps its copies out and back with its
  messages, chunk by chunk, with the same messages on the wire
  (docs/VERBS_TRANSPORT.md, "GPU memory"). `last_gpu["staging"]` and
  `staged_phases["overlapped_calls"]` say when it ran. A conditional candidate: no
  speedup is claimed; docs/verbs-perf-20261019.md, section 7, is its A/B. Not
  validated on GPUs.
- A bounded numerical gate for native NCCL, Gloo islands and staged verbs checks
  exact collectives, gradients, independent optimizer updates/state and parameters;
  includes bad-reduction/scale/data controls and actual endpoint/source attestation.
  `train_bench.py --controlled` resets the seeded model after its compute probe
  and fingerprints initial state/data for fair fixed-workload comparisons.
  The bounded tiny GPU gate passed at `f4146a4`; representative GPT
  equivalence remains open (item 43; hardware record above).

- `bridge="verbs"` takes GPU tensors, phase A3 (docs/VERBS_TRANSPORT.md): wire
  protocol 2 adds a rendezvous (an RDMA WRITE into memory registered for one
  receive, from `GPUBRIDGE_VERBS_RNDV_BYTES`, 256 KiB) and sends gathered from
  registered memory; GPU tensors go direct (NVIDIA dma-buf or peer memory, AMD
  dma-buf from ROCr, drivers loaded at run time, the caller's stream waited for
  and the GPU flushed) only when every check passes and registration succeeds,
  else staged through a registered pinned buffer, with the reason in telemetry;
  transport observations for the straggler detector (`GPUBRIDGE_VERBS_OBSERVE`,
  with `host`, `job` and the NIC's `pci_bus_id`); a prepared hardware kit
  (`VERBS_BRIDGE=1`, `scripts/gpu/verbs_kit.py`, items 38 to 42). Tested on mock
  providers, a mock GPU and CPU stand-ins; built on Explorer's login node (Linux
  clippy blocked by its limits). Host/staged loopback hardware results are
  recorded above; direct GPU and cross-node validation remain open.
  (PR pending)
- `bridge="verbs"`, opt-in, phase A2 of the raw-verbs bridge
  (docs/VERBS_TRANSPORT.md): a versioned wire protocol of RC SEND/RECV through
  each leader's own registered host buffers, with credits, chunking, a
  completion thread on a completion channel, deadlines, peer loss and teardown
  that never frees memory the device may still use; handshake through the c10d
  store, documented device, port and GID selection, and telemetry. If verbs
  can't run, setup fails with the reason; nothing falls back to TCP. Host
  tensors only. The multitcp bridge's collectives moved to a shared
  `TaggedBridge` that both use. Tested on mock providers (bit for bit against
  `flat-gloo` between processes on a test-only fabric) and built on Explorer's
  login node; host/staged loopback hardware results are recorded above,
  with cross-node validation still open. (PR pending)
- The foundation of a raw-verbs bridge, phase A1 of the opt-in `bridge="verbs"`
  above (docs/VERBS_TRANSPORT.md): a Rust crate,
  `gpubridge-verbs`, with a C shim compiled against rdma-core's headers and
  linked to `libibverbs.so.1`, owned device, protection domain, completion
  queue, queue pair and memory region types that release in a correct order,
  work tracking that keeps memory registered while posted work can use it,
  the queue pair state machine with InfiniBand and RoCE address checks, a
  bounded registration cache keyed by allocation identity, and a
  deterministic mock fabric with injectable faults. Tested on the mock
  provider (macOS and Linux) and built against rdma-core 57.0 on Explorer's
  login node; successful host/staged loopback traffic is recorded above.
  Cross-node and hardware failure cases remain open. A CI job tests the
  crate. (PR pending)
- Step 07's and the two-machine runner's `DP_RUNS` take `ddp-shares-<rows>`: DDP
  with the rows per rank as given, in rank order (`ddp-shares-5-3` runs
  `train_dp.py --shares 5,3`), and `-r<N>` at the end to name a repeat
  (`ddp-shares-5-3-r2`), so one job can sweep the split between a V100 and an
  MI50 on the same pair of GPUs. `train_dp.py` already took `--shares 5,3`;
  until now `DP_ARGS` was the only way in, and it gives every run of the job
  the same split. Opt in; the default `DP_RUNS` is unchanged. CPU-tested only.
- Megatron-Core's pipeline across vendors, experimental (docs/MEGATRON.md):
  `scripts/gpu/megatron_pp.py` runs Megatron-Core 0.14's GPT in two stages with
  its own 1F1B schedule through `backend="gpubridge"`, with no patch to
  Megatron. Megatron's groups build on the gpubridge default group; the
  configuration avoids Transformer Engine, Apex, the fused kernels and bf16.
  Step 07's `STEPS_07=megatron` runs it on one V100, then on a V100 + MI50.
  There: the two curves stay within 1.4e-4 over 200 steps, at 1.14x the V100's
  tokens/s. HARDWARE_VALIDATION.md item 37 (#93).
- One model's throughput, every way (docs/BENCHMARKS.md, "What mixing vendors
  costs, and whether it helps"). `llama-125m`, 500 steps, each setting run twice
  (spread at most 5.4%):
  - one V100 alone, 14,264 tokens/s; one MI50 alone, 9,614;
  - a V100 + MI50 pipeline, 15,686 with the 9 + 3 split (1.10x the V100) and
    15,316 with the automatic one;
  - V100 + MI50 data parallelism, 4,543 uncompressed (0.32x) and 9,689 with fp8
    on the bridge (0.68x).

  `submit.sh 07-nvidia-pair` and `07-amd-pair` run step 07's dp or pipeline
  runs on two GPUs of one type, one island (native NCCL or RCCL), for the
  same-vendor baselines. They're refused until `SAME_PARTITION_PAIRS_OK=1`
  (#92).
- The GPU kit refuses runs whose records would mislead. Every job stops before
  its first step if `import gpubridge` isn't the running checkout's `src/`
  (`scripts/gpu/import_check.py`, and the probe on every side of a mixed job).
  Steps that use the Triton compression kernels (`gpu_tests`, the kernel and
  compressed benchmarks, compressed collective runs) first check that Triton
  compiles and runs all four kernels where the step runs (both sides of a mixed
  job; `scripts/gpu/triton_check.py`, once per job); where it can't, those steps are recorded as not run and the rest
  of the job goes on (`mixed/node.sh`, whose machines decide alone, never skips a
  mixed step). The probe records where gpubridge was imported from,
  its commit, `PYTHONPATH`, and whether Triton compiles the kernels. A job whose
  results folder already holds another run's records writes to `<step>_job<id>/`
  instead. The AMD node job writes
  `job.jsonl` like the others, and a directory of wrapper scripts, such as
  `rocm_triton_overlay.sh`'s, works as `VENV` or `VENV_ROCM`.
- Automatic pipeline stage balancing: `gpubridge.balance`, an opt-in module
  nothing else imports. `profile_layers` times each layer's forward and
  backward on a device (CUDA or HIP events), with its output size, state and
  saved activations. `partition` picks the contiguous split whose slowest
  stage, transfers across each boundary included, is fastest within each
  device's memory: exact, by dynamic programming, and checked against brute
  force with injected speeds, links and memory limits. `train_pipeline.py
  --split auto` profiles every block on every stage's GPU and the links, and
  trains bit for bit like the same split given by hand. Step 07's
  `PP_SPLITS` runs several splits in one job. On a V100 + MI50 (1F1B, 500
  steps): even 13,062 tokens/s, 8 + 4 15,577, 9 + 3 16,021; the automatic
  split chose 8 + 4 (15,559, 1.19x even), 3.0% below the best hand-tuned one.
  docs/PIPELINE.md, HARDWARE_VALIDATION.md item 36 (#90).
- `examples/train_synthetic.py` (the kit's `train_ddp`) compares every rank's
  gradients after the all_reduce and parameters after the optimizer step, bit for
  bit, between steps, and records the first step each differed, in which tensors,
  and by how much (`first_grad_mismatch`, `first_param_mismatch`). The parameter
  verdict now comes from the bits; the earlier float64 checksum comparison stays
  as `params_checksums_equal`.
- Step 07 can repeat `train_ddp` (item 22) in one job: `STEPS_07=ddp` runs only
  the probe, the preflight and `train_ddp`, and `TRAIN_DDP_REPEATS=N` runs it N
  times, as steps `train_ddp_1` to `train_ddp_N`. Bridge compression comes from the
  environment at submit time, as before. `scripts/gpu/train_ddp_report.py` lists
  every `train_ddp` record with its verdict, including repeats the job planned but
  never ran, and counts the runs that diverged per compression setting and kind of
  run; the validation tables count each repeat as item 22's evidence. CPU-tested only.
- The two-machine runner (`scripts/gpu/mixed/node.sh`) also trains `llama.py`'s
  model: `train_dp.py` on one GPU of each machine alone (the baseline), then
  step 07's data-parallel runs (`DP_RUNS`, default DDP uncompressed, bf16 and
  fp8) and pipeline runs (`PP_RUNS`, GPipe and 1F1B) on both machines
  together, with step 07's `DP_*` and `PP_*` settings. The run names are now
  one shell function for both (`dp_run_options`). The runbook's 8+8 steps say
  how to size them for 16 GPUs. Step 07's `07-multi` with `STEPS_07=dp` and
  `pipeline` runs on the fake cluster with an NVIDIA island of two, and the
  runner on two pretend machines and in two containers (#85).
- Pipeline training across vendors: `scripts/gpu/train_pipeline.py` cuts
  `llama.py`'s model into contiguous stages, one per rank, and trains it with
  torch's GPipe or 1F1B through `backend="gpubridge"`'s point-to-point (or on a
  plain Gloo group, the reference). It records every step's loss and time,
  each stage's compute alone, the time an activation takes to arrive from the
  previous stage, and idle-time and bubble estimates. Step 07's
  `STEPS_07=pipeline`; `dp_report.py` reads its records too;
  docs/PIPELINE.md; HARDWARE_VALIDATION.md item 34. On CPU: bit for bit like
  plain Gloo (2 stages, and 4 over 2 + 2 islands), within 1e-4 of one process,
  on torch 2.8.0, 2.9.1 and 2.14.1. Every rank needs the same PyTorch release
  (2.14.1's stage setup doesn't match 2.9.1's), checked up front, and 2.9's
  int64 handshake goes as a float32, as in 2.14.1. On a V100 and an MI50
  (Explorer, both on 2.9.1, 1,000 steps): 13,115 tokens/s with GPipe and
  13,051 with 1F1B, 1.05x and 1.04x the V100 alone with the same 4
  micro-batches; over the last 100 steps 1F1B is within the spread of four
  one-V100 runs, GPipe 0.02 above it after a later drop (#81).
- `bridge="multitcp"`, an opt-in bridge transport over several TCP connections
  per pair of leaders (`GPUBRIDGE_MULTITCP_STREAMS`, 4), from a small Rust
  library (`src/gpubridge/multitcp/crate`, standard library only, a C interface
  loaded with ctypes; `python -m gpubridge.multitcp build`). Python runs the
  collectives and reductions as the UCX transport does; with two leaders each
  chunk is reduced as soon as it has arrived. Host tensors only. Tested on CPU
  only: every policy at two to four leaders bit for bit against `flat-gloo`,
  compression, subgroups, failures and teardown (`tests/test_multitcp.py`,
  `cargo test`, a CI job). Step 08 measures it against the Gloo bridge with
  `MULTITCP_BRIDGE=1`; HARDWARE_VALIDATION.md item 35; docs/RUST_TRANSPORT.md
  (#66).
- `scripts/gpu/bench_p2p.py`: point-to-point latency and bandwidth through
  `backend="gpubridge"`, ping-pong from 1 KB to 1 GB on each route rank 0 has (across
  islands through the bridge, and within its island), with a payload check per size and
  the staging (pinned, chunk size, transport) in each record. A dedicated untimed
  exchange verifies both directions before warmups and trials, including with zero
  warmups. CLI bounds fail before rendezvous; chunked integer residues preserve exact
  payload patterns beyond float32's consecutive-integer range. Explorer steps 02 and 07 run
  it (`bench_p2p`; step 07 also `bench_p2p_pinned`). CPU only so far: in simulation on one
  Mac, a cross-island message took about 3 ms one way at 1 KB to 1 MB, a native one 80 to
  320 us.
- GPU checks for point-to-point through `backend="gpubridge"`: `check.py --p2p-only`
  (step `check_p2p` in Explorer steps 02 and 07) and a GPU test in `tests/test_gpu.py`,
  both from `gpukit.p2p_checks`: six dtypes and odd sizes bit for bit across islands, the
  send's input written on a busy producer stream and dropped before the send completes, the
  received tensor read on another stream right after `wait()`, and two tags matched out of
  order. Uploads finish before the producer delay; its write is device to device,
  and a same-stream allocation pressures the dropped send's storage before wait.
  CPU scheduling-model tests reject missing stream waits and storage retention;
  item 32 counts only explicit `check_p2p` steps, excluding historical `gpu_tests`.
  CPU only so far (1+1 and 2+2 ranks in simulation).

- `scripts/gpu/rocm_triton_overlay.sh`: an env-only venv of one ROCm venv's
  torch with another's Triton linked in front, for an MI50 that trains on
  torch 2.9.1+rocm6.3 (gfx906 rocBLAS kernels) and needs Triton 3.8.0 for the
  compression kernels. Links, not copies; no Python runs.
- `scripts/gpu/amd/run_all.sh` measures training throughput (`train_bench.py`):
  each model on one GPU, and with 2+ GPUs as one RCCL island and split through
  the bridge. The runbook has the steps for an MI300X on AMD Developer Cloud.
- `submit.sh train-multi` and `07-multi`, for the day multigpu access arrives:
  `train_bench.py` on one GPU, on `GPUS_TRAIN_MULTI` GPUs (4) as one NCCL
  island, and on the same GPUs as two islands of half each through the bridge
  (a split test); and step 07 with `GPUS_MIXED_NVIDIA_MULTI` NVIDIA GPUs (4) on
  `multigpu` next to the AMD side. Both run end to end on the fake cluster
  (#64).
- Step 07: `STEPS_07=bench` runs only the probe, the preflight and the
  `train_bench_<model>` steps: no `train_ddp`, FSDP2 or HSDP (HSDP hung until
  its step limit in each of the four fp8 jobs above).
  `TRAIN_BENCH_BUCKETS="10 25 50 100"` runs each `train_bench_<model>` step
  once per DDP bucket size (`--bucket-mb`), as `train_bench_<model>_bucket<MB>`
  (step and JSON), for a bucket-size sweep in one job. Tested on the fake
  cluster, CPU only (#77).
- Data-parallel training across vendors on a LLaMA-style model:
  `scripts/gpu/llama.py` (RMSNorm, rotary embeddings, SwiGLU; 134M to 1.03B
  parameters; the same weights on every vendor; synthetic tokens with a
  learnable rule, the same global batch for any world size),
  `scripts/gpu/train_dp.py` (DDP with the comm hook, FSDP2 or HSDP through the
  backend, or one GPU; every step's loss over the whole batch; parameter checks
  across ranks; tokens/s and DDP's step split; batch shares by GPU speed with
  sample-weighted averaging), step 07's `STEPS_07=dp`, `submit.sh dp-baseline`
  and `scripts/gpu/dp_report.py` (the runs against the baseline, with the loss
  curves as CSV and SVG); HARDWARE_VALIDATION.md item 33. On a V100 and an
  MI50 (Explorer, `llama-125m`, 1,000 steps): 4,549 tokens/s uncompressed,
  5,753 with bf16 and 9,523 with fp8 on the bridge (Triton kernels on both
  GPUs), against 14,264 on the V100 alone; FSDP2 1,809. Shares by speed cut
  the gap between the GPUs' compute alone from 78 to 9 ms a step without
  raising tokens/s; the bridge, about 4x the slower GPU's compute, looks like
  what sets the step. 5, 25 and 100 MB buckets within 3.5%. Uncompressed DDP
  ends as close to one V100 as V100 runs do to each other; bf16 and fp8 end
  0.02 to 0.03 above in job 11111481 (later runs: HARDWARE_VALIDATION.md item
  33), after a later drop off the loss plateau, about the gap an uncompressed
  pipeline run left after a similarly late drop; these runs
  can't separate compression's error from the drop's timing. Parameters
  identical on both GPUs at every check.
  docs/BENCHMARKS.md, and docs/COMPRESSION.md's "Training with compression"
  (#67, #84).
- Host-staged UCX P2P for arbitrary island members, with independent
  directional lane workers and optional pinned/chunked staging. Tested on
  CPU over UCX TCP, including non-leaders and concurrent pairs; no GPU/RDMA
  validation or device-direct P2P claim.

- Native P2P batch coalescing for the torch backend, plus an explicit
  `gpubridge.batch_isend_irecv` helper for torch 2.8, whose batch function
  excludes Python process groups from coalescing. CPU tested; GPU unvalidated.

- Opt-in P2P pinned staging (`GPUBRIDGE_PINNED_STAGING`) and two-slot chunked
  staging (`GPUBRIDGE_P2P_CHUNK_BYTES`), with lazy-copy ordering mutation tests.
  CPU only; no GPU overlap or speedup measured.
  **2026-10-20 update:** the
  [split-test results](HARDWARE_VALIDATION.md#split-test-results-2026-10-20)
  add GPU payload checks and descriptive timing improvements for pinned/chunked
  P2P; direct copy/transfer overlap and significant speedups remain unproven.

- Torch backend point-to-point operations: native calls within an island and
  independent host-staged Gloo lanes across islands, including asynchronous
  and batched calls, tag matching, size/dtype checks and bounded teardown.
  CPU only; GPU stream behavior and native GPU batches are not validated.
  **2026-10-20 update:** the
  [split-test results](HARDWARE_VALIDATION.md#split-test-results-2026-10-20)
  add a full default P2P dtype/stream/tag check on separate GPUs over native
  NCCL and cross-island Gloo. Chunked/pinned benchmarks add payload checks,
  not the full stream suite; verbs P2P remains unvalidated.

- `scripts/gpu/rdma_report.sh`, a step of the network check (`net-nvidia`,
  `net-amd`): what RDMA from GPU memory would need on the node, one file each
  in `rdma/<side>/`: the kernel's dma-buf and peer-to-peer options, the RDMA
  and GPU kernel modules (and whether `nvidia_peermem` or `nv_peer_mem` exist),
  rdma-core and `ibv_reg_dmabuf_mr`, `rdma link`, UCX (`ucx_info -v` and `-d`)
  and the HSA runtime's dma-buf export, with paths cut from the records. It runs
  last and loads `RDMA_MODULES` inside its own step, so the steps before it are
  unchanged and a module that won't load fails only that step (#62, #65).
- A Kaggle runner for the GPU validation kit, for a free notebook with two
  T4s (`scripts/gpu/kaggle/`): a notebook and the same steps as one script,
  each step under a hard time limit. It checks what compute capability 7.5
  can run (CUDA events, bfloat16, Triton) and skips or adapts the rest with
  the reason. The results are the kit's usual records and validation tables,
  in one tarball (#43).
- `scripts/bench_per_call.py`: the per-call time of small all_reduces through
  each policy, sync and async, with and without an observer, against the same
  torch.distributed calls made directly. `--stub-comm` leaves only gpubridge's
  own work; `--profile` runs cProfile on rank 0 (#44).
- `scripts/gpu/triage.py`: a short plain-text summary of an Explorer or Kaggle
  results directory or tarball, under 80 lines, to paste into a review. It shows
  each failed, timed-out or skipped step and why, the GPUs and versions, the
  key benchmark numbers and the compression check (#45).
- `docs/EXPLORER_FIRST_SESSION.md`: a first Explorer session in order, from a
  fresh login: the clone in the project space, `local.env`, step 00, the setup
  job, the first rungs of the ladder, how to follow the jobs, where results
  land, what to do when a step fails and what to paste back (#46).
- `scripts/gpu/diag_fp8_residual.py`: reruns the fp8 error-feedback case of
  the GPU kernel test on one device and prints, for each value whose residual,
  code or scale differs from the CPU reference, every intermediate value with
  its bits and whether it is subnormal in float32. It also runs the kernel
  with multiply-add contraction on (gpubridge's own launches have it off since
  #50), a probe of what the device computes, and counts the compiled code's
  fused multiply-adds and flush-to-zero instructions (#48).
- `scripts/gpu/stream_mutations.py`: undoes each stream-ordering fix in turn
  (`GPUBRIDGE_TEST_MUTATION` in `tests/test_gpu.py`) and checks that the GPU
  test guarding it fails, and that the tests pass without one (#52).
- `scripts/gpu/time_triton_cache.py`: how long the compression kernels take
  to build and to load from Triton's cache, by cache location (#52).
- `probe.py` records whether BLAS works (`blas_available`, with the error),
  from a matmul in a child process, without failing the probe (#52).
- `scripts/gpu/bench_compression.py --fp-fusion`: launches the compression
  kernels with multiply-add contraction on, so a run each way times what
  gpubridge's default (off) costs (#50).
- Step 08, a comparison matrix of what gpubridge costs
  (`scripts/gpu/explorer/08_bench_matrix.sbatch`, `scripts/gpu/bench_matrix.py`):
  one heterogeneous job of two 1-GPU nodes runs the all_reduce benchmark
  (1 KB to 1 GB, CUDA events, phases) for native NCCL or RCCL on the
  transport each picks and over TCP on the bridge's interface (A1, A2, B1,
  B2), gpubridge between two islands of one vendor (C), gpubridge between a
  V100 and an MI50 under every policy, chunk size and compression mode (D),
  and one flat Gloo group (E). `matrix.md` has a table per size range with
  each run as a multiple of the two native baselines (marked where the
  baseline ran in another job, on other nodes) and the bridge phase against
  the preflight's prediction, each job's preflight, each node's InfiniBand
  ports, and the transport NCCL and RCCL chose. `submit.sh 08` submits the
  V100 + MI50 job;
  `08-nvidia-pair` and `08-amd-pair`, two GPUs of one type in one partition,
  are refused until `SAME_PARTITION_PAIRS_OK=1`.
- `submit.sh net-nvidia` and `net-amd` (`net_check.sbatch`,
  `scripts/gpu/net_report.sh`): one node's InfiniBand devices and ports, its
  interfaces, and the network NCCL or RCCL initializes there, as chosen and
  with `NCCL_IB_DISABLE=1`.
- `scripts/gpu/train_bench.py`: training throughput through DDP and
  `gpubridge.ddp_comm_hook` on GPT-2 small and ResNet-50 (built from
  `torch.nn`, synthetic data), with each step split into compute, waiting
  for the slower GPU and exposed communication. Step 07 runs it across both
  vendors (`STEPS_07=training` runs only its training steps), and
  `submit.sh train-nvidia` or `train-amd` on one GPU alone.
- `train_bench.py` per-rank batches (CPU only): `--batch-sizes a,b`,
  `--global-batch N` (split evenly; one rank alone takes all N) and
  `--global-batch N --balance` (a short calibration times each rank, then N is
  split by speed, at least one sample per rank). Each rank's summed loss is
  scaled by world size / N, so the hook's average is the gradient of the mean
  loss over all N samples: bit-identical to one process over the concatenated
  batch on CPU with integer data (2 and 3 ranks), within 1e-5 relative on float
  data. The JSON records the global batch, the split and each rank's batch,
  compute per sample and waiting. No `src/` change (#74).
- `train_bench.py --accumulate K` (CPU only): each rank's batch runs as K
  micro-batches with `ddp.no_sync()` for the first K-1, one gradient exchange
  per optimizer step; combines with `--batch-sizes`, `--global-batch` and
  `--balance` (calibration at the micro-batch size, at least K samples per
  rank). Bit-identical on CPU to one process over the same samples with
  integer data (2 and 3 ranks, K = 2 and 3), within 1e-5 relative on float
  data; BatchNorm normalizes over each micro-batch. The JSON records K, each
  rank's micro-batch sizes, samples per step and the exchanges per step the
  hook counted. No `src/` change (#75).
- `train_bench.py` on GPUs: each rank's `memory` in the JSON, for the compute,
  ddp and comm-alone phases: the caching allocator's peak allocated and
  reserved MiB (`torch.cuda.max_memory_allocated`, `max_memory_reserved`) and
  its allocation retries (`num_alloc_retries`), and the GPU's total. Null on
  CPU; the GPU path is tested only against a faked allocator (CPU only)
  (#78).

### Changed

- Cross-island point-to-point (the torch backend's `send` and `recv` between islands)
  looks for the peer's descriptor after pauses that start at 50 µs and double up to the
  2 ms it always paused, instead of 2 ms from the first look. Each look is still one short
  Store call, so a lane waiting for its peer holds the Store client that every lane shares
  only for that call and doesn't delay pairs that are ready. The exchange also times out
  when a Store call itself ran past the
  deadline. The health thread makes no Store call while nothing is pending; `submit()`
  wakes it, and it then checks on the same 50 ms grid as before. On CPU (two ranks on one
  machine, Gloo bridge, `scripts/gpu/bench_p2p.py`, 3 alternating runs per arm): one way
  3.9 ms → 0.63 ms at 1 KiB, 4.4 ms → 0.38 ms at 16 KiB. With transfers stubbed out
  (`scripts/bench_p2p_handshake.py`, 5 runs): the same Store calls per message, 30% (TCPStore)
  and 63% (FileStore) less CPU per ping-pong, 2 Store calls instead of about 70 over 2 s idle,
  and the same Store calls and CPU while lanes wait for a missing peer (equal over a 5 s
  wait, 7 runs; within run-to-run spread over 1 s). A blocking
  `store.wait` was tried and rejected: TCPStore serializes a client's requests, so one
  lane's wait held up the others past their deadlines and logged every timed-out slice.
  CPU evidence only, not GPU, RDMA or cross-node. On teardown, Engine._watch rechecks stop/closed state under lock before clearing the idle wake, preventing a concurrent close() wake from being cleared.
- `import gpubridge` no longer imports the opt-in bridges (`ucx`, `multitcp`,
  `verbs`): each registers itself when a job first names it (`init(bridge=...)`,
  `GPUBRIDGE_BRIDGE`, `resolve_transport`). The default Gloo job loads 15 gpubridge
  modules instead of 25; measured on CPU with `scripts/bench_import.py`, `import
  gpubridge` takes 4.75 ms instead of 7.04 ms with compiled bytecode and 19.6 ms
  instead of 31.3 ms without (torch's own import: about 390 ms). A built-in name
  still can't be registered by another class,
  and a built-in that fails to import is a `ValueError` like an unknown name
  (docs/RUNTIME_INVENTORY.md).
- On Explorer, the AMD side's default venv (`VENV_ROCM`) is now the MI50's
  Triton overlay, `venv-rocm63-triton38`: torch 2.9.1+rocm6.3, whose rocBLAS
  trains on gfx906, with Triton 3.8.0 in front, which compiles the compression
  kernels. It was `venv-rocm` (torch 2.14.1+rocm7.2). `submit.sh setup-rocm`
  still builds `venv-rocm` unless `VENV_ROCM` or `ROCM_SETUP_VENV` names another
  venv, and AMD jobs stop with the commands that build the overlay if it's
  missing (docs/GPU_VALIDATION_RUNBOOK.md, "Choosing the ROCm build").
- A rank's island and whether it leads, and a policy's supported reduce ops,
  are worked out once instead of on every collective. Every call still checks
  its op. On CPU, gpubridge's own work per small reduce-bridge-broadcast call
  went from 4.0 to 3.4 µs; the README has the per-call numbers (#44).
- For policy authors: `CollectivePolicy.supported_reduce_ops()` is asked once
  per policy instance and topology, at the first collective, and its answer
  must not change after that (#44).
- The README opens with a pitch, badges, a one-line status, an architecture
  diagram, a quickstart and a feature table with each feature's status. New
  diagrams for all_reduce and HSDP (docs/img, light and dark). New
  CONTRIBUTING.md, SECURITY.md and issue templates (#42).
- The GPU kit says why a step failed where you look first: the job's own log
  has the step's last 20 lines (before torchrun's failure report) after its
  `FAILED` line, and a failed `probe.py` ends with one line per failed check,
  naming it and its values (#49).
- `setup_env.sh`: the venv's Python is `VENV_PYTHON` (default 3.12). It was
  `PYTHON_VERSION`, which Python's Docker images set to their own full
  version (#49).
- `results/explorer/` is gitignored: its step logs, stack dumps and NCCL logs
  are not redacted (#50).
- Explorer jobs build Triton's kernels in a node-local cache, filled from
  `$WORK_DIR/triton-cache` at the start and saved back at the end, instead of
  Triton's default `~/.triton/cache` in the home directory
  (`TRITON_CACHE_STORE`; a `TRITON_CACHE_DIR` you set wins) (#52).
- The GPU kit's `gpu_tests` step (Explorer 02 and 04, `amd/run_all.sh`, the
  Kaggle runner) runs pytest with `--durations=15`, so the slowest tests and
  their times end its log (#51).
- `AMD_NODES_CONFIRMED` is a setting, read from `explorer/local.env` or the
  environment, instead of a line to edit in `submit.sh`, which left runs
  git_dirty (#51).
- `submit.sh` warns before it submits a job asking the `gpu` partition for
  more than one GPU: that partition's QOS allows one per job in total, so
  such a job sits PENDING with `QOSMaxGRESPerJob`. EXPLORER_FIRST_SESSION.md
  describes that limit and multigpu's access request (#51).

### Fixed

- The `gpubridge` torch.distributed backend's setup hung until its timeout when the
  ranks had created different numbers of groups before their first collective: its
  own groups took torch's names, which count each process's groups, so the ranks
  waited on different store keys. HSDP across torch 2.14.1 and 2.9.1 reproduced
  this on CPU (their `DeviceMesh` creates one group more on 2.9.1). The V100 + MI50
  jobs hung at the same step, likely for the same reason; the fix has not run on
  those GPUs yet. Each world setup now names its groups and transport links with
  a fresh generation, so destroying and recreating a backend group inside a Gloo
  world cannot reuse its old names or store keys. On a torch
  whose private naming function has changed, torch's names are kept, with a warning.
- Point-to-point lifecycle: future callbacks no longer run under the engine
  lock or on lane threads, so a callback waiting on another (or its own) P2P
  Work completes and teardown does not wait for it; idle lanes release
  completed payloads and their threads exit, keeping per-tag sequence order;
  a CPU request on a lane that earlier carried a CUDA tensor uses no stream.
  A batch's Work (`batch_isend_irecv`, `_coalescing_manager`) waits on its
  members rather than on futures behind user callbacks, so a slow callback
  doesn't delay it, a callback can wait on it, and `wait(timeout)` works.
  CPU tests only; the GPU stream paths are unvalidated.
- The runbook's 8+8 steps said the master's tarball holds the two-machine
  training records. Rank 0 writes them, and torchrun numbers the machines by
  address, not by role, so they can be on either machine. The two-machine
  test assumed the master too, and failed when the worker's torchrun started
  first; it now expects the record on exactly one of the two (#74).
- `examples/fsdp2_torch_backend.py --verify` is exact at any world size. It
  averaged each rank's unscaled loss, which divides by the world size, inexact
  unless that is a power of two, so on 3 or 5 ranks (2 NVIDIA GPUs and an AMD
  one, say) the check failed with nothing wrong. Each rank's loss is now scaled
  by the world size and the replay sums the ranks' losses; on CPU it matches bit
  for bit on 2 to 6 ranks, and with `--hsdp 2` on 4 and 6 (#64).
- The stream-ordering checks (`check.py`'s `busy_gpu` and `async` stages,
  `tests/test_gpu.py`) queued matmuls as busy work, and failed or aborted
  where rocBLAS has no kernels for the GPU (gfx906 on the ROCm 7.2 wheel).
  They queue a calibrated spin kernel instead (`gpukit.queue_gpu_delay`); no
  BLAS anywhere in the checks. `check.py --busy-iters` is now
  `--busy-seconds` (#52).
- The AMD driver was recorded as `rocm-smi`'s whole CSV output (`name,
  value "Driver version", "5.14.0-..."`). Records hold the version now, and
  triage reads older ones the same way (#52).
- ROCm runs printed `(null): No such file or directory` once per GPU per
  process: the ROCm wheel's bundled libdrm couldn't find its GPU-name table.
  The kit now points `AMDGPU_ASIC_ID_TABLE_PATHS` at the copy the wheel ships
  (#51).
- Bridge compression: a finite value near float32's maximum could arrive as
  inf, and leave a residual of -inf for the next step. In `fp8` a block above
  448 · 2^119 got the scale 2^120, so a code could decode to 2^128; in `bf16`
  a value from about 3.3961e38 (halfway past bfloat16's largest) rounded to
  inf. Both now saturate: `fp8` caps the scale exponent at 119 and its codes
  at 448, `bf16` its codes at bfloat16's largest, and a finite float64 beyond
  float32's range, or a value its residual pushes past it, counts as
  float32's largest. Every finite value is now sent as a finite float32 and
  leaves a finite residual. Payloads change only for blocks holding such
  values (#50).
- The Triton compression kernels are compiled without multiply-add
  contraction (`enable_fp_fusion=False`), so they round every product as the
  PyTorch reference does, whatever the compiler. On the V100, a fused
  `v - decoded * scale` gave a finite residual where the reference's was
  -inf (#50).
- `probe.py` took a CUDA GPU's build target from `gcnArchName`, which CUDA
  builds of PyTorch fill with the product name, so its build check could fail
  for a GPU the build covers. The target is `gcnArchName` without its feature
  flags on ROCm, and `sm_XY` from the compute capability on CUDA. A GPU counts
  as covered by its own target, a cubin for an earlier minor of its major, or
  PTX at or below its capability; `probe.json` records which in
  `target_match` (#49).
- `submit.sh setup` printed `<job id>` in its log's path. Every submission now
  prints its job id and log (#49).
- Step 01's probe had no limit of its own, and the kit's `STEP_TIMEOUT` (20
  minutes) was longer than the job's 10-minute Slurm limit. `submit.sh` now
  caps each job's `STEP_TIMEOUT` at its Slurm limit less `STEP_MARGIN` (120 s),
  and the probe runs under it (#49).
- `triage.py` couldn't see a job that stopped before or during its first step.
  Explorer steps 01 to 07 now write `job.jsonl` (their start, an error that
  stopped them, their end and exit code), and triage lists such a job as
  unfinished, with its error or exit code (#49).

## [0.4.0-alpha] - 2026-10-13

CPU-validated. New in this alpha: gpubridge as an experimental
`torch.distributed` backend (DDP, FSDP2 and HSDP through subgroups), opt-in
bridge compression and an opt-in UCX bridge transport. None of it has run on a
GPU or on RDMA hardware. Not on PyPI.

### Added

- Subgroups on the `torch.distributed` backend, **experimental, tested in
  simulation only**: `new_group` and multi-dimensional `DeviceMesh` groups get a
  topology of their own over their members, built at their first collective
  (native-only for one vendor, bridged for mixed members). HSDP with shard
  groups set to the vendor islands keeps all-gathers and reduce-scatters inside
  each vendor and sends one replicate all_reduce per step over the bridge.
  `examples/fsdp2_torch_backend.py --hsdp ISLANDS`. Design:
  docs/TORCH_BACKEND.md, "Subgroups and HSDP". Subgroup setup patches torch's
  private `distributed_c10d._process_group_name`. On a torch where that
  function or `dist.new_group`'s use of it has changed, creating a subgroup
  fails with an error naming the torch version, so the job doesn't risk a
  hang (#39, #40).

- gpubridge as a `torch.distributed` backend,
  `dist.init_process_group(backend="gpubridge")`, **experimental and not
  validated on GPUs.** DDP and FSDP2 scripts run their collectives through
  gpubridge across vendor islands with the backend name and `import
  gpubridge`, which registers it. On torch 2.14 and later, a lazy
  `torch.distributed.backends` entry point registers it without the import;
  nothing hooks `import torch`. gpubridge's groups are built at the first
  collective, with the usual discovery checks. `GPUBRIDGE_POLICY` and
  `GPUBRIDGE_BRIDGE` pick the policy and transport. CPU tensors in GPU jobs go
  to a Gloo group. Point-to-point, all-to-all, reduce,
  GPU-tensor gather and scatter, and other reductions are refused before any
  communication. New examples `ddp_torch_backend.py` and
  `fsdp2_torch_backend.py` (with a `--verify` mode); the GPU kit's
  `train_fsdp2_backend` and `verify_fsdp2_backend` steps. Design:
  docs/TORCH_BACKEND.md (#38).

- Opt-in bridge compression, `GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8`, **off by
  default, changes results (no longer bit-exact against the uncompressed sum)
  and not validated on GPUs.** The bridged policies (default, pipelined,
  sharded) compress the bridge step of floating-point SUM and AVG, over Gloo
  or UCX. Island steps and every other op and dtype stay exact. Leaders
  quantize on the GPU with Triton kernels (one source for CUDA and ROCm;
  PyTorch fallback without Triton), exchange the payloads, then dequantize and
  add them in float32. `fp8` is OCP E4M3 with a power-of-two scale per 256
  values, travelling as uint8 bytes plus a float32 scale tensor. Every rank
  still ends with the same bits, whichever GPU ran the kernels. Error feedback
  per buffer carries each step's rounding error into the next. New phases
  `compress` and `decompress` keep kernel time out of `bridge`. The GPU kit's
  `bench_compression.py` times the kernels against memory bandwidth, and
  `bench_all_reduce.py --compression` compares bridge time with and without
  compression. Design and error bounds: docs/COMPRESSION.md (#37).

- An opt-in UCX bridge transport, `init(bridge="ucx")`, **not validated on
  GPUs or RDMA hardware** (tested on CPU over UCX's TCP transport only); the
  Gloo bridge stays the default. It carries the leaders' bridge over
  InfiniBand or RoCE where UCX finds them. Leaders hand GPU tensors to UCX
  directly where its build has their memory type, and flush the NIC's writes
  into the GPU after every receive, since UCX doesn't. Elsewhere, GPU tensors
  are staged through pinned host memory. It is a small C++ library over UCX,
  built on first use (`python -m gpubridge.ucx build`). Design and open
  questions: docs/RDMA_TRANSPORT.md (#35).

### Fixed

- The two-node job tests (Explorer steps 05 and 07) no longer hang on machines
  whose own name doesn't resolve, such as macOS, where `socket.getfqdn()` can
  return the reverse name of `::1`. The test harness's fake `srun` gives each
  pretend node torchrun's `--local-addr=127.0.0.1`; the job scripts are
  unchanged (#36).

## [0.3.0-alpha] - 2026-10-12

CPU-validated, with the GPU validation kit ready for a first GPU run. Not on
PyPI.

### Breaking

- Requires PyTorch 2.8 or later. Tearing down a Gloo group could deadlock on
  older releases; the fix landed upstream in 2.8 (#22).
- `init()` refuses a layout that puts two ranks of one island on the same GPU,
  on every rank and with the same message, since NCCL and RCCL refuse it when
  the island first communicates. A rank whose GPU reports neither a UUID nor a
  PCI address leaves its island unchecked, with a warning from rank 0 (#25,
  #29).
- `Work.is_completed()` turns true only once the collective's GPU work is
  done, as with NCCL, instead of as soon as the worker thread had queued it
  (#24).

### Added

- `gpubridge.in_flight()`: the collective each thread is inside, with its
  phase, for debugging hangs (#21).
- `GPUBRIDGE_PINNED_STAGING=1` (opt-in, off by default, **not validated on
  GPUs**): the default policy's leaders stage the bridge step through one
  pinned host buffer instead of pageable `.cpu()` copies, ordered with CUDA
  events. Every rank must agree (#32).
- GPU-only correctness tests, `pytest -m gpu tests/test_gpu.py`: every policy
  against `flat-gloo` bit for bit, every op and dtype, stream ordering,
  pinned buffer reuse, `is_completed()` (#24, #32).
- A release workflow: every pull request and version tag builds the sdist and
  wheel and checks them with twine, and a tag must match the package version.
  Publishing to PyPI (trusted publishing) stays off until enabled (#34).

### Fixed

- The pipelined policy could reuse a pinned host slot in the next collective
  before the previous collective's copy back out of it had run. It now keeps
  each slot's last copy-back event across collectives (#24).

### GPU validation kit

- Every job script runs end to end on CPU against a fake Slurm cluster, with
  hard time limits and the last NCCL/RCCL log lines on failure (#23); those
  tests run in the Python 3.12 CI job only (#26).
- A ladder of runs from one GPU to a real mixed job, a checked split test on
  one NVIDIA node, and an Explorer AMD-node job (#25).
- A network preflight before multi-node runs: every routed pair of interfaces
  tested both ways, raw TCP bandwidth (one stream, a lower bound), and a
  `GLOO_SOCKET_IFNAME` chosen per node (#27, #29).
- `bench_all_reduce.py` times with CUDA events and puts native NCCL/RCCL,
  gpubridge and `flat-gloo` side by side; every row names its baseline (#28,
  #29).
- Step 02's deliberate failing run passes only on the duplicate-GPU error,
  within its own time limit (#29).
- Every record carries the versions, driver, GPU model, commit and interfaces
  of the run, with an allowlist of environment variables and no paths or
  user names; `validation_tables.py` builds the HARDWARE_VALIDATION.md tables,
  safe to publish, including predicted against measured bridge time (#30,
  #31).
- `check.py`'s `in_flight` stage orders its ranks through the store instead of
  sleeping (#31).
- `check_pinned` and `bench_pinned` compare pinned staging with pageable
  copies in the same allocation (#32).

## [0.2.0-alpha] - 2026-10-10

CPU-validated, opt-in policies not validated on GPUs.

### Added

- `ReduceOp.AVG`, `MAX` and `MIN` for `all_reduce`, plus
  `all_gather_into_tensor`, `reduce_scatter_tensor` and `async_op=True` on
  every collective (#15).
- `gpubridge.ddp_comm_hook` for `DistributedDataParallel` (comm hook only;
  **not validated on GPUs**) (#19).
- Opt-in collective policies, never chosen by `auto` (**not validated on
  GPUs**): `pipelined-reduce-bridge-broadcast` (#17), `sharded-bridge` (#18)
  and `auto-tuned` (#17).
- `BridgeTransport.create_groups` for policies with several bridge links (#18).
- Collective observers, `gpubridge.observe`: a record of every collective and
  the phases its policy marks (#9).
- GPU validation kit for Slurm clusters: setup on compute nodes, a
  `PROJECT_DIR` setting, a configurable ROCm build, per-island NCCL/RCCL
  baselines and a mixed heterogeneous-job path (#11, #12).
- Docs: `docs/ROADMAP.md` (bridge-only compression, an RDMA bridge transport,
  a `torch.distributed` backend for FSDP2; design only) and the 2026
  mixed-vendor systems in `docs/PRIOR_ART.md` (#10, #16, #20).

### Changed

- The default policy's island step is a reduce onto the island leader instead
  of an island all_reduce (#14).

### Fixed

- `init()` fails on every rank, with the same message, when `LOCAL_RANK` names
  a GPU the process can't see, instead of hanging (#13).

## [0.1.0-alpha] - 2026-10-07

CPU-validated, GPU validation pending.

### Added

- `gpubridge.init()`, `all_reduce` (SUM), `broadcast` and `barrier` over one
  job that mixes NVIDIA and AMD ranks: an NCCL or RCCL island per vendor,
  island leaders joined by a CPU Gloo bridge.
- Collective policies `auto`, `reduce-bridge-broadcast`, `native-only` and
  `flat-gloo`, with hooks for size-based selection, and an up-front dtype
  check (#7).
- A swappable bridge transport, Gloo by default; discovery shares every rank's
  probe results, so a bad setup fails `init()` on every rank (#3).
- Simulation mode (`GPUBRIDGE_VENDOR`), CPU-only mode, and a test-only
  split-test mode flagged in every log and result (#5).
- A GPU validation kit in `scripts/gpu` with a runbook (#5).
- A CPU-only matrix showing CUDA and ROCm builds of PyTorch can share one job
  (#1, #4), and prior-art notes (#2, #6).

[Unreleased]: https://github.com/DavidCzartoryski/gpubridge/compare/v0.4.0-alpha...main
[0.4.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.4.0-alpha
[0.3.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.3.0-alpha
[0.2.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.1.0-alpha
