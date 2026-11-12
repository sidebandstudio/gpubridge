# Benchmarks

What gpubridge costs, measured on GPUs. Every number here comes from the GPU kit's
records on the Explorer cluster on 2026-10-14, at commit 934f87c unless
a row or section says otherwise; [the verbs bridge on one node](#the-verbs-bridge-on-one-node-split-test)
is from 2026-10-20. Host names, user names and paths are left out; the raw
records keep them.

**"The default bridge" in these measurements is the Gloo bridge.** They predate
2026-11-11, when the default became `bridge="auto"` (a Rust bridge: verbs or
TCP) and Gloo became opt-in. Nothing here was measured on the new default; the
[multitcp rows](#a-multi-stream-tcp-bridge-prototype) are the TCP bridge it runs
on without RDMA, measured by name beside a Gloo default group.

The plan was a comparison of NVIDIA with NVIDIA, AMD with AMD and NVIDIA with AMD.
So far only the NVIDIA + AMD half has run: a pair of two GPUs of one type, in one
partition, waits for Research Computing to confirm it is within the partitions'
intended use (see [Waiting on Research Computing](#waiting-on-research-computing)).

| Run | What it measures | Status |
| --- | --- | --- |
| Network check | InfiniBand/RoCE on each node type; the transport NCCL and RCCL choose | ran, one V100 node and one MI50 node |
| RDMA check | what RDMA from GPU memory would need on each node type: peer-memory modules, dma-buf, UCX | ran, one V100 node and one MI50 node: no RDMA path between the nodes checked |
| A1, A2 | two V100s, native NCCL: as NCCL chooses, then TCP on the bridge's interface | waiting on Research Computing |
| B1, B2 | two MI50s, native RCCL: the same two | waiting on Research Computing |
| C, C-amd | two V100s (two MI50s) through gpubridge's bridge, one island each: gpubridge's own cost without vendor mixing | waiting on Research Computing |
| D | a V100 and an MI50 through gpubridge: every policy | ran |
| E | the same two ranks as one flat Gloo group | ran |
| Verbs, one node | four V100s as two 2-GPU islands (split test): the verbs bridge against the Gloo bridge, through one HCA in loopback | ran 2026-10-20: [below](#the-verbs-bridge-on-one-node-split-test) |
| Training | DDP with `gpubridge.ddp_comm_hook` on GPT-2 small and ResNet-50: each GPU alone, then both | ran: the V100 alone, the MI50 alone (torch 2.9.1+rocm6.3) and both together, uncompressed and with fp8 on the bridge (GPT-2 small at batches 4 to 32) |
| Data-parallel training | `train_dp.py` on a LLaMA-style model: DDP uncompressed, bf16 and fp8, FSDP2, shares by speed, bucket sizes; one V100 alone | ran: [below](#data-parallel-training-on-a-llama-style-model) |
| One model, every way | the same LLaMA-style model on one V100, one MI50, a V100 + MI50 pipeline and V100 + MI50 data parallelism, twice each; the same-vendor pairs | ran, but the pairs wait on Research Computing: [What mixing vendors costs](#what-mixing-vendors-costs-and-whether-it-helps) |

## What mixing vendors costs, and whether it helps

One model, data and seed for every run: `llama.py`'s `llama-125m` (134 M
parameters), 8 sequences x 512 tokens a step, AdamW 3e-4, seed 0, 500 steps.
Each setting ran twice, in separate Explorer jobs, at commit 1f372b1 (clean on
every rank):

- **GPUs and software.** The V100 (Tesla V100-SXM2-32GB) ran torch
  2.9.1+cu126 with NCCL 2.27.5, driver 545.23.08. The MI50 ran torch
  2.9.1+rocm6.3 with RCCL 2.21.5, under the amdgpu driver in the node's kernel
  (5.14.0-611, RHEL 9.7); for fp8 it had Triton 3.8.0 in front
  (`venv-rocm63-triton38`), so the compression kernels ran as Triton on both
  GPUs.
- **Network.** The bridge ran over TCP on `internal.2137`; the preflights
  measured 1151 to 1174 MB/s each way, 0.05 ms round trip.
- **Runs.** The pipeline is 1F1B with 4 microbatches ([PIPELINE.md](PIPELINE.md#automatic-stage-balancing)),
  both with the automatic split and with the best hand-tuned one.
  Data parallelism is DDP through `gpubridge.ddp_comm_hook`, 4 + 4 sequences,
  25 MB buckets ([below](#data-parallel-training-on-a-llama-style-model)).
- **Jobs:**

  | Setting | Run 1 | Run 2 |
  | --- | --- | --- |
  | V100 alone | 11115074 | 11116181 |
  | MI50 alone | 11115230 | 11116401 |
  | Pipeline | 11115487 | 11116512 |
  | Data parallel | 11115759 | 11116801 |

| Setting | Tokens/s, run 1 | Tokens/s, run 2 | Mean | Spread | Against the V100 alone | Step (ms, mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) one V100 alone | 14,197 | 14,331 | 14,264 | 0.9% | 1.00x | 287.2 |
| (b) one MI50 alone | 9,613 | 9,615 | 9,614 | <0.1% | 0.67x | 426.0 |
| (c) V100 + MI50, pipeline, automatic split (8 + 4) | 15,497 | 15,135 | 15,316 | 2.4% | 1.07x | 267.5 |
| (c) V100 + MI50, pipeline, best hand-tuned split (9 + 3) | 15,882 | 15,490 | 15,686 | 2.5% | 1.10x | 261.2 |
| (d) V100 + MI50, data parallel, uncompressed | 4,666 | 4,419 | 4,543 | 5.4% | 0.32x | 902.4 |
| (d) V100 + MI50, data parallel, fp8 on the bridge | 9,663 | 9,714 | 9,689 | 0.5% | 0.68x | 422.8 |
| (e) V100 + V100, pipeline and data parallel, native NCCL | - | - | - | - | - | waiting on Research Computing |
| (f) MI50 + MI50, the same with RCCL | - | - | - | - | - | waiting on Research Computing |

Where each step goes (means of the two runs, ms; "est." marks estimates from
each GPU's compute timed alone):

| Setting | Compute | Waiting on the slower GPU (est.) | Exposed communication (est.) |
| --- | --- | --- | --- |
| (a), (b) | the whole step | - | - |
| (c) automatic, 8 + 4 | 43.7 a microbatch on the V100, 46.6 on the MI50 | the V100 idles 92.8 of each step, the MI50 81.1 | one microbatch's activations arrive in 7.1 |
| (c) hand-tuned, 9 + 3 | 53.3 on the V100, 38.8 on the MI50 | the V100 idles 47.9, the MI50 105.9 | 8.1 a hop |
| (d) uncompressed | 231.2 on the MI50, the slower | the V100 waits 76.8 | 671.2 of the bridge's 835.6 alone; 20% hidden behind backward |
| (d) fp8 | 234.0 on the MI50 | the V100 waits 80.9 | 188.8 of the bridge's 370.1 alone; 48% hidden |

**Answers:**

- **Does adding the MI50 to the V100 beat the V100 alone?** Through the
  pipeline, yes: 1.10x with the best hand-tuned split and 1.07x with the
  automatic one, each run within 2.5% of the other. Through data parallelism,
  no: 0.32x uncompressed and 0.68x with fp8. Every step all_reduces 536 MB of
  gradients over a link of about 1.17 GB/s, against 3 MB of activations a
  microbatch for the pipeline.
- **How close to the two GPUs' sum?** The MI50 alone does 0.67x the V100's
  work, so the two alone add up to 23,878 tokens/s. The best pipeline reaches
  66% of that; data parallelism with fp8 reaches 41%.
- **What share of a same-vendor pair's throughput does the mixed pair
  reach?** Not measured yet. Explorer's same-partition GPU pairs wait on
  Research Computing ([below](#waiting-on-research-computing)). The jobs are
  ready, refused until `SAME_PARTITION_PAIRS_OK=1`:
  `STEPS_07=dp DP_RUNS=ddp` and `STEPS_07=pipeline PP_RUNS=1f1b` with
  `submit.sh 07-nvidia-pair` (two V100s) or `07-amd-pair` (two MI50s). Each
  runs one island of two GPUs, so DDP's all_reduce and the pipeline's hops are
  native NCCL or RCCL, from the same model, data and seed.

**Reading guide.** Tokens/s is the whole step's tokens (8 x 512) over the
median step after the first 5. "Against the V100 alone" divides means. The
pipeline's idle times and data parallelism's waiting and exposed times are
estimates from each GPU's compute timed alone, as their records say.
Everything else is measured. The unflattering results (data parallelism
below the V100 alone; the automatic split below the best hand-tuned one) are
reported the same way as the others.

## Setup

| | NVIDIA side | AMD side |
| --- | --- | --- |
| GPU | Tesla V100-SXM2-32GB (sm_70) | AMD Instinct MI50, 32 GB (gfx906) |
| Partition | `gpu` | `sharing` |
| Driver | 545.23.08 | 5.14.0-611.5.1.el9_7 (the kernel's amdgpu, as the records report it) |
| OS kernel | 5.14.0-362.13.1.el9_3 | 5.14.0-611.5.1.el9_7 |
| Python | 3.12.15 | 3.12.15 |
| torch | 2.14.1+cu126 | 2.14.1+rocm7.2 (all_reduce matrix); 2.9.1+rocm6.3 (training) |
| CUDA / HIP | CUDA 12.6 | HIP 7.2.53211; HIP 6.3.42134 with 2.9.1+rocm6.3 |
| NCCL / RCCL | NCCL 2.29.3 | RCCL 2.27.7; RCCL 2.21.5 with 2.9.1+rocm6.3 |
| gpubridge | 0.4.0a0, commit 934f87c, clean | the same |

One GPU and one rank on each side, as one heterogeneous Slurm job across the two
partitions (`submit.sh 08`). Every MI50 job today ran on the same node, and
step 08, the V100-alone training run and the mixed training run shared one V100
node.

## Network

From `submit.sh net-nvidia` and `net-amd` (one short job on each node type:
`ibv_devinfo`, `ibstat`, `/sys/class/infiniband`, `ip -br addr`, and NCCL or
RCCL initialized alone with `NCCL_DEBUG=INFO`, as it chooses and then with
`NCCL_IB_DISABLE=1`); step 08 recorded the same on its own two nodes.

| Node | RDMA ports (link layer, state, rate) | NCCL/RCCL as it chooses | With `NCCL_IB_DISABLE=1` |
| --- | --- | --- | --- |
| V100 (`gpu`) | `mlx5_0`: InfiniBand, active, 100 Gb/s | NET/IB on `mlx5_0` (InfiniBand); bootstrap on `internal` | NET/Socket on `internal` and `internal.2137` |
| MI50 (`sharing`) | `mlx5_0`: Ethernet (RoCE), active, 10 Gb/s; `mlx5_1`: Ethernet, down, 40 Gb/s; `mlx5_2`: InfiniBand, down | NET/IB on `mlx5_0` over RoCE; bootstrap on `ibs8` | NET/Socket on `ibs8` |

gpubridge's bridge is Gloo over TCP. Its preflight chose `internal.2137` (a VLAN
on `internal`) on both nodes and measured, with one TCP stream from Python,
1173.7 MB/s from the V100 to the MI50, 1172.1 MB/s back, and a 0.048 ms round
trip. Step 07's two earlier jobs measured the same bandwidth within 2.5%.

What this means for the comparison:

- **A1 would not use the bridge's wire.** Between two V100 nodes, NCCL picks
  InfiniBand at 100 Gb/s; the bridge's one TCP stream measured about 9.4 Gb/s.
  A1 against D would mostly measure the network, which is why A2 (NCCL forced
  onto TCP on the bridge's interface) is the baseline for gpubridge's cost.
- **B1 would use RDMA over the MI50 node's 10 Gb/s RoCE port**, also not the
  bridge's wire.
- **RCCL's own TCP choice on the MI50 node is `ibs8`, which `ip` reports as
  down** (it has an address). A TCP baseline there must name the interface:
  step 08's A2 and B2 set `NCCL_SOCKET_IFNAME` to exactly the preflight's
  interface.
- In the D runs each island has one rank, so NCCL and RCCL never cross the
  network; their lines above say what they would choose, not a transport in use.
- **No RDMA path between the V100 and MI50 nodes checked**; see
  [RDMA between the node types](#rdma-between-the-node-types).

### RDMA between the node types

A second run of `net-nvidia` and `net-amd` (commit 1ac6fa9, which adds
`rdma_report.sh` to the network check; the steps before it are unchanged from
934f87c) recorded what RDMA from GPU memory would need on each node: the kernel's
dma-buf and peer-to-peer options, the RDMA and GPU kernel modules, rdma-core,
`rdma link`, the HSA runtime's dma-buf export, and UCX, with the cluster's
OpenMPI 4.1.6 module loaded for its `ucx_info` (`RDMA_MODULES`). One node of
each type: a different V100 node from the first network check's, and the same
MI50 node. The V100 node's ports matched the first one's. Step 00's records list
two MI50 nodes in `sharing`, one of them drained; every MI50 job so far ran on
the other.

| | V100 node (`gpu`) | MI50 node (`sharing`) |
| --- | --- | --- |
| RDMA devices (`rdma link`) | `mlx5_0`: InfiniBand, active | `mlx5_0`: Ethernet (RoCE) on `internal`, active, 10 Gb/s; `mlx5_1`: disabled; `mlx5_2`: InfiniBand, disabled |
| The bridge's interface (`internal`, VLAN `internal.2137`) | no RDMA device behind it | `mlx5_0`, RoCE |
| Kernel | 5.14.0-362.13.1.el9_3 | 5.14.0-611.5.1.el9_7 |
| dma-buf and peer-to-peer in the kernel | `DMABUF_MOVE_NOTIFY`, `PCI_P2PDMA` | the same, and `HSA_AMD_P2P` |
| rdma-core | 46.0; libibverbs has `ibv_reg_dmabuf_mr` | 57.0; the same |
| GPU memory for the NIC | `nvidia_peermem` (545.23.08) installed for this kernel, not loaded; the NVIDIA kernel module is the proprietary one (license `NVIDIA`) | no `amdp2p` module; torch's HSA runtime (2.14.1+rocm7.2) has `hsa_amd_portable_export_dmabuf` |
| UCX | 1.16.0, in the OpenMPI 4.1.6 module: `ucx_info -d` found `self`, `tcp`, `sysv`, `posix`, `cma`, all host memory only | the same build, the same transports; host only |

What this means:

- **No RDMA path between the V100 and MI50 nodes checked** (two V100 nodes,
  counting the first network check's, and the MI50 node in service). The V100
  nodes' only RDMA device is on InfiniBand; the MI50 node's only active one is
  on Ethernet, and its InfiniBand port is disabled. An InfiniBand port can't
  reach a RoCE port, and RoCE needs an RDMA device at both ends: the V100 node
  reaches the MI50 node's Ethernet network through `internal`, which has none.
  So on these nodes the UCX transport's RDMA and GPUDirect paths (items 26 and
  27 in [HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md)) can't be
  validated; the bridge stays on TCP. Other nodes, or a change to their ports,
  could differ.
- **The OpenMPI module's UCX can't do RDMA or GPU memory, even between nodes
  of one type.** At run time it offered only TCP and shared-memory transports,
  host memory only, and its install has one transport module, `cma`: no `ib`,
  `rdmacm`, `cuda` or `rocm` (`lib/ucx` in the module, listed from the login
  node). It is the only UCX module in `module avail`, and the system has no
  UCX package. Some application modules bundle their own UCX: AMBER's has only
  `cma` too, while LAMMPS's and OpenFOAM's have `ib` and `rdmacm`, still with
  no `cuda` or `rocm`. A UCX like those could carry host-staged RDMA between
  nodes on one fabric (two V100 nodes), never GPU memory; none was tried, and
  their versions weren't checked. GPU memory over RDMA on Explorer would need
  a UCX built with CUDA or ROCm.
- **GPU side, for other hardware.** On the V100 nodes GPUDirect RDMA would
  need `nvidia_peermem` loaded, which it isn't; NVIDIA's dma-buf path needs the
  open kernel modules, which these nodes don't run. The MI50 node has the
  pieces dma-buf RDMA from AMD GPU memory would use (the kernel option, the
  runtime's export call, `ibv_reg_dmabuf_mr`); neither exporting a GPU buffer
  nor registering one with the NIC was tried.
- Not recorded: `ulimit -l` in jobs and `perftest`. Neither changes the
  answer above.

## The all_reduce matrix: a V100 and an MI50

`submit.sh 08` (`scripts/gpu/explorer/08_bench_matrix.sbatch`), then
`scripts/gpu/bench_matrix.py` over its results. Two ranks, one GPU each on two
nodes; float32 all_reduce from 1 KB to 1 GB in steps of 4x; 20 timed trials per
size; the median latency of the slower rank, timed with CUDA events on each GPU.
Bus bandwidth as nccl-tests computes it, which with 2 ranks is bytes / time. All
11 runs were in one job (all 15 steps passed, in 12 minutes), so they share the
two nodes and the network.

| Run | Policy |
| --- | --- |
| D-default | `reduce-bridge-broadcast`, pageable host staging |
| D-pinned | the same with `GPUBRIDGE_PINNED_STAGING=1` |
| D-pipelined-1M to -64M | `pipelined-reduce-bridge-broadcast`, chunks of 1, 4, 16 and 64 MB |
| D-sharded | `sharded-bridge`; with one GPU per island it has one link (k = 1), so it runs the default's steps |
| D-bf16, D-fp8 | the default with `GPUBRIDGE_BRIDGE_COMPRESSION=bf16` or `fp8` |
| D-default-repeat | D-default again, last, to show the spread between two runs |
| E | one flat Gloo group of the same two ranks, no gpubridge |

Latency at five sizes, and speed at 1 GB against D-default (above 1 is faster):

| Run | 1 KB | 64 KB | 1 MB | 16 MB | 1 GB | Bus bandwidth at 1 GB (GB/s) | Speed-up over D-default at 1 GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-default | 731 us | 906 us | 2.7 ms | 32.3 ms | 1.67 s | 0.64 | 1.00x |
| D-pinned | 701 us | 1.12 ms | 3.67 ms | 27.2 ms | 1.21 s | 0.89 | 1.39x |
| D-pipelined-1M | 784 us | 1.14 ms | 3.27 ms | 26.2 ms | 1.66 s | 0.65 | 1.01x |
| D-pipelined-4M | 761 us | 895 us | 3.84 ms | 27.9 ms | 2.00 s | 0.54 | 0.84x |
| D-pipelined-16M | 836 us | 955 us | 3.22 ms | 26.6 ms | 1.07 s | 1.00 | 1.56x |
| D-pipelined-64M | 781 us | 1.22 ms | 3.19 ms | 29.3 ms | 1.22 s | 0.88 | 1.37x |
| D-sharded | 671 us | 1.22 ms | 3.93 ms | 29.8 ms | 1.58 s | 0.68 | 1.06x |
| D-bf16 | 921 us | 1.05 ms | 2.96 ms | 26 ms | 1.23 s | 0.88 | 1.37x |
| D-fp8 | 919 us | 952 us | 1.97 ms | 12 ms | 609 ms | 1.76 | 2.75x |
| D-default-repeat | 648 us | 1.01 ms | 4.35 ms | 34.6 ms | 1.60 s | 0.67 | 1.05x |
| E | 424 us | 562 us | 2.81 ms | 29.9 ms | 1.64 s | 0.66 | 1.02x |

What the records show:

- **With one GPU per vendor, the default bridge costs about what one flat Gloo
  group does from 1 MB up**: 1.67 s and 1.60 s (the repeat) at 1 GB, against
  1.64 s for E, all near 0.65 GB/s. With one rank per island there is nothing
  for NCCL or RCCL to reduce, so the bridge is Gloo over one TCP connection, as
  E is. From 1 KB to 16 KB the gpubridge runs took 650 to 930 us against 390
  to 500 us for E: the island phases add about 0.15 ms together, and the bridge
  phase alone took 0.36 to 0.64 ms.
- **Pinned staging is 1.3 to 1.4x faster than either default run from 64 MB
  up** (1.21 s at 1 GB, 0.89 GB/s; 304 ms against 417 and 389 ms at 256 MB; 77 ms
  against 104 and 100 ms at 64 MB).
  Step 07's earlier job measured 1.28 s at 1 GB with pinned staging.
- **The pipelined bridge with 16 MB chunks was the fastest uncompressed run**
  at 64 MB, 256 MB and 1 GB (1.07 s at 1 GB, 1.00 GB/s, 1.56x faster than the
  default). The other chunk sizes don't order consistently: 4 MB was the slowest
  run at 1 GB (2.00 s) but faster than the default at 256 MB. One run each, so
  the chunk size's effect is not settled beyond 16 MB being best here.
- **The sharded bridge with one link is the default** (1.58 s at 1 GB), as
  expected with k = 1.
- **fp8 compression was 2.75x faster than the default at 1 GB** (609 ms, an
  effective 1.76 GB/s), of which compressing and decompressing took 6.1 and
  4.4 ms. **bf16 was 1.37x faster** (1.23 s), not 2x: with two islands each
  leader's payload goes in turn, one direction at a time, so halving the bytes
  gives the same predicted time as the uncompressed ring (916 ms at 1 GB). bf16
  took 1.33x that prediction, the default 1.83x of the same number.
- **Two runs of one configuration differed by 23% and 61% at 4 MB and 1 MB**,
  by 3 to 15% below that, and by 4 to 7% from 16 MB up (table below). A
  difference between two runs in the middle sizes smaller than that says
  nothing.

| Size | D-default | D-default-repeat | Difference |
| --- | ---: | ---: | ---: |
| 1 KB | 731 us | 648 us | -11% |
| 4 KB | 755 us | 850 us | +13% |
| 16 KB | 812 us | 930 us | +15% |
| 64 KB | 906 us | 1.01 ms | +11% |
| 256 KB | 1.57 ms | 1.52 ms | -3% |
| 1 MB | 2.7 ms | 4.35 ms | +61% |
| 4 MB | 11.7 ms | 14.4 ms | +23% |
| 16 MB | 32.3 ms | 34.6 ms | +7% |
| 64 MB | 104 ms | 99.6 ms | -4% |
| 256 MB | 417 ms | 389 ms | -7% |
| 1 GB | 1.67 s | 1.60 s | -4% |

### Predicted against measured

The prediction is the job's preflight (one TCP stream: its slower direction and
the round trip) applied to a ring all_reduce, the whole tensor each way at once;
with compression, each leader's payload broadcast in turn. The measured bridge
phase also includes the copies between GPU and host, which the prediction
doesn't. At 1 GB the bridge took 1.17x (pipelined, 16 MB chunks) to 2.18x
(pipelined, 4 MB) the prediction, the default 1.74x and 1.83x. From 1 KB to
16 KB it took 6 to 12x: 0.36 to 0.64 ms per call against a 0.05 ms round trip.

| Run | Size | Predicted bridge (ms) | Measured bridge (ms) | Measured / predicted |
| --- | --- | ---: | ---: | ---: |
| D-default | 1 MB | 0.943 | 2.4008 | 2.55x |
| D-default | 1 GB | 916.132 | 1671.96 | 1.83x |
| D-pinned | 1 MB | 0.943 | 3.474 | 3.69x |
| D-pinned | 1 GB | 916.132 | 1204.97 | 1.32x |
| D-pipelined-1M | 1 MB | 0.943 | 2.7342 | 2.90x |
| D-pipelined-1M | 1 GB | 916.132 | 1658.01 | 1.81x |
| D-pipelined-4M | 1 MB | 0.943 | 3.6174 | 3.84x |
| D-pipelined-4M | 1 GB | 916.132 | 1997.72 | 2.18x |
| D-pipelined-16M | 1 MB | 0.943 | 2.8851 | 3.06x |
| D-pipelined-16M | 1 GB | 916.132 | 1072.77 | 1.17x |
| D-pipelined-64M | 1 MB | 0.943 | 2.9686 | 3.15x |
| D-pipelined-64M | 1 GB | 916.132 | 1220.44 | 1.33x |
| D-sharded | 1 MB | 0.943 | 3.6382 | 3.86x |
| D-sharded | 1 GB | 916.132 | 1577.45 | 1.72x |
| D-bf16 | 1 MB | 0.943 | 2.4076 | 2.55x |
| D-bf16 | 1 GB | 916.132 | 1217 | 1.33x |
| D-fp8 | 1 MB | 0.502 | 1.3871 | 2.76x |
| D-fp8 | 1 GB | 465.247 | 600.001 | 1.29x |
| D-default-repeat | 1 MB | 0.943 | 3.9748 | 4.22x |
| D-default-repeat | 1 GB | 916.132 | 1596.91 | 1.74x |

### Phases

Median ms of each phase on the slower rank, at 1 MB and 1 GB. The island phases
are NCCL or RCCL on one GPU, so they stay under 0.3 ms, except the sharded
bridge's reduce-scatter and all-gather (3 ms each at 1 GB).

| Run | Size | island-reduce | island-reduce-scatter | bridge | island-broadcast | island-all-gather | compress | decompress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-default | 1 MB | 0.075 | - | 2.4008 | 0.0842 | - | - | - |
| D-default | 1 GB | 0.1657 | - | 1671.96 | 0.1935 | - | - | - |
| D-pinned | 1 MB | 0.075 | - | 3.474 | 0.0242 | - | - | - |
| D-pinned | 1 GB | 0.1121 | - | 1204.97 | 0.0126 | - | - | - |
| D-pipelined-1M | 1 MB | 0.0751 | - | 2.7342 | 0.0495 | - | - | - |
| D-pipelined-1M | 1 GB | 0.106 | - | 1658.01 | 0.2616 | - | - | - |
| D-pipelined-4M | 1 MB | 0.0758 | - | 3.6174 | 0.063 | - | - | - |
| D-pipelined-4M | 1 GB | 0.1288 | - | 1997.72 | 0.1746 | - | - | - |
| D-pipelined-16M | 1 MB | 0.0756 | - | 2.8851 | 0.0609 | - | - | - |
| D-pipelined-16M | 1 GB | 0.1141 | - | 1072.77 | 0.0144 | - | - | - |
| D-pipelined-64M | 1 MB | 0.0701 | - | 2.9686 | 0.0532 | - | - | - |
| D-pipelined-64M | 1 GB | 0.1075 | - | 1220.44 | 0.0144 | - | - | - |
| D-sharded | 1 MB | - | 0.1096 | 3.6382 | - | 0.087 | - | - |
| D-sharded | 1 GB | - | 2.9712 | 1577.45 | - | 3.0598 | - | - |
| D-bf16 | 1 MB | 0.0714 | - | 2.4076 | 0.0829 | - | 0.1147 | 0.0924 |
| D-bf16 | 1 GB | 0.0952 | - | 1217 | 0.165 | - | 5.7661 | 2.8808 |
| D-fp8 | 1 MB | 0.079 | - | 1.3871 | 0.086 | - | 0.1184 | 0.1113 |
| D-fp8 | 1 GB | 0.1386 | - | 600.001 | 0.2202 | - | 6.0975 | 4.3595 |
| D-default-repeat | 1 MB | 0.0786 | - | 3.9748 | 0.1103 | - | - | - |
| D-default-repeat | 1 GB | 0.1429 | - | 1596.91 | 0.2032 | - | - | - |

### How to read the full matrix

`bench_matrix.py` writes this guide with the tables. Until A1, A2, B1, B2 and C
run, the x A2 and x A1 columns are empty.

- **A1 against A2 is the network**: the same NCCL on the same two GPUs in one
  job, on the transport NCCL picks and then over TCP on the bridge's interface.
- **A2 against C is gpubridge's own cost**: the same job, the same two V100s
  and the same interface; C takes gpubridge's path (each GPU an island, the sum
  staged through host memory and all_reduced by Gloo over one connection)
  instead of NCCL's socket transport, which may open several. B2 against C-amd
  is the same on two MI50s.
- **C against D is the vendor mixing, across jobs**: the same bridge with an
  MI50 and RCCL on one side, but on other nodes. A ratio marked † divides by a
  run of another job, so check the two jobs' networks before putting a
  difference down to the vendor.
- **D's variants against D-default** are what each policy buys on this pair,
  and **D-default against E** what gpubridge does better than one flat Gloo
  group.

### Full tables

#### Small: under 256 KB (latency)

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1KB | D-default | 731 us | 0.0014 | - | - | 0.049 | 0.4552 | 9.31x |
| 1KB | D-pinned | 701 us | 0.0015 | - | - | 0.049 | 0.4372 | 8.95x |
| 1KB | D-pipelined-1M | 784 us | 0.0013 | - | - | 0.049 | 0.5174 | 10.59x |
| 1KB | D-pipelined-4M | 761 us | 0.0013 | - | - | 0.049 | 0.4799 | 9.82x |
| 1KB | D-pipelined-16M | 836 us | 0.0012 | - | - | 0.049 | 0.5621 | 11.50x |
| 1KB | D-pipelined-64M | 781 us | 0.0013 | - | - | 0.049 | 0.5233 | 10.71x |
| 1KB | D-sharded | 671 us | 0.0015 | - | - | 0.049 | 0.3552 | 7.27x |
| 1KB | D-bf16 | 921 us | 0.0011 | - | - | 0.049 | 0.3763 | 7.70x |
| 1KB | D-fp8 | 919 us | 0.0011 | - | - | 0.048 | 0.364 | 7.51x |
| 1KB | D-default-repeat | 648 us | 0.0016 | - | - | 0.049 | 0.3855 | 7.89x |
| 1KB | E | 424 us | 0.0024 | - | - | - | - | - |
| 4KB | D-default | 755 us | 0.0054 | - | - | 0.051 | 0.4666 | 9.06x |
| 4KB | D-pinned | 758 us | 0.0054 | - | - | 0.051 | 0.4936 | 9.59x |
| 4KB | D-pipelined-1M | 773 us | 0.0053 | - | - | 0.051 | 0.5156 | 10.01x |
| 4KB | D-pipelined-4M | 807 us | 0.0051 | - | - | 0.051 | 0.5474 | 10.63x |
| 4KB | D-pipelined-16M | 793 us | 0.0052 | - | - | 0.051 | 0.509 | 9.88x |
| 4KB | D-pipelined-64M | 805 us | 0.0051 | - | - | 0.051 | 0.5371 | 10.43x |
| 4KB | D-sharded | 692 us | 0.0059 | - | - | 0.051 | 0.3853 | 7.48x |
| 4KB | D-bf16 | 928 us | 0.0044 | - | - | 0.051 | 0.3989 | 7.75x |
| 4KB | D-fp8 | 923 us | 0.0044 | - | - | 0.05 | 0.361 | 7.25x |
| 4KB | D-default-repeat | 850 us | 0.0048 | - | - | 0.051 | 0.5775 | 11.21x |
| 4KB | E | 391 us | 0.0105 | - | - | - | - | - |
| 16KB | D-default | 812 us | 0.0202 | - | - | 0.062 | 0.5576 | 9.00x |
| 16KB | D-pinned | 912 us | 0.018 | - | - | 0.062 | 0.639 | 10.31x |
| 16KB | D-pipelined-1M | 850 us | 0.0193 | - | - | 0.062 | 0.5558 | 8.97x |
| 16KB | D-pipelined-4M | 831 us | 0.0197 | - | - | 0.062 | 0.5663 | 9.14x |
| 16KB | D-pipelined-16M | 809 us | 0.0203 | - | - | 0.062 | 0.5463 | 8.81x |
| 16KB | D-pipelined-64M | 902 us | 0.0182 | - | - | 0.062 | 0.6295 | 10.16x |
| 16KB | D-sharded | 822 us | 0.0199 | - | - | 0.062 | 0.511 | 8.24x |
| 16KB | D-bf16 | 905 us | 0.0181 | - | - | 0.062 | 0.3881 | 6.26x |
| 16KB | D-fp8 | 923 us | 0.0178 | - | - | 0.055 | 0.3819 | 6.93x |
| 16KB | D-default-repeat | 930 us | 0.0176 | - | - | 0.062 | 0.6226 | 10.05x |
| 16KB | E | 495 us | 0.0331 | - | - | - | - | - |
| 64KB | D-default | 906 us | 0.0723 | - | - | 0.104 | 0.64 | 6.16x |
| 64KB | D-pinned | 1.12 ms | 0.0587 | - | - | 0.104 | 0.8387 | 8.07x |
| 64KB | D-pipelined-1M | 1.14 ms | 0.0574 | - | - | 0.104 | 0.8042 | 7.74x |
| 64KB | D-pipelined-4M | 895 us | 0.0732 | - | - | 0.104 | 0.6369 | 6.13x |
| 64KB | D-pipelined-16M | 955 us | 0.0686 | - | - | 0.104 | 0.6897 | 6.64x |
| 64KB | D-pipelined-64M | 1.22 ms | 0.0538 | - | - | 0.104 | 0.9281 | 8.93x |
| 64KB | D-sharded | 1.22 ms | 0.0535 | - | - | 0.104 | 0.8981 | 8.64x |
| 64KB | D-bf16 | 1.05 ms | 0.0625 | - | - | 0.104 | 0.5325 | 5.12x |
| 64KB | D-fp8 | 952 us | 0.0689 | - | - | 0.076 | 0.4065 | 5.32x |
| 64KB | D-default-repeat | 1.01 ms | 0.0652 | - | - | 0.104 | 0.7301 | 7.03x |
| 64KB | E | 562 us | 0.1166 | - | - | - | - | - |

#### Medium: 256 KB to under 64 MB

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256KB | D-default | 1.57 ms | 0.1667 | - | - | 0.272 | 1.2728 | 4.69x |
| 256KB | D-pinned | 1.68 ms | 0.1559 | - | - | 0.272 | 1.4286 | 5.26x |
| 256KB | D-pipelined-1M | 1.67 ms | 0.1574 | - | - | 0.272 | 1.4126 | 5.20x |
| 256KB | D-pipelined-4M | 1.69 ms | 0.1552 | - | - | 0.272 | 1.2455 | 4.58x |
| 256KB | D-pipelined-16M | 1.28 ms | 0.2046 | - | - | 0.272 | 0.9994 | 3.68x |
| 256KB | D-pipelined-64M | 1.5 ms | 0.1753 | - | - | 0.272 | 1.0957 | 4.03x |
| 256KB | D-sharded | 1.76 ms | 0.1491 | - | - | 0.272 | 1.5129 | 5.57x |
| 256KB | D-bf16 | 1.43 ms | 0.1832 | - | - | 0.272 | 0.8858 | 3.26x |
| 256KB | D-fp8 | 1.28 ms | 0.2053 | - | - | 0.162 | 0.7177 | 4.44x |
| 256KB | D-default-repeat | 1.52 ms | 0.1722 | - | - | 0.272 | 1.2502 | 4.60x |
| 256KB | E | 1.19 ms | 0.2207 | - | - | - | - | - |
| 1MB | D-default | 2.7 ms | 0.3886 | - | - | 0.943 | 2.4008 | 2.55x |
| 1MB | D-pinned | 3.67 ms | 0.2857 | - | - | 0.943 | 3.474 | 3.69x |
| 1MB | D-pipelined-1M | 3.27 ms | 0.3204 | - | - | 0.943 | 2.7342 | 2.90x |
| 1MB | D-pipelined-4M | 3.84 ms | 0.273 | - | - | 0.943 | 3.6174 | 3.84x |
| 1MB | D-pipelined-16M | 3.22 ms | 0.326 | - | - | 0.943 | 2.8851 | 3.06x |
| 1MB | D-pipelined-64M | 3.19 ms | 0.3284 | - | - | 0.943 | 2.9686 | 3.15x |
| 1MB | D-sharded | 3.93 ms | 0.2667 | - | - | 0.943 | 3.6382 | 3.86x |
| 1MB | D-bf16 | 2.96 ms | 0.3547 | - | - | 0.943 | 2.4076 | 2.55x |
| 1MB | D-fp8 | 1.97 ms | 0.5321 | - | - | 0.502 | 1.3871 | 2.76x |
| 1MB | D-default-repeat | 4.35 ms | 0.241 | - | - | 0.943 | 3.9748 | 4.22x |
| 1MB | E | 2.81 ms | 0.3726 | - | - | - | - | - |
| 4MB | D-default | 11.7 ms | 0.3574 | - | - | 3.626 | 10.4676 | 2.89x |
| 4MB | D-pinned | 8.19 ms | 0.512 | - | - | 3.626 | 8.0503 | 2.22x |
| 4MB | D-pipelined-1M | 11 ms | 0.3817 | - | - | 3.626 | 10.7753 | 2.97x |
| 4MB | D-pipelined-4M | 12.3 ms | 0.3415 | - | - | 3.626 | 12.1204 | 3.34x |
| 4MB | D-pipelined-16M | 8.45 ms | 0.4965 | - | - | 3.626 | 7.0456 | 1.94x |
| 4MB | D-pipelined-64M | 8.15 ms | 0.5144 | - | - | 3.626 | 8.0108 | 2.21x |
| 4MB | D-sharded | 9.22 ms | 0.4551 | - | - | 3.626 | 8.849 | 2.44x |
| 4MB | D-bf16 | 7.48 ms | 0.5605 | - | - | 3.626 | 6.7036 | 1.85x |
| 4MB | D-fp8 | 4.44 ms | 0.9454 | - | - | 1.865 | 3.8332 | 2.06x |
| 4MB | D-default-repeat | 14.4 ms | 0.2906 | - | - | 3.626 | 14.2158 | 3.92x |
| 4MB | E | 9.24 ms | 0.454 | - | - | - | - | - |
| 16MB | D-default | 32.3 ms | 0.5192 | - | - | 14.362 | 30.9117 | 2.15x |
| 16MB | D-pinned | 27.2 ms | 0.6174 | - | - | 14.362 | 27.0202 | 1.88x |
| 16MB | D-pipelined-1M | 26.2 ms | 0.6396 | - | - | 14.362 | 25.8242 | 1.80x |
| 16MB | D-pipelined-4M | 27.9 ms | 0.6016 | - | - | 14.362 | 26.7146 | 1.86x |
| 16MB | D-pipelined-16M | 26.6 ms | 0.6318 | - | - | 14.362 | 25.9052 | 1.80x |
| 16MB | D-pipelined-64M | 29.3 ms | 0.5718 | - | - | 14.362 | 28.3955 | 1.98x |
| 16MB | D-sharded | 29.8 ms | 0.5627 | - | - | 14.362 | 28.4848 | 1.98x |
| 16MB | D-bf16 | 26 ms | 0.6441 | - | - | 14.362 | 25.2471 | 1.76x |
| 16MB | D-fp8 | 12 ms | 1.3957 | - | - | 7.317 | 11.1805 | 1.53x |
| 16MB | D-default-repeat | 34.6 ms | 0.4844 | - | - | 14.362 | 33.5045 | 2.33x |
| 16MB | E | 29.9 ms | 0.5618 | - | - | - | - | - |

#### Large: 64 MB and up (bandwidth)

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64MB | D-default | 104 ms | 0.6459 | - | - | 57.303 | 103.312 | 1.80x |
| 64MB | D-pinned | 76.7 ms | 0.8745 | - | - | 57.303 | 76.5565 | 1.34x |
| 64MB | D-pipelined-1M | 99.1 ms | 0.6769 | - | - | 57.303 | 98.7684 | 1.72x |
| 64MB | D-pipelined-4M | 114 ms | 0.5894 | - | - | 57.303 | 113.6335 | 1.98x |
| 64MB | D-pipelined-16M | 73.1 ms | 0.918 | - | - | 57.303 | 72.8996 | 1.27x |
| 64MB | D-pipelined-64M | 80.8 ms | 0.8308 | - | - | 57.303 | 79.6711 | 1.39x |
| 64MB | D-sharded | 104 ms | 0.6479 | - | - | 57.303 | 101.8749 | 1.78x |
| 64MB | D-bf16 | 85.3 ms | 0.7866 | - | - | 57.303 | 83.9264 | 1.46x |
| 64MB | D-fp8 | 44 ms | 1.5249 | - | - | 29.123 | 42.8263 | 1.47x |
| 64MB | D-default-repeat | 99.6 ms | 0.6741 | - | - | 57.303 | 98.3242 | 1.72x |
| 64MB | E | 103 ms | 0.6527 | - | - | - | - | - |
| 256MB | D-default | 417 ms | 0.6432 | - | - | 229.069 | 416.2925 | 1.82x |
| 256MB | D-pinned | 304 ms | 0.8842 | - | - | 229.069 | 303.401 | 1.32x |
| 256MB | D-pipelined-1M | 418 ms | 0.6428 | - | - | 229.069 | 417.1843 | 1.82x |
| 256MB | D-pipelined-4M | 298 ms | 0.9001 | - | - | 229.069 | 297.8138 | 1.30x |
| 256MB | D-pipelined-16M | 280 ms | 0.9598 | - | - | 229.069 | 279.4966 | 1.22x |
| 256MB | D-pipelined-64M | 293 ms | 0.9149 | - | - | 229.069 | 293.2173 | 1.28x |
| 256MB | D-sharded | 401 ms | 0.6688 | - | - | 229.069 | 399.582 | 1.74x |
| 256MB | D-bf16 | 332 ms | 0.8083 | - | - | 229.069 | 320.4423 | 1.40x |
| 256MB | D-fp8 | 153 ms | 1.7522 | - | - | 116.348 | 149.5369 | 1.29x |
| 256MB | D-default-repeat | 389 ms | 0.69 | - | - | 229.069 | 387.9319 | 1.69x |
| 256MB | E | 407 ms | 0.6589 | - | - | - | - | - |
| 1GB | D-default | 1.67 s | 0.6412 | - | - | 916.132 | 1671.9611 | 1.83x |
| 1GB | D-pinned | 1.21 s | 0.8909 | - | - | 916.132 | 1204.9681 | 1.32x |
| 1GB | D-pipelined-1M | 1.66 s | 0.6475 | - | - | 916.132 | 1658.0089 | 1.81x |
| 1GB | D-pipelined-4M | 2 s | 0.5373 | - | - | 916.132 | 1997.7178 | 2.18x |
| 1GB | D-pipelined-16M | 1.07 s | 1.0003 | - | - | 916.132 | 1072.7695 | 1.17x |
| 1GB | D-pipelined-64M | 1.22 s | 0.8796 | - | - | 916.132 | 1220.4363 | 1.33x |
| 1GB | D-sharded | 1.58 s | 0.6781 | - | - | 916.132 | 1577.4468 | 1.72x |
| 1GB | D-bf16 | 1.23 s | 0.8759 | - | - | 916.132 | 1217.0022 | 1.33x |
| 1GB | D-fp8 | 609 ms | 1.7631 | - | - | 465.247 | 600.0012 | 1.29x |
| 1GB | D-default-repeat | 1.6 s | 0.6713 | - | - | 916.132 | 1596.9077 | 1.74x |
| 1GB | E | 1.64 s | 0.656 | - | - | - | - | - |

## A multi-stream TCP bridge (prototype)

#66 (merged, opt-in) is a prototype bridge transport, `bridge="multitcp"`: a small Rust library that
carries each leader pair's traffic over several TCP connections (4 by default), with
Python reducing each 4 MiB chunk as soon as it has arrived. Before any run, its gate
was set: it stays only if, with pinned staging, the whole float32 all_reduce's median
takes at most 0.8 times the pinned Gloo bridge's at every size from 64 MB to 1 GB, in
each of two jobs. **It doesn't meet that**, so the prototype stops here.

Two step-08 jobs (11104565, 11105069), each one V100 + one MI50, as in the matrix
above, at commit 23180fe of #66's branch (clean): torch 2.14.1+cu126 and NCCL 2.29.3
on the V100, torch 2.14.1+rocm7.2 on the MI50, the bridge on `internal.2137` on both.
Median of the whole all_reduce, gpubridge's default policy:

| Run | 1 MB | 16 MB | 64 MB | 256 MB | 1 GB |
| --- | --- | --- | --- | --- | --- |
| Gloo bridge, pageable (D-default) | 4.5 / 3.9 ms | 33.4 / 30.8 ms | 99.8 / 99.6 ms | 394.4 / 398.0 ms | 1.581 / 1.630 s |
| Gloo bridge, pinned (D-pinned) | 3.1 / 3.1 ms | 28.1 / 32.7 ms | 80.0 / 85.3 ms | 313.5 / 307.2 ms | 1.250 / 1.241 s |
| multitcp, pageable | 2.9 / 4.0 ms | 30.9 / 33.9 ms | 110.1 / 114.9 ms | 388.6 / 387.7 ms | 1.551 / 1.580 s |
| multitcp, pinned | 2.7 / 2.3 ms | 28.3 / 27.5 ms | 85.9 / 87.4 ms | 281.9 / 275.4 ms | 1.099 / 1.098 s |
| multitcp, one connection | 3.9 / 4.1 ms | 31.1 / 31.8 ms | 104.3 / 105.2 ms | 397.4 / 435.4 ms | 1.656 / 1.603 s |

(first job / second job.) Pinned multitcp against pinned Gloo: 1.07 and 1.02 times
its time at 64 MB, 0.90 and 0.90 at 256 MB, 0.88 and 0.88 at 1 GB. So several
connections bought about 12% at 1 GB and nothing at 64 MB, and one connection was no
faster than Gloo's. The path itself wasn't the limit at these sizes: the library alone
moved about 1.15 GB/s between such a pair of nodes (256 MiB in 4 MiB messages, one
direction at a time), and Gloo's pinned bridge at 1 GB already reaches about 0.86 GB/s.

**A finding for every bridge that reduces in Python.** The first run of the prototype
(job 11102041) was about seven times slower than these: the reduction of each chunk,
a `torch.Tensor.add_` on host memory, took 12 to 27 ms instead of 0.3 to 1 ms. On
these 4-CPU jobs, torch's intra-op threads (`OMP_NUM_THREADS=4`) competed with the
transport's network threads; with one torch thread for the duration of the
collective it ran as above. The UCX transport also reduces with torch ops in Python
and may meet the same contention on GPU nodes; that isn't measured.

## The verbs bridge on one node (split test)

One node of four Tesla V100-SXM2-32GB (driver 545.23.08, torch 2.14.1+cu126,
NCCL 2.29.3) with one Mellanox ConnectX HDR InfiniBand HCA (`mlx5_0`), 2026-10-20.
`GPUBRIDGE_SPLIT_TEST=half` makes ranks 0-1 and 2-3 two 2-GPU NCCL islands (both
V100s), so the bridge joins two leaders on the same host: verbs traffic goes
through the one HCA in loopback. GPU tensors were staged through registered pinned
host memory (no dma-buf or peer-memory path on that node). This is **not**
cross-node, mixed-vendor or GPUDirect evidence.

`bench_all_reduce.py`, float32, default policy (reduce-bridge-broadcast): the
CUDA-event median of each trial's slowest rank, for the whole collective (island
reduce, staging, the bridge, island broadcast and gpubridge's Python), not the
NIC's own latency. Two samples, separate jobs: `4c13375` and `c92684b`, whose
`src/` and benchmark scripts are identical. Native is one NCCL group of all four
GPUs, the homogeneous baseline; it has no bridge.

| Size | Native NCCL, 4 GPUs | Gloo bridge | Verbs bridge | Verbs speedup over Gloo, `4c13375` | Same, `c92684b` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 0.093 ms | 0.557 ms | 0.696 ms | 0.80x | 0.85x |
| 4 KiB | - | - | - | 0.79x | 0.82x |
| 16 KiB | - | - | - | 0.78x | 0.87x |
| 64 KiB | - | - | - | 0.82x | 0.80x |
| 256 KiB | - | - | - | 1.05x | 0.93x |
| 1 MiB | 0.130 ms | 2.247 ms | 1.087 ms | 2.07x | 1.74x |
| 4 MiB | - | - | - | 2.48x | 2.42x |
| 16 MiB | 0.344 ms | 20.21 ms | 7.42 ms | 2.72x | 2.73x |
| 64 MiB | - | - | - | 2.70x | 2.68x |
| 256 MiB | 3.52 ms | 286.3 ms | 106.3 ms | 2.69x | 3.07x |
| 1 GiB | - | not run | 420.8 ms | - | - |

Times are `4c13375`'s (job 11259197; the Gloo bridge from its Gloo step, verbs
from its verbs step, native from the Gloo step's control). A dash is a size whose
times aren't repeated here. The Gloo bridge step stopped at 256 MiB by plan, so
there is no 1 GiB ratio. One sample per configuration and job: the two samples'
speedups differ by up to 18.5% (at 1 MiB), so these are descriptive, not a
significant or general speedup. Verbs is slower than the Gloo bridge up to 64 KiB
in both samples and faster from 1 MiB on; native NCCL on the same four GPUs is 6
to 81 times faster than either bridge at these sizes.

Each sample also had a companion job (11259198 at `4c13375`), so the following
are separate launches from the table above. There, the pipelined policy (1, 4, 16
and 64 MiB chunks) helped the Gloo bridge at every matched size from 1 to 256 MiB
and generally slowed the verbs bridge. GPT-2 small DDP training (global batch 32,
50 steps, 25 MB buckets) took 301.6 ms a step on native NCCL, 741.2 ms over the
Gloo bridge and 419.4 ms over the verbs bridge (one run each); `train_bench.py`
estimated 19% of the Gloo bridge's time and 41% of the verbs bridge's hidden
behind backward. Those are estimates from separately timed phases, not traces. Correctness, the no-silent-TCP
gate and what remains open are in
[HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20).

## Training throughput

`scripts/gpu/train_bench.py`: DDP with `gpubridge.ddp_comm_hook` on a real
model with synthetic data, float32, AdamW, 25 MB buckets; 3 warm-up steps, then
10 timed steps. Per rank: GPT-2 small (124 M parameters, 498 MB of gradients)
at batch 4 x 512 tokens, ResNet-50 (25.6 M parameters, 102 MB) at batch 32 x
224 x 224. The step splits into the slower rank's compute, the time the faster
rank waits for it, and the communication not hidden behind compute ("exposed").
"Comm alone" is the same buckets all_reduced with no compute alongside. Each GPU
alone (`submit.sh train-nvidia`, `train-amd`) is one rank, so its communication
is the hook with nothing to sync.

| Model | Run | Ranks | Samples/s | Step (ms) | Slower GPU's compute | Faster GPU waits | Exposed comm | Comm alone |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | V100 alone | 1 | 25.3 | 158.1 | 152.4 | - | 5.7 | 2.8 |
| GPT-2 small | MI50 alone (2.9.1+rocm6.3) | 1 | 16.4 | 243.6 | 238.9 | - | 4.7 | 2.8 |
| GPT-2 small | V100 + MI50 through gpubridge | 2 | 8.9 | 899.4 | 240.9 (MI50) | 88.6 (V100) | 658.6 | 714.6 |
| ResNet-50 | V100 alone | 1 | 315.8 | 101.3 | 98.1 | - | 3.2 | 0.9 |
| ResNet-50 | MI50 alone (2.9.1+rocm6.3) | 1 | 247.3 | 129.4 | 128.0 | - | 1.3 | 0.9 |
| ResNet-50 | V100 + MI50 through gpubridge | 2 | 327.7 | 195.3 | 128.3 (MI50) | 29.8 (V100) | 67.0 | 145.2 |

Medians of 10 timed steps. Samples/s counts every rank's batch, so the mixed
runs process twice the samples per step. The mixed runs are step 07 with
`STEPS_07=training` and the MI50 on `venv-rocm63` (torch 2.9.1+rocm6.3 there,
2.14.1+cu126 on the V100), the default bridge (`reduce-bridge-broadcast`,
pageable staging) over TCP on `internal.2137`; their preflight measured
1173.8 MB/s from the V100 to the MI50, 1169.7 MB/s back and a 0.046 ms round trip.

**Does adding the MI50 raise throughput? Not on this link.**

- **GPT-2 small: no, it cuts it to 0.35x** the V100 alone (8.9 against 25.3
  samples/s). Each step all_reduces 498 MB of gradients over the bridge, 715 ms
  on its own, and only 56 ms of it hides behind the backward pass; the MI50's
  slower compute adds 89 ms of waiting on the V100.
- **ResNet-50: about the same**: 327.7 against 315.8 samples/s, 4% more, from
  one run of 10 steps that ranged from 187 to 240 ms (the V100 alone: 100.8 to
  101.6 ms). Its 102 MB of gradients take 145 ms alone, 78 ms of it hidden
  behind the backward pass; the V100 waits 30 ms a step for the MI50.
- Within the mixed job each GPU's compute matched its single-GPU run (V100
  152.2 and 98.5 ms, MI50 240.9 and 128.3 ms, against 152.4, 98.1, 238.9 and
  128.0 ms alone), so the loss is the communication and the wait, not slower
  GPUs.
- Every rank ended each run with identical, finite parameters.

These runs used the default policy, which the matrix above measured at about
0.65 GB/s; pinned staging, pipelining and fp8 compression were 1.4x to 2.8x
faster at 1 GB there. fp8 is measured in training below; pinned staging and
pipelining aren't.

### fp8 on the bridge, and larger batches

The same mixed job and the same `train_bench.py`, at commit 4e7f04e (clean on
both ranks), with the bridge compressing to fp8
(`GPUBRIDGE_BRIDGE_COMPRESSION=fp8`). The MI50's Triton in `venv-rocm63`
can't compile gpubridge's compression kernels for gfx906, so both ranks ran
the PyTorch versions instead (`GPUBRIDGE_COMPRESSION_KERNELS=torch`): the same
bits, but slower, so these runs understate what fp8 saves. Both settings are
in every rank's record. Everything else as above: float32, the default policy
and pageable staging over TCP, 25 MB buckets (13 for GPT-2 small at every
batch), 3 warm-up and 10 timed steps, one run per row. Each run's preflight
measured 1172 to 1174 MB/s from the V100 to the MI50 and 1164 to 1174 MB/s
back. The MI50 was the same node in every run; the V100 was one of two nodes.

Batch is per rank, so a mixed run processes twice the V100-alone run's
samples per step.

| Model | Batch | V100 alone | V100 + MI50, fp8 | Mixed / alone | Step (ms) | MI50 compute | V100 compute | V100 waits | Exposed comm | Comm alone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 4 | 25.3 | 12.495 | 0.49x | 640.3 | 243.1 | 152.0 | 91.1 | 397.2 | 482.5 |
| GPT-2 small | 8 | 27.669 | 19.686 | 0.71x | 812.8 | 441.8 | 282.6 | 159.1 | 371.0 | 472.4 |
| GPT-2 small | 16 | 28.395 | 26.274 | 0.93x | 1217.9 | 960.5 | 551.2 | 409.2 | 257.5 | 484.9 |
| GPT-2 small | 32 | 29.247 | 28.311 | 0.97x | 2260.6 | 1815.4 | 1087.0 | 728.4 | 445.2 | 605.3 |
| ResNet-50 | 32 | 315.8 | 366.849 | 1.16x | 174.5 | 128.4 | 98.4 | 30.0 | 46.0 | 102.2 |

Samples/s, then milliseconds per step (medians of the 10 timed steps). The
V100-alone runs are `submit.sh train-nvidia` at the same per-rank batch; the
batch-4 GPT-2 and ResNet-50 ones are the 934f87c runs above. Without fp8
(above), GPT-2 small at batch 4 reached 8.9 samples/s (0.35x) and ResNet-50
327.7 (1.04x).

- **ResNet-50 with fp8: the pair is 1.16x the V100 alone** (366.8 against
  315.8 samples/s), 65% of the two GPUs' throughputs alone added together
  (315.8 + 247.3).
- **GPT-2 small: below the V100 alone at every batch measured**, closing from
  0.49x at batch 4 to 0.97x at batch 32.
- **Communication stayed between 472 and 485 ms a step from batch 4 to 16**,
  so a growing share of it hid behind the backward pass (exposed: 397, 371,
  then 258 ms). At batch 32 it rose to 605 ms alone and 445 ms exposed. That
  isn't explained yet. The bucket count didn't change, so the buckets aren't
  the cause. That run's step times also spread more (2130 to 2419 ms, against
  1216 to 1270 at batch 16).
- **From batch 16 on, the wait for the slower GPU costs more than the bridge**:
  the V100 waits 409 ms a step at batch 16 and 728 at batch 32, against 258 and
  445 ms of exposed communication. Per sample, the V100 computed in 34 to 35 ms
  and the MI50 in 55 to 60 ms (compute over batch, batches 8 to 32).
- `train_bench.py` found every rank's parameters identical after each run.

Job IDs: mixed 11089348 (batch 4 and ResNet-50), 11090051 (8), 11090726
(16), 11091146 (32); V100 alone 11090049 (8), 11090725 (16), 11091145 (32).
The batch-16 V100-alone record was overwritten by the batch-32 run's; its
numbers come from that job's log.

**The other training steps of these four jobs** (each job runs them all):

- `train_ddp` (`examples/train_synthetic.py`, DDP with the comm hook):
  **the parameter check failed in the batch-16 job** (11090726); it passed in
  the other three. That was the old float64-checksum check;
  under the later bit-for-bit check, 24 of 24 runs stayed in sync, and it was
  most likely the checksum's false alarm (HARDWARE_VALIDATION.md item 22).
- `verify_fsdp2_backend`: parameters differed from the single-process replay
  in all four, against a bit-for-bit match without compression (above). fp8
  is lossy, so this check may not apply with compression on; whether it does
  is open.
- `train_hsdp_backend` printed nothing until its 20-minute step timeout in
  all four, before any NCCL or RCCL communicator was created, as in the
  uncompressed run; each job then hit its time limit during
  `verify_hsdp_backend`. Where it stops, from a CPU reproduction with the
  same two torch releases: [HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md#items),
  item 31's note.

### fp8 with Triton kernels on the MI50

`venv-rocm63`'s Triton (pytorch-triton-rocm 3.5.1) can't compile gpubridge's
compression kernels for gfx906, so the runs above used the PyTorch versions.
The default ROCm venv's Triton (triton-rocm 3.8.0) can. A venv of
`venv-rocm63`'s torch 2.9.1 with that Triton in front of its own
(`scripts/gpu/rocm_triton_overlay.sh`; the runbook's "Choosing the ROCm
build") gives the MI50 both: rocBLAS kernels for gfx906 and compression
kernels. The 2026-10-15 runs used a manual copy of the same Triton files where
the script now makes links; the script itself has not yet run on the MI50.

- **The kit's GPU tests on one MI50**, as the `amd` step runs them
  (`tests/test_gpu.py`, `test_compression_kernels.py`, `test_torch_backend.py`),
  with torch 2.9.1+rocm6.3 and Triton 3.8.0: 26 passed, 8 skipped, the same as
  the default venv; with `venv-rocm63`'s own Triton 3.5.1, 20 failed. This held
  both with Triton's bundled HIP runtime and with `TRITON_LIBHIP_PATH` at
  torch's HIP 6.3 (job 11091760, commit 4e7f04e).
- **GPT-2 small through the bridge with fp8**, as in the table above but
  `GPUBRIDGE_COMPRESSION_KERNELS=triton` (in every rank's record), at commit
  4e7f04e:

| Batch | V100 alone | Mixed, PyTorch kernels | Mixed, Triton kernels | Triton / alone | Step (ms) | MI50 compute | V100 compute | V100 waits | Exposed comm | Comm alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 28.395 | 26.274 | 29.631 | 1.04x | 1079.9 | 985.9 | 556.6 | 429.4 | 94.0 | 346.0 |
| 32 | 29.247 | 28.311 | 32.726 | 1.12x | 1955.6 | 1827.5 | 1077.8 | 749.7 | 128.1 | 361.8 |

Samples/s, then milliseconds per step (medians of the 10 timed steps); the
first two columns are the runs above. Jobs 11091811 (16) and 11093097 (32);
preflight 1170 to 1174 MB/s each way, a 0.05 ms round trip, on
`internal.2137` on the V100 and `internal` on the MI50.

- **The first GPT-2 runs where the pair beats the V100 alone**: 1.04x at
  batch 16, 1.12x at 32. One run each, and the steps spread (1067 to 1400 ms
  at 16, 1913 to 2289 ms at 32), so the 1.04x is within that spread.
- **Communication alone fell from 485 to 346 ms at batch 16 and from 605 to
  362 ms at 32**; exposed communication from 258 to 94 and 445 to 128 ms. With
  the Triton kernels, batch 32 needs about what batch 16 does, so the rise at
  batch 32 above went away with Triton kernels in these runs. The cause of
  the difference has not been isolated.
- **What's left is mostly the wait for the MI50**: 429 ms a step at batch 16
  and 750 at 32, against 94 and 128 ms of exposed communication. The MI50's
  compute was within 3% of the PyTorch-kernel runs'.
- In the same jobs, `train_ddp`'s parameters stayed in sync, FSDP2's
  bit-for-bit check differed from its uncompressed replay as in every fp8 run,
  and `train_hsdp_backend` hung (batch 16: until its timeout; batch 32: the job
  was cancelled during it).

## Data-parallel training on a LLaMA-style model

[`scripts/gpu/train_dp.py`](../scripts/gpu/train_dp.py) on `llama.py`'s
`llama-125m` (134 M parameters, 536 MB of float32 gradients a step), a V100
and an MI50 in step 07's heterogeneous job with `STEPS_07=dp`. The global batch
is 8 sequences x 512 tokens, 4 per GPU unless a row says otherwise; AdamW 3e-4,
seed 0. DDP runs through `gpubridge.ddp_comm_hook` with 25 MB buckets (14),
FSDP2 through `backend="gpubridge"`; the default bridge policy over TCP on
`internal.2137`, whose preflights measured 1159 to 1174 MB/s each way. The
V100 ran torch 2.14.1+cu126, the MI50 2.9.1+rocm6.3; in the 1,000-step job the
MI50's venv had Triton 3.8.0 in front (`venv-rocm63-triton38`,
`scripts/gpu/rocm_triton_overlay.sh`), so the compression kernels ran as
Triton on both GPUs. One V100 alone is `submit.sh dp-baseline`. Medians of
the steps after the first 5; "exposed" is the median step less the slower
GPU's compute alone, "hidden" the share of the bridge's time alone that isn't
exposed, and "V100 waits" the slower GPU's compute alone less the V100's:
estimates from each GPU's compute timed alone, as in `train_dp.py`, not waits
measured during training.

| Run | Steps | Tokens/s | Against the V100 alone | Step (ms) | MI50 compute | V100 waits (est.) | Bridge alone | Exposed (est.) | Hidden (est.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V100 alone | 1,000 | 14,264 | 1.00x | 287.1 | - | - | - | - | - |
| V100 + MI50, DDP | 1,000 | 4,549 | 0.32x | 900.4 | 231.8 | 78.4 | 792.2 | 668.7 | 16% |
| V100 + MI50, DDP, bf16 bridge | 1,000 | 5,753 | 0.40x | 711.9 | 231.8 | 77.5 | 627.7 | 480.2 | 24% |
| V100 + MI50, DDP, fp8 bridge | 1,000 | 9,523 | 0.67x | 430.1 | 232.4 | 77.7 | 326.0 | 197.7 | 39% |
| V100 + MI50, FSDP2 | 500 | 1,809 | 0.13x | 2,264.5 | - | - | - | - | - |

Jobs: 11111481 (the three DDP runs), 11110598 (FSDP2), 11109239 (the V100).

Shares, bucket sizes and fp8 with the PyTorch kernels on the MI50, side by
side in one job (11111037), 100 steps each:

| Run | Shares (V100 + MI50) | Buckets | Tokens/s | Step (ms) | V100 compute | MI50 compute | V100 waits (est.) | Bridge alone | Exposed (est.) | Hidden (est.) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDP | 4 + 4 | 14 x 25 MB | 4,391 | 932.8 | 153.5 | 231.3 | 77.8 | 836.6 | 701.5 | 16% |
| DDP, shares by speed | 5 + 3 | 14 x 25 MB | 4,274 | 958.4 | 183.3 | 192.8 | 9.4 | 860.7 | 765.7 | 11% |
| DDP | 4 + 4 | 50 x 5 MB | 4,278 | 957.4 | 153.8 | 240.4 | 86.7 | 873.2 | 717.0 | 18% |
| DDP | 4 + 4 | 5 x 100 MB | 4,429 | 924.9 | 153.9 | 231.9 | 78.0 | 832.6 | 693.0 | 17% |
| DDP, fp8 bridge, PyTorch kernels on the MI50 | 4 + 4 | 14 x 25 MB | 6,544 | 625.9 | 153.8 | 231.7 | 78.0 | 492.3 | 394.2 | 20% |

- **Adding the MI50 lowers throughput in every data-parallel setting on this
  link.** The best, DDP with fp8 on the bridge, reaches 0.67x the V100
  alone. Each step sends 536 MB of gradients through a link of about
  1.17 GB/s, and backward hides 16% of it uncompressed.
- **Compression: bf16 is 1.26x uncompressed DDP's tokens/s and fp8 2.09x**
  (5,753 and 9,523 against 4,549, one job). The bridge alone fell from 792
  to 628 and 326 ms, and the share hidden rose to 24% and 39%. With the
  PyTorch kernels on the MI50, fp8 was 1.49x (6,544 against 4,391, another
  job), so Triton on the MI50 matters here as it did for GPT-2 above.
- **Shares by speed even out the compute but don't raise throughput here.**
  With 5 + 3 the two GPUs' compute alone differs by 9 ms a step instead of
  78, but tokens/s fell 2.7%: in that run the bridge alone took 861 ms
  against 837, and the share hidden fell from 16% to 11%. The bridge takes
  about 4x the slower GPU's compute, so it, more than the compute gap, looks
  like what sets the step here. Balancing may pay where compute is a larger
  share (bigger batches, or a faster bridge), which these runs didn't
  measure.
- **Bucket size matters little**: 100 MB, 25 MB and 5 MB buckets ran within
  3.5% of each other (4,429, 4,391 and 4,278 tokens/s), the 50 small buckets
  slowest.
- **FSDP2 is 0.40x DDP** (1,809 against 4,549 tokens/s, different jobs): each
  step gathers the parameters through the bridge for the forward and again
  for the backward, and scatters the gradients. At 500 steps it needed
  1,205 s, more than the kit's default 20-minute step limit (`STEP_TIMEOUT`).
  `train_dp.py` records no step split or parameter checks for FSDP2. HSDP
  didn't run: with one GPU per vendor it would shard nothing.
- **Every DDP run's parameters were identical on both GPUs at every check**
  (every 50 steps and the last), compressed runs included.
- **Loss**: uncompressed DDP tracks one V100 about as closely as V100 runs track
  each other over the last 100 of 1,000 steps, and so does FSDP2 over steps 401
  to 500, the last it ran (0.007 to 0.069 from the four V100 runs, which are
  0.024 to 0.093 apart there, still falling). Compressed DDP ended 0.02 to 0.03
  above after 1,000 steps in job 11111481; four later runs ended 0.033 to 6.92
  above, each after a late drop (one `fp8` run still falling at step 1,000,
  8.73 over its last 100 steps; HARDWARE_VALIDATION.md item 33). The
  comparison, and why it needs 1,000 steps rather than 500, is in
  [COMPRESSION.md](COMPRESSION.md#training-with-compression).

## Which ROCm wheels can train on an MI50 (gfx906)

The MI50's rank aborts at its first matmul with the kit's default ROCm wheel
(torch 2.14.1+rocm7.2): its rocBLAS has no kernels for gfx906. To find a wheel
whose rocBLAS does, a CPU job read the zip directory of every torch ROCm wheel
from 2.8 on for Python 3.12 (the PyTorch index and AMD's repo.radeon.com, by
HTTP range requests, not whole downloads) and counted
`torch/lib/rocblas/library/TensileLibrary*gfx906*`:

| torch | ROCm build | rocBLAS files in the wheel | gfx906 Tensile files |
| --- | --- | --- | --- |
| 2.8.0, 2.9.0, 2.9.1 | rocm6.3 | 1620 to 1675 | **155** |
| 2.8.0, 2.9.0, 2.9.1 | rocm6.4 | 1391 to 1446 | 0 |
| 2.10.0 | rocm7.0, rocm7.1 | 1794 | 0 |
| 2.11.0 to 2.13.0 | rocm7.1 | 1794 | 0 |
| 2.11.0, 2.12.x, 2.14.x | rocm7.2 | 1794 | 0 |
| 2.14.0, 2.14.1 | rocm7.14 | none: rocBLAS comes in separate `rocm-sdk-device-gfx*` wheels | the index has none for gfx906 (it has gfx908, gfx90a, gfx942, gfx950 and gfx10xx to gfx12xx) |
| AMD's builds of 2.8.0 to 2.10.0 | rocm 6.4.4 to 7.2 | 1542 to 1698, or none in the `.lw` builds | 0 |

Not read: 2.13.0+rocm7.2 (the server sent the whole file instead of the range;
the scan now refuses such replies) and two AMD development builds of 2.8.0 (no
end-of-directory record in the last 128 KB).

**hipBLASLt and MIOpen**, from the same zip directories: none of the 38 wheels
read has hipBLASLt files for gfx906, and training ran on the MI50 with 2.9.1+rocm6.3 anyway
(below). MIOpen's tuning databases for gfx906
(`torch/share/miopen/db/gfx906_60.db.txt` and its find-dbs, 5 files) are in
every wheel that ships MIOpen files except the rocm7.14 and AMD's `.lw`
builds; no wheel read has precompiled MIOpen kernel archives for gfx906, and the
MI50's conv2d and ResNet-50 runs worked without them. Not read for MIOpen: the PyTorch index's 2.10 to 2.13
rocm7.1 builds and 2.11 to 2.14.0 rocm7.2 builds, whose server ignored the
range requests this time. The default wheel (2.14.1+rocm7.2) lists gfx942
(MI300X) in its build and ships rocBLAS and hipBLASLt files for it.

**torch 2.9.1+rocm6.3 on one MI50**, in a second venv (`venv-rocm63`; the
kit's default ROCm venv is unchanged), with HIP 6.3.42134 and RCCL 2.21.5:

- `setup-rocm` with `ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.3`: the probe's
  `blas` stage passed (on the default wheel it aborts, exit -6).
- A short check on the GPU: a 1024 x 1024 float32 matmul (largest error 1.6e-6
  of the largest value, against float64 on the CPU), a float16 matmul, a conv2d,
  causal scaled-dot-product attention and a linear layer's backward all ran.
- The single-node AMD step (`submit.sh amd`, at dbed5c9) passed 7 of 9 steps:
  the probe, `device_map`, `local_rank_check`, `check_1gpu`, `rccl_version`,
  `check_split_shared_gpu` and its summary. `gpu_tests` had 20 failed, 6
  passed, 8 skipped, and `bench_compression_kernels` failed: every failure is
  this wheel's Triton (pytorch-triton-rocm 3.5.1) failing to compile
  gpubridge's compression kernels for gfx906 (`PassManager::run failed` in
  `ConvertTritonAMDGPUToLLVM`). The default venv's Triton (3.8.0) compiles
  them.
- So on an MI50, training needs torch 2.9.1+rocm6.3, and bridge compression
  on the GPU a newer Triton in front of it ([above](#fp8-with-triton-kernels-on-the-mi50));
  the runbook's "Choosing the ROCm build" has the commands.

## Waiting on Research Computing

These need two GPUs of one type in one partition, as one heterogeneous job.
The job scripts are ready; `submit.sh` refuses them until
`SAME_PARTITION_PAIRS_OK=1` is set in `scripts/gpu/explorer/local.env` (or the
environment):

| Command | Runs |
| --- | --- |
| `scripts/gpu/explorer/submit.sh 08-nvidia-pair` | A1, A2, C and E-nvidia on two V100s |
| `scripts/gpu/explorer/submit.sh 08-amd-pair` | B1, B2, C-amd and E-amd on two MI50s |

Then `bench_matrix.py` over all three jobs' results fills the x A2 and x A1
columns; ratios between jobs are marked †.

## How to reproduce

```bash
scripts/gpu/explorer/submit.sh net-nvidia
scripts/gpu/explorer/submit.sh net-amd
RDMA_MODULES=OpenMPI/4.1.6 scripts/gpu/explorer/submit.sh net-nvidia   # the RDMA check's
RDMA_MODULES=OpenMPI/4.1.6 scripts/gpu/explorer/submit.sh net-amd      # ucx_info
scripts/gpu/explorer/submit.sh 08
scripts/gpu/explorer/submit.sh train-nvidia
VENV_ROCM=$WORK_DIR/venv-rocm63 scripts/gpu/explorer/submit.sh train-amd
VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=training TIME_07=00:40:00 scripts/gpu/explorer/submit.sh 07
# fp8 on the bridge, GPT-2 small at a larger batch (N = 8, 16, 32), each GPU alone then both:
TRAIN_BENCH_MODELS=gpt2-small TRAIN_BENCH_ARGS="--batch-size N" scripts/gpu/explorer/submit.sh train-nvidia
GPUBRIDGE_BRIDGE_COMPRESSION=fp8 GPUBRIDGE_COMPRESSION_KERNELS=torch \
    TRAIN_BENCH_MODELS=gpt2-small TRAIN_BENCH_ARGS="--batch-size N" \
    VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=training scripts/gpu/explorer/submit.sh 07
python scripts/gpu/bench_matrix.py results/explorer/08_matrix_nvidia-amd \
    results/explorer/net_check_nvidia results/explorer/net_check_amd
```

Tonight's rocm6.3 jobs also set `RESULTS_ROOT=results/explorer/rocm63`, to keep
their results apart from the default venv's.
