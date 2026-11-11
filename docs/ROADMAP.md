# Roadmap: designs waiting on GPU results

(Since 2026-11-11 the default bridge is `bridge="auto"`, one of the Rust bridges
of section 2, and Gloo is opt-in: docs/RUNTIME_INVENTORY.md, section 6. Where the
sections below say "the default bridge" or "Gloo over TCP" for the bridge step,
they describe the job as it was when they were written.)

These are designs, not commitments. All three now have experimental
implementations, tested on CPU and, so far, on few and narrow GPU runs (each
status below says which). Each one
waits for specific results from the GPU validation kit
([GPU_VALIDATION_RUNBOOK.md](GPU_VALIDATION_RUNBOOK.md)). Without those
results there is no way to tell whether the problem it solves is the one that
matters. Any of them that gets built will follow the rules the opt-in
policies follow:

- off by default, and never picked by `auto`;
- selectable with `--policy` (or `bridge=`) in `check.py` and
  `bench_all_reduce.py`;
- any setting that changes how many messages a collective sends is checked
  across ranks during discovery;
- a numbered HARDWARE_VALIDATION.md item for every claim only GPUs can check.

Contents:

1. [Bridge-only compression](#1-bridge-only-compression)
2. [An RDMA bridge transport](#2-an-rdma-bridge-transport)
3. [A torch.distributed backend for FSDP2 and DeviceMesh](#3-a-torchdistributed-backend-for-fsdp2-and-devicemesh)

## 1. Bridge-only compression

**Problem.** Inside an island, NCCL/RCCL move data over NVLink, Infinity
Fabric or PCIe. On the bridge, the same bytes go through host memory and
Gloo over TCP. If GPU runs show the bridge step dominates large all_reduces,
sending fewer bytes over the bridge alone would help without touching the
fast island links.

**Status: implemented, opt-in.** On GPUs its kernels matched the CPU reference on a
V100, an H200 and an MI50, and compressed DDP trained on a V100 + MI50
([BENCHMARKS.md](BENCHMARKS.md)); its accuracy on representative training is not
established (HARDWARE_VALIDATION.md items 28 and 29). `GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8`;
design, numerics and error bounds in [COMPRESSION.md](COMPRESSION.md). It
differs from the design first sketched here in three ways:

- **A setting, not a policy.** It applies to the bridge step of the three
  bridged policies (default, pipelined, sharded) and both bridge transports,
  instead of being a separate "lossy" policy. It is still off by default,
  checked across ranks in discovery, and limited to floating-point SUM and
  AVG.
- **fp8 as well as bf16.** OCP E4M3 with per-block power-of-two scales,
  quantized and dequantized by Triton kernels on the leaders' GPUs, so the
  copy to host memory carries fewer bytes too. Codes are computed with
  integer arithmetic, so every vendor's GPU gives the same bits and every
  rank still ends with the same result.
- **Error feedback for every buffer, not only DDP.** Each bridge member keeps
  a residual per buffer (found by address, size and dtype), which carries a
  step's rounding error into the next. That needs no comm-hook state, so
  plain all_reduce gets it too.

**What's left, and depends on GPU results:**
- Whether it pays off, and from which size (HARDWARE_VALIDATION.md item 29,
  the kit's `bench_compressed` step). If items 13, 18 and 21 show the bridge
  isn't the bottleneck, it isn't worth its loss of precision.
- That the compiled kernels match the reference bit for bit on NVIDIA and AMD
  GPUs (item 28).
- Training accuracy: loss curves from `examples/train_synthetic.py` with and
  without it, over enough steps to see drift.
- **Not planned:** lossless compression of float gradients saves little, and
  it costs CPU time on the leaders, which already do the host copies and run
  Gloo.

## 2. An RDMA bridge transport

**Problem.** The Gloo bridge copies GPU data to host memory, sends it over
TCP, and copies it back on the other side. Three 2026 papers move
cross-vendor data with RDMA instead, and report far better cross-vendor
performance than a Gloo bridge. The second HetCCL reports more than 6x Gloo's
point-to-point bandwidth (17 to 19x in its abstract). Joint Training reports
539.6 TFLOPs per GPU with its RDMA path, against 160.8 to 236.5 with Gloo
carrying the cross-vendor pipeline traffic. Details are in [PRIOR_ART.md](PRIOR_ART.md):

- HetCCL ([arXiv 2601.22585](https://arxiv.org/abs/2601.22585)) registers
  `cudaMalloc` and `hipMalloc` memory with `ibv_reg_mr` and uses InfiniBand
  verbs. It relies on NVIDIA GPUDirect RDMA and AMD DirectGMA, with no driver
  changes.
- The other HetCCL ([arXiv 2605.31000](https://arxiv.org/abs/2605.31000), a
  different group) copies data into registered RDMA buffers in GPU memory and
  sends 4 MB chunks with verbs, under a CPU proxy thread.
- "Joint Training on AMD and NVIDIA GPUs"
  ([arXiv 2602.18007](https://arxiv.org/abs/2602.18007)) adds a Device-Direct
  path with GPUDirect RDMA on both sides, for pipeline-parallel traffic.

None of them publishes code today.

**Design.**
- **A transport, not a policy.** `RdmaTransport` would subclass
  `BridgeTransport` and register as `bridge="rdma"`. Every policy would use
  it unchanged, including the sharded policy's extra links (through
  `create_groups`).
- **Device memory in, no staging.** Today a transport only sees CPU tensors.
  A class attribute such as `device_tensors = True` would let policies skip
  the host copy and pass device tensors straight to the transport. Transports
  without it keep today's behavior.
- **Two ways to register GPU memory with the NIC:**
  - `ibv_reg_mr` on device pointers, through the peer-memory kernel modules
    (`nvidia-peermem` on NVIDIA, AMD's peer memory support on ROCm). This is
    the papers' route.
  - `ibv_reg_dmabuf_mr` with a dma-buf exported by the GPU runtime
    (`cuMemGetHandleForAddressRange` on CUDA, `hsa_amd_portable_export_dmabuf`
    on ROCm), which uses the upstream kernel interface. None of the papers
    uses it. It may be the only option where the peer-memory modules aren't
    installed.
- **Capabilities in the probe.** `Probe` (shared during discovery) would
  record what each rank can do: `libibverbs` present, RDMA devices
  (`/sys/class/infiniband`), the peer-memory module loaded, dma-buf support
  in the runtime. With `bridge="rdma"`, `init()` would fail on every rank
  before any group is created, naming the rank, the missing piece, and
  `bridge="gloo"` as the fallback.
- **Not a core dependency.** The transport would load `libibverbs` at run
  time (ctypes or a small optional extension) and ship as an extra, e.g.
  `pip install gpubridge[rdma]`. The core package keeps PyTorch as its only
  dependency.
- **Reductions.** RDMA moves bytes; it doesn't add them. The second HetCCL
  reduces with each vendor's own library. gpubridge would add on the GPU
  after each receive, for example in a ring all_reduce between the
  participating ranks.

**How we'd measure it.**
- First, before writing any code: the hardware ceiling. perftest's
  `ib_write_bw` can read and write GPU memory (`--use_cuda`, `--use_rocm`, and
  dma-buf variants in recent releases). Run it between an NVIDIA node and an
  AMD node. If cross-vendor GPU-to-GPU RDMA doesn't work there, the design
  stops.
- A new kit step: `scripts/gpu/rdma_probe.py` records `ibv_devinfo`, loaded
  modules and the perftest numbers on each node, in the style of `probe.py`.
- `bench_all_reduce.py --phases` with `bridge="gloo"` and `bridge="rdma"`:
  the bridge phase alone, then the whole all_reduce, against the per-island
  baselines (item 11) and the papers' numbers.
- Correctness: `check.py` with `bridge="rdma"`, bit for bit against
  `flat-gloo`, like every other bridge.

**Depends on GPU results.**
- Hardware first: RDMA NICs between GPU nodes of both vendors, with peer
  memory or dma-buf usable. Explorer's answer is unknown until the GPU survey
  (step 0a) and step 05 run. A cloud pair needs NICs that support it.
- HARDWARE_VALIDATION.md item 8 (separate hosts on a real network) and item
  13 (how much of a bridged all_reduce the bridge takes): RDMA only pays off
  where the bridge dominates.

**Status: implemented twice, both opt-in.** First as `bridge="ucx"`: UCX instead
of raw verbs, and a ctypes-loaded library instead of a torch extension; the design
and why are in [RDMA_TRANSPORT.md](RDMA_TRANSPORT.md). It is tested on CPU over TCP
only, never on RDMA hardware or a GPU (HARDWARE_VALIDATION.md items 26 and 27). Then
as `bridge="verbs"`: raw libibverbs in a Rust library
([VERBS_TRANSPORT.md](VERBS_TRANSPORT.md)). On 2026-10-20 it ran on one node of four
V100s through one InfiniBand HCA in loopback, host and staged GPU tensors exact and
the no-silent-TCP gate passed; GPU memory direct (dma-buf, peer memory), two hosts,
AMD and point-to-point have not run (items 38 to 42). Explorer's V100 and MI50 nodes
have no RDMA path between them ([BENCHMARKS.md](BENCHMARKS.md#rdma-between-the-node-types)).

## 3. A torch.distributed backend for FSDP2 and DeviceMesh

**Problem.** `gpubridge.ddp_comm_hook` covers DistributedDataParallel.
FSDP2, DeviceMesh and DTensor have no hook point: they call a process group's
collectives directly. That takes a `torch.distributed` backend.

**Status: implemented, experimental.** On GPUs, DDP, FSDP2 and HSDP ran through it
on a V100 + MI50 and on two 2-GPU V100 islands of one node; FSDP2 matched a replay
bit for bit on both, HSDP on the V100 node (HARDWARE_VALIDATION.md items 30 and 31).
`dist.init_process_group(backend="gpubridge")`; design, measurements and limits in
[TORCH_BACKEND.md](TORCH_BACKEND.md). DDP and FSDP2 scripts train through it
with the backend name and `import gpubridge`, bit for bit like Gloo on
integer-valued data in simulation mode, on torch 2.8.0 and 2.14.1. What
changed from the CPU prototype recorded here before:

- **A Python `dist.ProcessGroup` subclass, not a `c10d.Backend` one.** A
  `Backend` subclass can't be constructed in Python on torch 2.8.0, the floor.
  DeviceMesh's "ProcessGroup name not set" goes away once the subclass defines
  `group_name` as a property, as PyTorch's own threaded test backend does.
- **gpubridge's groups are built at the first collective**, with
  `dist.new_group`, once the default group exists; discovery checks every rank
  then, so a bad setting fails that collective on every rank instead of hanging
  the others.
- **The world group only.** Subgroups (HSDP, 2-D meshes) fail where they are
  created; each would need its own islands and bridge (the prototype's
  per-group `PrefixStore` idea), which is left for later.
- **Reductions.** FSDP2 picks AVG for float32 and bfloat16 and SUM for float16
  and float64; PREMUL_SUM appears only with a custom gradient divide factor,
  which the backend refuses. Both 2.8's `_allgather_base` /
  `_reduce_scatter_base` and 2.14's `all_gather_single` /
  `reduce_scatter_single` (and their coalesced forms) are implemented.
- **Errors.** Argument errors are raised before any communication, on every
  rank alike. A collective that fails at run time reaches DDP as an exception
  in its future, which in the prototype crashed DDP's backward. With a Python
  process group it surfaces as a RuntimeError instead (found in review, not
  yet a test).

**What's left, and depends on GPU results:** HARDWARE_VALIDATION.md item 30
(DDP and FSDP2 through the backend on GPUs: the `train_fsdp2_backend` step in
steps 04 and 07, `amd/run_all.sh` and `mixed/node.sh`), step time against
FSDP2 on one vendor's NCCL/RCCL, and subgroups if HSDP is wanted. The
torchcomms route (PRIOR_ART.md, torchcomms idea 2) remains an alternative once
its API and ROCm wheels settle.
