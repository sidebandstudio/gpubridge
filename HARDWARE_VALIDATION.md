# What still needs real-hardware validation

The library itself is tested in CPU simulation mode. Simulation runs the same
orchestration code, but it cannot answer the questions here, which depend on
real CUDA and ROCm builds of PyTorch running side by side. Items 1 and 2 are now
answered for the CPU side by running real builds without GPUs; see
[Mixed build results](#mixed-build-results) below.

**On GPUs:** [docs/GPU_VALIDATION_RUNBOOK.md](docs/GPU_VALIDATION_RUNBOOK.md)
maps every item below to a scripted step: Explorer (Slurm) for NVIDIA, a rented
AMD machine, and a real mixed job between two cloud machines. If Explorer's
`sharing` partition turns out to have AMD GPUs, the mixed job can instead run
inside Explorer as one heterogeneous Slurm job. For a first session there,
[docs/EXPLORER_FIRST_SESSION.md](docs/EXPLORER_FIRST_SESSION.md) has the
commands in order, from a fresh login to the ladder's first rungs.

**GPU results:** from 2026-10-14 on Explorer, in [GPU results](#gpu-results)
below, made by [`scripts/gpu/validation_tables.py`](scripts/gpu/validation_tables.py)
from the kit's results: what each job ran on (hosts, GPUs, driver, torch, CUDA
or ROCm, NCCL or RCCL, gpubridge commit, interface), a verdict per item below
from the steps and stages that test it, and the preflight's network numbers.
Runs in CPU simulation never count towards an item. The tables number hosts
instead of naming them and show no paths. See the runbook's
[The validation tables](docs/GPU_VALIDATION_RUNBOOK.md#the-validation-tables).
The *Status* lines under each item describe the CPU work and scoped hardware
updates; the GPU verdicts are in the tables. The
[2026-10-20 split-test results](#split-test-results-2026-10-20) qualify the older
step-exit verdicts. What the bridge costs is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Blocking: can the job start at all?

1. **CUDA-PyTorch and ROCm-PyTorch processes rendezvous in one job.** Both
   builds must connect to one c10d store (TCPStore through `torchrun` or
   `tcp://`) and form one Gloo default group. This is expected to work because
   the store and Gloo run on CPU and are compiled into both builds, but nobody
   has checked it. Test it with a launcher that starts one `torchrun` per node,
   each with its own PyTorch build.

   **Status: works on CPU.** CUDA-build and ROCm-build processes rendezvoused
   through torchrun's c10d backend and completed discovery, all_reduce and
   broadcast in every run, with either build hosting the store. Rechecked on
   `main` after discovery started sending each rank's full probe record; see
   [Recheck on current main](#recheck-on-current-main-nested-probe-record).
   Still open: separate hosts on a real network, and the same with GPUs attached.
2. **Version matching between builds.** Use the same PyTorch release on both
   sides, e.g. `2.x.y+cu12x` and `2.x.y+rocm6.x`, and the same Python minor
   version, since discovery pickles objects with `all_gather_object`. Every
   rank needs PyTorch 2.8 or later: older releases can deadlock while tearing
   down Gloo groups
   ([pytorch/pytorch#154976](https://github.com/pytorch/pytorch/pull/154976)
   fixed it in 2.8). Pick both builds from 2.8 on.
   `init()` warns when releases differ but does not block. Things to find out:
   - Which release pairs actually work.
   - Whether a minor-version mismatch breaks the store or Gloo protocol.
   - Whether ROCm wheels lag behind CUDA wheels for the release you want.

   **Status: looser than feared on CPU.** Mixed releases from 2.9.1 to 2.14.1
   worked, in both directions, with Python 3.12.12 on every rank. CUDA and ROCm
   wheels exist for every release from 2.6 to 2.14 (Python 3.12, Linux x86_64),
   with different CUDA/ROCm versions per release. Still open: other Python
   versions, releases outside 2.9.1 to 2.14.1, and the GPU side.
3. **`new_group(backend="nccl")` on ranks outside the group.** Every rank
   creates every island group, so ROCm ranks register the NVIDIA island and the
   other way round. That should only touch the store, without initializing the
   backend. Confirm nothing tries to load NCCL on AMD ranks or RCCL on NVIDIA
   ranks.

## Correctness on GPUs

4. **`"nccl"` on ROCm really runs RCCL.** Check with `NCCL_DEBUG=INFO`; RCCL
   logs its own version.
5. **Device selection on AMD nodes.** Ranks use `cuda:$LOCAL_RANK`. Confirm this
   maps to the intended GPU under `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`
   and the cluster's scheduler.
6. **Stream ordering in the bridge step.** A leader's `tensor.cpu()` must see
   the finished island all_reduce, and the island broadcast must see the
   leader's `copy_` back to the GPU. PyTorch's synchronous collectives should
   guarantee both. Check it on both vendors with large tensors and a busy GPU.
   *Code audit (2026-10-10):* the island reduce is a synchronous `dist.reduce`,
   whose `work.wait()` makes the caller's current stream wait for it; the
   copy (`tensor.cpu()`) then runs on that same stream and blocks the host
   until it finishes. The `copy_` back and the island broadcast follow on the
   same stream. With `async_op=True` the same sequence runs on the worker's
   stream, which first waits on an event recorded on the caller's stream. The
   pipelined policy records its start event after the island reduce and makes
   the stream wait on its last copy back before the island broadcast.
   *Second audit (2026-10-11), three more cases:* (a) The only
   `non_blocking` device-to-host copies are the pipelined policy's, into
   pinned slots; the host waits on each chunk's copy event
   (`event.synchronize()`) before the bridge reads the slot. Correct.
   (b) A pinned slot reused while a copy back out of it is still queued:
   within one collective a slot's next copy in waits on the device for its
   last copy back, and a leader that only receives waits for it on the host
   before the bridge writes the slot. But a collective returns with copies
   back still queued, and the next pipelined collective started with no
   record of them, so it could overwrite a slot first (on the host when it
   only receives, from any stream; on the device from another stream).
   Fixed: the policy keeps each slot's last copy-back event from one
   collective to the next. (c) See item 16: `wait()` was correct, but
   `is_completed()` was not. The 2026-10-20 split test at `c92684b` passed
   the GPU stream/readiness suite and Gloo/staged-verbs checks; the collective
   `wait-skips-stream` mutation gap remains (item 16).
   *Step:* `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`):
   `test_bridge_copies_wait_for_the_callers_stream_and_the_island_reduce`
   writes the input at the end of a queue of matmuls on a side stream, sync
   and async, for every bridge policy (a pipelined bridge step that read a
   slot before its copy event would fail it), and
   `test_pinned_slots_are_not_reused_before_their_copies_back` holds a
   receiving island's copies back behind matmuls and starts the next
   broadcast at once, on the same stream and on another. On CPU,
   `test_the_next_collective_waits_for_a_slots_last_copy_back`
   (`tests/test_pipelined.py`) fails without the fix.
7. **float16 and other dtypes over Gloo between real hosts**, compared bit for
   bit with a single-vendor NCCL/RCCL all_reduce on integer-valued data.

## Operational

8. **Gloo networking.** Leaders on different nodes reach each other on the right
   interface; set `GLOO_SOCKET_IFNAME` on nodes with several NICs.
   *Status:* `scripts/gpu/preflight.py` chooses it by measurement: TCP both
   ways between every routed pair of interfaces of the two leader nodes, raw
   bandwidth each way over one stream, then a Gloo group on the pair whose
   slower direction is fastest. Steps 05 and 07 run it first and give each
   node its own choice, even when the two names differ. The bandwidth is one
   TCP stream driven from Python, a lower bound that may understate fast
   links. Works on CPU on one machine (`tests/test_gpu_kit.py`,
   `tests/test_gpu_jobs.py`, where two different names are faked). Still
   open: Explorer's real interfaces between `multigpu` nodes and between
   `gpu` and `sharing`.
9. **Timeouts.** While leaders work over the bridge, the other ranks wait inside
   an island broadcast. With large tensors the bridge step can be slow enough to
   trip the NCCL/RCCL watchdog. Check whether the `timeout` passed to `init()` is
   enough or whether the watchdog needs its own setting.
10. **`barrier()` on ROCm**, which calls `torch.cuda.synchronize`.

## Not correctness, but worth measuring early

11. Bridge throughput and latency compared with a single-vendor all_reduce, to
    set a baseline before any performance work.
    *Step:* `bench_all_reduce.py`, in Explorer steps 03 to 05 and 07,
    `amd/run_all.sh` and `mixed/node.sh`. For each size (1 KB to 1 GB, x4) it
    runs warmup calls, then times trials with CUDA events on the current
    stream. By default it puts three ops side by side: native NCCL/RCCL over
    every rank (or each island's own, where one group is impossible),
    gpubridge, and flat-gloo, with busbw as nccl-tests computes it and
    gpubridge and flat-gloo as multiples of the baseline. Every row names its
    baseline; each island's own is labeled `island-native (fewer ranks, best
    case)`, since it is not a like-for-like comparison. Where a preflight ran,
    `validation_tables.py` sets the bridge phase against the time the
    preflight's bandwidth predicts (runbook, "Predicted vs measured").
    *Status: GPU native/island/Gloo/staged-verbs timing baselines recorded
    at `c92684b` on the 2026-10-20 split test; see
    [results](#split-test-results-2026-10-20). Single-run timings do not
    establish statistical significance or cross-node performance.*

## Added with later features

Each item below came with a feature that is tested on CPU but makes a claim
only real GPUs or NICs can check. Each names the kit step that checks it.

12. **A rank whose `LOCAL_RANK` has no GPU fails `init()` on every rank.**
    `resolve_config` compares `LOCAL_RANK` with `torch.cuda.device_count()` and
    discovery reports a mismatch as that rank's problem; `set_device` now waits
    until discovery succeeds.
    *Status: works on CPU, with a faked device count
    (`test_an_out_of_range_local_rank_fails_init_on_every_rank`).* Still open:
    that the count PyTorch reports honours `CUDA_VISIBLE_DEVICES`,
    `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` and Slurm's GPU binding, so
    the check fires on real nodes. *Step:* `check.py --expect-init-error`,
    launched with one more process than the node has GPUs (Explorer step 01,
    `amd/run_all.sh`).
13. **The island step of reduce-bridge-broadcast is a reduce onto the leader**,
    not an all_reduce: only the leader needs the island sum, and a ring reduce
    moves about half the bytes.
    *Status: works on CPU: every layout still matches `flat-gloo` exactly, and
    a test counts one island reduce and one island broadcast per all_reduce.*
    On GPUs (2026-10-20, `c92684b`): nontrivial two-rank NCCL island-reduce
    phases and native island all_reduce baselines were measured with Gloo and
    staged verbs bridges in the [split test](#split-test-results-2026-10-20).
    This establishes measurement coverage, not a universal time saving; RCCL
    remains untested in these runs. Small tensors
    may see none, because both ops are latency-bound there. *Step:*
    `bench_all_reduce.py --ops native,island,gpubridge,flat-gloo --phases` puts the
    `island-reduce` phase next to a native island all_reduce (`island:<vendor>`)
    of the same size (Explorer steps 04, 05 and 07, `amd/run_all.sh`).
14. **Where NCCL/RCCL and Gloo agree on reductions.** gpubridge mixes the island
    backend and Gloo in one reduction, so a result is only well defined where
    both agree. PRODUCT is left out until they are shown to agree in every
    supported dtype, including 8-bit overflow. MAX and MIN with NaN in the data
    follow whatever the backends do.
    *Status: not verifiable on CPU, where islands are Gloo too.* *Step:*
    `check.py` stage `reduction_agreement` (informational) compares a raw
    island-backend all_reduce with a Gloo one: PRODUCT in every dtype, MAX/MIN
    with a NaN on one rank. Read `product_agree` and `nan_max_min` in the rank
    records.
15. **MAX, MIN, AVG, `all_gather_into_tensor` and `reduce_scatter_tensor` on
    NCCL/RCCL islands.** Native-only uses NCCL's own all_gather_into_tensor
    and reduce_scatter_tensor; reduce-bridge-broadcast uses island gather and
    scatter, which torch builds from NCCL send/recv.
    *Status: works on CPU, bit for bit against `flat-gloo` in every layout and
    policy (`tests/test_ops.py`).* Still open: the same on GPUs. *Step:*
    `check.py` stages `reductions`, `all_gather` and `reduce_scatter`, and
    `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`), which compares every policy with `flat-gloo` bit for
    bit for every op, fp32/fp16/bf16 and odd sizes. Bit for bit holds although
    NCCL, RCCL and Gloo add in different orders: the data are small integers,
    so every partial sum is exact in every dtype, and the tests use 1, 2 or 4
    ranks so every average is exact too (`tests/test_gpu.py` explains).
16. **Async collectives keep their stream semantics.** The worker's stream
    waits on an event recorded on the caller's stream; `wait()` makes the
    caller's stream wait on an event recorded after the collective, on the
    worker's stream. `is_completed()` used to turn True as soon as the worker
    thread had queued the last kernels (the island broadcast), before they
    ran; with NCCL it means the GPU work is done. Fixed (2026-10-11): it also
    asks that event (`test_is_completed_waits_for_the_gpu_work_too`).
    *Status: ordering, results and failure handling work on CPU
    (`tests/test_ops.py`), where there are no streams.* Still open: the
    decisive coverage of `wait()` stream ordering: the `wait-skips-stream`
    mutation remains uncaught. The 2026-10-20 split test at `c92684b` passed
    Gloo and staged-verbs async checks and the GPU stream/readiness suite;
    these passes do not close that gap. *Step:* `check.py` stage `async`
    queues the all_reduce right behind GPU work and checks the result on the
    caller's stream after `wait()`, with no host sync in between. Also
    `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`), which submits from one non-default stream and waits
    on another, and reads a large result right after `is_completed()` turns
    True, without `wait()`
    (`test_is_completed_means_the_result_is_ready`).
17. **The pipelined bridge is correct on GPUs.** `pipelined-reduce-bridge-broadcast`
    (opt-in) copies chunks through pinned host memory on two side streams,
    ordered by CUDA events, waiting on the host only for the chunk the bridge
    needs next.
    *Status: works on CPU, bit for bit against `flat-gloo`, including with a
    staging layer that runs each copy only when something waits for its event,
    so a missing wait corrupts the result (`tests/test_pipelined.py` shows it
    does), and one that leaves copies back queued when a collective returns,
    as a GPU does, so the next collective must wait for them (item 6 (b)).*
    On GPUs (2026-10-20, `c92684b`): the full pipelined Gloo correctness
    check passed on two 2-GPU NCCL islands. Verbs pipeline timings ran,
    but a full verbs-pipeline payload correctness suite remains open.
    *Step:* `check.py --policy pipelined-reduce-bridge-broadcast` (its
    `busy_gpu` stage queues work ahead of the collective), in Explorer steps
    04, 05 and 07 and `amd/run_all.sh` (`OPT_IN_POLICIES`).
18. **The overlap pays off.** For large tensors the pipelined bridge should
    beat reduce-bridge-broadcast without hurting small ones; the best chunk
    size depends on the machine.
    *Status: measured on the 2026-10-20 split test at `c92684b`; the full
    performance goal is not established.* Gloo pipeline timings beat the
    primary default at matched 1–256 MiB points, but the sweep omitted sub-MiB
    sizes and the Gloo default at 1 GiB. Candidate timings include small-size
    regressions. Verbs generally regressed; only 64 MiB chunks at 1 GiB beat
    default. There is no direct overlap trace or statistical significance.
    See [results](#split-test-results-2026-10-20). *Step:* `bench_all_reduce.py
    --candidates` (or `--write-thresholds`), rerun with a few
    `GPUBRIDGE_CHUNK_BYTES` values.
19. **Thresholds measured on GPUs drive `auto-tuned`.**
    *Status: on CPU, a file written by `--write-thresholds` loads, and
    `auto-tuned` routes each size to the rule's policy.* Still open: the
    portability of the thresholds to other hardware. The 2026-10-20 split
    test at `c92684b` wrote GPU-derived thresholds and passed the full
    `auto-tuned` correctness recheck using them. *Step:* the `bench_candidates` and
    `check_auto_tuned` steps write `thresholds.json` next to the results and
    rerun the check under `auto-tuned` with it.
20. **The sharded bridge is correct on GPUs.** `sharded-bridge` (opt-in) runs
    NCCL/RCCL `reduce_scatter_tensor` and `all_gather_into_tensor` on islands
    of exactly k ranks, k NCCL/RCCL reduces and broadcasts on larger ones, and
    k Gloo bridge groups at once, one per island rank below k.
    *Status: works on CPU, bit for bit against `flat-gloo` for every
    supported dtype and SUM/AVG/MAX/MIN, in 2+2, 3+1, 3+2, 2+4, 3+3,
    interleaved and three-island layouts, with sizes that need padding and
    sizes smaller than k (`tests/test_sharded.py`).* On GPUs (2026-10-20,
    `c92684b`): full Gloo-bridge correctness passed with k=2 on two 2-GPU
    NCCL islands. Earlier GPU checks had k=1. Separate-host bridge groups
    and verbs-sharded correctness remain open.
    *Step:* `check.py --policy sharded-bridge` in Explorer steps 04, 05 and 07
    and `amd/run_all.sh` (`OPT_IN_POLICIES`).
21. **More bridge links mean more cross-vendor bandwidth.** Each link carries
    1/k of the bytes, and each of the k ranks copies its own segment to the
    host, so with the bridge as the bottleneck, k links should beat one.
    *Status: not shown on CPU.* In CPU simulation (8 ranks on one 10-core
    machine), `sharded-bridge` is 1.3x to 1.8x slower than
    reduce-bridge-broadcast. Every rank shares one CPU and memory, so the
    bridge phase barely changes with 4 links (10.8 ms against 10.9 ms at
    64 MB). The island steps are Gloo there, and on CPU islands the
    reduce_scatter is an all_reduce plus a slice, which makes them slower.
    GPU islands use NCCL/RCCL for those steps. The 2026-10-20 split test
    at `c92684b` adds k=2 Gloo candidate timings on one host, but no controlled
    k=1 versus k=2 scaling study or independent network-link bandwidth result.
    Earlier GPU step passes used k=1 and did not test this claim.
    *Step:* `bench_candidates` in step 05, where the bridge crosses the
    network, and in step 07: compare the `policy:sharded-bridge` and
    `policy:reduce-bridge-broadcast` columns. Step 05's
    `bench_phases_sharded-bridge` times the bridge phase alone, next to the
    `bench_split_by_node` phases of reduce-bridge-broadcast.
22. **DDP with `gpubridge.ddp_comm_hook` trains correctly on GPUs.** Two
    things only GPUs show. First, DDP's own traffic runs on the Gloo world
    group with device tensors: the shape check and parameter broadcast when it
    wraps the model, on CUDA and ROCm builds. Second, the stream handoff: DDP
    calls the hook on the autograd thread, the bucket's all_reduce starts after
    the work queued on that thread's stream, and DDP reads the result through
    the device-aware future.
    *Status: works on CPU. Gradients match the exact average bit for bit (with
    integer-valued gradients) under the default, sharded, pipelined,
    native-only and flat-gloo policies, with one bucket per parameter. A
    failed bucket makes `backward()` raise on every rank, and later
    collectives refuse to start (`tests/test_ddp.py`).* *Step:*
    `examples/train_synthetic.py`, which exits non-zero if parameters diverge:
    `train_ddp` in Explorer steps 04 (split test) and 07 (mixed NVIDIA +
    AMD), `train_ddp_split` in `amd/run_all.sh`, and the single-island
    scaling runs in step 06. *On GPUs (V100 on torch 2.14.1+cu126, MI50 on
    2.9.1+rocm6.3): passes. 0 of 24 mixed runs diverged, 16 uncompressed and
    8 with fp8 and the PyTorch kernels: the ranks' gradients and parameters
    were the same bit for bit after every step
    ([below](#items-22-and-33-2026-10-15-and-2026-10-16)). That bit check is
    the step's verdict; the float64 checksum is kept beside it, for
    information only. Two earlier runs had failed: job 11090726 (commit
    4e7f04e, fp8 with the PyTorch kernels) and job 11112713 (commit 1c7d107,
    uncompressed). Both were judged by the checksum alone, a sum of each
    rank's parameters taken on its own GPU, which flagged 4 of the 24 runs
    whose parameters were identical; so the two failures were most likely
    its false alarms (inferred: their records hold only the sums).*
23. **What `gpubridge.in_flight()` shows on GPUs.** It reflects the host: a
    thread is listed while it is inside a gpubridge call. NCCL and RCCL calls
    return once their work is queued, so on GPUs a non-leader whose island
    collective is still waiting on the device may already have left the
    call, while a leader blocks in the bridge's device-to-host copy or in
    Gloo. Record what each rank shows while one rank is late.
    *Status: works on CPU: a late rank shows nothing; the others show the
    collective's seq, op and current phase (bridge on the waiting leader,
    island-broadcast and island-reduce on the others), with and without
    observers; async collectives show on the worker thread; nested phases and
    failed calls are tracked (`tests/test_observe.py`).* *Step:* the
    `in_flight` stage of `check.py`, in every step that runs it. The late rank
    joins only once every other rank's view has stayed the same for
    `--in-flight-settle` seconds (1 by default), in an order the job's store
    enforces, so a loaded machine changes how long the stage takes, not what
    it records.
24. **No island puts two ranks on one GPU.** NCCL and RCCL refuse two ranks
    of one communicator on the same GPU, and an island is one communicator.
    Split-test mode lets ranks share GPUs, so a layout such as 4 ranks on a
    2-GPU node with `alternate` would fail inside NCCL. `init()` compares the
    ranks of each island by the GPU each one drives (its UUID and PCI
    address: device indices differ between processes under
    `CUDA_VISIBLE_DEVICES` or Slurm's GPU binding) and fails on every rank,
    before creating any group, naming the ranks. Every layout in the runbook
    passes: steps 02 (two single-rank islands share the GPU), 04 (a GPU per
    rank, both layouts) and 05, the AMD script, the GPU-only tests, and
    `half` on a 2-GPU node with up to 4 ranks. The only NCCL/RCCL groups
    gpubridge creates are the island groups; the world group, the bridge and
    the sharded bridge's links use Gloo.
    *Status: works on CPU with faked GPU identities
    (`tests/test_split_test.py`).* A GPU that reports neither a UUID nor a
    PCI address (some ROCm builds) can't be compared: `init()` then warns on
    rank 0, naming the island, and skips that island's check instead of
    failing; tested on CPU with every rank reporting nothing. Still open: that
    PyTorch reports a UUID or PCI address on both vendors. *Step:* Explorer
    step 02, `shared_gpu_check`: 4 ranks on its one GPU with `alternate` must
    fail `init()` on every rank with the duplicate-GPU error, within 60 s;
    `steps.jsonl` records `"expected failure confirmed"` or `"check did not
    fire"`.
25. **Pinned staging for the default bridge pays off, and stays correct.**
    With `GPUBRIDGE_PINNED_STAGING=1`, the leaders of
    `reduce-bridge-broadcast` copy into one pinned host buffer with
    `non_blocking=True`, wait on that copy's event (not the whole device)
    before Gloo reads it, and leave the copy back queued; the next
    collective's copy out waits for it on the device, and a larger tensor
    replaces the buffer only after it. Check that results stay bit for bit
    those of `flat-gloo`, including back-to-back collectives on different
    streams, and whether the `bridge` phase gets shorter than with pageable
    `.cpu()` copies, at which sizes.
    *Status: works on CPU, where the copies are checked with lazy pretend
    streams that run a copy only when something waits for it
    (`tests/test_pinned_staging.py`); the GPU tests are written
    (`tests/test_gpu.py`: every op against `flat-gloo`, streams, completion,
    buffer reuse).* *Step:* `check_pinned` and `bench_pinned` in Explorer
    steps 04, 05 and 07 and `amd/run_all.sh`, right after the pageable
    benchmark in the same allocation (`bench`, or `bench_split_by_node` in
    step 05); compare their gpubridge rows and `bridge` phases.
26. **The UCX bridge (`bridge="ucx"`) is correct on GPUs, between real NVIDIA
    and AMD leaders.** Each side builds the library against its own UCX
    (CUDA-built on NVIDIA, ROCm-built on AMD); the handshake refuses
    mismatched UCX versions. GPU tensors go to UCX directly where its build
    has the memory type, and through pinned host memory where it doesn't; the
    run record says which (`run.bridge_path`). Check results bit for bit
    against `flat-gloo`, and that `direct` is what was used where GPUDirect
    RDMA (or ROCm peer memory, or dma-buf) is available. On the direct path
    the transport flushes the NIC's writes into the GPU after every receive
    (`run.bridge_flush` says how; UCX doesn't, see docs/RDMA_TRANSPORT.md
    section 4): check that received data is fresh with the NIC writing GPU
    memory (`test_data_ucx_writes_into_gpu_memory_is_what_the_next_kernel_reads`
    with `GPUBRIDGE_TEST_UCX_TLS=rc,cuda_copy`, or across nodes), and how long
    a flush takes.
    *Status: works on CPU over UCX's TCP transport, every bridge policy on two,
    three and four leaders (`tests/test_ucx.py`); the flush's placement is
    tested on CPU against a fake worker whose receives land only when flushed
    (`tests/test_gdr_flush.py`); the device path's GPU tests are written
    (`tests/test_gpu.py`). Nothing has run on RDMA hardware. On Explorer, the
    V100 and MI50 nodes checked have no RDMA path between them, and no UCX
    there has a `cuda` or `rocm` transport
    ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#rdma-between-the-node-types)).*
    *Step:* `ucx_build` and `check_ucx` in Explorer steps 05 and 07, with
    `UCX_BRIDGE=1`. See [docs/RDMA_TRANSPORT.md](docs/RDMA_TRANSPORT.md) for
    the questions to settle first (InfiniBand on the GPU nodes, GPUDirect
    RDMA, a UCX module).
27. **The UCX bridge against the Gloo bridge.** The bridge phase over UCX
    (RDMA where available) next to the Gloo bridge's, at each size, and against
    the preflight's TCP prediction, which RDMA should beat.
    *Status: not run.* *Step:* `bench_ucx` in steps 05 and 07 with
    `UCX_BRIDGE=1`; compare with the Gloo benchmark of the same job.
28. **Bridge compression's Triton kernels give the reference's bits on NVIDIA
    and AMD GPUs.** `GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8` (opt-in) promises
    that every rank still ends with the same bits, which needs the compiled
    kernels on both vendors to match the plain-PyTorch reference exactly
    (integer arithmetic on bit patterns, power-of-two scales, explicit
    flushing below 2^-100; docs/COMPRESSION.md). Check that the GPU tests pass,
    that `bench_compression.py` reports "same bits as CPU" for every mode and
    dtype, that its payload and result digests are identical on the NVIDIA
    and the AMD node, and that `bench_compressed` reports every rank the same.
    Also how close each kernel gets to the GPU's memory bandwidth.
    *Status: works on CPU, where Triton's interpreter runs the kernels and
    they match the reference bit for bit (`tests/test_compression_kernels.py`);
    in simulated jobs one leader on Triton and the other on PyTorch agree
    (`tests/test_compression.py`). The GPU tests are written
    (`test_compression_kernels.py`, marked `gpu`).* *Step:* `gpu_tests` and
    `bench_compression_kernels` in Explorer steps 04 and 07, `amd/run_all.sh`
    and `mixed/node.sh`; in step 07 compare `bench_compression_gpu` with
    `bench_compression_amd`.
29. **Where bridge compression pays off.** Compression adds two kernels and
    saves bridge bytes (2x for `bf16`, 3.9x for `fp8` on float32). From which
    size, if any, is a compressed all_reduce faster, per policy, and how much
    shorter does the `bridge` phase get? The `compress` and `decompress`
    phases show the kernels' share. And the cost in accuracy, from the error
    `bench_compressed` reports, and from loss curves with and without it.
    *Status: single-host split-test timings measured on 2026-10-20 at
    `c92684b`: `bench_compressed` passed with 40 rows and across-rank identity
    checks; compression kernels passed 3/3 CPU-reference checks. These are
    descriptive timings, not representative compressed-training accuracy
    validation. Cross-node and mixed-vendor coverage remain open; on CPU the
    PyTorch reference kernels are slower than the bridge they save.* See
    [dated results](#split-test-results-2026-10-20).
    *Step:* `bench_compressed` (`--compression
    none,bf16,fp8 --phases`) in Explorer steps 04 and 07, `amd/run_all.sh`
    and `mixed/node.sh`; its summary.md says from which size each mode is
    faster.
30. **DDP and FSDP2 through `backend="gpubridge"` on GPUs.** The
    torch.distributed backend (docs/TORCH_BACKEND.md) carries a script's
    collectives as gpubridge async collectives. Check that
    `examples/fsdp2_torch_backend.py --verify` passes across a split test and
    across real NVIDIA + AMD ranks: on exact data, its parameters must equal a
    single-process replay bit for bit, which a collective reading or writing a
    tensor at the wrong moment would break even when every rank agrees. Also
    that its plain training run ends with every rank's parameters identical,
    and that the GPU tests pass (DDP on two GPUs split into two islands, with a
    CPU tensor carried by Gloo; `init_process_group(device_id=...)`). Step time
    against FSDP2 on one vendor's NCCL/RCCL isn't measured by the kit yet.
    *Status: works on CPU in simulation mode on torch 2.8.0 and 2.14.1: DDP and
    FSDP2 bit for bit like Gloo on integer-valued data, every collective the
    backend carries, every refusal (`tests/test_torch_backend.py`), and the
    example's `--verify`; the GPU tests are written.* *Step:*
    `verify_fsdp2_backend` and `train_fsdp2_backend` in Explorer steps 04 and
    07, `amd/run_all.sh` and `mixed/node.sh`; `gpu_tests`.
31. **HSDP through the backend: shard within islands, replicate across
    vendors.** Subgroups give each vendor island a native-only shard group
    and each replicate group a bridged topology (docs/TORCH_BACKEND.md,
    "Subgroups and HSDP"). Check that `examples/fsdp2_torch_backend.py
    --verify --hsdp 2` passes on the split test and on real NVIDIA + AMD
    ranks, bit for bit against a single-process replay, and that the plain
    `--hsdp 2` run ends with every rank's parameters identical. Its JSON has
    the median step time; set it against `train_fsdp2_backend`'s from the same
    job (plain FSDP2 over every rank, whose all-gathers and reduce-scatters
    cross the bridge). On the mixed job this needs as many NVIDIA GPUs as AMD;
    with one of each, the shard groups are single ranks and only the replicate
    all_reduce is exercised. *Status: works on CPU in simulation mode on torch
    2.14.1 and in CI on torch 2.8.0 and 2.14.1: bit for bit like one process,
    every all-gather and reduce-scatter native-only with no bridge phase
    (`tests/test_torch_backend.py`), and the example's `--hsdp 2` and
    `--verify --hsdp 2` on 4 simulated ranks. On GPUs: passes since #80. Job
    11112713, commit 1c7d107, V100 on torch 2.14.1+cu126 and MI50 on
    2.9.1+rocm6.3 (`venv-rocm63`): `train_hsdp_backend` and
    `verify_hsdp_backend` passed. With one GPU per vendor the shard groups are
    single ranks, so only the replicate all_reduce crossed the bridge.* *Step:*
    `train_hsdp_backend` and `verify_hsdp_backend` in Explorer steps 04 and 07.
33. **Data-parallel training across vendors tracks one GPU, compressed or
    not.** `train_dp.py` on `llama.py`'s model, 1,000 steps (at 500 some runs
    are still mid-drop off the loss plateau), the same global
    batch and seed as one V100 alone: DDP through `gpubridge.ddp_comm_hook`
    uncompressed, with bf16 and with fp8 (error feedback), FSDP2 through the
    backend, and DDP with batch shares by GPU speed. Uncompressed and
    speed-weighted runs should track the single GPU's loss curve closely (the
    same mean gradient each step, reduced in another order on other hardware);
    compressed ones within a tolerance to be justified from the measured gaps.
    Check that the ranks' parameters stay bit-identical (every 50 steps), and
    record tokens/s, the step split and the share of communication backward
    hides. *Status: works on CPU in simulation mode: DDP, uneven shares and
    FSDP2 within 1e-4 of one process over 12 steps, fp8 within 0.05
    (`tests/test_gpu_kit.py`). On a V100 and an MI50 (Explorer, 1,000 steps,
    two runs per mode against eight one-V100 runs; the rule was fixed before
    the new runs, and by it S = 0.0131, the widest gap between two one-V100
    runs, which measures uncompressed run-to-run noise;
    [below](#items-22-and-33-2026-10-15-and-2026-10-16)): uncompressed DDP
    passes, at 0.32 S and 0.34 S. DDP with shares by speed and FSDP2 fail by
    the rule, one run of each outside S (1.30 S and 10.2 S); both reached the
    plateau exit (10-step mean loss below 2.0) late, at steps 655 and 893
    against 557 to 591 for nine of eleven one-V100 runs, which points at
    timing (inferred; untested at 2,000 steps). bf16 and fp8 fail by the rule,
    all four runs outside S (2.51 S to 528 S): a lossy mode is expected to
    land outside a measure of uncompressed noise, and whether these gaps are
    acceptable is open. The ranks' parameters were bit-identical at every
    check in every DDP run, compressed or not; FSDP2's sharded parameters
    aren't compared. At 2,000
    steps one V100 alone also falls outside S against its own runs (one run
    of three, at 25 S), so the planned next step is a redesigned test:
    several seeds per mode, the end loss once every run has reached the
    plateau exit, and the steps and time to a target loss (below).
    Bucket sizes ran 100 steps each (docs/BENCHMARKS.md).* *Step:* `dp_*` in
    Explorer step 07 with `STEPS_07=dp`, and `dp_single` in `submit.sh
    dp-baseline`.
34. **Pipeline training across vendors tracks one GPU.** `train_pipeline.py`
    (docs/PIPELINE.md): `llama.py`'s model in two stages, the first on the
    NVIDIA GPU and the second on the AMD one, with torch's GPipe and 1F1B
    schedules through `backend="gpubridge"`'s point-to-point, 500 steps.
    Check that the loss curve tracks one V100's with the same global batch,
    microbatches and seed, and record tokens/s, each stage's compute, its idle
    time and the time an activation takes to cross the bridge. *Status: works
    on CPU in simulation mode: bit for bit like the same pipeline on a plain
    Gloo group (2 stages, and 4 over 2 + 2 islands), within 1e-4 of one
    process (`tests/test_gpu_kit.py`). On a V100 and an MI50 (Explorer job
    11109376, both on torch 2.9.1, 1,000 steps): over the last 100 steps 1F1B
    is as close to four one-V100 runs (0.0036 to 0.0052) as they are to each
    other (0.0041 to 0.0056), GPipe 0.02 above them after a later drop off
    the loss plateau; 1.04x to 1.05x the tokens/s of the V100 alone with the
    same micro-batches (docs/PIPELINE.md).* *Step:* `pp_gpipe`
    and `pp_1f1b` in Explorer step 07 with `STEPS_07=pipeline`, and
    `dp_single` in `submit.sh dp-baseline` with `DP_MICRO` set to the
    microbatches.

35. **The multitcp bridge against the Gloo bridge.** `bridge="multitcp"`
    (docs/RUST_TRANSPORT.md) carries the bridge over several TCP connections
    per pair and reduces each chunk as it arrives. Between two machines, its
    bridge phase at each size against the Gloo bridge's (default and pinned
    staging) in the same job, and against the preflight's prediction; one
    connection against several, for the connections' share. It takes host
    tensors only, so the GPU side is the policies' usual staging. Stage 1 stays
    only if, in each of two jobs, the median time of the whole float32
    all_reduce with pinned staging is at most 0.8 times the pinned Gloo
    bridge's at every size from 64 MB to 1 GB.
    *Status: works on CPU over loopback TCP, every bridge policy on two, three
    and four leaders, bit for bit against `flat-gloo` (`tests/test_multitcp.py`).
    Not run between two machines.* *Step:* `multitcp_build` and the
    `matrix_D-multitcp*` runs in Explorer step 08, with `MULTITCP_BRIDGE=1`.

36. **Automatic stage balancing against even and hand-tuned splits.**
    `train_pipeline.py --split auto` (docs/PIPELINE.md, "Automatic stage
    balancing"): `gpubridge.balance` profiles each block on each stage's GPU
    and the links between stages, then picks the contiguous split whose
    slowest stage, transfers included, is fastest within each GPU's memory. On
    a V100 + MI50, 500 steps: tokens/s, per-stage times and idle time for the
    even split, the best of a few hand-tuned splits and the automatic one, same
    model and seed. *Status: works on CPU: the partitioner matches brute force
    on random cases with injected device speeds, links and memory limits
    (`tests/test_balance.py`), and `--split auto` trains bit for bit like the
    same split given by hand (`tests/test_gpu_kit.py`). On a V100 + MI50
    (Explorer job 11113912, 1F1B, 500 steps): even 13,062 tokens/s, 7 + 5
    14,216, 8 + 4 15,577, 9 + 3 16,021, and the automatic split chose 8 + 4,
    15,559: 1.19x even, 3.0% below the best hand-tuned split. The cost model
    misses a per-microbatch cost on the MI50's stage (docs/PIPELINE.md).*
    *Step:* `pp_<schedule>_<split>` in Explorer step 07 with
    `STEPS_07=pipeline PP_SPLITS="even 7,5 8,4 9,3 auto"`.

37. **Megatron-Core's pipeline across vendors tracks one GPU.**
    `megatron_pp.py` (docs/MEGATRON.md): Megatron-Core's GPT in two stages, one
    on the NVIDIA GPU and one on the AMD GPU, with Megatron's own 1F1B schedule
    through `backend="gpubridge"`, 200 steps. Check that it runs at all, and
    that its loss curve tracks the same model, weights and data in one stage on
    the V100; record tokens/s. *Status: Megatron's group setup on
    `backend="gpubridge"` and a `batch_isend_irecv` across its pipeline group
    work on CPU (checked by hand); Megatron's pipeline code needs CUDA, so no
    CPU test. On a V100 + MI50 (Explorer job 11117506, Megatron-Core 0.14.0,
    200 steps): it runs, the two-stage loss curve stays within 1.4e-4 of one
    stage on the V100 at every step, at 1.14x its tokens/s (docs/MEGATRON.md).*
    *Step:* `megatron_pp1` and `megatron_pp2` in
    Explorer step 07 with `STEPS_07=megatron`.

38. **The verbs bridge between hosts on RDMA hardware.** `bridge="verbs"`
    (docs/VERBS_TRANSPORT.md) moves the bridge over RC queue pairs on
    libibverbs: its wire protocol (version 2) of RC SEND/RECV with credits and a
    completion thread, and RDMA WRITE into memory registered for one receive
    (rendezvous, from `GPUBRIDGE_VERBS_RNDV_BYTES`). Between two machines on one
    InfiniBand or RoCE fabric (step 05 with `VERBS_BRIDGE=1`): the endpoint each
    leader selects (device, port, GID) as telemetry reports it, and that the
    provider is libibverbs, not MOCK; host-to-host verbs without collectives
    (`verbs_endpoint`: latency, a bandwidth sweep, SHA-256 of every buffer on
    both sides, CPU time); host tensors through the collectives exactly against
    a Gloo reduction (`verbs_host`); every bridge policy (`check_verbs`); a
    deliberately lost peer and a short deadline fail with their reasons;
    teardown with work outstanding leaks nothing; no bridge payload on TCP
    (every bridge leader's `verbs_kit` `tcp_evidence` is `pass`: the bridge's own
    libibverbs telemetry moved the payload with completions and no errors, and
    the process's own TCP sockets, by `ss -tinp`, moved no more than 5% of it;
    port and netdev counters are recorded as context, never as proof); then its bridge phase against the Gloo, multitcp and UCX bridges at
    each size (`bench_verbs`). *Status: real libibverbs host and staged-GPU
    traffic passed on one HCA on one host (loopback), not between hosts.*
    At `c92684b` the staged kit, default-policy check and benchmark passed;
    host collectives failed before payload because CPU tensors met CUDA
    communicators. The `0cf32a3` recheck passed exact host collectives with
    CPU/Gloo islands. Before per-process attribution, same-host host-mode
    no-silent-TCP evidence was `not-decisive` and `--require-no-tcp` failed
    closed (at `0cf32a3`, again at `4c13375`). With the kit's per-process
    attribution of same-host sockets, the `4ffcf93` rerun on one host was
    decisive: strict host collectives passed (28/28 exact, verbs evidence
    passed, 1728 B of counted TCP), and both real-TCP negatives (a flat
    world-Gloo policy and the Gloo bridge) recorded verdict `fail` on counted
    TCP over the limit. That is one-host loopback evidence, not
    cross-node no-TCP acceptance. Cross-node RDMA,
    hardware failure/teardown cases and transport comparisons beyond the
    measured Gloo/verbs pair remain open. See
    [results](#split-test-results-2026-10-20).

39. **GPU tensors staged through the verbs bridge.** With
    `GPUBRIDGE_VERBS_GPU_PATH=staged`, every call on a GPU tensor copies it
    into a pinned host buffer registered with the NIC on the caller's stream,
    waits for that copy before the bridge reads, and queues the copy back
    (`verbs_gpu_staged`, steps 05 and 07). Every call's record says `path`
    `staged` and its reason; results exact against Gloo; no payload on TCP;
    the same with async collectives and busy GPUs (`check_verbs`). Same-vendor
    (step 05) and cross-vendor (step 07) separately. *Status: staged GPU
    kit and default-policy correctness passed on the 2026-10-20 split test
    at `c92684b`, with two 2-GPU NCCL islands of identical V100s and
    single-HCA loopback.* The exact kit's largest payload was 512 MiB
    despite a 1 GiB cap; the benchmark reached 1 GiB. GPU-mode strict TCP
    evidence passed, including the `0cf32a3` regression recheck; this does
    not prove cross-node no-fallback behavior. Cross-node, AMD/mixed-vendor
    and direct/GPUDirect paths remain unproven by these runs.
    *Prerequisites for the remaining two-host and direct-path checks:* two
    allocated nodes with an RDMA device each on one fabric, permission to open
    `/dev/infiniband/uverbs*` from the job, a verbs library built with
    rdma-core headers (`python -m gpubridge.verbs build` or
    `GPUBRIDGE_VERBS_INCLUDE_DIR`), GPUs and a CUDA or ROCm build of PyTorch
    on those nodes, and authorization to run the job. Items 40 and 41 add
    their vendor-specific direct-memory requirements.
    `GPUBRIDGE_VERBS_STAGED_OVERLAP=1` (copies overlapped chunk by chunk, two
    leaders) must pass the same checks with the same messages on the wire, and a
    call that fails part way raises its own error only after its copies on the
    caller's stream have passed (or, if that wait fails too, with the wait's
    error chained); whether it is faster is the conditional A/B of
    docs/verbs-perf-20261019.md, section 7. *Status of the option: not run on
    hardware; the 2026-10-20 results above predate it. Its stream ordering is
    tested with CPU tensors standing in and lazy copies
    (`tests/test_verbs_staged_overlap.py`), not on a GPU.* Its GPU check needs
    the GPUs, RDMA device, verbs library and authorization listed above.

40. **NVIDIA GPU memory direct through the verbs bridge.** With
    `GPUBRIDGE_VERBS_GPU_PATH=direct` on NVIDIA leaders, every call registers the
    tensor (and the reduction's scratch) with the NIC: a dma-buf from
    `cuMemGetHandleForAddressRange` into `ibv_reg_dmabuf_mr` where the driver
    reports `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`, else `ibv_reg_mr` on the
    device pointer with `nvidia_peermem` loaded; `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`
    set once per registration; the cache keyed by `CU_POINTER_ATTRIBUTE_BUFFER_ID`;
    the host waits for the caller's stream before the NIC touches the memory;
    `cuFlushGPUDirectRDMAWrites` after receives where the GPU doesn't order
    remote writes itself (`verbs_gpu_direct`). The record names the method and
    the flush; `direct` must fail loudly, not stage, where a check fails.
    Results exact against Gloo; registrations and dma-buf descriptors released
    at close (telemetry). *Status: selection, descriptor lifetimes, cache
    invalidation on reallocation and the rendezvous into device memory are
    tested on a mock GPU and the mock provider (`tests/direct.rs`); the CUDA
    calls are checked against CUDA 12.8's cuda.h (`shim/gpu_abi_check.c`), never
    run.* *BLOCKED: hardware/authorization:* item 39's
    prerequisites on NVIDIA GPUs, plus either dma-buf support (the open GPU
    kernel modules, a kernel with dma-buf, and an RDMA driver that imports
    dma-bufs: `rdma_report.sh` records each) or the `nvidia_peermem` module
    loaded on both nodes.

41. **AMD GPU memory direct through the verbs bridge.** As item 40 on AMD
    leaders: the dma-buf comes from ROCr (`hsa_amd_portable_export_dmabuf`, closed
    with `hsa_amd_portable_close_dmabuf` after deregistering) where ROCr reports
    `HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED`; allocation identity from HIP's
    `HIP_POINTER_ATTRIBUTE_BUFFER_ID` and `hipMemGetAddressRange`; after
    receives, `hipDeviceFlushGPUDirectRDMAWrites` or the HDP flush register
    (as the UCX bridge's direct path). No peer-memory path on AMD. Cross-vendor
    (one NVIDIA and one AMD leader, step 07) separately from AMD pairs. *Status:
    the ROCm calls are checked against ROCm 6.3's HIP and ROCr headers
    (`shim/gpu_abi_check.c`), never run; the selection and lifetimes are the
    mock GPU's tests.* *BLOCKED:
    hardware/authorization:* item 39's prerequisites on AMD GPUs with ROCm 6.x,
    a kernel and amdgpu driver with dma-buf export, and an RDMA driver that
    imports dma-bufs.

42. **Training over the verbs bridge against the Gloo bridge.**
    `train_bench.py` with `GPUBRIDGE_BRIDGE=verbs` and with `gloo` on the same
    nodes (`train_verbs`, `train_gloo`, steps 05 and 07): the loss curves agree
    as the Gloo bridge's runs do, throughput and the step split side by side,
    the verbs runs' observations (`verbs_observations/`) cover every step.
    *Status: native/Gloo/staged-verbs GPT-2-small training ran on the
    2026-10-20 split test at `c92684b`, at 10/25/100 MB bucket caps; every
    cell had finite, rank-identical parameters.* Earlier split-test runs
    on the same node also exercised staged-verbs training. These are
    within-run checks and descriptive timings, not cross-arm numerical
    equivalence or significant speedups. Cross-node and mixed-vendor verbs
    training remain open. See [results](#split-test-results-2026-10-20).

43. **Deterministic numerical comparison before performance.**
    [NUMERICAL_GATE.md](docs/NUMERICAL_GATE.md) freezes a four-rank native NCCL,
    equal Gloo-island and staged-verbs comparison with compression off. Exact
    collective probes, independent FP64 gradient/update/optimizer-state references,
    cross-arm parameters and reduced bad-reduction/scale/data controls must pass
    before representative throughput. The actual verbs endpoints must attest
    non-mock libibverbs, staged traffic and matching native-library hashes.
    *Status: the bounded tiny gate was accepted on real GPUs at `f4146a4`,
    with 48/48 positive rank records, zero violations and all three bad
    controls rejected for their intended reasons; recorded in the
    [changelog](CHANGELOG.md#unreleased).* Four identical NVIDIA
    GPUs establish only single-host split-island evidence, not mixed-vendor or
    cross-node acceptance.
    This is a separate strict comparison report; legacy kit step-exit tables
    cannot infer its acceptance from training exits. Representative GPT
    numerical equivalence remains unproven.

44. **A job with no Gloo group starts and runs on GPUs.** Since 2026-11-11 a job
    that doesn't ask for Gloo (`bridge="auto"`, `"tcp"`, `"verbs"`, `"ucx"`)
    has no Gloo process group: discovery is a record per rank in the job's
    c10d store, the default process group is gpubridge's own
    (`GpubridgeProcessGroup`), and the NCCL/RCCL island groups are created
    under it by their own members (`new_group(backend="nccl",
    use_local_synchronization=True)` with gpubridge's group names). Check, on
    one vendor, in a split test and on real NVIDIA + AMD ranks, on torch 2.8
    with Python 3.10 and on torch 2.14 or 2.15 with Python 3.12:
    `gpubridge.init()` returns on every rank; `topology.gloo_free` is true and
    `dist.get_backend()` is `gpubridge`; every collective of `check.py` is
    exact; and no rank has a thread named `pt_gloo_runloop` or `gloo_tcp_loop`
    (`cat /proc/<pid>/task/*/comm` on each rank after its first collectives). A
    rank with no GPU, a `LOCAL_RANK` with no GPU behind it and ranks that ask
    for different bridges must still fail `init()` on every rank with one
    message. And at torch's debug level DETAIL: with
    `TORCH_DISTRIBUTED_DEBUG=DETAIL` the job must be refused on every rank
    before any NCCL group exists; with `GPUBRIDGE_ALLOW_DEBUG_GLOO=1` added it
    must start, `topology.debug_gloo` be true and the Gloo threads be there
    (torch's helper groups around the NCCL islands); without DETAIL, none.
    *Status: tested on CPU (`tests/test_no_gloo.py`, and every
    existing suite on the default; the DETAIL guard in
    `tests/test_debug_gloo.py`, where simulation's Gloo islands stand in for
    NCCL ones: that torch wraps an NCCL group the same way is read from its
    source, 2.8.0 and 2.15.0). NCCL island groups under gpubridge's own
    default group are what `backend="gpubridge"` has always created, and that
    ran on GPUs at `c92684b` (torch 2.14.1, two 2-GPU V100 islands, with the
    Gloo bridge and the Gloo groups of every rank beside them). Not run on a
    GPU: the same under `gpubridge.init()`, without any Gloo group, on torch
    2.8 or 2.15, on AMD or mixed.* *Step:* `check.py --bridge auto` in each
    kit step that runs `check.py` (the kit's own default stays `--bridge
    gloo`, the path its recorded results were measured on).

45. **`barrier()` through the c10d store.** Without a Gloo group, `barrier()`
    is a counter in the store that every rank adds to and then watches, with
    pauses from 50 µs doubling to 2 ms between looks. Check that
    `check.py barrier` passes, and measure one barrier's time against the Gloo
    barrier's (`--bridge gloo`) at 2, 4 and as many ranks as the job has: each
    look is a round trip to the store, which is on rank 0's node. *Status:
    tested on CPU (it returns only once the last rank arrives; a missing rank
    times out saying how many arrived). Its cost between machines, and with
    many ranks, is not measured.* *Step:* `check.py barrier` with `--bridge
    auto`; no timing step yet.

46. **CPU tensors through the backend on a GPU job, without Gloo.** A
    collective on CPU tensors in a GPU job (object collectives such as
    `all_gather_object`, a library's CPU flags) used to go to a Gloo group of
    the backend's own. It is now staged: a copy on the rank's GPU, the gpubridge
    collective, a copy back before the returned Work completes; `gather` and
    `scatter` run as an all-gather and a broadcast of bytes. Check on GPUs that
    `all_gather_object`, `broadcast_object_list`, `gather_object` and
    `scatter_object_list` return every rank's object, that an `all_reduce` of a
    CPU tensor is exact, and that the result is there when `wait()` returns
    (the copy back runs on gpubridge's worker stream). The same on the default
    group after `gpubridge.init()`. *Status: tested on CPU with the staging
    forced and a clone standing in for the device copy
    (`tests/test_no_gloo.py`); never run on a GPU.* *Step:* `gpu_tests`
    (`test_gpu_ddp_on_two_gpus_split_into_two_islands` all-reduces a CPU tensor
    through the backend, which now takes this path); the object collectives
    have no GPU step yet.

47. **The TCP bridge as the default: collectives and point-to-point between
    machines.** `bridge="tcp"` (the Rust library, `multitcp` until now) is what
    `auto` runs on between a V100 node and the MI50 node. Check that every
    policy of `check.py --bridge tcp` is exact; that the leaders listen on the
    intended interface (`GPUBRIDGE_MULTITCP_IFNAME`, else the route to
    `MASTER_ADDR`: `topology.bridge.telemetry()["address"]`); that
    `check_p2p` and `bench_p2p.py` pass with `GPUBRIDGE_BRIDGE=tcp`, whose
    cross-island messages run over the new TCP lanes; and the bridge's time
    against the Gloo bridge's (step 08's `D-multitcp` rows are the same bridge
    by its older name). *Status: collectives ran between a V100 and an MI50 at
    23180fe by name, beside a Gloo default group (0.88 to 1.07 times the pinned
    Gloo bridge's time). As the default, and the TCP point-to-point lanes: CPU
    only (`tests/test_multitcp.py`, `tests/test_p2p.py`).* *Step:* `check.py`,
    `check_p2p` and step 08 with `--bridge tcp` / `GPUBRIDGE_BRIDGE=tcp`.

48. **`bridge="auto"` on real RDMA devices.** `auto` asks each island leader
    whether the verbs bridge could run there (`gpubridge.verbs.availability()`:
    the library loads or builds, an endpoint opens on a device with an active
    port and an address) and picks `verbs` only if every leader can and all are
    on one link layer. Check on a V100 pair (InfiniBand on both) that it picks
    `verbs`, and on a V100 + MI50 pair (InfiniBand and RoCE) that it picks `tcp`
    with that reason, the same on every rank (`topology.selection`); that the
    probe's open and close of an endpoint leaves the device usable for the
    bridge that follows; and what the probe costs at `init()`. Where `auto`
    picks verbs and the leaders then can't connect (two fabrics of one link
    layer), `init()` fails on every rank and names `bridge="tcp"`: check the
    message. *Status: the rule and the reports are tested on CPU with fixed
    reports, and the bridge it picks on the MOCK fabric
    (`tests/test_bridge_selection.py`); the device check has never met an RDMA
    device.* *Step:* `check.py --bridge auto` in steps 05 and 07.

49. **A rank that exits without `destroy()` on GPUs.** An exit handler closes
    the bridge (joining its native threads) and, on a job without Gloo,
    destroys the process groups, NCCL/RCCL islands included, before the
    interpreter finalizes. Check that a training script that returns without
    `gpubridge.destroy()` exits 0 on every rank, repeatedly (the CPU run is
    300 repetitions: `scripts/exit_abort.py`), on Python 3.12 with the
    cluster's torch; and the same script with `GPUBRIDGE_BRIDGE=gloo` as the
    control. *Status: on CPU (Linux, Python 3.12.3, torch 2.15.0) the TCP arm
    had no bad exit in 300 repetitions with and without `destroy()`; the
    CHANGELOG has both arms' counts. Never run on a GPU: whether NCCL's or
    RCCL's own threads can need the GIL at exit is not established.* *Step:*
    none yet.

50. **The Rust bridges give Gloo's results on GPUs.** On CPU every bridge
    reproduces hashes the Gloo bridge generated (`tests/test_parity_fixture.py`)
    in jobs with no Gloo group. On GPUs the bridge sees the same host bytes
    (staged by the policy for `tcp`; staged or direct for `verbs`), so the
    check is the kit's exactness against `flat-gloo` (`check.py`, which makes
    its own Gloo group for the reference) and the numerical gate with a `tcp`
    arm beside `gloo` and `verbs`. *Status: staged verbs was exact against Gloo
    on one node (`c92684b`, item 39); the TCP bridge's GPU jobs at 23180fe were
    timing runs. The numerical gate has no `tcp` arm yet.* *Step:* `check.py
    --bridge tcp`; a gate arm to add.

32. **Point-to-point across islands through the backend on GPUs.** `send`,
    `recv`, `isend`, `irecv` and `batch_isend_irecv` on `backend="gpubridge"`
    go native within an island and through the bridge across islands. On GPUs,
    check that six dtypes and odd sizes arrive bit for bit, that a send whose
    input was copied device to device on a busy producer stream and dropped before
    it completes still sends what was written despite a same-stream allocation
    filled with other data, that the received tensor read on another
    stream right after `wait()` holds the data, and that two tags between a
    pair match out of order. *Status: works on CPU in simulation mode (1+1 and
    2+2 ranks, `tests/test_gpu.py`'s simulation test and check.py `--p2p-only`).
    On GPUs: passes, commit 1c7d107. `check_p2p` on the split test (one V100,
    torch 2.14.1+cu126, job 11112711) and across vendors (V100 on 2.14.1+cu126,
    MI50 on 2.9.1+rocm6.3 in `venv-rocm63`, job 11112713); the stream-order test
    in `gpu_tests` on the V100 (job 11112711) and the MI50 (job 11113623).* *Step:* `check_p2p` in Explorer steps 02 and 07, and
    `test_point_to_point_across_islands_keeps_stream_order_and_storage` in
    `gpu_tests`. The results table counts `check_p2p` only; historical `gpu_tests`
    records cannot show whether this test existed in the run.

## GPU results

### Split-test results, 2026-10-20

Hardware: **one node with 4x Tesla V100-SXM2-32GB, driver 545.23.08,
torch 2.14.1+cu126, NCCL 2.29.3, Mellanox ConnectX (mlx5) HDR InfiniBand,
2026-10-20**. Bridge GPU runs split four distinct identical V100s into two
2-GPU NCCL islands; native baselines use all four GPUs. Both half and alternate
layouts passed Gloo correctness. Phase medians are independent per-rank
maxima and cannot be added or treated as pure transport cost. Verbs traffic
uses one HCA on one host:
**loopback, not cross-node, mixed-vendor or GPUDirect evidence**. Host-mode
kit rechecks instead use CPU tensors and two Gloo islands.

These rows summarize recorded checks and timings, separately from the older
kit-generated counts below. A benchmark pass means execution and expected
records, not a performance or arbitrary-float correctness gate.

| Date | GPUBridge commit tested | Run scope | Outcome |
| --- | --- | --- | --- |
| 2026-10-20 | `c92684b` | Gloo bridge, two 2-GPU NCCL islands | All 24 planned steps passed: both layouts, pinned staging, pipelined correctness, sharded k=2, GPU-derived auto-tuned thresholds, DDP and exact FSDP2/HSDP replay. GPU tests: 29 passed, 6 skipped (2 UCX, 4 inapplicable bf16-compression cases), no failures; compression kernels: 3/3 reference checks. Benchmarks through 256 MiB. |
| 2026-10-20 | `c92684b` | Staged verbs, same split, single-HCA loopback | GPU kit and default-policy full check passed; real non-mock libibverbs, zero completion errors, staged paths and GPU strict TCP gate passed. Exact kit payloads through 512 MiB; benchmark through 1 GiB. Original host-mode kit failed before payload (CPU/CUDA topology mismatch); this run remains partial. |
| 2026-10-20 | `c92684b` | Native/Gloo/staged-verbs GPT-2-small, 10/25/100 MB bucket caps | All nine cells passed execution gates, with finite and rank-identical parameters. Overlap is estimated from separate timed phases; final digests differ between arms, so this is not cross-arm equivalence. |
| 2026-10-20 | `c92684b` | Gloo and verbs pipeline, 1/4/16/64 MiB chunks | All timing sweeps completed through 1 GiB. Gloo improves at matched 1–256 MiB points; verbs generally regresses. Item 18's unrestricted goal is not established; ordinary timing rows do not replace full verbs-pipeline payload checks. |
| 2026-10-20 | `c92684b` | P2P native NCCL and Gloo bridge on separate GPUs | Full default check passed; all 66 benchmark payload rows passed across default, chunked and pinned configurations. No verbs P2P or direct overlap trace. |
| 2026-10-20 | `0cf32a3` (production fix in `54f701e`) | Host/Gloo-island verbs kit and GPU-staged regression, single-HCA loopback | Host collectives exact, all ranks passed, zero mismatches/completion errors; host TCP evidence `not-decisive`. Strict host run failed closed as intended; GPU-staged strict run passed. `54f701e` adds test isolation only and was not the hardware checkout. |
| 2026-10-20 | `f4146a4` | Separate bounded tiny gate: native NCCL, Gloo islands and staged verbs, same four-V100 node class | Accepted: 48/48 positive rank records, zero violations; all three bad controls rejected for their intended reasons. This is not representative GPT equivalence. |
| 2026-10-20 | `4ffcf93` | Host/Gloo-island verbs kit with per-process TCP attribution, and GPU-staged regression, single-HCA loopback | Strict host collectives passed decisively: 28/28 exact, verbs evidence pass, island TCP attributed, 1728 B counted. The flat world-Gloo and Gloo-bridge negatives recorded verdict `fail` on counted TCP over the limit. GPU-staged strict was unchanged. Not cross-node no-TCP acceptance. |

Earlier split-test runs on the same node (2026-10-19, `616f2f7`) passed a
2-rank (1+1) staged-verbs correctness kit and staged-verbs training comparisons;
the 1+1 kit did not exercise a nontrivial NCCL island. The separate tiny numerical
gate on 2026-10-20 at `f4146a4` accepted native NCCL, Gloo-island and staged-verbs arms with
negative controls (item 43); it establishes only that bounded workload.

No result here establishes cross-node verbs, AMD/mixed-vendor verbs,
GPUDirect/direct verbs, UCX, 8-GPU scaling, independent link scaling, statistical
significance, representative GPT equivalence or a direct copy/communication
overlap trace. The collective `wait-skips-stream` mutation remains uncaught.
Co-resident host leaders get decisive no-silent-TCP evidence only with per-process
socket attribution (`4ffcf93`, one host).
Training and pipeline observations corroborate native-library verbs activity,
but their archives contain no separate in-process provider/completion snapshot
or strict TCP gate; the staged kit supplies the explicit provider telemetry.
See the [Unreleased hardware record](CHANGELOG.md#unreleased).

### Historical generated results

From `scripts/gpu/validation_tables.py` over every Explorer result of
2026-10-14 (the kit's `results/explorer`, with tonight's rocm6.3 jobs in
`results/explorer/rocm63`). Hosts are numbered, not named. A folder's
`steps.jsonl` keeps every attempt run into it, so a job below can count a step
twice, and a FAIL can be an early attempt that a later one in the same folder
passed; the notes say which. GPU names are as the driver reports them: the
MI50 shows as "AMD Instinct MI60 / MI50", or "AMD Radeon Graphics" with
torch 2.9.1+rocm6.3.

### Runs

| Job | Steps passed | Failed | Hosts | GPUs | Driver | torch | CUDA / ROCm | NCCL / RCCL | gpubridge | Gloo interface |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_probe | 5 of 6 | `probe` | host 1 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 14ed60c | auto |
| 02_a100_df80abb | 19 of 20 | `gpu_tests` | host 2 | NVIDIA A100-SXM4-80GB | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_l40s_df80abb | 24 of 25 | `gpu_tests` | host 3 | NVIDIA L40S | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_split_shared_gpu | 29 of 30 | `gpu_tests` | host 4 | Tesla T4 | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_t4_df80abb | 29 of 30 | `gpu_tests` | host 4 | Tesla T4 | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_v100_df80abb | 14 of 15 | `gpu_tests` | host 5 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb | auto |
| 07_mixed_hetjob | 15 of 20 | `train_ddp`, `train_fsdp2_backend`, `verify_fsdp2_backend`, `train_hsdp_backend`, `verify_hsdp_backend` | host 6, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 4014234 | internal.2137, auto |
| 07_mixed_hetjob_4014234_job11084294 | 10 of 16 | `bench`, `check_pinned`, `summarize_pinned`, `bench_pinned`, `bench_compression_kernels`, `train_ddp` | host 8, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 4014234 | internal.2137, internal, auto |
| 08_matrix_nvidia-amd | 15 of 15 | - | host 9, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 934f87c | internal.2137, auto |
| amd_node | 9 of 9 | - | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 4014234 | auto |
| amd_node_4014234_job11083835 | 14 of 18 | `check_1gpu`, `gpu_tests`, `check_split_shared_gpu`, `summarize` | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 4014234 | auto |
| net_check_amd | 4 of 4 | - | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 934f87c | auto |
| net_check_nvidia | 4 of 4 | - | host 10 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 934f87c | auto |
| rocm63/07_mixed_hetjob | 7 of 8 | `train_hsdp_backend` | host 9, host 7 | Tesla V100-SXM2-32GB, AMD Radeon Graphics | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.9.1+rocm6.3 | CUDA 12.6, ROCm ? (HIP 6.3.42134-a9a80e791) | NCCL 2.29.3, RCCL 2.21.5 | 934f87c | internal.2137, auto |
| rocm63/amd_node | 7 of 9 | `gpu_tests`, `bench_compression_kernels` | host 7 | AMD Radeon Graphics | 5.14.0-611.5.1.el9_7.x86_64 | 2.9.1+rocm6.3 | ROCm ? (HIP 6.3.42134-a9a80e791) | RCCL 2.21.5 | dbed5c9 | auto |
| rocm63/train_bench_amd | 3 of 3 | - | host 7 | AMD Radeon Graphics | 5.14.0-611.5.1.el9_7.x86_64 | 2.9.1+rocm6.3 | ROCm ? (HIP 6.3.42134-a9a80e791) | RCCL 2.21.5 | 934f87c | auto |
| train_bench_nvidia | 3 of 3 | - | host 9 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 934f87c | auto |

### Items

`pass (n)` counts GPU runs only; one failed record makes an item `FAIL (k of
n)`. The `gpu_tests` step tests items 6, 15, 16, 17, 28 and 30 together, so its
failures count against all six whichever test failed.

| Item | Tested by | Verdict |
| --- | --- | --- |
| 1. CUDA and ROCm processes rendezvous in one job | check.py `init` (gpu-mixed) | pass (7) |
| 2. Version matching between builds | check.py `init` (gpu-mixed) | pass (7) |
| 3. `new_group(backend="nccl")` on ranks outside the group | check.py `init` (gpu-mixed, split-test) | pass (15) |
| 4. `"nccl"` on ROCm really runs RCCL | step `rccl_version` | pass (4) |
| 5. Device selection | step `device_map` | pass (4) |
| 6. Stream ordering in the bridge step | check.py `busy_gpu` (gpu-mixed, split-test); step `gpu_tests` | FAIL (7 of 43) |
| 7. float16 and other dtypes over Gloo | check.py `dtypes` (gpu-mixed, split-test) | pass (15) |
| 8. Gloo networking | step `preflight`; step `check_split_by_node` | pass (4) |
| 9. Timeouts | check.py `busy_gpu` (gpu-mixed, split-test) | pass (15) |
| 10. `barrier()` on GPUs | check.py `barrier` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 11. Bridge throughput and latency against native | step `bench`; step `bench_split_by_node` | FAIL (1 of 2) |
| 12. A rank whose `LOCAL_RANK` has no GPU fails `init()` | step `local_rank_check` | pass (6) |
| 13. The island step is a reduce onto the leader | step `bench`; step `bench_split_by_node` | Historical FAIL (1 of 2): step-creation failure; nontrivial two-rank measurements now recorded above, no universal speed claim |
| 14. Where NCCL/RCCL and Gloo agree on reductions | check.py `reduction_agreement` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 15. MAX, MIN, AVG, all_gather and reduce_scatter on GPUs | check.py `reductions` (gpu-mixed, gpu-single-vendor, split-test); check.py `all_gather` (gpu-mixed, gpu-single-vendor, split-test); check.py `reduce_scatter` (gpu-mixed, gpu-single-vendor, split-test); step `gpu_tests` | FAIL (7 of 82) |
| 16. Async collectives keep their stream semantics | check.py `async` (gpu-mixed, gpu-single-vendor, split-test); step `gpu_tests` | Historical FAIL (7 of 46); current split async checks pass, wait-stream mutation gap remains |
| 17. The pipelined bridge is correct on GPUs | step `check_pipelined-reduce-bridge-broadcast`; step `gpu_tests` | FAIL (7 of 30) |
| 18. The overlap pays off | step `bench_phases_pipelined-reduce-bridge-broadcast`; step `bench_candidates` | Historical step passes (2); unrestricted performance goal not established (see current results) |
| 19. Thresholds measured on GPUs drive `auto-tuned` | step `bench_candidates`; step `check_auto_tuned` | pass (2) |
| 20. The sharded bridge is correct on GPUs | step `check_sharded-bridge` | Historical step passes (2), k=1; current k=2 Gloo correctness passes on 2+2 |
| 21. More bridge links mean more cross-vendor bandwidth | step `bench_phases_sharded-bridge`; step `bench_candidates` | Historical step passes (2), k=1; k=2 timings now exist, controlled link scaling still unproven |
| 22. DDP with `gpubridge.ddp_comm_hook` trains correctly | step `train_ddp(_split)?`; step `train_n\d+` | FAIL (2 of 3) |
| 23. What `gpubridge.in_flight()` shows on GPUs | check.py `in_flight` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 24. No island puts two ranks on one GPU | step `shared_gpu_check` | pass (24) |
| 25. Pinned staging for the default bridge | step `check_pinned`; step `bench_pinned` | FAIL (2 of 4) |
| 26. The UCX bridge is correct on GPUs | step `ucx_build`; step `check_ucx` | not run |
| 27. The UCX bridge against the Gloo bridge | step `bench_ucx` | not run |
| 28. Bridge compression kernels give the reference's bits on GPUs | step `gpu_tests`; step `bench_compression_kernels` | FAIL (9 of 34) |
| 29. Where bridge compression pays off | step `bench_compressed` | pass (2) |
| 30. gpubridge as a torch.distributed backend: DDP and FSDP2 on GPUs | step `verify_fsdp2_backend`; step `train_fsdp2_backend`; step `gpu_tests` | FAIL (9 of 32) |
| 31. HSDP through the backend: shard within islands, replicate across vendors | step `verify_hsdp_backend`; step `train_hsdp_backend` | FAIL (3 of 3) |
| 33. Data-parallel training across vendors tracks one GPU, compressed or not | step `dp_(ddp\|fsdp2)(-.+)?`; step `dp_single` | not run |
| 34. Pipeline training across vendors tracks one GPU | step `pp_(gpipe\|1f1b)`; step `dp_single` | not run |
| 35. The multitcp bridge against the Gloo bridge | step `multitcp_build`; step `matrix_D-multitcp(-pinned\|-1stream)?` | not run |
| 36. Automatic stage balancing against even and hand-tuned splits | step `pp_(gpipe\|1f1b)_(even\|auto\|[0-9]+(-[0-9]+)+)` | not run |
| 37. Megatron-Core's pipeline across vendors tracks one GPU | step `megatron_pp2`; step `megatron_pp1` | not run |
| 38. The verbs bridge between hosts on RDMA hardware | step `verbs_build`; step `verbs_endpoint`; step `verbs_host`; step `check_verbs`; step `bench_verbs` | Single-host loopback passed, host recheck qualified above; between-host claim unproven |
| 39. GPU tensors staged through the verbs bridge | step `verbs_gpu_staged` | Staged split-test kit/check passed; cross-node and mixed-vendor unproven |
| 40. NVIDIA GPU memory direct through the verbs bridge | step `verbs_gpu_direct` | not run |
| 41. AMD GPU memory direct through the verbs bridge | step `verbs_gpu_direct` | not run |
| 42. Training over the verbs bridge against the Gloo bridge | step `train_verbs`; step `train_gloo` | Split-test training passed within-run gates; cross-arm equivalence and significance unproven |

Where the FAILs come from:

- **`gpu_tests`, 7 records (items 6, 15, 16, 17, 28, 30).** Five are one
  morning attempt on a V100 at af51b20, recorded in each step 02 folder: two
  fp8 error-feedback residual tests failed, fixed in #50; every later attempt in
  those folders passed. One is the first MI50 attempt at df80abb: 2 tests
  failed in a matmul the tests used as busy work, which the wheel's rocBLAS
  can't run on gfx906, replaced in #52; the folder's later attempt passed. The
  last is the MI50 with torch 2.9.1+rocm6.3: its Triton (pytorch-triton-rocm
  3.5.1) can't compile the compression kernels for gfx906, so the 20
  compression-kernel tests failed; the other 6 passed.
- **`bench` (items 11 and 13), `check_pinned` and `bench_pinned` (item 25),
  one of the two `bench_compression_kernels` FAILs (item 28)**: step 07's first
  run, where Slurm couldn't create the job steps; all passed in the second run.
  The other `bench_compression_kernels` FAIL is the rocm6.3 Triton above.
- **`train_ddp` (item 22)**: failed in step 07's first two runs, on the default
  ROCm wheel (a Gloo timeout in the first, cause unknown; the MI50 aborting in
  rocBLAS in the second); passed with the MI50 on torch 2.9.1+rocm6.3. Item 22
  now passes on GPUs: 24 of 24 later mixed runs, judged bit for bit
  ([below](#items-22-and-33-2026-10-15-and-2026-10-16)).
- **`train_fsdp2_backend`, `verify_fsdp2_backend` (item 30)**: failed in step
  07's second run (the same rocBLAS abort); passed with torch 2.9.1+rocm6.3,
  bit for bit against a single-process replay.
- **`train_hsdp_backend`, `verify_hsdp_backend` (item 31)**: failed in step
  07's second run (the same rocBLAS abort). With the MI50 on torch
  2.9.1+rocm6.3, `train_hsdp_backend` printed nothing until the kit's step
  timeout stopped it (exit 124) or the job was cancelled, in this job and
  the later mixed jobs recorded here.
  **Where it stops**, from a CPU reproduction (gpubridge's simulation mode,
  two ranks on one node, `examples/fsdp2_torch_backend.py --hsdp 2`): with
  both ranks on torch 2.14.1, or both on 2.9.1, it completes; with rank 0 on
  2.14.1 and rank 1 on 2.9.1 it hangs the same way. Python stacks taken at
  the hang (faulthandler) show both ranks in gpubridge's one-time world setup,
  inside `dist.new_group` (`torch_backend._world_topology` -> `setup`),
  reached from FSDP's first gradient collective: an `all_reduce` on rank 0
  and a `reduce_scatter` on rank 1. The two torch releases' `DeviceMesh`
  creates a different number of groups for the (2, 1) mesh (2.14.1 reuses the
  default group for the replicate dimension; 2.9.1 calls `new_group`), and
  the world setup's groups take torch's default names, which count the groups
  each process has made, so the ranks likely wait on different store keys.
  A setup-naming fix in `src/` (#80) completed this mixed-version CPU example
  and its bit-for-bit replay check, and on 2026-10-15 both steps passed on
  the GPU pair with it (job 11112713, below). Plain `init_device_mesh`
  and a barrier with the same mixed versions, without gpubridge, don't hang.

### 2026-10-15, commit 1c7d107

Not yet in the tables above: their folders now mix these jobs' attempts with
the 2026-10-14 ones (a folder's `steps.jsonl` doesn't say which job ran a
step), so they are left as generated until the next clean run. From each job's
log, every job from the same checkout, with gpubridge imported from it:

| Job | What | Torch (V100 / MI50) | MI50 venv | Passed | Failed |
| --- | --- | --- | --- | --- | --- |
| 11112710 | 01 probe | 2.14.1+cu126 | - | every step | - |
| 11112711 | 02 split test, one V100 | 2.14.1+cu126 | - | 8 of 8, incl. `check_p2p`, `bench_p2p`, `gpu_tests` (27 passed, 8 skipped) | - |
| 11112713 | 07 V100 + MI50 | 2.14.1+cu126 / 2.9.1+rocm6.3 | `venv-rocm63` | 24 of 27, incl. `check_p2p`, `bench_p2p`, FSDP2, HSDP | `bench_compression_kernels`, `bench_compressed`, `train_ddp` |
| 11112712 | AMD node | 2.14.1+rocm7.2 | `venv-rocm` | 9 of 11 | both `train_bench` runs (rocBLAS has no gfx906 kernels) |
| 11112824 | AMD node | 2.9.1+rocm6.3 | `venv-rocm63` | 9 of 11 | `gpu_tests`, `bench_compression_kernels` (Triton 3.5.1 can't compile for gfx906) |
| 11113623 | AMD node | 2.9.1+rocm6.3, Triton 3.8.0 overlay | `venv-rocm63` + overlay | 11 of 11, `gpu_tests` 27 passed, 8 skipped | - |

- **Items 31 and 32** pass on GPUs (their entries above).
- **Item 22**: `train_ddp` failed its parameter check in job 11112713, the
  float64 checksum; most likely that check's false alarm. Item 22 passes on
  GPUs under the bit-for-bit check ([below](#items-22-and-33-2026-10-15-and-2026-10-16)).
- **Item 28 on the MI50** is an environment limit: `venv-rocm63`'s Triton
  (pytorch-triton-rocm 3.5.1) can't compile the compression kernels for
  gfx906, and the default ROCm venv's rocBLAS can't train on it. With Triton
  3.8.0 in front of `venv-rocm63` (`scripts/gpu/rocm_triton_overlay.sh`), every
  step passed (job 11113623). That job's records don't name the overlay's
  Triton version; it is inferred from the kernels compiling where 3.5.1 had
  failed half an hour earlier on the same node (job 11112824). Later jobs
  with the overlay record Triton 3.8.0: the kit's Triton check in jobs
  11139974 and 11139976 (item 33, below) reports it compiling and running
  all four kernels on the MI50. The overlay links to a venv that can be
  rebuilt, so that shows the version at their time, not necessarily at job
  11113623's.

### Items 22 and 33, 2026-10-15 and 2026-10-16

Not in the tables above either. Every job ran from one clean checkout at the
commit given, with gpubridge imported from it (each probe records the commit;
the V100 side's log shows the import path): the V100 on torch 2.14.1+cu126, the
MI50 on 2.9.1+rocm6.3 (`venv-rocm63`, or with Triton 3.8.0 in front of it where
noted).

**Item 22 passes.** `train_synthetic.py` now compares every rank's gradients (after
the all_reduce) and parameters (after the optimizer step) bit for bit after
every step (#88), and keeps the old check beside it as
`params_checksums_equal`: the float64 sum of each rank's parameters, taken on
its own GPU. With it, `train_ddp` on a V100 + MI50:

| Jobs | Commit | Bridge compression | Runs | Parameters bit-identical at every step | Old checksums differed |
| --- | --- | --- | ---: | ---: | ---: |
| 11121389, 11121391, 11121393, 11121395, 11121802, 11122013, 11122173, 11122372 | 1659f4f | none | 8, one per job | 8 | 1 (11121389) |
| 11139538 | 48840ba | none | 8 in one job (`TRAIN_DDP_REPEATS=8`) | 8 | 2 |
| 11139551 | 48840ba | `fp8`, PyTorch kernels on both GPUs | 8 in one job | 8 | 1 |

No run diverged: in all 24, the gradients and the parameters were the same
bit for bit on both ranks after every step, uncompressed or `fp8`. That bit
check is what passes or fails the step; the checksum is recorded for
information only. It still reported a difference in 4 of the 24, with the
parameters identical: the V100 and the MI50 each sum their own copy, and the float64
totals can differ in the last bits. The two earlier failures, jobs 11090726
and 11112713, were judged by the checksum alone (2 of 8 runs then, 4 of 24
now), and the code on DDP's path, `ddp.py`, `collectives.py` and the
compression codec, hasn't changed since either job. So they were most likely
the same false alarm. Their records hold only the sums, so that can't be
confirmed for them; and the new runs add the bit check's two gathers across
ranks after every step, which the old runs didn't have (outside the timed
step and outside backward).

**Item 33.** The rule was fixed before the new runs. d is the mean absolute
difference of two runs' losses over their last 100 steps, the measure in
docs/COMPRESSION.md. S is the largest d between any two one-V100 runs of
1,000 steps. A mixed run is within S if the median of its d to the one-V100
runs is at most S and the ranks' parameters were bit-identical at every check
(every 50 steps). By the rule, a mode passes if at least two of its runs
finished and all of them are within S; if one isn't, it fails by the rule.
Every run: `train_dp.py`, `llama-125m`, global batch 8 x 512 tokens, AdamW
3e-4, seed 0.

Eight one-V100 runs (11109239, 11109665, 11109872 and 11110316 with 1, 2, 4 and
8 micro-batches, the first at abd9d74 and the rest at dc47b6a: 11109239's
checkout moved to dc47b6a, its child, which changes only `train_pipeline.py`,
during the run, so its training record names dc47b6a; 11122896 to 11122899
with 1, at fe84663; the training scripts are the same at all three) are 0.0033
to 0.0131 apart, so by the rule **S = 0.0131**. S measures how far uncompressed
runs land from each other: the same mean gradient summed in another order, and
run-to-run nondeterminism. A lossy mode adds its own error on top of that, so
`bf16` and `fp8` are expected to land outside S; the rule is applied to them as
written, and their gaps are given as multiples of S.

At 1,000 steps (the plateau exit is the first step, counted from 1, at which
the mean loss of the last 10 steps is below 2.0):

| Mode | Job | Commit | Median d to the one-V100 runs | Parameters identical | Plateau exit | Within S |
| --- | --- | --- | ---: | --- | ---: | --- |
| DDP, uncompressed | 11122900 | fe84663 | 0.0042 (0.32 S) | 20 of 20 checks | 560 | yes |
| DDP, uncompressed | 11122902 | fe84663 | 0.0044 (0.34 S) | 20 of 20 | 576 | yes |
| DDP, shares by speed | 11122900 | fe84663 | 0.0041 (0.32 S) | 20 of 20 | 584 | yes |
| DDP, shares by speed | 11122902 | fe84663 | 0.0170 (1.30 S) | 20 of 20 | 655 | no |
| FSDP2 | 11125340 | fe84663 | 0.1334 (10.2 S) | not compared (sharded) | 893 | no |
| FSDP2 | 11125342 | fe84663 | 0.0043 (0.32 S) | not compared (sharded) | 589 | yes |
| DDP, `bf16` | 11139974 | 48840ba | 0.1547 (11.8 S) | 20 of 20 | 918 | no |
| DDP, `bf16` | 11139976 | 48840ba | 0.0329 (2.51 S) | 20 of 20 | 722 | no |
| DDP, `fp8` | 11139974 | 48840ba | 0.0407 (3.11 S) | 20 of 20 | 734 | no |
| DDP, `fp8` | 11139976 | 48840ba | 6.92 (528 S) | 20 of 20 | none by 1,000 | no |

By the rule: uncompressed DDP **passes** (2 of 2 within S). DDP with shares
by speed **fails by the rule** (1 of 2), and so does FSDP2 (1 of 2). `bf16`
**fails by the rule** at 2.51 S and 11.8 S, and `fp8` at 3.11 S and 528 S.
Whether gaps like these are acceptable for compressed training is a
separate question the rule doesn't answer, and it is open. FSDP2 shards the
parameters, so its ranks hold different ones and only its loss is judged.
In the compressed runs both GPUs ran the Triton kernels: the kit's Triton
check found Triton 3.8.0 compiling and running them on the V100 and on the
MI50 (`venv-rocm63` with `scripts/gpu/rocm_triton_overlay.sh`'s overlay,
`DP_KERNELS_ON_AMD=auto`).

**Plateau timing.** Every run sits on a plateau near ln(32000), then falls to
about 2; when the fall starts depends on float rounding (docs/COMPRESSION.md,
"Training with compression"). The eleven one-V100 runs here (the eight above
and three of 2,000 steps, below) reached the plateau exit (10-step mean below
2.0) at steps 557 to 591, except two: 11122898 at 652 and 11141231 at 1,991.
Against that:

- DDP with shares by speed: 584 (11122900, within S) and 655 (11122902,
  1.30 S). 11122902 is 0.0073 from 11122898, the one-V100 run that exited at
  652, and closed on the other runs the same way: over 100-step windows from
  step 501 to 1,000, its median gap to the one-V100 runs fell 0.150, 0.074,
  0.042, 0.027, 0.017; 11122898's to the other seven fell 0.143, 0.064,
  0.036, 0.020, 0.011.
- FSDP2: 589 (11125342, within S) and 893 (11125340, 10.2 S), later than all
  but one one-V100 run; its gap fell from 4.73 to 0.13 over the same windows.
- The compressed runs: 722 to 918, and `fp8` in 11139976 hadn't reached it by
  step 1,000 (8.73 over its last 100 steps).

So each run outside S reached the plateau exit late, or (`fp8` in 11139976) not
by step 1,000, and a late one-V100 run closes its gap the same way. That points
at timing rather than at something the bridge computes wrong, and at compression
delaying the fall; both are inferred, not tested.

**At 2,000 steps** (jobs at 48840ba, one run per mode): S stays 0.0131, and
with one run per mode the rule gives no verdict at this length. The three
one-V100 runs are the references.

| Run | Job | Median d to the one-V100 runs | Parameters identical | Plateau exit |
| --- | --- | ---: | --- | ---: |
| One V100 | 11141232, 11141233 | 0.0031 apart | - | 588, 576 |
| One V100 | 11141231 | 0.327 from those two (25 S) | - | 1,991 |
| DDP, uncompressed | 11141122 | 0.0033 (0.26 S) | 40 of 40 checks | 580 |
| DDP, `bf16` | 11141229 | 0.517 (39 S) | 40 of 40 | none by 2,000 (2.21 over the last 100 steps) |
| DDP, `fp8` | 11141646 | 0.0221 (1.69 S) | 40 of 40 | 951 |

One V100 alone, with no bridge, produced a run slower than all but two of
the 13 mixed runs (both compressed): 11141231 first fell below 9 at step 521
and ended 25 S from the other two. So a late run is not specific to the
bridge or to compression, but it is far more common with compression: the
loss first fell below 9 after step 250 in all 6 compressed runs here (and 3
of the 4 in docs/COMPRESSION.md), against 2 of 11 one-V100 runs and 2 of 7
uncompressed mixed ones here. After their fall the compressed runs keep
closing: `fp8`'s gap fell from 0.059 over steps 1,301 to 1,400 to 0.022 over
the last 100. That is about where the earlier 1,000-step `fp8` run ended
(job 11111481, 0.022 to 0.027 from four one-V100 runs), so part of it may be
`fp8`'s own error rather than timing; these runs can't separate the two
(inferred).

**Not tested yet, and next.** The timing explanation for shares by speed and
FSDP2 is untested at 2,000 steps: the 2,000-step runs above include neither.
And at 2,000 steps one V100 alone falls outside S against its own runs
(11141231, 25 S), so the rule measures when a run falls off the plateau as
much as anything the bridge does. The planned next step is a redesigned
test rather than more runs of this one: at least four seeds per mode, each
with one-V100 runs of the same seed; the loss at the end, judged only once
every run has reached the plateau exit; and the steps and wall-clock time to
a target loss, against uncompressed runs. For `bf16` and `fp8` any slower
convergence is to be reported as a measured cost, not a pass or a fail. The
MI50s are only in Explorer's `sharing` partition, whose one-hour limit bounds
a mixed job: about 2,700 steps of DDP, but only about 1,150 of FSDP2.

### Network

| Job | Master (host, interface) | Worker (host, interface) | MB/s master to worker | MB/s worker to master | Round trip (ms) |
| --- | --- | --- | --- | --- | --- |
| 07_mixed_hetjob | host 6 internal.2137 | host 7 internal.2137 | 1173.4 | 1147.3 | 0.069 |
| 07_mixed_hetjob_4014234_job11084294 | host 8 internal.2137 | host 7 internal | 1171.4 | 1173.8 | 0.047 |
| 08_matrix_nvidia-amd | host 9 internal.2137 | host 7 internal.2137 | 1173.7 | 1172.1 | 0.048 |
| rocm63/07_mixed_hetjob | host 9 internal.2137 | host 7 internal.2137 | 1173.8 | 1169.7 | 0.046 |

The benchmark tables (gpubridge against each run's baseline, and the predicted
against the measured bridge time) are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) with the full matrix;
`validation_tables.py` writes them too.

What has not run on GPUs at all: the UCX bridge (items 26 and 27), more than
one GPU in an island of a mixed job (Explorer's `multigpu` partition isn't
open to this account), and more than two islands.

## Mixed build results

Run on 2026-10-06 with [`scripts/mixed_build_rendezvous.sh`](scripts/mixed_build_rendezvous.sh),
which reproduces everything below with one command. Raw results:
[`results/mixed_build/summary.md`](results/mixed_build/summary.md) (one row per run),
[`summary.json`](results/mixed_build/summary.json) (per-rank detail) and
[`envs.json`](results/mixed_build/envs.json) (builds and sizes).

### Setup

- No GPUs. A linux/amd64 Docker image (emulated on an Apple Silicon Mac) holds
  one venv per PyTorch build, installed from the official wheel indexes. Every
  venv uses Python 3.12.12.
- `GPUBRIDGE_CPU_ONLY=1` on every rank: the vendor comes from the real build
  (`torch.version.cuda` or `torch.version.hip`), islands use Gloo, tensors stay
  on CPU. `GPUBRIDGE_VENDOR` is not set.
- Each run starts two `torchrun` "nodes" on localhost, one per venv, with
  `--nnodes=2` and a c10d rendezvous. `--rdzv-conf is_host=...` chooses which
  node hosts the rendezvous store; that node also ends up as node 0, which
  hosts the store the workers use.
- Each rank runs [`scripts/rendezvous_probe.py`](scripts/rendezvous_probe.py)
  and checks, in order: rendezvous, `gpubridge.init()` (Gloo world group,
  `all_gather_object` discovery, island and bridge groups), that every rank got
  the same vendor map and that it matches each rank's own build, `all_reduce`
  SUM (exact, float32 and float16), `broadcast` from every rank, and a clean
  `destroy()` and exit.
- Negative control: with the bridge `all_reduce` removed from `collectives.py`,
  the same harness reports `FAIL (all_reduce)` with broadcast still passing,
  so the checks do catch wrong results.

| venv | torch | CUDA / HIP | Index | Download | Installed |
| --- | --- | --- | --- | --- | --- |
| cu-2.14.1 | 2.14.1+cu130 | 13.0 | `whl/cu130` | 3.0 GB | 5.6 GB |
| rocm-2.14.1 | 2.14.1+rocm7.2 | 7.2.53211 | `whl/rocm7.2` | 6.6 GB | 15.7 GB |
| cu-2.13.0 | 2.13.0+cu130 | 13.0 | `whl/cu130` | 2.7 GB | 4.8 GB |
| cu-2.9.1 | 2.9.1+cu128 | 12.8 | `whl/cu128` | 4.1 GB | 7.0 GB |
| rocm-2.9.1 | 2.9.1+rocm6.4 | 6.4.43484 | `whl/rocm6.4` | 4.8 GB | 15.4 GB |

ROCm builds are the large ones: the 2.14.1 ROCm torch wheel alone is 6.2 GB.

### Compatibility

Layouts are CUDA ranks + ROCm ranks; the build in brackets hosted the store.

| CUDA build | ROCm build | Runs | Result | Notes |
| --- | --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release |
| 2.9.1+cu128 | 2.9.1+rocm6.4 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release, older CUDA and ROCm generation |
| 2.13.0+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 1+1 (ROCm), 2+2 (CUDA) | **PASS** 3/3 | One minor release apart; `init()` warns |
| 2.14.1+cu130 | 2.9.1+rocm6.4 | 1+1 (CUDA 2.14.1) | **PASS** | Five minor releases apart, newer side hosts; warns |
| 2.9.1+cu128 | 2.14.1+rocm7.2 | 1+1 (CUDA 2.9.1) | **PASS** | Five minor releases apart, older side hosts; warns |
| 2.13.0+cu130 | (2.14.1+cu130, CUDA) | 1+1 | **PASS** | Control: one vendor, no bridge; warns |

Every run passed on its first attempt, with no timeouts and so no reruns.
Each took 8 to 9 seconds end to end, emulation included.

### Recheck on current main (nested probe record)

Run on 2026-10-06 against `main` at `3a815c0` (after #3). Discovery there sends
each rank's full `probe()` record nested inside its `PeerInfo`, which the
earlier runs above did not. Same setup, 2.14.1 pair only, with the wheels
installed offline from the host cache:

```
GPUBRIDGE_ENVS=2.14.1 scripts/mixed_build_rendezvous.sh --group 2.14.1 \
    --results /gpubridge/results/mixed_build_recheck
```

Raw results:
[`results/mixed_build_recheck/summary.md`](results/mixed_build_recheck/summary.md)
and [`summary.json`](results/mixed_build_recheck/summary.json).

| CUDA build | ROCm build | Layout (store host) | Result |
| --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 2+2 (ROCm) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 3+1 (CUDA) | **PASS** |

Every check passed (rendezvous, init, vendor map, all_reduce, broadcast,
shutdown), each run on its first attempt, in 8 seconds.

**The nested record crosses builds intact.** Every rank received every peer's
probe record, and all ranks held identical tables. A CUDA rank saw the ROCm
rank as:

```
build_vendor="amd", torch_version="2.14.1+rocm7.2", cuda_version=None,
hip_version="7.2.53211", gpu_count=0, nccl_available=True,
gloo_available=True, errors=()
```

The ROCm ranks saw the mirror image (`build_vendor="nvidia"`,
`cuda_version="13.0"`). No rank reported problems.

`nccl_available` is True on both builds even without GPUs, because it reports
whether the backend is compiled into the build. `gpu_count` is what shows the
missing hardware.

### What this means for the design

- **The bridge design stands.** The parts of a mixed job that both builds must
  share all run on CPU: the c10d store, Gloo, and pickled builtins in
  `all_gather_object`. All of them interoperate between CUDA and ROCm builds,
  and across releases from 2.9.1 to 2.14.1. No workaround was needed.
- **Version matching is a recommendation, not a requirement, on the CPU side.**
  `init()` keeps warning when releases differ, because the GPU side and other
  Python versions are untested.

### What this does not show

- Anything on GPUs: NCCL, RCCL, device selection, stream ordering (items 4 to 7, 10).
- Separate hosts on a real network. Gloo only ran over loopback (item 8).
- Item 3 (`new_group(backend="nccl")` on ranks outside the group). In CPU-only
  mode islands use Gloo, and on a machine without GPUs a member rank cannot
  create an NCCL group at all: it raises `ValueError: ProcessGroupNCCL is only
  supported with GPUs, no GPUs found!` (checked on 2.14.1+cu130).
- Python versions other than 3.12, releases outside 2.9.1 to 2.14.1, and
  non-x86 hosts.

### Setup problems worth knowing about

- Installing the 6.2 GB ROCm wheel straight from the index with uv failed once
  with "Failed to read from zip file". Downloading it with `curl` and checking
  the index's sha256 worked, and uv then installed the file without trouble.
  Whether the first download was cut off or uv mishandled the large file was
  not pinned down. ROCm wheels are now always fetched that way
  ([`scripts/fetch_wheel.sh`](scripts/fetch_wheel.sh)).
- The wheels first lived in a BuildKit cache mount, but Docker Desktop's
  BuildKit garbage collection caps cache mounts at about 2.76 GiB, so the 6+ GB
  ROCm wheels were evicted. They now live in a permanent cache on the host
  (`~/.cache/gpubridge/wheels`, filled by
  [`scripts/fetch_wheels.sh`](scripts/fetch_wheels.sh)), and the image
  installs offline from it.
- ROCm builds of torch need `libatomic.so.1`, which slim Debian images lack
  (package `libatomic1`).
- Recent macOS ships a BSD `sha256sum` that has no `--status` flag. An early
  version of `fetch_wheel.sh` used it, mistook the usage error for a checksum
  mismatch, and deleted a complete 6.2 GB wheel. The script now compares
  digests as text, preferring `shasum`.
- `download-r2.pytorch.org` answers 403 to Python's default `urllib`
  User-Agent; set one when scripting against it.
