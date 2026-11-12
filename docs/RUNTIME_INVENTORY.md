# Where Python and Gloo run, and what a native-first deployment keeps

A map of what executes in a gpubridge job, in which language and runtime, and when;
then a keep / demote / remove decision for each part, with the callers and
measurements behind it. Written on 2026-10-20 against `38b1e20` (the reviewed
`main`), whose `src/` tree `ea6c77b9` and verbs crate tree `63a58a08` are the ones
run on hardware at `c92684b`, `4c13375` and `4ffcf93`. File and line references are
to that tree unless they say otherwise.

"Native-first" means: NCCL or RCCL inside each island, the Rust verbs bridge between
island leaders where an RDMA fabric qualifies, and no Python or Gloo work on a path
that doesn't need it. It does **not** mean deleting Python or Gloo: Python is the
PyTorch integration itself, and Gloo is a native C++ library that gpubridge also uses
for bootstrap, barriers and subgroups, and as the real-TCP negative control.

The hardware evidence behind any "qualified" statement here is one node of four
V100s, as two 2-GPU islands, with verbs through one HCA in loopback
([HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20)).
Nothing below is cross-node, AMD, mixed-vendor or GPUDirect evidence.

> **2026-11-11: three decisions of section 3 are reversed.** The Gloo world group, the
> Gloo bridge as the default, and the plan to deprecate the Rust TCP bridge no longer
> stand. Sections 1 to 5 are kept as written on 2026-10-20, as the record of what ran
> where then and why it was decided; the rows that changed are marked **REVERSED**, and
> [section 6](#6-the-gloo-decisions-reversed-2026-11-11) says what replaced them and
> why. Where the two disagree, section 6 is the current state.

## 1. What runs where

| Area | What runs | Runtime | When | Where |
| --- | --- | --- | --- | --- |
| Training integration | `ddp_comm_hook`: one async `all_reduce(AVG)` per DDP bucket | Python on the training thread, then gpubridge's worker thread | every bucket | `ddp.py`; `work.py:128` (`gpubridge-async`, started at the first async call) |
| | `backend="gpubridge"`: a Python `ProcessGroup` subclass; every collective of DDP, FSDP2 or HSDP enters Python, is queued as a gpubridge async collective, and completes through a torch future | Python | every call | `torch_backend.py` (`GpubridgeProcessGroup._run`) |
| Island collectives | reduce, broadcast, gather, scatter inside each island | NCCL / RCCL on GPUs (Gloo in CPU simulation) | every call | `policies.py`, `topology.py:564` |
| Bulk bridge payload (default) | the leaders' all_reduce or broadcast of the staged island result | Gloo (C++) over TCP; Python only stages and calls | every bridged call | `transport.py:137` (`GlooTransport`), `policies.py` (`ReduceBridgeBroadcast`) |
| Bulk bridge payload (opt-in) | `verbs`: RC send/recv and RDMA WRITE in a Rust library; the collective's chunking, ring order and reductions in Python with torch ops on host tensors | Rust + libibverbs; Python reduces | every bridged call | `verbs/transport.py`, `tagged.py` |
| | `multitcp`: several TCP streams in a Rust library; reductions as for verbs | Rust; Python reduces | every bridged call | `multitcp/transport.py`, `tagged.py` |
| | `ucx`: a C++ UCX shim; reductions in Python | C++ + UCX; Python reduces | every bridged call | `ucx/transport.py` |
| Reference path | `flat-gloo`: every rank stages to host and runs one world Gloo collective | Gloo | when chosen | `policies.py:461` |
| Discovery | every rank's `PeerInfo` (probe, settings, GPU identity) pickled and all-gathered once | Gloo default group (`init()`), or a Gloo world group (backend) | startup | `__init__.py:175`, `topology.py:534` |
| | under the backend, one extra all-gather of setting errors before discovery, so a bad setting fails every rank | Gloo | startup | `torch_backend.py:701` |
| Barriers | `gpubridge.barrier()`: a device sync, then a Gloo barrier of every rank | Gloo | every call | `collectives.py:239` |
| Subgroups | each backend subgroup builds its own topology: two Gloo groups of its members, its islands and its bridge, named alike on every member by a patch of torch's private group naming | Gloo + native | first collective on the subgroup | `topology.py:695-737`, `topology.py:645-692` |
| Bootstrap of native bridges | addresses, queue pair numbers, settings and per-member status through the c10d store | TCPStore (or FileStore) | bridge setup | `tagged.py` (`TaggedBridge.create`), `verbs/transport.py` (`_connect`) |
| Point-to-point | same island: native `isend`/`irecv`; across islands: a store handshake (a JSON size/dtype descriptor per message), then Gloo `isend`/`irecv` on a world Gloo group, one Python lane thread per (peer, tag, direction) | Python + store + Gloo | every cross-island message | `torch_backend.py:408-455`, `p2p.py` |
| Serialization and copies | `PeerInfo` → `asdict` → pickle, once; D2H/H2D staging (`.cpu()`, or one pinned buffer with `GPUBRIDGE_PINNED_STAGING`); verbs staged through registered pinned memory; `reduce_scatter_tensor` clones its input (it must stay unmodified); all-gather assembles host rows | torch | per bridged call | `policies.py`, `staging.py`, `verbs/gpu.py` |
| Observation | `CollectiveObserver`: without observers, a sequence counter and the in-flight table only; with observers, CUDA events and records on the calling thread | Python | every call | `observe.py` |
| | `GPUBRIDGE_VERBS_OBSERVE`: one JSON record per call, peer and direction for the straggler detector, checked on the calling thread and written by a background thread | Python | every verbs call, opt-in | `verbs/observe.py:138` |
| Background workers | `gpubridge-async` (one per rank, lazy); `gpubridge-p2p-completion`, `gpubridge-p2p-health` (polls the store every 50 ms; since the P2P handshake change, only while operations are pending) and `gpubridge-p2p` lanes (lazy, at the first cross-island message); `gpubridge-verbs-observe` (opt-in, wakes every 0.5 s); the verbs library's completion thread; multitcp's I/O threads; Gloo's and NCCL's own threads | Python and native | as noted | `work.py:128`, `p2p.py:103,176,181,296`, `verbs/observe.py:138,157` |
| Imports | `import gpubridge`: 15 gpubridge modules after this branch (25 before); an opt-in bridge's modules load when a job names it | Python | each process | `__init__.py`, `transport.py` (`_BUILT_IN`) |

## 2. Measurements behind the decisions

**Imports** (this Mac mini, Apple silicon, torch 2.14.1 CPU wheel, Python 3.12.15;
`scripts/bench_import.py --before <38b1e20's src>`: fresh processes alternating
between that tree and this branch's, `python -X importtime` for the `gpubridge`
package). At
`38b1e20`, `import gpubridge` loaded 25 gpubridge modules, ten of them the opt-in
bridges' (`ucx`, `multitcp`, `verbs`, and `tagged`, which two of them share), and no
third-party module that torch hadn't already loaded. Every process loaded the ten,
although the default bridge is Gloo.

| `import gpubridge` (median) | `38b1e20` | this branch | change |
| --- | ---: | ---: | ---: |
| gpubridge modules loaded | 25 | 15 | -10 |
| with compiled bytecode (30 runs each) | 7.04 ms | 4.75 ms | -2.29 ms (-33%) |
| without bytecode, `--no-bytecode` (20 runs each) | 31.31 ms | 19.57 ms | -11.74 ms (-37%) |
| peak RSS of `import torch, gpubridge`, with bytecode | 193.07 MiB | 192.62 MiB | -0.45 MiB |
| torch's own import, for scale | 389 ms | 389 ms | - |

So the repair is small: about 0.6% of a process's import time with bytecode. Its
value is that a Gloo job loads no opt-in bridge, a verbs job no longer loads the
multitcp prototype, and a broken optional bridge can no longer break `import
gpubridge` for jobs that don't use it. (An earlier single-tree reading of
13.6 of 36 ms was taken while the test suite ran, with stale bytecode; it is not
used.)

**Bridge paths** (accepted hardware evidence, one node, end-to-end collective
timings; [docs/BENCHMARKS.md](BENCHMARKS.md#the-verbs-bridge-on-one-node-split-test)):
verbs ran at 0.78 to 0.87 times the Gloo bridge's speed from 1 to 64 KiB and 1.7 to
3.1 times from 1 to 256 MiB, over two samples; native NCCL over the same four GPUs was 6
to 81 times faster than either bridge. Pipelining helped the Gloo bridge and generally
slowed verbs. Cross-island P2P over the Gloo bridge took 2.85 ms one way at 1 KiB
(native route 0.14 ms; a whole Gloo-bridge all_reduce of 1 KiB, 0.56 ms).

**Unreferenced code.** Every function and class in `src/gpubridge` is referenced
somewhere else in the repository, except four that torch calls by name
(`_source_rank`, `pg_name`, `alltoall_base`, `recv_anysource`). No module in
`src/gpubridge` is unused: `balance.py` serves `scripts/gpu/train_pipeline.py`,
`_compression_kernels.py` loads only when compression runs on a GPU with Triton.

## 3. Keep, demote, remove

| Part | Decision | Evidence | Before changing it further |
| --- | --- | --- | --- |
| Thin Python API, policies, DDP hook, `backend="gpubridge"` | **Keep** | They are the PyTorch integration: tensors, streams, DDP/FSDP2/HSDP semantics. Exact on two 2-GPU V100 islands (`c92684b`) | A native rewrite would not remove a Python caller: PyTorch calls in from Python |
| NCCL / RCCL islands | **Keep** | The island collectives; 6 to 81 times faster than either bridge on one node | Never replaced without matched evidence |
| Gloo default / world group: discovery, `barrier()`, `flat-gloo` | ~~Keep~~ **REVERSED 2026-11-11** (section 6): no Gloo group of every rank unless the job asks for Gloo | The only cross-vendor group that exists before any bridge does: `__init__.py:175`, `topology.py:534`, `collectives.py:239`, `torch_backend.py:690` | A replacement must pass startup, membership, subgroup, barrier, failure and teardown checks first |
| Gloo bridge (`bridge="gloo"`, the default) | ~~Keep as default~~ **REVERSED 2026-11-11** (section 6): opt-in for one release; the default is `auto` (a Rust bridge) | Runs on any network; one of the two real-TCP negative controls of the verbs kit's no-silent-TCP gate (`4ffcf93`) | The default changes only after cross-node qualification of another bridge |
| `flat-gloo` policy | **Keep** | The bit-for-bit reference of every test (CONTRIBUTING.md) and the other real-TCP negative control | - |
| Subgroup Gloo groups and the private-naming patch | **Keep** (the Gloo groups only with `bridge="gloo"` since 2026-11-11; the naming patch for every setup) | HSDP and DeviceMesh dimensions; refused clearly on a torch whose naming differs (`topology.py:625`) | Recheck at each new torch release |
| CPU simulation and CPU-only modes | **Keep** | The whole CPU suite runs on them | - |
| Verbs bridge (Rust, libibverbs) | **Keep; the preferred bulk bridge to qualify** | Only native RDMA path that has moved payload on hardware; faster than Gloo from 1 MiB on one node; slower up to 64 KiB | Cross-node, AMD and direct-GPU qualification (items 38 to 42); a small-message measure before it becomes any default |
| Python reductions in `tagged.py` (verbs, multitcp) | **Keep for now; measure** | Each chunk is reduced with torch ops in Python, on one torch thread | Profile at 1 to 64 KiB, where verbs loses to Gloo, before moving reductions native |
| Verbs observation export (`GPUBRIDGE_VERBS_OBSERVE`) | **Keep, opt-in** | The detector's transport channel; its cost was A/B-measured and trimmed (docs/VERBS_PERF.md) | - |
| Verbs phase timers & write window tracking (PR #101, `195d7b9`) | **Keep, opt-in in telemetry** | `staged_phases` in `VerbsTransport.telemetry()` and write window timing in `direct.rs`; `scripts/gpu/verbs_staged_ab.py` harness | Host clocks and flag check only; default path unchanged |
| Staged copy overlap (PR #103, `26da9f9`) | **Keep, opt-in (`GPUBRIDGE_VERBS_STAGED_OVERLAP`)** | Opt-in overlap of D2H/H2D copies with 2-leader verbs all_reduce; default-off identity verified upstream in PR #103 (`results/verbs-perf/ab-4-staged-overlap/probes/default-path.json`, commit `d153bf3`), not re-executed on hardware here | Multi-node GPU benchmark required before considering enabled by default |
| Opt-in bridges' eager import | **Demote: lazy (done on this branch)** | Ten modules in every process for code a Gloo job never runs; lazy: 2.3 ms (33%) less import with bytecode, 11.7 ms without | Kept: a built-in name can't be taken by another class, and a broken built-in is a `ValueError`, as an unknown name is, which the torch.distributed backend reports from every rank (`tests/test_transport.py`) |
| UCX bridge | **Demote: opt-in, lazy, unrecommended** | Never run on a GPU or RDMA (items 26, 27); verbs covers the RDMA goal | Keep loadable; decide after one hardware run or a usage check |
| multitcp bridge | ~~Demote: opt-in, lazy; deprecate after a usage check~~ **REVERSED 2026-11-11** (section 6): it is `bridge="tcp"`, what the default runs on without RDMA | Missed its own gate (0.88 to 1.07 times pinned Gloo's time, needed ≤ 0.8; BENCHMARKS.md); not a negative control | A deprecation notice, then removal in a later release, if no user needs it |
| `tagged.py` and verbs importing helpers from `ucx.transport` | **Demote the coupling** (done 2026-11-11: the shared helpers live in `tagged.py` and `rendezvous.py`; verbs still imports the GPU flush code from `ucx.gdr`) | A verbs-only job still loads `gpubridge.ucx` and `gpubridge.ucx.transport` (`tagged.py:37`, `verbs/transport.py:59-60`); `ucx.gdr` is genuinely shared | Move `_REDUCE`, `_segments`, `_wait_for` and `tag` to a neutral module; worth about 3.6 ms (`ucx` + `ucx.transport` cumulative import) |
| Pipelined policy over verbs | **Demote: don't recommend with verbs** | Generally slower than the default policy over verbs on one node; `auto` never picks it | Measure across nodes before any change |
| Pinned staging, sharded bridge, auto-tuned | **Keep, opt-in** | Correct on two 2-GPU islands; no general speedup established | - |
| P2P store handshake and health poller | **Demoted (P2P handshake change): shorter first pauses, no idle polling** | 2.85 ms one way at 1 KiB over the bridge on hardware (native 0.14 ms); `match` paused 2 ms between Store looks (`p2p.py:259`), the health thread polled every 50 ms per rank for the engine's life (`p2p.py:181`). Now pauses start at 50 µs and double to 2 ms, and the health thread is parked while nothing is pending. On CPU: 1 KiB one way 3.9 → 0.63 ms (`bench_p2p.py`), the same Store calls per message, 2 instead of about 70 Store calls per 2 s idle; a blocking `store.wait` was rejected because lanes share one serialized TCPStore client | A peer that never comes still costs each waiting lane a Store look every 2 ms (about 45 ms CPU per lane-second on CPU), as before; a longer pause after a long wait would cut it at a late peer's latency. Hardware latency unmeasured |
| Backend setup's extra all-gather of setting errors | **Keep** (since 2026-11-11 a rank's setting error is its discovery record in the store, with the same effect) | One small Gloo round at startup; it is what makes a bad setting fail every rank instead of hanging the others | Merge into discovery only with the same failure tests |
| Build-on-first-use of native libraries (`rustc --version` and source hashing at the first load) | **Demote for deployments** | Needs a Rust toolchain on the node; skipped when `GPUBRIDGE_VERBS_LIB` / `GPUBRIDGE_MULTITCP_LIB` name a prebuilt library | Deploy prebuilt libraries with their SHA-256 recorded (`scripts/baseline_identity.py`) |
| Anything unreferenced | **Remove: nothing found** | Section 2 | - |

## 4. What this does not decide

- Whether the verbs bridge beats Gloo between machines, on AMD, or with GPU memory
  direct: not run.
- Whether gpubridge's cost on one vendor justifies it against plain DDP on NCCL: the
  matched 1/2/4-GPU series is missing.
- The straggler detector's own Python/Rust split: a separate repository.

## 5. Upstream PR #101 and PR #103 reconciliation (`26da9f9`)

Between the baseline at `38b1e20` and upstream `main` (`26da9f9`), two pull requests merged:
- **PR #101** (`195d7b9`, `friend/verbs-measure-20261019`): Added instrumentation for verbs write window registration and release timing (`direct.rs`), breakdown of host staged collective time into phases (`staged_phases` in `verbs/transport.py`, `verbs/gpu.py`), and a matched one-node A/B benchmark harness (`scripts/gpu/verbs_staged_ab.py`).
- **PR #103** (`26da9f9`, `friend/verbs-pipeline-20261019`): Added opt-in staged copy overlap (`GPUBRIDGE_VERBS_STAGED_OVERLAP=1`) for the two-leader verbs `all_reduce` path, overlapping host-to-device and device-to-host copies with bridge execution where streams permit.

**Runtime impact on this audit:**
1. Both features operate exclusively under the opt-in `verbs` transport (`bridge="verbs"`). Neither is loaded by default jobs nor affects the default Gloo bridge (`bridge="gloo"`).
2. Default-off behavior of staged overlap was verified bit-for-bit against the unoverlapped path upstream in PR #103 (`results/verbs-perf/ab-4-staged-overlap/probes/default-path.json`, commit `d153bf3`; not re-executed on hardware here); timers add one flag check to staged calls and two clock reads per write window only when a completion releases memory.
3. Neither PR alters the lazy import boundaries, dependency inventory, or Python/Gloo execution boundaries established here.
4. Neither PR provides cross-node, multi-node, or multi-vendor hardware qualification; the qualification boundaries in Section 4 and [DAY1-HANDOFF.md](../DAY1-HANDOFF.md) remain intact.

## 6. The Gloo decisions, reversed (2026-11-11)

**What was decided on 2026-10-20** (section 3): keep a Gloo group of every rank for
discovery, `barrier()` and `flat-gloo`; keep the Gloo bridge as the default until
another bridge is qualified between machines; deprecate the Rust TCP bridge, which
had missed its speed gate.

**What changed.** On 2026-11-11 an exit abort was diagnosed (a rank ending with signal
6 and "terminate called without an active exception" after its training had finished
and its results were written). The cause is in torch's Gloo backend on CPython 3.12:
with torch 2.10 to 2.15, a Gloo collective's work object owns a Python object, the
Gloo worker thread (`pt_gloo_runloop`) can hold the last reference to it, and if that
thread asks for the GIL after the interpreter has begun to finalize, CPython 3.12
ends the thread in the middle of C++ destructors, which terminates the process. It
reproduces with a program that imports only torch (2 to 3 jobs in 100 of four DDP
ranks on Gloo). The runtime on the cluster gpubridge is validated on is Python
3.12.15 with torch 2.14.1, inside that range, and every gpubridge job created Gloo
groups. Nothing in gpubridge can fix a thread torch owns; what gpubridge can do is
not create it. That was not known on 2026-10-20, and it outweighs the reasons given
then. David reversed the decision the same day: the Gloo group goes from every job
that doesn't ask for it, and a Rust bridge is the default.

**What replaced each use of Gloo.**

| Gloo was used for | Now | Where |
| --- | --- | --- |
| The default process group of `gpubridge.init()` | gpubridge's own `GpubridgeProcessGroup`, the Python group `backend="gpubridge"` always had: `torch.distributed` calls on it run through gpubridge | `__init__.py` (`init`), `torch_backend.py` (`adopt`) |
| Discovery: one `all_gather_object` of every rank's `PeerInfo` | each rank writes its record as JSON under a key in the job's c10d store and reads all of them, with a deadline that names the ranks that never reported; it runs before any process group exists, so ranks that asked for different bridges are refused instead of waiting in two different rendezvous | `rendezvous.py` (`open_store`, `exchange`), `topology.py` (`discover`) |
| The backend's extra all-gather of setting errors | the same discovery record, carrying the error | `torch_backend.py` (`setup`) |
| `barrier()` | a counter in the store that every rank adds to and watches, one key kept per topology | `rendezvous.py` (`barrier`), `topology.py` (`barrier`) |
| The bridge by default | `bridge="auto"`: `verbs` if every island leader passes the verbs bridge's own check for a usable RDMA device and they are on one link layer, else `tcp` (the Rust TCP bridge, `multitcp` until now). Decided once at `init()`, from reports every rank reads alike; recorded in `Topology.selection` and the bridge's `telemetry()`, logged at INFO; never changed during the job | `transport.py` (`choose_bridge`), `topology.py` (`select_bridge`), `verbs/__init__.py` (`availability`) |
| CPU tensors on a GPU job under the backend (a Gloo group of its own) | staged onto the rank's GPU, carried by the topology, copied back; a pure copy travels as bytes, so any dtype does | `torch_backend.py` (`_on_device`, `_copy_back`) |
| `gather` and `scatter` of CPU tensors under the backend | an all-gather of which the root keeps the rows; a broadcast of the root's rows of which each rank keeps its own | `torch_backend.py` |
| A subgroup's two Gloo groups of its members | none | `topology.py` (`build_subtopology`) |
| Point-to-point across islands (a Gloo group of every rank) | lanes over TCP, one two-member worker per (source, destination, tag), as the UCX lanes | `multitcp/p2p.py` |
| A one-rank island in CPU simulation (a Gloo group of one) | `SoloGroup`: a Python group with no backend, connection or thread | `torch_backend.py`, `topology.py` (`island_backend`) |

**Where Gloo still is**, and only there:

- `bridge="gloo"` (by argument or `GPUBRIDGE_BRIDGE=gloo`): the whole job as it was
  before, default group, barriers and bridge on Gloo. Kept for one release, as the
  reference the other bridges are tested against and one of the verbs kit's real-TCP
  negative controls.
- The `flat-gloo` policy (by name, as an `auto-tuned` rule, or as the reference inside
  a test): it makes a Gloo group of the topology's members when it is set up or first
  used.
- CPU simulation and CPU-only mode, for an island of several ranks: there is no
  NCCL on CPU, so such an island is a Gloo group. Simulation is a test device, not a
  deployment. Such a group carries the exit abort above for as long as the rank holds
  its topology (README, "Simulation mode"): one test's rank did, and aborted in 11 of
  380 runs until it let go before returning (0 of 300).
- torch's own debug wrapper, and only if the job accepts it. At torch's distributed
  debug level DETAIL (`TORCH_DISTRIBUTED_DEBUG=DETAIL`, or
  `torch.distributed.set_debug_level`) torch's `new_group` wraps every group of a C++
  backend, NCCL and RCCL ones included, in a checker (`_ProcessGroupWrapper`) that runs
  on a `ProcessGroupGloo` helper of the same ranks. A GPU job would have Gloo groups and
  threads without naming Gloo. So `init()` refuses a job that didn't ask for Gloo at
  that level: every rank reports its level in discovery, and every rank raises the same
  error, before any island group exists, saying what DETAIL does and the two ways on.
  One is to lower the level. The other is the explicit opt-in `allow_debug_gloo=True`
  (`GPUBRIDGE_ALLOW_DEBUG_GLOO=1` for either init API; the same on every rank, or
  `init()` fails naming the ranks): the job then starts with the helper groups, their
  `pt_gloo_runloop` threads and the exit-abort exposure above, and
  `Topology.debug_gloo` says so. gpubridge never changes the level itself. A job
  whose island groups are all Python groups (one-rank simulated islands) is not
  refused: torch wraps none of them.

So a GPU job has no Gloo group and no thread named `pt_gloo_runloop` or `gloo_tcp_loop`
when `init()` returns if it did not ask for Gloo in any of the ways above (its bridge is
`auto`, `tcp`, `verbs` or `ucx` and not `gloo` or a `GlooTransport` subclass, and its
policy is not `flat-gloo`, by name or as an `auto-tuned` rule) and did not accept
torch's debug helpers with `allow_debug_gloo`. Both conditions are needed: a job on
`bridge="tcp"` with `policy="flat-gloo"` has a Gloo group because of its policy, though
the Rust bridge under it still has no Gloo thread of its own.
Two limits of that sentence:

- **A debug level raised after `init()` is outside the guard.** The islands exist by
  then and are not wrapped. A subgroup set up later (HSDP, a DeviceMesh dimension)
  checks the level again on each of its members and is refused there, but only where
  the level was raised: members that didn't raise it wait for the others until their
  timeout. Anything else the program then creates with torch's own `new_group` on a
  C++ backend is torch's, wrapped, and not seen by gpubridge.
- **What is established, and how.** That torch wraps a C++ group at DETAIL and not a
  Python one is tested on CPU with simulation's Gloo islands standing in for NCCL ones
  (`tests/test_debug_gloo.py`, torch 2.8.0 and 2.15.0); that the rule refuses a layout
  of native islands is tested as a function of the ranks' reports. That torch wraps an
  NCCL group the same way rests on reading `_new_process_group_helper` in those two
  releases (NCCL and Gloo take the same branch). No GPU run has shown the refusal, the
  helper under NCCL, or its absence without DETAIL
  (HARDWARE_VALIDATION.md item 44).

**The checks the 2026-10-20 row asked a replacement to pass**, and what covers each,
all on CPU:

| Check | Covered by |
| --- | --- |
| Startup | `tests/test_no_gloo.py` (discovery through the store, a record of another version, a rank that never reports), the existing `init()` failure tests, now run on store discovery (`tests/test_probe.py`, `tests/test_topology.py`, `tests/test_transport.py`) |
| Membership | `tests/test_topology.py`; ranks that ask for different bridges or policies fail `init()` on every rank before any group exists |
| Subgroups | `tests/test_no_gloo.py` (a bridged subgroup with no Gloo group), `tests/test_torch_backend.py` (HSDP, subgroup naming), `tests/test_multitcp.py` and `tests/test_verbs.py` (subgroups on the Rust bridges) |
| Barrier | `tests/test_no_gloo.py`: it returns only once the last rank has reached it, it times out saying how many arrived, it retires the previous barrier's key |
| Failure | discovery's deadline and reported problems; a bridge that can't be set up fails `init()` on every rank (`tests/test_multitcp.py`, `tests/test_verbs.py`, `tests/test_bridge_selection.py`); an explicit `bridge="verbs"` never drops to TCP |
| Teardown | `tests/test_exit.py`: `destroy()`, the exit handler for a script that never called it, a forked child; `scripts/exit_abort.py`: the same job 300 times, with the Gloo bridge as the control |
| The same bits | `tests/test_parity_fixture.py`: every bridge reproduces hashes the Gloo bridge generated, in jobs with no Gloo group |

**What this reversal did not have**, said plainly:

- The 2026-10-20 row said the default would change "only after cross-node
  qualification of another bridge". It changed without that. The TCP bridge ran
  between two machines (a V100 node and an MI50 node, two step-08 jobs, commit
  23180fe), where it took 0.88 to 1.07 times the pinned Gloo bridge's time: about
  the same speed, not faster. The verbs bridge has run on one node only, through one
  InfiniBand adapter in loopback, staged through host memory.
- The default changed for the missing Gloo thread, not for speed.
- **Nothing in this change was run on a GPU**: not store discovery, not gpubridge's
  own default group over NCCL or RCCL islands, not the store barrier, not `auto`'s
  device check on a real RDMA device, not the TCP point-to-point lanes, not the exit
  handler with NCCL groups. Creating NCCL island groups under gpubridge's own default
  group is what `backend="gpubridge"` has always done, and that ran on GPUs on
  2026-10-20 (torch 2.14.1, DDP and exact FSDP2 and HSDP replays at `c92684b`); the
  rest is in [HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md) as items 44 to 50.
- Bit for bit against Gloo holds for two bridge members, which is every NVIDIA + AMD
  job, on any data; and for more members on integers, MAX, MIN, broadcasts and sums
  that are exact. A floating-point SUM over three or more members that rounds is added
  in ring order by the Rust and UCX bridges, which is not Gloo's order, and can differ
  from Gloo's result in the last bits (`tests/_parity.py`, "Order").
