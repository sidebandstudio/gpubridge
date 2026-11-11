# A raw-verbs bridge transport in Rust

**Status: phases A1 to A3. `bridge="verbs"` exists, opt-in, for host and GPU
tensors. Host-memory and staged-GPU verbs have passed real-hardware checks on
single-host, single-HCA loopback; cross-node and direct GPU paths remain open.**
The 2026-10-20 runs used one node with 4x Tesla V100-SXM2-32GB, driver 545.23.08,
torch 2.14.1+cu126, NCCL 2.29.3, Mellanox ConnectX (mlx5) HDR InfiniBand.
GPU bridge runs used two 2-GPU NCCL islands of identical V100s, not mixed vendors.
At `c92684b`, the staged kit, default-policy check and benchmark passed; the
original host kit failed before payload due to CPU tensors meeting CUDA
communicators. At `0cf32a3` (production fix retained by `54f701e`), exact host
collectives passed with CPU/Gloo islands, and GPU-staged strict checks passed.
Host-mode no-silent-TCP evidence on one host was `not-decisive` until per-process
socket attribution; at `4ffcf93` strict host mode passed decisively on one host. See [hardware results](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20)
and the [changelog](../CHANGELOG.md#unreleased).

**Integration update (2026-10-18; hardware status updated 2026-10-20):** selectable
native progress policies, operation-specific observations, allocation-window
identity checks and bounded async submissions are implemented and tested in
software. The final integration Linux all-target type-check was killed with
SIGKILL; Linux Clippy was not run after that kill. Those native validation gates
remain open independently of the successful loopback hardware runs. Progress
policy timing, hardware failure injection, cross-node, AMD/mixed-vendor,
GPUDirect, UCX, 8-GPU scaling and statistical significance remain unproven by
these runs. The collective `wait-skips-stream` mutation remains uncaught.
See the [capability matrix](VERBS_CAPABILITIES.md) for software coverage.

The plan has three phases, each its own change:

- **A1:** the binding, a C shim compiled against rdma-core's headers, owned
  verbs resources (device context, protection domain, completion queue, queue
  pair, memory region) with their release order, the queue pair state machine,
  a bounded registration cache, and a deterministic mock provider with
  injectable faults.
- **A2:** a versioned reliable-connected (RC) wire protocol on top of
  it (["Wire protocol"](#wire-protocol-version-2)), the endpoint selection rules,
  completion channels and asynchronous events in the shim, a completion thread,
  the Python `BridgeTransport` as `bridge="verbs"` with telemetry, and the
  collective tests bit for bit against `flat-gloo` on a test-only fabric.
- **A3, this one:** registered caller memory and an RDMA WRITE rendezvous (wire
  protocol version 2); the host-staged GPU path; the NVIDIA dma-buf and peer-memory
  and the AMD dma-buf registration paths behind run-time capability checks
  (["GPU memory"](#gpu-memory)); the detector's transport observations
  (["Transport observations"](#transport-observations)); and the hardware kit
  (["Hardware kit"](#hardware-kit)).

Nothing here inherits a validation claim from the UCX transport
([RDMA_TRANSPORT.md](RDMA_TRANSPORT.md), CPU and TCP only) or from the
multitcp transport ([RUST_TRANSPORT.md](RUST_TRANSPORT.md), TCP, which missed
its hardware gate).

## Status

IMPLEMENTED: the code exists. SOFTWARE-VERIFIED: tests pass against a mock
provider. EMULATED: tested over a software RDMA provider (`rdma_rxe`, `siw`).
HARDWARE-VERIFIED: tested on RDMA hardware.

The transport-observations row below records the historical MOCK schema audit,
as does the [2026-10-18 capability matrix](VERBS_CAPABILITIES.md). Separately,
the [2026-10-20 loopback results](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20)
include real host/staged kit observations and staged training/pipeline
observations. Explicit in-process provider/completion telemetry and strict TCP
gates belong to the kit; training/pipeline observations corroborate activity
without independently supplying those gates. Cross-node and direct paths
remain unproven.

| Feature | Status |
| --- | --- |
| C shim over libibverbs (open, query, PD, CQ, QP, MR, dma-buf MR, post, poll; completion channels, `ibv_req_notify_cq`, asynchronous events, batched posts) | IMPLEMENTED; compiles and links against rdma-core 57.0 headers and the system `libibverbs.so.1` on Linux (build evidence, not a transfer) |
| Owned resources, release order, cleanup after a failed setup | IMPLEMENTED, SOFTWARE-VERIFIED |
| Queue pair state machine, IB (LID) and RoCE (GID) addressing checks | IMPLEMENTED, SOFTWARE-VERIFIED |
| Work tracking: no deregistration while work is outstanding, drain before release | IMPLEMENTED, SOFTWARE-VERIFIED |
| Bounded registration cache, keyed by allocation identity (buffer ID or owner token) | IMPLEMENTED, SOFTWARE-VERIFIED (mock GPU: invalidation when an address is reused) |
| Wire protocol v2: handshake, chunking, credits, ordering, goodbye; rendezvous by write grant or "in chunks"; gathered sends from registered memory | IMPLEMENTED, SOFTWARE-VERIFIED (in-process mock; the test fabric carries RDMA WRITE too) |
| Completion thread on a completion channel, asynchronous events | IMPLEMENTED, SOFTWARE-VERIFIED; real libibverbs completions observed in successful host/staged loopback runs; hardware failure cases remain open |
| Deadlines, completion errors, lost peers, teardown with work outstanding | IMPLEMENTED, SOFTWARE-VERIFIED |
| Failure is cancellation: once a wait reports a failure, nothing copies to or from the caller's plain memory | IMPLEMENTED, SOFTWARE-VERIFIED (mock: a peer breaks the protocol mid-message) |
| Sends queued to a member that said goodbye fail at once with "member N closed its end of the bridge" | IMPLEMENTED, SOFTWARE-VERIFIED (mock) |
| Remote write only through one receive's window: exactly its bytes, registered for it, deregistered once written; cached and send-side registrations local-only | IMPLEMENTED, SOFTWARE-VERIFIED (mock: writes before, after and past "written" fail, for host, dma-buf and peer memory); successful host/staged rendezvous exercised on loopback, adversarial hardware tests still open |
| Fork safety checked at open (`ibv_is_fork_initialized`, `ibv_fork_init` before any registration, or a failure with the reason) | IMPLEMENTED, SOFTWARE-VERIFIED (mock statuses); predecessor native check/Clippy passed at `75922a0`; final integration Linux all-target check was killed with SIGKILL, so its native validation gate remains BLOCKED |
| Device, port and GID selection | IMPLEMENTED, SOFTWARE-VERIFIED; mlx5 InfiniBand endpoint recorded on loopback, other selection cases remain open |
| `bridge="verbs"`: every policy, compression, subgroups, async, bit for bit against `flat-gloo` | IMPLEMENTED, SOFTWARE-VERIFIED (the MOCK test fabric between processes, rendezvous on); default-policy GPU check passed on loopback; full pipelined/sharded verbs payload suites remain open |
| GPU tensors, host-staged: a registered pinned buffer, copies on the caller's stream | IMPLEMENTED, SOFTWARE-VERIFIED; staged kit and default-policy GPU correctness passed on the 2026-10-20 split test at `c92684b` (loopback only; item 39) |
| NVIDIA direct, dma-buf (`cuMemGetHandleForAddressRange`, `ibv_reg_dmabuf_mr`) | IMPLEMENTED (selection, descriptor lifetime and reuse on a mock GPU); HARDWARE BLOCKED (item 40) |
| NVIDIA direct, peer memory (`ibv_reg_mr` on device memory with `nvidia_peermem`) | IMPLEMENTED (selection on a mock GPU; module detection on fixtures); HARDWARE BLOCKED (item 40) |
| AMD direct, dma-buf (ROCr `hsa_amd_portable_export_dmabuf`) | IMPLEMENTED (calls checked against ROCm 6.3's headers; selection on a mock GPU); HARDWARE BLOCKED (item 41) |
| Ordering: stream wait before the NIC touches GPU memory, flush after it wrote there, SYNC_MEMOPS, BUFFER_ID identity | IMPLEMENTED, SOFTWARE-VERIFIED (a recording fake endpoint); HARDWARE BLOCKED |
| Direct or staged, with the reason, in telemetry; `direct` only when nothing bounced (else `direct+bounce` with the bounced bytes); `GPUBRIDGE_VERBS_GPU_PATH=direct` fails rather than stage | IMPLEMENTED, SOFTWARE-VERIFIED (fake endpoint) |
| Transport observations (`straggler.transport_observation/1` with `host`, `job`, `nic.pci_bus_id`) | IMPLEMENTED, SOFTWARE-VERIFIED (written on the MOCK test fabric, checked against the schema) |
| Hardware kit: host verbs, GPU staged and direct, training, against Gloo and UCX | PARTIALLY RUN: staged kit/check/benchmark and training passed on identical-V100 split islands; host collectives passed at `0cf32a3`. Cross-node, direct and UCX comparisons remain open (items 38 to 42) |
| Any of the above on a software RDMA provider | not run: none available, and loading one needs privileges (EMULATED would need it) |
| Any of the above on RDMA hardware or GPUs | Host and staged-GPU paths exercised on one HCA on one host; between-host, AMD/mixed-vendor and GPUDirect hardware gates remain open. See the dated hardware results. |

## Bindings

The integration audit's [capability matrix](VERBS_CAPABILITIES.md) maps these
claims to implementation symbols, software checks and remaining hardware gates.

What the transport needs from libibverbs: device listing and opening, port
and GID queries (with the GID type, for RoCE v2), protection domains,
completion queues, RC queue pairs and `ibv_modify_qp`, `ibv_reg_mr`, and for
A3 `ibv_reg_dmabuf_mr`, plus the hot-path calls `ibv_post_send`,
`ibv_post_recv` and `ibv_poll_cq`. Those last three, and `ibv_reg_mr` and
`ibv_query_gid_ex` in current rdma-core, are `static inline` functions in
`<infiniband/verbs.h>`: they reach the provider through function pointers in
`struct ibv_context`, so a binding either compiles C against the header or
re-implements the inline code against a copy of the struct layout.

Checked on 2026-10-18 (crates.io and GitHub):

| | `ibverbs` / `ibverbs-sys` (jonhoo/rust-ibverbs) | `sideway` / `rdma-mummy-sys` (RDMA-Rust) | `rdma-sys`, `async-rdma` (datenlord) | C shim (chosen) |
| --- | --- | --- | --- | --- |
| Release | 0.9.2 / 0.3.2+55.0, 2025-03-08; master moved on (rdma-core 65.0, edition 2024) unreleased | 0.4.3, 2026-06-04 / 0.2.6, 2026-10-10 | 0.3.0 / 0.5.0, 2023-02-01; no pushes since 2023 | ours |
| License | MIT or Apache-2.0 | MPL-2.0 / MIT | MIT / GPL-3.0 | MIT, the repo's |
| MSRV | 1.82 (0.3.2); 1.85 on master | not declared | not declared | 1.77 (`offset_of!`, for the layout checks) |
| Build | vendors rdma-core 55.0 as a submodule, runs cmake on it, runs bindgen (libclang) | builds a static "mummy" libibverbs + mlx5 with cmake that loads the real providers at run time; bindgen | bindgen + pkg-config against installed headers | a C compiler and the headers; no build dependencies |
| Static inlines | re-implemented in Rust through bindgen's layout of the vendored header (`ops.post_send`, `ops.poll_cq`) | written by hand case by case ("inline function and nested structure are not handled properly", its README) | bindgen + hand-written inlines | compiled from the real header |
| `ibv_reg_dmabuf_mr` | not in 0.9.2; master has an mlx5 "data direct" method only | yes | no | yes |
| Dependencies | bindgen, cmake (+ serde) | bindgen, cmake, libc, memoffset, paste, serde, tabled, thiserror, ... | bindgen, libc, memoffset, paste | none (std) |

**Decision: a minimal C shim** (`src/gpubridge/verbs/crate/shim/gbverbs.c`),
compiled by the crate's `build.rs` with the system C compiler against
rdma-core's real headers and linked dynamically to `libibverbs.so.1`. Why:

- **Header-compiled, no copied layouts.** Every libibverbs call, inline or
  not, is compiled by a C compiler against `<infiniband/verbs.h>`, so the
  inline code that runs is the header's own. Rust sees only opaque pointers
  and the few structs the shim defines itself (attributes in, completions
  out), each with `_Static_assert`s on its size and field offsets in C, the
  same constants asserted in Rust at compile time, and a test that compares
  the two at run time. No `ibv_*` struct is mirrored in Rust and there are no
  hand-written offsets. The shim also translates every enum and flag (access
  flags, MTU, QP state, opcodes, completion status), so no libibverbs
  constant is copied either.
- **The cluster ABI.** It links the system's `libibverbs.so.1` by its
  SONAME (`-l:libibverbs.so.1`), so it builds on nodes with the runtime but
  no `-devel` package (Explorer's login node: rdma-core 57.0, no headers) and
  runs against whatever providers the node has. libibverbs keeps its ABI
  within SONAME 1 with versioned symbols. The built library needs
  `IBVERBS_1.1`, `1.8` (`ibv_reg_mr_iova2`, which the header's `ibv_reg_mr`
  calls), `1.11` (`_ibv_query_gid_ex`) and `1.12` (`ibv_reg_dmabuf_mr`)
  (`readelf -V`). The headers should be the installed release's (here
  rdma-core v57.0); a runtime too old for them fails at load time with a
  missing symbol version, not later.
- **Builds offline with the repo's toolchain.** Like the multitcp crate, it
  uses the standard library only (`build.rs` runs the C compiler and `ar`
  itself rather than through the `cc` crate), so `cargo build --offline`
  works on a node with a Rust toolchain and a C compiler. `ibverbs-sys` needs
  cmake (not on Explorer's login node) and libclang, and its newest release
  fixes the bindings to rdma-core 55.0's header.
- **Coverage.** The shim wraps exactly what the transport uses, including
  `ibv_reg_dmabuf_mr` and `ibv_query_gid_ex`, which the released `ibverbs`
  crate lacks (A3 registers GPU memory through `ibv_reg_dmabuf_mr`). `sideway` has them, but brings MPL-2.0, a statically linked
  copy of rdma-core's providers and nine other crates.
- **Cost.** About 400 lines of C to keep correct, each wrapper a direct call.
  Adding a verb is a C function and its Rust declaration. If a maintained
  crate later covers dma-buf, builds without cmake and compiles the inlines
  from the installed header, it can replace the shim behind the same
  `Provider` trait.

## The crate

`gpubridge-verbs`, in [`src/gpubridge/verbs/crate`](../src/gpubridge/verbs/crate),
laid out like the multitcp crate. It builds and tests anywhere with only the
mock provider; the libibverbs provider is behind the `ibverbs` feature.

**Build switch.**

```bash
cd src/gpubridge/verbs/crate
cargo test                          # mock provider only: macOS, or Linux without rdma-core headers
cargo test --features ibverbs       # also the libibverbs provider; needs rdma-core's headers
cargo test --features test-fabric   # also the cross-process test fabric (any Unix)
```

| Variable | Meaning |
| --- | --- |
| `GPUBRIDGE_VERBS_INCLUDE_DIR` | a directory holding `infiniband/verbs.h` (and `rdma/`), when the headers aren't in the compiler's default path; rdma-core's source tree has them in `libibverbs/` and `kernel-headers/rdma/` |
| `GPUBRIDGE_VERBS_LIB_DIR` | where `libibverbs.so.1` is, when not in the linker's default path |
| `GPUBRIDGE_VERBS_CACHE_DIR` | where `python -m gpubridge.verbs build` (and the first `load()`) puts the built library: `~/.cache/gpubridge` by default. (The Python side used to read `GPUBRIDGE_VERBS_LIB_DIR` for this too, so setting it as above also put the build in the `libibverbs` directory.) |
| `CC`, `AR` | the C compiler and archiver (`cc` and `ar` by default) |

Without the headers, `--features ibverbs` fails at build time and says
which variable to set.

**Layers.**

- `Provider` (`provider.rs`): the verbs calls, on opaque handles. Two
  implementations: `IbvProvider` (`ibverbs` feature: the shim) and
  `MockProvider` (always).
- Owned resources (`resources.rs`): `Context`, `ProtectionDomain`,
  `CompletionQueue`, `QueuePair`, `MemoryRegion`. Each holds a reference to
  what it was made from, so the release order is structural: a queue pair
  holds its completion queues and protection domain, a memory region its
  protection domain, and both of those the context. Dropping things in any
  order releases them in the right one, and a setup that fails halfway drops
  what it had made, in reverse.
- Work tracking: every posted work request is filed on its completion queue
  with references to the memory regions it reads or writes, and is removed
  only when its completion is polled. A memory region (and, for one that owns
  its buffer, the buffer) is released only after the last such reference
  goes. A queue pair that is closed or dropped with work outstanding first
  moves to the error state, which flushes everything posted on it, and polls
  until each request has completed; only then is it destroyed. If that drain
  runs out of time, the regions it still references are leaked on purpose
  (never deregistered, never freed) and the close reports it: a leak is
  recoverable, the NIC writing into freed memory is not.
- Bounded queues: posting more sends or receives than the queue pair was
  created for fails with `QueueFull` rather than overrunning it, and a queue
  pair can only be created on completion queues with room for all its work.
- `MrCache` (`cache.rs`): the registration cache.
- `gpu.rs`: the GPU drivers, loaded at run time (`CudaDriver`, `RocmDriver`), the
  peer-memory module probe, and `MockGpu` for tests (["GPU memory"](#gpu-memory)).
- `Registrar` (`direct.rs`): registers device memory (dma-buf, then peer memory,
  with the reasons when neither works) and the caller's host buffers, each through
  its own bounded `MrCache`.
- `Endpoint` (`endpoint.rs`): one bridge member's end of the wire protocol,
  over any provider; `protocol.rs` has the header, the handshake line and the
  endpoint selection; `ffi.rs` the C interface Python loads (`gb_verbs_*`).
- `UdsProvider` (`uds.rs`, the `test-fabric` feature only): the test fabric
  between processes (["Testing"](#testing)).

**Queue pair states.** `Reset -> Init -> Rtr -> Rts`, and `Err` from any
of them. Each transition checks its attributes first: queue pair numbers and
PSNs fit in 24 bits, retry counts in 3 bits, timeouts in 5; the path MTU is
at most the port's active MTU; an InfiniBand port takes a nonzero LID (and an
optional global route), an Ethernet (RoCE) port must have a global route with
a GID index inside the port's table and a nonzero destination GID. Receives
can be posted from `Init`, sends only in `Rts`. Going back to `Reset` is
refused while work is outstanding (verbs would discard it without
completions).

**The registration cache.** Keyed by the allocation's identity: address,
length, access flags and an owner token (for device memory the driver's buffer
ID; for host buffers a token the caller supplies), so the same address handed
out again is a different key. A lookup may register more than it asks for (a
covering range: device memory is rounded out to 2 MiB inside its allocation).
Bounded by entries and by bytes, evicting the least recently used idle
entry. A lookup inside a cached registration of the same allocation reuses
it. Registering an allocation drops cached entries of other owners that
overlap it (that memory was freed and reused). `invalidate` drops entries
when an allocation is freed. Nothing is evicted or deregistered while it is
in use: an entry handed out or referenced by posted work stays registered
until both are gone, and a cache with no idle entry to evict refuses the new
registration (`Exhausted`) instead of growing. Holding the address is not
holding the registration; holding the `MemoryRegion` is.

**Unsafe code.** Every `unsafe` block has a `// SAFETY:` comment. They are
in the libibverbs provider (calls into the shim), in the mock providers (their
"DMA" copies between registered memory), in `MemoryRegion`'s owned
buffers, in the endpoint's copies between its slots and the caller's memory
(`Endpoint::send` and `recv` are `unsafe fn`: the caller keeps the memory valid
until the operation finishes, fails or is aborted), and in the C interface.
Registering memory the region doesn't own is an `unsafe fn`: the caller
promises the memory stays valid until the region is released. A dma-buf region
is safe to make: the kernel pins the exported memory while it is registered. The
GPU driver calls (`gpu.rs`) are `unsafe` calls through function pointers looked
up by name; each signature is the header's, checked by
`shim/gpu_abi_check.c` against cuda.h 12.8, HIP 6.3's `hip_runtime_api.h` and
ROCr's `hsa.h`/`hsa_ext_amd.h` (rocm-6.3.0).

## Wire protocol (version 2)

**Verbs operations used:** `IBV_WR_SEND` with `IBV_SEND_SIGNALED` (every send
completes visibly) and receives (`ibv_post_recv`), on RC queue pairs; a send has
one scatter-gather entry (a slot), or two (a slot's header, then a payload
gathered from registered memory); a receive one (a slot). With rendezvous on
(`rndv_bytes` > 0), also `IBV_WR_RDMA_WRITE`, signaled, from a slot or from
registered memory, into memory the peer granted. No READ or atomic is posted.
Without rendezvous the queue pairs grant no remote access (`qp_access_flags` 0)
and no remote key exists; with it they grant remote write, and a remote key
crosses the fabric only in a write grant: the key of a **write window**, a
registration of exactly that receive's bytes made for it alone and deregistered
once it is written (never logged or reported). No other registration is remotely
writable: not the slots, and not the cached registrations of caller memory, which
are local-only. Several requests are posted as one
chained list (one doorbell).

**Guarantees this rests on (RC):** messages on one queue pair arrive once, in
order, intact; a send's successful completion means the peer's adapter has the
message in a posted receive buffer (not that the peer's program has read it); the
responder executes the requests of one queue pair in order, so the data of an
RDMA WRITE has been placed in the responder's memory before a SEND posted after
it on the same queue pair completes there; a failed request moves the queue pair
to the error state, which completes everything after it with `WR_FLUSH_ERR`.
Placement in host memory is then visible to the responder's CPU; placement in
GPU memory needs the flush in ["GPU memory"](#gpu-memory) before GPU work reads
it.

**Endpoints.** Each member has one completion channel and one completion queue,
and one RC queue pair per other member. Per peer it registers `send_slots`
send buffers and `credits + ctrl_credits` receive buffers of `slot_bytes` each
(defaults 32, 32 + 4 and 128 KiB), owned by the endpoint and page-aligned.
Every receive buffer is posted while the queue pair is in `Init`, before the
member publishes its handshake. **Plain caller memory is never registered**: a
message is copied from the caller's tensor into a send slot, and out of a receive
slot into the caller's tensor (or a heap buffer, see below), so the device never
reads or writes it and a drain that fails can only leak the endpoint's slots.
**Registered memory** (A3) is the exception the caller asks for: device memory
(["GPU memory"](#gpu-memory)) and host buffers the caller owns and registers
(the Python bridge's pinned staging buffer and scratch buffer). Sends gather
their payload from it; with rendezvous on, receives are written into it.

**Handshake (through the c10d store).** Each member publishes one line under
`gpubridge/verbs/<link>/addr/<rank>`: `<protocol>|<chunk bytes>|` then

```
gbverbs 2;ll=ib|eth;lid=<LID>;gid=<32 hex digits>;grh=0|1;mtu=<active MTU>;
slot=<slot_bytes>;credits=<n>;ctrl=<n>;rndv=<bytes>;qpn=<per member>;psn=<per member>;
dev=<name>;port=<n>
```

(`qpn` and `psn`: the queue pair number and first send PSN this member uses for
each member index, 0 for itself.) Readers ignore keys they don't know. Each
member refuses to connect, and setup fails on every rank with the reason, when
the protocol version, the link layer (InfiniBand against Ethernet), or
`slot`/`credits`/`ctrl`/`rndv` differ, when an InfiniBand peer has no unicast LID, or
a GID-addressed peer sent no GID. The path MTU is the smaller of the two active
MTUs. Then `Init -> Rtr -> Rts` (RTR: `min_rnr_timer` 12, no RDMA reads; RTS:
ACK timeout exponent 14 and 7 retries by default, `rnr_retry` 6). Each member
then publishes `ready` and **sends nothing until every member is ready**: a
message reaching a queue pair still in `Init` is dropped, and its sender's
retries can run out before the peer connects. Finally one 8-byte message each
way checks the data path.

**Header.** Every message starts with 48 bytes, little-endian:

| Bytes | Field |
| --- | --- |
| 0-1 | magic `0x4256` |
| 2 | protocol version (2) |
| 3 | kind: 1 data, 2 credit, 3 goodbye, 4 clear to write, 5 clear to send in chunks, 6 written |
| 4-7 | sequence number of this message on this queue pair, from 0, wrapping |
| 8-11 | payload bytes after the header (0 for credit and goodbye) |
| 12-15 | data receives the sender has reposted for its peer, cumulative, wrapping |
| 16-19 | control receives the sender has reposted for its peer, cumulative, wrapping |
| 20-23 | clear to write: the remote key of the granted memory; otherwise zero |
| 24-31 | tag |
| 32-39 | the whole message's length |
| 40-47 | data: this chunk's offset in the message; clear to write: the granted address |

A receiver checks the magic, version, sequence number, that the completion's
byte count is the header plus the payload, that a chunk lies inside its message
and follows the last one, and that credits never come back beyond what was
sent; that only a write grant carries a key and an address, that a rendezvous
message names a non-empty message, and that a "written" message answers a grant
this member made; any violation fails the endpoint ("member N broke the
protocol: ...").

**Messages and chunking.** A message is `(tag, bytes)` to one peer, split into
chunks of up to `slot_bytes - 48` bytes, sent in order (an empty message is one
empty chunk). Messages to one peer go out in the order they were posted, each
whole before the next; messages with the same tag from one peer are received
in the order sent. A posted receive takes the next message with its tag and
must be exactly its length ("member N sent X bytes for tag T, posted for Y
bytes" otherwise). The Python collectives use 64-bit tags of collective number,
step, sender and chunk (the multitcp bridge's).

**Rendezvous.** With `rndv_bytes` > 0 (the same on every member; the Python
bridge sets 256 KiB unless `GPUBRIDGE_VERBS_RNDV_BYTES` says otherwise), a
message of at least that many bytes is not sent until its receiver has posted
the receive and cleared it:

- into registered memory (`recv_registered`), with a **write grant** (kind 4:
  tag, length, address, key): the sender RDMA WRITEs the payload there (from its
  registered memory directly, or slot by slot from plain memory), then SENDs a
  **written** message (kind 6). The receive completes when that message is
  polled, by which time RC has placed the writes before it. The key is the
  receive's own write window (`Registrar::expose`): a separate, uncached
  registration of `[address, address + length)` with remote write, made when the
  receive is posted (for a dma-buf, from a fresh export of the pages around it)
  and deregistered as soon as its "written" message is polled, so a write before,
  after or past "written" fails with `REM_ACCESS_ERR` instead of landing in a
  neighbouring tensor or a later call's memory.
- into plain memory, with **clear to send in chunks** (kind 5): the message then
  goes as data chunks like any other.

A sender binds grants to its queued messages with the same tag in order, and a
grant whose length differs from the message is a protocol error on both sides. A
message waiting for its grant holds up only later messages with its tag; others
to that peer go ahead. Grants and "written" messages are data-class messages:
each takes a data receive and a data credit, so the credit rules below cover
them. Device memory is only ever written by grant: a receive into it below the
rendezvous size, or with rendezvous off, is refused when posted.

**Credits and receive provisioning.** A sender may have at most `credits` data
messages and `ctrl_credits` control messages to a peer that the peer hasn't
reported reposting. A receiver reposts a receive buffer as soon as it has
copied the message out (or taken its grant or notice), and reports the cumulative counts in the header of
every message it sends that peer. When it owes at least `credits / 2` data
credits and has no data to send that peer (or no credits to send it with), it
sends a credit message, which uses a control credit. So a send always finds a
posted receive (running out of RNR retries is a protocol error, reported), no
credit message is needed to return control credits (every credit message
returns at least one data credit, so the peer's next data carries the count),
and credit messages can't deadlock. Data credits returned beyond those sent
fail the endpoint.

**Unexpected messages and backpressure.** A message that arrives before its
receive is posted is copied into a heap buffer, up to `unexpected_bytes` (256
MiB by default) per endpoint; an empty message counts as one byte. Past that, it and every later data message from
that peer stay in their receive buffers, unreposted, so their credits don't go
back and the peer stops sending; posting receives drains them in arrival
order. Posted operations are bounded too (`max_pending`, 65536); beyond that a
post fails. Sends wait for credits and free send slots inside the library, not
in the caller's thread.

**Completion thread.** One thread per endpoint: poll the completion queue
until empty, ask for a notification, poll once more (a completion between the
two would otherwise wait for the next), then block in `poll(2)` on the
completion channel's descriptor and the device's asynchronous event descriptor
(both non-blocking), at most 100 ms so it notices a close. Notifications and
events are acknowledged as they are taken. It does everything a completion
causes (copy out, repost, return credits, send what credits now allow, wake
waiters) in batches; no Python runs per completion and nothing spins.

**Deadlines.** Every wait has one (the collective's, from `init()`'s timeout);
it fails with "timed out: k of n operations unfinished after s". The Python
bridge then aborts the endpoint: a timed-out collective may have left messages
half sent or received, so the bridge refuses later calls.

**Errors and peer loss.** A completion with an error status fails every
operation with that peer and stops using its queue pair, with the status and
vendor error in the message (`RETRY_EXC_ERR`: "the peer didn't acknowledge
within the retry budget (it exited, or the path to it is down)"). An
asynchronous event that breaks something (`QP_FATAL`, `QP_REQ_ERR`,
`QP_ACCESS_ERR`, `PATH_MIG_ERR`, `CQ_ERR`, `PORT_ERR`, `DEVICE_FATAL`) fails
that peer, or the whole endpoint when it names no queue pair. A lost peer is
noticed when something is sent to it; a receive from a peer that died silently
waits for its deadline. Which operations report a failure as their own (`error`)
and which as `cancelled` is in "Transport observations".

**Failure is cancellation.** An operation fails (a completion error, a lost or
departed peer, a protocol error, a deadline the caller turns into `abort`) only
by being cancelled, under the endpoint's lock, before anyone can be told: it
leaves every queue, a chunk arriving for it is dropped, its posted chunks stop
counting for it. A wait that sees a failed operation fails the endpoint, and a
failed endpoint cancels every operation, posts nothing more (no chunks, no
credits) and moves every queue pair to the error state: a failure is an abort. So once `wait` has reported a failure, nothing copies to or from the
caller's plain memory again, which is what `send`/`recv`'s contracts (Rust and C)
promise; it no longer rests on the Python bridge calling `abort` first.

**Registered memory and failures.** A receive granted a write that fails
(deadline, lost peer, abort) gives its write window to its peer's queue pair: the
peer may have been writing until that queue pair reached the error state, so the
window, and the registration it lies in, are deregistered only once the queue
pair is closed (and never, if it didn't drain). A failure (any failed wait, not
only the Python bridge's abort) moves every queue pair to the error state, so once
it is reported no peer can write a window any more; a write the device had already
begun may still finish (how promptly is hardware-dependent), so the Python bridge
keeps such tensors until the endpoint closes, and Rust and C callers must keep
registered memory allocated until close too. Sends from registered memory hold their regions through their work
requests, as slots do.

**Teardown.** Closing first sends each connected peer a goodbye (best effort,
up to a second for it to complete), so the peer takes what follows as a close,
not a failure: messages that arrived before it stay receivable, receives that
can no longer be met fail at once, new sends to it are refused, and sends queued
to it or part-way fail at once with "member N closed its end of the bridge"
(posted chunks still complete). Then every
operation is cancelled under the endpoint's lock (from then on nothing reads
or writes caller memory), every queue pair moves to the error state, the
completion thread takes the flushed completions until none is outstanding or
the close's timeout passes, the thread stops (it is then joined with no time
limit of its own: the stop flag and its 100 ms poll are what end it, and it is
never detached, since it can touch caller memory; a device call that never
returned would hold the close, which no hardware test has provoked), and the
queue pairs close (A1's
drain-or-leak: work still outstanding keeps its slots registered and allocated,
and the close says so). `abort` is the same without the goodbye or the release. When the drain
times out, registrations still held by outstanding work are leaked too (never
deregistered), and the Python bridge then never frees their memory.

`gb_verbs_close` frees the endpoint's handle, so no other thread may be in the
library with it meanwhile; every `gb_verbs_*` function refuses a null handle. The
Python `Endpoint` counts the calls each thread has in the library. Its `close`
refuses new ones, and if one is still in (an async collective blocked in a wait
while the process groups are destroyed), it aborts first so that call returns at
once, and waits for it, within the close's timeout, before closing. A call still
there after that leaves the endpoint open and never freed (leaked, like undrained
work: `close` returns False and the bridge keeps its buffers).

## Choosing the endpoint

`GPUBRIDGE_VERBS_DEVICE`, `GPUBRIDGE_VERBS_PORT` and `GPUBRIDGE_VERBS_GID_INDEX`
first: a named device or port is used or setup fails with the reason; it is
never swapped for another. Otherwise: devices in name order, their ports in
order, the first `Active` port with a usable address. An InfiniBand port needs
a unicast LID (a subnet manager) and is addressed by LID without a global
route, unless a GID index is set. A RoCE port needs a GID: the set index, or
the best in its table: RoCE v2 with an IPv4-mapped address, then other RoCE v2,
then link-local RoCE v2, then RoCE v1, the lowest index among equals. Members
on different link layers refuse to connect. Telemetry reports what was chosen.

## GPU memory

A3 moves GPU tensors one of two ways per call, and says which and why
(`telemetry()["path"]` and `["path_reason"]`; `paths` counts them):

**Host-staged** (`staged`). The tensor is copied into a pinned host buffer on the
caller's current stream; the host waits for an event recorded after that copy
(only that copy and the work queued before it) before the bridge reads the
buffer; the collective runs on it; the copy back is queued on the same stream,
so nothing the caller queues after the collective sees the buffer's old contents;
the next call's copy out waits for that copy back on the device before the host
overwrites the buffer. This is `PinnedBridgeStaging`'s protocol (`staging.py`),
unchanged; the verbs bridge registers the buffer (and its scratch buffer) with the
NIC once, so sends gather from it and receives are written into it with no host
copy. A larger tensor replaces the buffer after its last copy back has passed,
the old registration dropped first.

**Staged, chunk by chunk** (`GPUBRIDGE_VERBS_STAGED_OVERLAP=1`; opt-in, off by
default). For a two-leader all_reduce of more than one chunk the staged call
overlaps its copies with its messages: every chunk's copy out is queued at once (the
first after the last copy back, on the device); then per chunk the receive is
posted, the host waits for that chunk's copy out and posts its send; then per
chunk, once its send and receive have completed, it is reduced and its copy back
queued. Each copy back queued is the one the next call's first copy out waits for.
The messages are the same as a whole call's (the same buffers, registrations,
tags and sizes), so leaders with it on and off interoperate. Broadcasts, more than
two leaders and tensors of one chunk are staged whole; `last_gpu["staging"]` says
`whole`, `overlapped`, or `whole (<why not>)` when it is on. Two differences: an
observation's `duration_s` starts before the first chunk's copy out has passed, and
a call that fails part way may already have copied some reduced chunks back into
the caller's tensor. It still raises, and the bridge refuses later calls. Whatever
fails, queueing a copy included, the worker is aborted first, then the host waits
for every copy the call queued on the caller's stream (that stream only), so none
still writes the caller's tensor or the staging buffer once the error is raised,
async collectives included. If that wait fails too (on a GPU, a sticky error), the
call's own error is still the one raised, with the wait's as its `__cause__`; its
copies may then still be pending, so the staging buffer stays kept.
SOFTWARE-VERIFIED (CPU tensors standing in, lazy copies,
`tests/test_verbs_staged_overlap.py`); not validated on GPUs (the 2026-10-20
staged results predate it), and no speedup is claimed: docs/verbs-perf-20261019.md, section 7, is the measurement that would show
one.

**Direct** (`direct`). The tensor's memory, and the reduction's scratch buffer,
are registered with the NIC, and the NIC reads and writes them in place. Only
when every check passes and both registrations succeed; otherwise the call is
staged and the reason recorded, or, with `GPUBRIDGE_VERBS_GPU_PATH=direct`, the
call fails with the reason before anything is posted (`staged` forces staging).
The checks, once per GPU and cached:

1. rendezvous is on (a peer can only write into device memory by grant);
2. the GPU driver loads at run time: `libcuda.so.1` (CUDA driver API), or the HIP
   runtime and ROCr the process already mapped (`libamdhip64.so*`,
   `libhsa-runtime64.so*` from `/proc/self/maps`, never a second copy); the crate
   has no build dependency on either;
3. a registration method is available (below);
4. the NIC's writes can be made visible to the GPU's kernels
   (`gpubridge.ucx.gdr.for_device`, shared with the UCX bridge): not needed where
   `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING` is at least `OWNER`, else
   `cuFlushGPUDirectRDMAWrites` where `..._FLUSH_WRITES_OPTIONS` has `HOST`; on ROCm
   `hipDeviceFlushGPUDirectRDMAWrites` where the runtime has it, else the HDP flush
   register. Neither: staged, with that reason.

**Registration methods**, tried in order for each new registration (`direct.rs`):

| | NVIDIA dma-buf | NVIDIA peer memory | AMD dma-buf |
| --- | --- | --- | --- |
| Capability | `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` = 1 and `cuMemGetHandleForAddressRange` present (CUDA 11.7+) | `nvidia_peermem` or `nv_peer_mem` in `/sys/module` or `/proc/modules`; `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` not 0; the allocation's `CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE` not 0 | `HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED`; `hsa_amd_portable_export_dmabuf`/`close_dmabuf` present |
| Registration | `cuMemGetHandleForAddressRange(&fd, start, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)` (page-aligned range), then `ibv_reg_dmabuf_mr(pd, 0, len, iova=start, fd, access)` | `ibv_reg_mr` on the device address | `hsa_amd_portable_export_dmabuf(start, len, &fd, &offset)`, then `ibv_reg_dmabuf_mr(pd, offset, len, iova=start, fd, access)` |
| Descriptor | closed with `close(2)` | none | closed with `hsa_amd_portable_close_dmabuf` |
| Identity | `CU_POINTER_ATTRIBUTE_BUFFER_ID`, `cuMemGetAddressRange` | as dma-buf | `HIP_POINTER_ATTRIBUTE_BUFFER_ID`, `hipMemGetAddressRange` (not HIP's `RANGE_SIZE`, a 32-bit value in HIP 6.3); HIP 6.0 or later |
| Once per registration | `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` set to 1 | as dma-buf | nothing (NVIDIA's rule; ROCm's export asks for none) |

Each method's failure is kept: when all fail, the reason reads `dma-buf: ...;
peer memory: ...`. CUDA calls run with the device's primary context pushed
(retained once, released when the driver goes). HIP 6.3 has no dma-buf export of
its own, so ROCr's is used, as ROCm's headers document it.

**Lifetime rules enforced.**

- *Allocation identity.* Registrations are cached under the allocation's buffer
  ID, which the driver never reuses within a process: memory freed and handed
  out again at the same address is a different owner, and its stale entry is
  dropped (or the registration refused while posted work still holds it, as the
  memory was freed too early). The cached range is rounded out to 2 MiB inside
  the allocation, so tensors of one allocator segment share it; that is why it is
  registered local-only, and a peer writes only through a receive's own window.
- *Fork safety.* When an endpoint opens, before it registers anything, it asks
  `ibv_is_fork_initialized` (looked up at run time; rdma-core 35 and later). If
  the kernel keeps registered pages with the parent on `fork()` (Linux 5.13 and
  later), nothing is done; otherwise, or when the library can't say, it calls
  `ibv_fork_init`, and if that fails (memory was already registered elsewhere in
  the process) it refuses to open, saying to set `RDMAV_FORK_SAFE=1`. Telemetry's
  `fork_safety` says which. The cache is
  bounded (`GPUBRIDGE_VERBS_DEVICE_CACHE_BYTES`, 8 GiB, 256 entries; host 8 GiB,
  64 entries), evicting idle entries only.
- *Holding a pointer is not holding the memory.* A dma-buf import pins the
  exported memory while registered; the peer-memory module pins GPU pages and
  revokes the registration if the allocation is freed. The NIC touches a
  registration only while work posted with it is outstanding or a peer holds the
  window of a pending receive in it; the Python bridge holds the tensors until then
  (and, after a failure, until the endpoint closes).
- *Descriptors.* An exported dma-buf descriptor is owned by its region and closed
  exactly once, after `ibv_dereg_mr` returns; if registering fails it is closed at
  once; if deregistering fails it is never closed (the region is leaked, like an
  owned buffer).

**Ordering on the direct path** (NVIDIA's GPUDirect RDMA guide; the ROCm flush as
the UCX bridge does it):

- before the first message after a wait (and at the start) touches device
  memory, the host waits for an event recorded on the caller's stream: the NIC
  must not read what a kernel is still writing, nor write what a queued kernel
  still reads;
- a receive into device memory completes only when the sender's "written"
  message is polled: the writes are complete with respect to this CPU thread;
- then, before `wait` returns and before any dependent GPU work is launched, the
  GPU's flush runs (nothing where the GPU orders such writes itself);
- a message below the rendezvous size can't be written into device memory: it
  arrives in a pinned bounce buffer and is copied to the GPU on the caller's
  stream, after the wait and before the kernels that read it. The call is then
  not `direct` but `direct+bounce`, its reason giving the bounced receives and
  bytes (also in `last_gpu` and in the observations' `bounced_bytes`).

## Native progress policies

`Config.progress` (`progress=...` in the native endpoint settings) and
`GPUBRIDGE_VERBS_PROGRESS` select a **local** policy; peers may choose different
policies without changing the wire protocol. All use the same dedicated native
completion thread and the same ordering, deadline, cancellation and DMA-safe
teardown code. No shared async-runtime worker busy-polls.

| Policy | Behavior and CPU cost |
| --- | --- |
| `economy` (default) | Original behavior: drain ready CQEs, arm notifications, re-poll, then block on completion/async events for at most 100 ms. Low idle CPU; no reserved core. |
| `balanced` | After work or an event wake, spin for at most **50 microseconds or 64 empty polls**, whichever is reached first, then use the same arm/re-poll/event wait. Extra CPU during brief idle gaps, no continuously reserved core. |
| `latency` | Continuously polls the CQ on the native thread. Can consume approximately one CPU per bridge endpoint (normally one per island leader, and more for extra subgroup endpoints), including while idle. Opt-in only; no core is automatically reserved or pinned. |

Every policy services queued completion notifications and asynchronous device
errors at least once per 1 ms of active polling (subject to scheduler/handler
delays). Failed endpoints continue draining posted work with event waits rather
than spinning. Event policies always re-arm **then re-poll before sleeping**.
The shim consumes queued events and acknowledges each successful get immediately;
shutdown and CQ drop also drain queued notifications before releasing resources.
Spurious events with an empty CQ are normal. These choices follow
[rdma-core's completion-event contract](https://github.com/linux-rdma/rdma-core/blob/v57.0/libibverbs/man/ibv_get_cq_event.3).

Optional `GPUBRIDGE_VERBS_PROGRESS_CPU=<integer>` (`Config.progress_cpu`) applies
only to `latency` in a Linux `ibverbs` build. Setup checks membership in the live
calling thread's allowed set using `sched_getaffinity`; the new progress thread
checks its inherited set again before applying `sched_setaffinity` to itself.
An out-of-set CPU, unsupported platform/build, or affinity failure rejects setup.
The setting never widens a job's allowed CPUs or changes a training thread's
affinity. Choose a permitted CPU near the selected NIC/GPU NUMA node, considering
the job's existing CPU allocation and memory placement. CPU topology and optimal
placement must be measured on the allocation; a CPU number alone implies no NUMA
locality. Hardware performance and native affinity execution remain BLOCKED.

The software tests use strict one-shot notifications (arming never creates a
retroactive event), injected poll/arm-boundary CQEs and spurious notifications.
They verify bursts, empty messages, errors, disconnect, deadlines, abort and
shutdown for all policies, including registered GPU-memory pairings and keeping
a granted write window alive after abort. These are MOCK evidence only.

## Telemetry

`VerbsTransport.telemetry()` (and `state`: `ok`, `failed`, `closed`) returns
the endpoint's JSON: `protocol`, `provider` (`libibverbs (<the loaded file>)`,
whose name carries rdma-core's version, or `MOCK: ...`), `mock`, `library`
(the crate's version), `device`, `port`, `link_layer`, `gid_index`, `gid_type`,
`active_mtu`, `firmware`, `vendor_id`, `vendor_part_id`, `operations` (`RC
SEND/RECV`, plus `RDMA WRITE (rendezvous)` when on), `rndv_bytes`, `gather`, the
protocol settings, `fork_safety` (what the open found and did), `state`,
`progress` and `progress_cpu` (the local policy and optional native-thread CPU),
`failure`, counts (operations, bytes, chunks, control
messages, credit stalls, unexpected and held bytes, completions, errors,
notifications, asynchronous events, batches, grants and "go"s sent and received,
writes and their bytes, "written" messages, bytes moved in place from registered
host and device memory, bytes the CPU copied, write windows deregistered once
written and kept after a failure), `direct` (the GPU driver and its version or why
it didn't load, which methods are enabled, registrations by method,
`cached_access` (`local`), write windows made, their bytes and the nanoseconds
spent making them, windows released once written and the nanoseconds that took
(on the completion thread), failures and the last reason, both caches' entries, bytes, hits, misses,
evictions, stale entries, invalidations, refusals), and per peer its queue pair
number, path MTU, whether it said goodbye and why it was lost. The Python bridge
adds `path` (`host`, `direct`, `direct+bounce`, `staged`) and `path_reason` of
the last call, `last_gpu` (device, method, flush, stream waits and flushes,
bounced receives and bytes), `paths`, `gpu_path_setting`, `gpus` (each GPU's
direct-path support: driver, flush, capabilities or reason), operation counts by
memory kind, `staged_phases` (for the staged calls that succeeded: `calls`, of
them `overlapped_calls`, and host seconds in `to_host_s`, the copy out queued and
waited for, `bridge_s`, and inside it `post_s` and `wait_s`, the rest being
reductions and Python, and `to_device_s`, queueing the copy back; an overlapped
call's are the same split of its host time, its waits for each chunk's copy out and
its queueing of each copy back added up; host clocks only, no GPU time),
`staged_overlap_setting`, and the observation emitter's counts. A remote key is never kept
where telemetry reads, and the tests check no key-like name appears.

## Transport observations

`GPUBRIDGE_VERBS_OBSERVE=<file>` (`{rank}` in it becomes the global rank) makes
each leader append `straggler.transport_observation/1` records, the
straggler-detector's NIC collector input (its `docs/NIC_TELEMETRY.md`), one JSON
object per line:

- `schema`, `t_wall`, `t_mono`, `rank`, `peer` (global ranks), `bridge`
  (`verbs`), `op` (`send` or `recv`), `bytes`, `duration_s` (first post to last
  completion), `status` (`ok`, `error`, `timeout`, `cancelled`), `error`;
- top-level `host` (this node's hostname) and `job` (`GPUBRIDGE_JOB_ID`, else
  `SLURM_JOB_ID` or `TORCHELASTIC_RUN_ID`, else an id the members agree on through
  the c10d store);
- `nic`: `device`, `port`, `gid_index`, `link_layer` (`InfiniBand` or
  `Ethernet`), `rate_gbps` and `pci_bus_id` (from `/sys/class/infiniband`, null
  where absent, as on the test fabric); `peer_nic` from the peer's handshake;
- `path` (`direct`, `direct+bounce` or `staged`) for GPU tensors, and on a
  `direct+bounce` call's receive records `bounced_bytes`; `step` when the caller sets
  `transport.step`; `collective` (the call's sequence number; unknown fields are
  ignored by the detector).

MOCK/test-fabric records always carry `synthetic: true`. When an operation's own
work completion fails, its record has `status: error` and `wc_status` using the
stable libibverbs spelling (for example `IBV_WC_RETRY_EXC_ERR`). A fault that no
single operation caused is the `error` of every operation it reaches, too:
- the device's or a port's (`DEVICE_FATAL`, `PORT_ERR`, `CQ_ERR`, a failed poll,
  arm or event wait): every unfinished operation;
- one peer's: an asynchronous error on its queue pair (`QP_FATAL`, `QP_REQ_ERR`,
  `QP_ACCESS_ERR`, `PATH_MIG_ERR`), a protocol error by it, a failed repost or
  post to it, its leaving, or a failed receive or control message completion on
  its queue pair: that peer's unfinished operations. A failed receive or control
  completion gives them its `wc_status`, unless it was only flushed; the others
  have none.

Other unfinished operations cancelled by such a failure (an operation's own, or
a fault with another peer that then fails the endpoint) or by a local abort or
close have `status: cancelled`; operations already completed remain `ok`. So a
call that failed for a fault with a peer it moved nothing with has only
`cancelled` records, with the fault's text as their `error`. A native timeout
return code (never an error-message substring) marks outstanding operations
`timeout`. Aggregating a peer/direction gives an error precedence over cancellation,
and cancellation over success.

`bytes` is the requested total for a non-ok record, including a failed post; it is
not a measurement of moved traffic. Successful records report completed payload
bytes. **Pre-transfer refusals**, including forced-direct rejection in `_plan`,
are exceptions/telemetry diagnostics and produce no transport observation: no
operation was attempted. They must not be inferred as successful transfers or
as zero-traffic hardware evidence.

**Rate:** one record per collective call, per peer and direction that moved
anything; a failed call gives one per peer and direction with its status.
Nothing per chunk or completion. **Off the critical path:** the call appends
dicts to a bounded queue (`GPUBRIDGE_VERBS_OBSERVE_QUEUE`, 10000 records); a
background thread writes; a full queue or a write error drops records and counts
them (`observations.dropped`), never failing a call. No record carries a key or a
raw GID; any key-like field name at any depth is refused before queueing.

## The Python bridge

The shared async worker's submission queue is bounded to 1024 waiting collectives
per rank, plus the running call (`GPUBRIDGE_ASYNC_QUEUE`, a positive integer read
when the worker starts). Overflow raises before enqueueing the new call; it does
not discard accepted work. Wait for accepted work before retrying, preserving the
same collective order on every rank. The limit bounds queue entries/tensor
references, not tensor bytes; users must budget outstanding tensor memory too.

`bridge="verbs"` (`src/gpubridge/verbs/transport.py`): by name, or since
2026-11-11 what the default `bridge="auto"` resolves to when every island leader
passes `gpubridge.verbs.availability()` (the library loads or builds, an
endpoint opens on this process's `GPUBRIDGE_VERBS_*` settings: a device, an
active port, an address; the MOCK test fabric never counts) and all leaders are
on one link layer. Otherwise `auto` is the TCP bridge. A job on verbs creates no Gloo group (torch would add Gloo helper groups to the
islands at its debug level DETAIL, so `init()` refuses that level unless the job
sets `allow_debug_gloo` / `GPUBRIDGE_ALLOW_DEBUG_GLOO=1`: RUNTIME_INVENTORY.md,
section 6). `device_tensors` is True: GPU
leaders hand GPU tensors straight to it, and it moves them direct or staged
(["GPU memory"](#gpu-memory)). Its collectives are
`gpubridge.tagged.TaggedBridge`'s, shared with `bridge="multitcp"`: with two
leaders a chunked exchange, with more a ring, so every element is reduced in one
fixed order. The worker under them is a router (`verbs/gpu.py`) that sends each
chunk from, and receives it into, whatever memory it lies in: registered device
memory, the bridge's registered host buffers (staging and scratch: host tensors'
reductions receive into the registered scratch by write grant), or plain memory
copied through the slots. Chunks of `GPUBRIDGE_VERBS_CHUNK_BYTES` (4 MiB).

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUBRIDGE_VERBS_RNDV_BYTES` | 256K | messages of at least this size go by rendezvous; `0`: off (and with it receives into GPU memory). The same on every leader |
| `GPUBRIDGE_VERBS_PROGRESS` | `economy` | native progress policy: `latency`, `balanced`, `economy`; local to each endpoint |
| `GPUBRIDGE_VERBS_PROGRESS_CPU` | unset | optional latency-thread CPU, inside the inherited allowed set; Linux `ibverbs` only |
| `GPUBRIDGE_VERBS_GPU_PATH` | `auto` | `auto`, `direct` (fail instead of staging) or `staged` |
| `GPUBRIDGE_VERBS_STAGED_OVERLAP` | `0` | `1`: a staged two-leader all_reduce overlaps its copies with its messages, chunk by chunk ("GPU memory"); local to each leader |
| `GPUBRIDGE_VERBS_DMABUF`, `GPUBRIDGE_VERBS_PEER_MEMORY` | `1` | `0` keeps the direct path from trying that method |
| `GPUBRIDGE_VERBS_DEVICE_CACHE_BYTES` | 8G | bound on GPU memory registered at once |
| `GPUBRIDGE_VERBS_OBSERVE`, `_OBSERVE_QUEUE`, `GPUBRIDGE_JOB_ID` | off, 10000 | transport observations |
| `GPUBRIDGE_VERBS_SLOT_BYTES`, `_CREDITS`, `_SEND_SLOTS`, `_UNEXPECTED_BYTES` | the library's | the protocol's settings |

The library is built like multitcp's: `python -m gpubridge.verbs build`
(Linux, the `ibverbs` feature, rdma-core's headers; `GPUBRIDGE_VERBS_LIB` for a
library built elsewhere), and `python -m gpubridge.verbs info` loads it without
opening a device. If it can't be built or loaded, or there is no usable device
or port, setup fails on every rank with the reason. Once verbs is the job's
bridge, by name or by `auto`'s choice at `init()`, payload never moves any
other way. (`auto` decides before any bridge exists; it never switches after.)

## Hardware kit

Partially run on single-host loopback: staged-GPU collectives passed at
`c92684b`, and repaired host collectives passed at `0cf32a3`; see the
[2026-10-20 results](../HARDWARE_VALIDATION.md#split-test-results-2026-10-20).
Host-mode TCP evidence was `not-decisive`, with strict host mode failing closed,
until per-process attribution; at `4ffcf93` strict host collectives passed
decisively on one host. The two-host/cross-node, direct-GPU, mixed-vendor and UCX-comparison
portions remain unrun. The full planned matrix follows
(`scripts/gpu/verbs_kit.py`; runbook: `VERBS_BRIDGE=1` in steps 05 and 07;
HARDWARE_VALIDATION.md items 38 to 42). Same-vendor (step 05) and cross-vendor
(step 07) runs are separate records:

1. `verbs_build`, `verbs_rdma_report` (each node's dma-buf options, peer-memory
   modules, `ibv_reg_dmabuf_mr` in its libibverbs, ROCr's dma-buf export);
2. `verbs_endpoint`: host-to-host verbs between one rank per node: latency,
   bandwidth sweep, SHA-256 of every buffer on both sides, CPU time;
3. `verbs_host`, `verbs_gpu_staged`, `verbs_gpu_direct`: the collectives on host
   tensors (the kit sets `GPUBRIDGE_CPU_ONLY=1` before importing torch or gpubridge,
   using CPU/Gloo islands and the verbs bridge between leaders; not NCCL island coverage), then GPU tensors
   staged and direct, exactly against Gloo, each call's
   path, reason and bounced bytes recorded and the path required. `direct` means
   nothing bounced (`direct+bounce` fails the step), so `verbs_gpu_direct` runs
   with `GPUBRIDGE_VERBS_RNDV_BYTES=1`: every message by rendezvous, written in
   place;
4. `check_verbs`, `bench_verbs`: every check stage and the bridge-phase benchmark
   over `bridge="verbs"`, the Gloo and UCX bridges' runs in the same job as the
   matched references;
5. `train_verbs`, `train_gloo`: `train_bench.py` over each.

Host collective records include `cpu_only: true` and `island_backend: "gloo"`.
The kit refuses inherited CPU-only settings that contradict `--memory` (including
`--memory gpu` with `GPUBRIDGE_CPU_ONLY=1`).
An inherited `GPUBRIDGE_VERBS_GPU_PATH=staged` is preserved and accepted in host
mode; CPU tensors still use the verbs host path.

Every `verbs_kit` record holds the endpoint telemetry, completion errors, a window
of every `/sys/class/infiniband/*/ports/*/{counters,hw_counters}` file and every
interface's `/proc/net/dev` bytes, and `tcp_evidence`. Port and netdev counters are
context only: a port is shared with every process and job on the node, and a RoCE
netdev carries traffic that is not TCP, so they can't show which transport carried
this job's payload. `tcp_evidence` is `pass` only with positive evidence attributable
to the process: the bridge's own telemetry over the window (provider `libibverbs`, not
MOCK; its sent plus received bytes at least the payload; completions from its own
queue pairs; no completion errors), and the process's own counted TCP sockets
(`ss -tinp`, by pid: Gloo's and the store's) moving no more than 5% of the payload
plus 1 MiB. Its RC queue pairs as the kernel lists them (`rdma resource show qp`, by
pid) are recorded where the `rdma` tool exists, and contradict a pass when it lists
none. Without `ss` the verdict is `unknown`, never a pass; Gloo reference collectives
run before the window. The TCP check excludes loopback unless host mode attributes
it (below). Only host mode with CPU-only Gloo islands also excludes peers whose
addresses appear as local socket endpoints in the same `ss` snapshot. IPv4, IPv4-mapped IPv6 and native IPv6 addresses
are normalized; unknown addresses still count. GPU mode with NCCL islands counts
same-host non-loopback TCP, preserving its existing acceptance check.

Host mode attributes those loopback and same-host sockets by process: both ends of a
same-host connection are in one `ss` snapshot, each with its owning pids. Every rank
gathers its hostname, kernel boot ID, pid and pid namespace; when every other rank of
a leader's island is positively identified as sharing its host and pid namespace (or
as remote), the leader excludes only sockets whose other end is owned by those island
peers alone (`island`). Sockets to any other process (`outside island`: the other
leader, the other island, torchrun's store) count, loopback included; sockets whose
other end has no owner in the snapshot, or a mix, are `unattributed`. Each record's
`tcp_evidence.local_tcp_attribution` lists the island pids, every socket's class,
owners and bytes, and the bytes per class (or why attribution was unavailable).
Host-mode ranks take their `ss` snapshots one at a time, the others waiting at a
barrier, and none tears down before every rank's closing snapshot: on one host a
peer's teardown closes, and its forked `ss` briefly holds, sockets another rank reads.

Leader co-residence is determined from gathered actual hostnames and kernel boot
IDs, independently of split-test labels. When local TCP is excluded and any bridge
peer is co-resident or cannot be positively identified as remote, otherwise positive
evidence is `not-decisive`, with `no_silent_tcp: null` and the reason
"local bridge: island Gloo traffic and a bridge TCP fallback are indistinguishable
on one host", unless attribution is available: then the counted sockets decide, and
the evidence stays `not-decisive` when the unattributed bytes, counted too, would
exceed the limit (the reason then says how many bytes), or when a leader with island
peers had none of its TCP attributed to them (its island's Gloo traffic unseen). The
positive verbs checks remain required and are reported separately in
`verbs_evidence`; any contradictory evidence still yields `fail`. Without
`--require-no-tcp`, a correct local host run may succeed while printing and recording
this limitation. With `--require-no-tcp`, its leaders fail with that reason.
Cross-node host runs remain decisive: local island TCP is excluded, remote TCP is
counted. A one-host host run, even a decisive one, cannot establish cross-node no-TCP
acceptance. Socket attribution has been checked on hardware on one host only: strict host
collectives passed with island TCP attributed, and real same-host TCP was counted and
rejected.
`--require-no-tcp` fails a bridge leader's step unless its decisive
verdict is `pass`; other ranks record `not applicable`. A job with no bridge leader
at all (one island: an all-NVIDIA node without `GPUBRIDGE_SPLIT_TEST`, say) never
runs the bridge, so `--require-no-tcp` fails every rank of it (each record says
`leaders: 0`).

## Testing

Rust, on the in-process mock (`tests/protocol.rs`): every message size around
the chunk boundary and 64 KiB, both directions and many tags at once, ordering
within a tag, a 50 KB message through 2 credits (stalls and credit messages),
unexpected messages buffered to the limit and then held, a receive of the wrong
size, the bound on posted operations, deadlines, a message arriving after the
receive's deadline and abort (it never lands), a completion surfacing after an
abort, a lost peer, a failed completion, a fatal device event, a port going
down, closing with sends and receives outstanding (everything released, no
violation), goodbye semantics, three members, mismatched versions, settings and
link layers, the path MTU, and the selection rules. With a bare queue pair as a
member (`tests/common`): a peer breaking the protocol while another member's plain
send and receive are part-way, after which nothing more is copied though credits
come back and chunks keep arriving; sends queued to a member that said goodbye
failing at once with its reason; and fork safety at open for each status and a
failing `ibv_fork_init`, before any registration.

Rust, registered memory on the in-process mock with a mock GPU (`tests/direct.rs`):
dma-buf chosen first, peer memory when dma-buf is unsupported or its export
fails, and both reasons when neither works or a setting disables one; the
capabilities report; a dma-buf descriptor closed exactly once, only after its
region is deregistered (the mock GPU records a close while imported as a
violation), and at once when registering fails; SYNC_MEMOPS once per new
registration, not on cache hits; an address reallocated with a new buffer ID
getting a new registration and the stale one dropped (refused while still in
use); device memory read in place by sends and written by grant, with messages
below the rendezvous size refused into it; every pairing of plain, registered
host and device memory at a size spanning several slots; a waiting message
holding up only its own tag; a grant of the wrong size failing both sides;
receives into device memory refused with rendezvous off; aborting with a write
granted keeping the window and its registration until close; cached
registrations local-only; a bare queue pair granted a write into host, dma-buf and
peer memory failing to write the byte before or after the receive and, once its
"written" message is in, anything at all (the window deregistered), the
neighbours untouched; and telemetry with no key-like name. `gpu.rs`'s own tests: the peer-memory module found in sysfs or
`/proc/modules` fixtures, a missing driver an `Unsupported` error, a mock
reallocation getting a new buffer ID.

**The test fabric** (`uds.rs`, feature `test-fabric`): a MOCK provider whose
nodes are processes on one machine, carrying RC queue pairs over Unix sockets
in a directory. It emulates RC's ordering, acknowledgements and failures
(a send to a missing, unconnected or mismatched queue pair, or to a node that
went away, completes with `RETRY_EXC_ERR`), and RDMA WRITE in queue pair order
where the target queue pair and region allow it (`REM_ACCESS_ERR` otherwise). It exists only in a separate test
build of the library (`gpubridge.verbs.build(test_fabric=True)`, without
libibverbs) that only `GPUBRIDGE_VERBS_TEST_FABRIC=<directory>` loads; the
production library can't select it, and telemetry and a warning say MOCK.
`tests/test_verbs.py` runs every bridge policy on two, three and four leaders
with every dtype and reduction, bridge compression, large tensors in many
chunks, subgroups through `backend="gpubridge"`, async collectives, empty and
bad tensors, a receiver after the sender closed, a bad setting and mismatched
settings failing init everywhere, a deadline, and teardown with a receive
outstanding, on it, bit for bit against `flat-gloo`; and that without it
`bridge="verbs"` fails init with the reason. These are software tests of the
code above the provider, not of any fabric. With A3 they run with rendezvous on
(256 KiB), so large tensors go by write grant into the registered scratch.

`tests/test_verbs_gpu.py` (no GPU): the router's ordering against a recording
fake endpoint (one stream wait before the first device message and again after
each wait, the flush after a wait that completed writes into device memory, a
small receive bounced and copied only once waited for, registered host memory
sent in place, plain memory copied); the direct-or-staged choice against a fake
endpoint, each failing check giving staged with its reason (rendezvous off, no
driver, no method, no flush, a refused registration, a refused scratch, which
gives the first registration back), `GPUBRIDGE_VERBS_GPU_PATH=direct` failing
with the reason, an empty tensor, driver checks once per GPU; the staged path end
to end on the test fabric with CPU tensors standing in and lazy copies (a copy
runs only when waited for, so reading the staging buffer early would move zeros):
all_reduce and broadcast exact, every copy out waited for before the bridge used
the buffer, the buffer registered and written by grant; and the observations
those runs wrote, checked field by field against the detector's schema, plus
key-like names refused at any depth, the NIC identity from sysfs fixtures, the
job id from the environment or the store, and an emitter that can't write
dropping and counting. `tests/test_verbs_staged_overlap.py`: with
`GPUBRIDGE_VERBS_STAGED_OVERLAP` off (unset or `0`) the staged call step for step as
before; on, against a fake wire that plays the peer, each chunk's copy out waited
for before its send is posted (the send sees the copied bytes), each copy back
queued only after its chunk is reduced, the first copy out after the last copy back
on another stream, a bigger tensor's buffer replaced only after it, the same
messages (buffers, offsets, sizes, tags, per-direction order) and results as off for
sum, max and min from one element to 3.5 chunks, broadcasts, three leaders and one
chunk staged whole with the reason, a failed wait part way (the last copy back
queued stays the one waited for, the chunks before it copied back, the rest
untouched, later calls refused), every copy of a failed call passed when it raises
(a caller reusing the tensor at once keeps what it wrote), a copy out that can't be
queued (the worker aborted, the copies queued before it drained, later calls on
another stream refused), a wait for those copies that fails too (the call's own
error raised, the wait's chained, the buffer kept, later calls refused) and a failed
post (no copy back), and the phases
splitting the call; on the test fabric, exact with it on, off and on one leader
only, under every progress policy, with equal wire counts and observations.

`shim/gpu_abi_check.c` (compile-only): every constant and signature `gpu.rs` takes
from cuda.h, HIP's runtime API and ROCr's headers.

## Building on a node without rdma-core's headers

Explorer's login node has rdma-core 57.0's runtime (`/usr/lib64/libibverbs.so.1`
and its providers) but not its `-devel` headers. The headers of the same
release, from rdma-core's own tag, give the same ABI:

```bash
git clone --depth 1 --branch v57.0 https://github.com/linux-rdma/rdma-core
mkdir -p inc/infiniband inc/rdma
cp rdma-core/libibverbs/verbs.h rdma-core/libibverbs/verbs_api.h \
   rdma-core/libibverbs/tm_types.h \
   rdma-core/kernel-headers/rdma/ib_user_ioctl_verbs.h inc/infiniband/
cp rdma-core/kernel-headers/rdma/*.h inc/rdma/
GPUBRIDGE_VERBS_INCLUDE_DIR=$PWD/inc cargo test --features ibverbs \
    --manifest-path src/gpubridge/verbs/crate/Cargo.toml
```

The tests that run there use the mock provider. The libibverbs provider's
own tests that need no device (layouts, status strings) run too; the one that
opens a device is ignored, and belongs on an allocated node:
`GPUBRIDGE_VERBS_DEVICE=mlx5_0 cargo test --features ibverbs -- --ignored`.

## Results so far

- **A3, macOS (mock providers, mock GPU, CPU stand-ins):** `cargo fmt --check` and
  `cargo clippy --all-targets -D warnings` with and without `test-fabric` are clean;
  `cargo test` passes 68 tests, 71 with `test-fabric`. `tests/test_verbs.py`: 34
  passed (now with rendezvous on); `tests/test_verbs_gpu.py`: 9 passed; the
  multitcp bridge's tests and the pinned-staging tests still pass after the shared
  `TaggedBridge` gained its scratch hook (61); `tests/test_gpu_kit.py`: everything
  but the two Triton-cache tests that fail on this Mac on every commit. The kit's
  `verbs_kit.py` ran in both modes on the MOCK test fabric (its `tcp_evidence`
  correctly reports no RDMA counters there). `gpu_abi_check.c` against ROCr's
  rocm-6.3.0 headers: clean.
- **A3, Linux build (2026-10-18), explorer-02:** at commit a9e61ab, rdma-core v57.0
  headers and the node's 57.0 runtime, debug builds at `cargo -j 1`: `cargo test
  --features ibverbs` 72 passed (the device test ignored; its first compile was
  SIGKILLed, the one retry passed), `--features test-fabric` 71; the shim has no
  warnings; `gpu_abi_check.c` is clean against CUDA 12.8's cuda.h and HIP 6.3's
  headers. The mock-only test build was SIGKILLed too (the second kill), so the
  rest was not run: **clippy on Linux and the library's readelf check are
  BLOCKED (login-node limits)**. Build evidence, not a transfer: nothing opened the
  node's device or loaded a GPU driver.
- **A2, macOS (mock providers):** `cargo fmt --check` and `cargo clippy
  --all-targets -D warnings` with and without `test-fabric` are clean; `cargo
  test --features test-fabric` passes 57 tests (55 without the feature).
  `tests/test_verbs.py`: 34 passed (every policy on two to four leaders,
  compression, large tensors, subgroups, async, failures and teardown on the
  test fabric; init failing with the reason without it). The multitcp bridge's
  tests still pass after its collectives moved to `TaggedBridge` (41).
- **A2, Linux build (2026-10-18), explorer-01:** at commit bf80da8, against
  rdma-core v57.0 headers and the node's rdma-core 57.0 runtime, debug builds at
  `cargo -j 1`: `cargo test --features ibverbs` 59 passed (the device test
  ignored), `--features test-fabric` 57, mock only 55 (its first compile was
  SIGKILLed by the login node's limits; the one retry passed); clippy clean with
  either feature; the shim has no warnings. The library needs `libibverbs.so.1`
  (`IBVERBS_1.0` to `1.12`) and resolves it to the system library; the
  completion channel and event calls are linked, the post and poll calls still
  inline. Build evidence, not a transfer: nothing opened the node's device.
- **A1:** the same checks at A1's commit (33 mock tests; release build on
  explorer-01).
- **CI:** the `verbs crate` job also lints and tests the `test-fabric` feature;
  the pytest jobs run `tests/test_verbs.py` where cargo is installed (not run
  yet with this change).
- Nothing has run over a software RDMA provider, RDMA hardware or a GPU: not
  validated on RDMA hardware or GPUs. That needs allocated GPU nodes with usable
  RDMA devices on one fabric and the prerequisites of HARDWARE_VALIDATION.md items
  38 to 42.

## Open risks

- **Untested on hardware**, all of it: real `poll(2)` on the channel descriptors,
  retry/RNR settings, RoCE GID choice, `IBVERBS_1.12`'s `ibv_reg_dmabuf_mr`, RDMA
  WRITE and its access checks on a real NIC, every GPU driver call, dma-buf import by
  the NIC's driver, and whether the flush rules are enough on each GPU.
- **Remote write exposure that remains.** With rendezvous on, the queue pairs
  allow remote write, and a granted peer can write any bytes into the receive's
  window (exactly the receive's range) until its "written" message is polled; a
  wrong grant of the right length bound by tag (a peer bug) still lands there. If
  the receive fails, that peer can keep writing its window until its queue pair
  is in the error state (the Python bridge aborts, which does that at once; a Rust
  or C caller that doesn't must keep the memory until `abort` or `close`). No
  cached or send-side registration is remotely writable. The bridge trusts its
  peers, as RDMA jobs do.
- **A registration per rendezvous receive.** Each window is an `ibv_reg_mr` (for
  dma-buf memory, an export and `ibv_reg_dmabuf_mr` too) and a deregistration:
  tens of microseconds or more per receive of at least `rndv_bytes`, not measured.
  Type-2 memory windows bound per grant would be cheaper and are not used.
- **`ibv_fork_init`'s cost.** Where the kernel isn't fork-safe, `ibv_fork_init`
  marks every registered range's pages `MADV_DONTFORK`: a forked child (a
  DataLoader worker) that touches other data on those pages faults. Kernels from
  5.13 don't need it, and then it isn't called.
- **Freed GPU memory under a cached registration.** The analysis that this is safe
  rests on dma-buf imports pinning the exported memory and on the peer-memory
  module revoking registrations; both are drivers' documented behaviour, not
  something tested here. The cache also keeps freed segments pinned until evicted
  or found stale (bounded by `GPUBRIDGE_VERBS_DEVICE_CACHE_BYTES`).
- **AMD ordering.** The HDP flush (or `hipDeviceFlushGPUDirectRDMAWrites`) is what
  the UCX bridge uses; HIP 6.3 reports no RDMA ordering attributes, so whether a
  kernel reads the NIC's writes without stale data is a hardware question (item 41).
- **Throughput unknown.** One completion thread and one CQ per endpoint, the CPU
  copy for plain host tensors (staged GPU and registered buffers avoid it), a
  rendezvous round trip per large message, a stream wait per step on the direct
  path; peer loss is still noticed only when sending; messages buffered then held
  can stall pathological orderings until the deadline (A2's risks stand).
- **Linux clippy** for this phase is BLOCKED (login-node limits), so lints of the
  `ibverbs` cfg are checked only by A2's run and the CI job. Of the fork-safety
  calls added after the review, the shim's compiled cleanly with gcc against
  rdma-core v57.0 on an Explorer login node and reported that node fork-safe
  (`unneeded`, kernel 5.14); `ibv.rs`'s were not compiled there (rustc was killed
  twice: BLOCKED, login-node limits).
