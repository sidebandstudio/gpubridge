# A multi-stream TCP bridge transport in Rust

**Status: `bridge="tcp"` (`"multitcp"` is its older name), and since 2026-11-11
what the default, `bridge="auto"`, runs on wherever the island leaders don't all
have a usable RDMA device.** Tested on CPU (loopback TCP between processes),
where it reproduces the Gloo bridge's results bit for bit, and compared with the
Gloo bridge in two step-08 jobs between a V100 node and an MI50 node, where it
ran at about the Gloo bridge's speed and so missed the gate it was first judged
by, which asked for a speed-up ([below](#stage-1-on-hardware-and-the-gate)). As
the default, with gpubridge's own default process group, it has not run on a GPU.
The Gloo bridge is opt-in (`bridge="gloo"`), and stays the reference in tests.

## Why it is the default

Not for speed. On Python 3.12 with torch 2.10 to 2.15 a Gloo worker thread can
abort a process at interpreter exit, after training has finished
([RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md#6-the-gloo-decisions-reversed-2026-11-11)).
A job on this bridge creates no Gloo group: the bridge is not a process group,
its rendezvous is the c10d store, and its threads never touch a Python object
(["Threading"](#design)). What it had to show to take Gloo's place is the same
results and no worse speed, not the 20% the first gate asked:

- the same bits as the Gloo bridge: `tests/test_parity_fixture.py` (SHA-256 hashes
  generated once by the Gloo bridge for every dtype, SUM, MAX, MIN and AVG, two to
  four leaders, and sizes on each side of the 4 MiB chunk; this bridge reproduces
  them in jobs with no Gloo group), with the one difference the fixture documents:
  a floating-point SUM over three or more leaders that rounds is added in ring
  order, not Gloo's order;
- 0.88 to 1.07 times the pinned Gloo bridge's time from 64 MB to 1 GB between two
  machines (the two step-08 jobs below);
- no rank exiting with a signal in 300 repetitions of a job that returns straight
  into interpreter exit, where the same job on the Gloo bridge does
  (`scripts/exit_abort.py`; the CHANGELOG has the counts).

## Why

On Explorer, between a V100 node and an MI50 node, the bridge crosses one TCP
path of about 1.17 GB/s (one stream, measured by the preflight). A 1 GB float32
all_reduce took 1.67 s through the default Gloo bridge, 1.21 s with pinned
staging and 1.07 s pipelined in 16 MB chunks
([BENCHMARKS.md](BENCHMARKS.md#the-all_reduce-matrix-a-v100-and-an-mi50)).
Moving 1 GB each way at the measured rate takes about 0.9 s, so roughly
0.2 to 0.8 s of each of those is something other than the wire: Gloo's one
connection per pair, and its transfer and reduction running one after the
other. This transport attacks both, on an ordinary network first. RDMA would
be stage 2, but Explorer has no RDMA path between those two node types
([BENCHMARKS.md](BENCHMARKS.md#rdma-between-the-node-types)), so stage 2 isn't
started.

## Design

**Shape.** A Rust crate, `gpubridge-transport`
([`src/gpubridge/multitcp/crate`](../src/gpubridge/multitcp/crate)), with a C
interface (`gb_tcp_*`), loaded from Python with ctypes like the UCX transport.
It uses the standard library only, so it builds offline on any node with a
Rust toolchain (Explorer's login node has cargo 1.88 from the OS packages;
the crate builds and passes its tests with 1.70 and 1.98). `python -m gpubridge.multitcp build` compiles it
into `~/.cache/gpubridge` (`GPUBRIDGE_MULTITCP_LIB_DIR`), keyed by the source,
the toolchain, the machine and libc, always without `RUSTFLAGS` (a cache shared
by nodes of different CPUs must not hold a `target-cpu=native` build); `load()` builds on first use;
`GPUBRIDGE_MULTITCP_LIB` takes a library built elsewhere, for nodes without
cargo.

**Split of work.** The Rust side moves tagged byte ranges between bridge
members and nothing else. The collectives (an exchange for two leaders, a ring
for more, a fan-out for broadcast) and every reduction stay in Python with
torch ops, exactly as in the UCX transport, so the results are the same bits
as Gloo's with the same code paths that already pass every policy against
`flat-gloo`.

**Connections.** Each member listens on one port of its chosen address
(`GPUBRIDGE_MULTITCP_IFNAME`, else `GLOO_SOCKET_IFNAME`, else the address this
machine reaches `MASTER_ADDR` from) and publishes `protocol|chunk|streams|host:port`
in the job's c10d store. For each pair, the member with the lower index dials
`GPUBRIDGE_MULTITCP_STREAMS` connections (4 by default), each opened with a
hello naming the sender and the connection's number; an accepting thread on
the other side files them. Leaders whose protocol, chunk size or stream count
differ fail `init()` on every rank, and a one-message handshake each way checks
that every link carries data, as the UCX transport does.

**Threading.** Every connection has a sending thread and a receiving thread
(`gb-tcp-send`, `gb-tcp-recv`), started at connect and joined by `close()`;
`gb-tcp-accept` takes the connections lower-index members dial, and ends when
it has them. Python calls in on one thread
(ctypes releases the GIL for the call), posts sends and receives, and waits on
the ids they return. **The native threads never call into Python and never
hold a Python object**, so no work depends on the GIL, and no thread here can
need it while the interpreter finalizes: the library takes no callback, links
nothing of Python's, and sees only the raw byte ranges of posted operations.
The rule is written in the crate's docs (`src/lib.rs`, "Threads never touch
Python") and checked by `tests/test_native_threads.py`. gpubridge closes the
bridge in `destroy()`, and from an exit handler for a script that never called
it, so the threads are joined before the interpreter is torn down.

**Point-to-point.** The torch backend's cross-island `send` and `recv` (and CPU
tensors on a GPU job) run over lanes of this library too
(`src/gpubridge/multitcp/p2p.py`): each matched (source, destination, tag) lane
has a two-member worker of its own with one connection, the sender dialing the
receiver, set up through the backend group's store by the two ranks of the lane
only. The common engine (`src/gpubridge/p2p.py`) stages, chunks and orders the
messages, as for the Gloo and UCX links. Tested on CPU by the whole of
`tests/test_p2p.py`; not run on GPUs.

**Messages.** A message is one frame, a 16-byte header (tag, length) then the
bytes, on one connection: sends to a peer go round-robin over its connections,
so the chunks of one tensor travel in parallel. Python splits each tensor into
chunks of `GPUBRIDGE_MULTITCP_CHUNK_BYTES` (4 MiB) and tags each one (the
collective's number, its step, the sender and the chunk). The receiving thread
matches a frame to the receive posted for that (peer, tag) and reads its bytes
straight into the destination; a frame whose receive isn't posted yet holds
back that one connection until it is. Each collective posts all its receives
before it waits on any, and tags never repeat within a collective, so a
connection held back this way is always freed by the next post.

**Buffers.** No copies in the transport: a send writes from the caller's
tensor and a receive reads into it. The caller (the transport's Python side)
keeps both alive until `wait()` says the operation finished. A send is
finished once its bytes are in the kernel's socket buffer, so the memory can
be reused. With two leaders, the reduction runs chunk by chunk: chunk *i* is
reduced as soon as it has arrived and this side's copy of it has gone out
(the send reads the same memory), while later chunks are still on the wire.

**GPU memory and stream ordering.** The transport takes host tensors only
(`device_tensors` is False). The policies stage GPU tensors to host memory
before the bridge step and back after it (pageable, or pinned with
`GPUBRIDGE_PINNED_STAGING=1`), exactly as for the Gloo bridge, so every
ordering guarantee the GPU tests enforce for the default bridge holds
unchanged: the transport never sees a device pointer or a stream.

**Failures and time limits.** Every wait has the deadline `init()` gave its
groups (30 minutes without one). A connection that ends at a frame boundary has
delivered everything it carried, so a member closing fails only what can no
longer arrive: a receive from it once all its connections have ended, a send on
a connection whose writer broke (and those queued behind it), a receive cut off
mid-frame. A leader that has finished its part of a collective and closed fails
nothing still arriving on its other connections or from other leaders. A frame of the wrong size fails the worker; a wait that runs out of time,
or any wait after a failure, raises, and the transport refuses every later call.
**Before a failed wait returns, the worker stops**: every connection is shut
down and every sending and receiving thread joined, without a time limit (shutting
a socket down wakes a thread blocked on it), so no thread can still read or write
a collective's tensors once Python sees the error and frees them. Only the
accepting thread, which never touches tensor memory, may outlive the stop's time
limit. If Python's own
code raises in the middle of a collective, the transport stops the worker the
same way (`gb_tcp_abort`) before the error propagates. An empty message never
touches its pointer (an empty tensor's is null). The accepting thread files
only the connections lower-index members dial, each number once. `close()`
stops the worker and frees it. Every C function catches Rust panics, so none
unwinds into Python.

## Tests (CPU)

- The crate's own (`cargo test`): messages of 0 bytes to 3 MB over three
  connections, matched by tag in any order; a frame that arrives before its
  receive is posted; a missing message timing out, and the frame that comes
  after it never reaching the buffer; empty messages with null pointers; a
  frame of the wrong size and a closed peer failing the wait; a member that
  closes failing only the waits on it, with three members, and frames it sent
  before closing still arriving on its other connections, and one that closed
  before the other side finished connecting seen as closed; stray and
  repeated dials not taking a real connection's place; bad arguments refused.
- `tests/test_multitcp.py`, between processes over loopback TCP: every bridge
  policy at two, three and four leaders, bit for bit against `flat-gloo`, for
  every dtype and reduction, broadcast, reduce-scatter and an async collective
  (16-byte chunks over three connections, so most tensors are several
  messages); bf16 and fp8 bridge compression; tensors of a million elements in
  64 KB chunks; subgroups (the torch.distributed backend); a bad setting or
  mismatched settings on one leader failing `init()` on every rank; a leader
  that never joins tripping the deadline, then every later call refused;
  teardown with a receive still posted, and a new bridge after it; a receiver
  that posts only after the broadcast's root has closed; a message of
  the wrong size; empty tensors, and a bad argument leaving the bridge working;
  the address and setting rules; and the library's build and load errors.
- CI: `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`, then
  the Python tests over the built library (job `multitcp bridge`).

## Stage 1 on hardware, and the gate

Step 08 with `MULTITCP_BRIDGE=1` builds the library on both nodes and adds three
runs to the comparison matrix, in the same job as the Gloo bridge's:

| Run | What |
| --- | --- |
| `D-multitcp` | the default policy over `bridge="multitcp"`, pageable staging |
| `D-multitcp-pinned` | the same with pinned staging: against `D-pinned` |
| `D-multitcp-1stream` | one connection: what the extra connections add, apart from the chunked overlap |

```bash
MULTITCP_BRIDGE=1 scripts/gpu/explorer/submit.sh 08
```

Explorer's compute nodes have no Rust toolchain (a login node has cargo 1.88), so
build the library once there and hand it to the jobs: `cargo build --release
--offline --manifest-path src/gpubridge/multitcp/crate/Cargo.toml`, then
`GPUBRIDGE_MULTITCP_LIB=<the .so>` with the command above (`multitcp_build` then only
checks that it loads). The venvs must import this branch's gpubridge. The kit's setup installs gpubridge in
editable mode from the clone that ran it, so from any other clone add
`PYTHONPATH=$PWD/src` to that command; otherwise the jobs run that clone's scripts
against the other clone's library.

**The gate, fixed before any run:** the metric is the whole float32
all_reduce's median time, `median_us` in each run's `bench.json` (gpubridge's op,
default policy), at each of step 08's sizes from 64 MB to 1 GB. Stage 1 stays only
if, in each of two step-08 jobs on its own, `D-multitcp-pinned` takes at most 0.8
times `D-pinned`'s time at every one of those sizes. Otherwise the measurements
get written up and the work stops there; it doesn't go on to stage 2 on that
basis alone. Either way the numbers go in
[BENCHMARKS.md](BENCHMARKS.md) and the CHANGELOG, measured, not predicted.

**Outcome: the gate was not met.** In the two step-08 jobs (a V100 and an MI50,
commit 23180fe of #66's branch), pinned multitcp took 1.07 and 1.02 times the
pinned Gloo bridge's time at 64 MB, 0.90 and 0.90 at 256 MB and 0.88 and 0.88 at
1 GB ([BENCHMARKS.md](BENCHMARKS.md#a-multi-stream-tcp-bridge-prototype)). Stage 2
did not start on this basis, and `import gpubridge` loads the transport only when
a job runs on it. That gate judged the bridge as a speed-up over Gloo, and as one
it failed. Since 2026-11-11 it is judged as what lets Gloo leave the job
(["Why it is the default"](#why-it-is-the-default)): for that, about Gloo's speed
is enough. Those two jobs ran it by name, beside a Gloo default group; nothing
has run on GPUs since it became the default.

## Not done

- Stage 2 (ibverbs with GPUDirect RDMA on NVIDIA and dma-buf on AMD, with
  registration caching): no RDMA path between Explorer's V100 and MI50 nodes.
- Device tensors. Taking GPU memory directly would need the stream-ordering
  work the UCX transport does (an event per step, a flush after receives), for
  a TCP path that copies through host memory anyway.
- Several leaders per pair of islands (the sharded policy's links each get
  their own transport, as with Gloo).
