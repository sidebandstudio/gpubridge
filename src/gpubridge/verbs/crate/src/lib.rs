//! gpubridge's raw-verbs bridge (docs/VERBS_TRANSPORT.md): owned libibverbs resources over
//! a [`Provider`], the queue pair state machine, work tracking that keeps memory registered
//! while the device can use it, a bounded registration cache, the bridge's RC wire protocol
//! (version 2, with an RDMA WRITE rendezvous) and its endpoint, GPU memory registration,
//! the C interface the Python bridge loads, and deterministic mock providers.
//!
//! - [`Context`], [`ProtectionDomain`], [`CompletionQueue`], [`QueuePair`] and
//!   [`MemoryRegion`] own their verbs resources and release them in a correct order
//!   whatever order they are dropped in (`resources.rs` says how).
//! - [`MrCache`] caches registrations by allocation identity, within fixed bounds.
//! - [`MockFabric`] / [`MockProvider`]: an in-process fabric with injectable faults.
//! - [`IbvProvider`] (feature `ibverbs`, Linux): libibverbs through a C shim compiled
//!   against rdma-core's headers.
//! - [`Endpoint`] (`endpoint.rs`, [`protocol`]): one member's end of the bridge, an RC
//!   queue pair to every other member, credits, a completion thread, and the
//!   rendezvous; the `gb_verbs_*` C functions (`ffi.rs`) are what `gpubridge.verbs`
//!   calls.
//! - [`Registrar`] (`direct.rs`, [`gpu`]): device memory through dma-buf or peer memory,
//!   host buffers, and each rendezvous receive's write window.
//!
//! # Threads never touch Python
//!
//! The rule, for every thread this crate starts (the endpoint's completion thread,
//! `gpubridge-verbs`, and in the test build the MOCK fabric's `verbs-test-fabric-*`
//! threads): **it never calls into Python and never holds, reads or releases a Python
//! object.** A native thread that needs the interpreter's lock (the GIL) while the
//! interpreter finalizes is ended by CPython 3.12 in the middle of its work, which is how
//! a Gloo worker thread aborts a process at exit; no thread here can ever need the GIL:
//!
//! - the interface takes no callback from Python: no `gb_verbs_*` function has a
//!   function-pointer parameter. (The function pointers in `gpu.rs` are the CUDA and
//!   ROCm driver entry points this library looks up itself.)
//! - the crate depends on the standard library only and links no Python library;
//! - what a thread sees of the caller is a posted operation's raw memory (an address
//!   and a length, never a tensor or any other object).
//!
//! `gb_verbs_close` stops and joins the completion thread before it returns. gpubridge
//! closes the bridge in `destroy()`, and from an exit handler for a job that never called
//! it. `tests/test_native_threads.py` checks the rule against the source and the built
//! library.
//!
//! Nothing here has run on RDMA hardware or a GPU.

mod cache;
mod direct;
mod endpoint;
mod error;
mod ffi;
pub mod gpu;
#[cfg(feature = "ibverbs")]
mod ibv;
mod mock;
mod progress;
pub mod protocol;
mod provider;
mod resources;
mod types;
#[cfg(feature = "test-fabric")]
mod uds;

pub use cache::{CacheConfig, CacheKey, CacheStats, Invalidated, MrCache};
pub use direct::{
    DirectConfig, Method, Registered, Registrar, Window, CACHED_ACCESS, DEVICE_ROUND, WINDOW_ACCESS,
};
pub use endpoint::{Config, Counters, Endpoint, WaitError, TICK, WRITE_CHUNK};
pub use error::{Error, Result};
pub use gpu::{Allocation, GpuMemory, MockGpu, PeerModules, Vendor};
#[cfg(feature = "ibverbs")]
pub use ibv::{check_layouts, library_path, wc_status_text, IbvProvider};
pub use mock::{
    Call, Fault, LiveCounts, MockDevice, MockFabric, MockLimits, MockPort, MockProvider, NodeId,
};
pub use progress::Progress;
pub use provider::{BatchResult, Handle, MrKeys, Provider, QpCreated};
pub use resources::{
    CompletionChannel, CompletionQueue, Context, DmabufCloser, DmabufFd, MemoryRegion,
    ProtectionDomain, QueuePair, SendRequest, Sge, StatsSnapshot, DEFAULT_DRAIN_TIMEOUT,
};
pub use types::{
    Access, Address, AsyncEvent, AsyncEventKind, Completion, DeviceAttr, Events, ForkStatus, Gid,
    GidEntry, GidType, GlobalRoute, InitAttr, LinkLayer, Mtu, PortAttr, PortState, QpCaps, QpState,
    RawCompletion, RawRecv, RawSend, RawSge, RemoteTarget, RtrAttr, RtsAttr, SendOp, Transition,
    WcStatus, WorkId, WorkKind, MAX_SGE,
};
#[cfg(feature = "test-fabric")]
pub use uds::UdsProvider;

use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// What this build is, for the Python side to check before it uses the library:
/// `"gpubridge-verbs <version> providers=mock[,libibverbs][,test-fabric]"`, NUL-terminated
/// into `out`.
/// Returns 0, or -1 if `out` is too small or null. Opens nothing.
///
/// # Safety
///
/// `out` must be null or point to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_build_info(out: *mut c_char, len: usize) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // The layout check fails only if the shim was built from another source.
        #[cfg(feature = "ibverbs")]
        let providers = if check_layouts().is_ok() {
            "mock,libibverbs"
        } else {
            "mock"
        };
        #[cfg(not(feature = "ibverbs"))]
        let providers = "mock";
        #[cfg(feature = "test-fabric")]
        let providers = format!("{providers},test-fabric");
        let text = format!(
            "gpubridge-verbs {} providers={providers}",
            env!("CARGO_PKG_VERSION")
        );
        if out.is_null() || text.len() + 1 > len {
            return -1;
        }
        // SAFETY: out has len writable bytes (the caller's contract) and text.len() + 1
        // of them are written.
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), out.cast::<u8>(), text.len());
            *out.add(text.len()) = 0;
        }
        0
    }))
    .unwrap_or(-1)
}
