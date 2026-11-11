//! Multi-stream TCP messaging for gpubridge's `bridge="tcp"` (docs/RUST_TRANSPORT.md), the
//! bridge `bridge="auto"` runs on without RDMA; `"multitcp"` is its older name.
//!
//! A [`Worker`] is one bridge member's end: a listening socket, and `streams` TCP
//! connections to every other member (the member with the lower index dials). A message
//! is a tagged byte range in the caller's memory. It travels as one frame (a 16-byte
//! header, the tag and the length, then the bytes) on one of that peer's connections,
//! chosen round-robin, so consecutive messages to one peer go out in parallel. Every
//! connection has its own sending and receiving thread; bytes go straight from the
//! sender's memory to the socket and from the socket into the receiver's memory.
//!
//! A receive is matched by (peer, tag), and must be posted before the frame can be read:
//! a frame whose receive isn't posted yet waits for it, holding back its connection, not
//! the others. Python numbers its messages so tags never repeat within a collective, and
//! posts a collective's receives before waiting on any of them.
//!
//! A wait that fails (a timeout, a broken connection, a frame of the wrong size) first
//! stops the worker: every connection is shut down and every sending and receiving thread
//! joined, so once the error reaches the caller no thread can read or write the caller's
//! memory again, and
//! the buffers of the failed collective can be freed. `gb_tcp_abort` does the same when
//! the caller fails on its own between posting and waiting.
//!
//! The C functions below are what gpubridge.multitcp loads with ctypes. Every one catches
//! panics, so a bug here can't unwind into Python. Not thread-safe across callers: one
//! Python thread drives a worker, as the bridge does.
//!
//! # Threads never touch Python
//!
//! The rule, for every thread this crate starts (`gb-tcp-accept`, and per connection
//! `gb-tcp-send` and `gb-tcp-recv`): **it never calls into Python and never holds, reads
//! or releases a Python object.** That is what lets a job on this bridge exit cleanly: a
//! native thread that needs the interpreter's lock (the GIL) while the interpreter
//! finalizes is ended by CPython 3.12 in the middle of its work, which is how a Gloo
//! worker thread aborts a process at exit. No thread here can ever need the GIL:
//!
//! - the interface takes no callback: no `gb_tcp_*` function has a function-pointer
//!   parameter, so there is nothing of Python's for a thread to call;
//! - the crate depends on the standard library only and links no Python library;
//! - what a thread sees of the caller is the raw byte range of a posted operation (a
//!   pointer and a length, never a tensor or any other object), and it reads or writes
//!   that range only between the post and the wait that reports the operation finished
//!   or failed.
//!
//! `gb_tcp_close` and `gb_tcp_abort` join every sending and receiving thread before they
//! return (`Worker::stop`). gpubridge closes the bridge in `destroy()`, and from an exit
//! handler for a job that never called it, so those threads are gone before the
//! interpreter finalizes. `tests/test_native_threads.py` checks the rule against the
//! source and the built library.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Wire protocol version: both ends must agree (Python also checks it at rendezvous).
pub const PROTOCOL: u32 = 1;
const MAGIC: u32 = 0x6762_7470; // "gbtp"
pub const FAILED: i64 = -1;
pub const TIMED_OUT: i64 = -2;
pub const PEER_FAILED: i64 = -3;
/// How long a failed wait waits for the worker's threads to stop.
const STOP_TIMEOUT: Duration = Duration::from_secs(10);

struct Job {
    id: i64,
    tag: u64,
    ptr: usize,
    len: usize,
}

struct Recv {
    id: i64,
    ptr: usize,
    len: usize,
}

#[derive(Default)]
struct State {
    next_id: i64,
    finished: HashSet<i64>,
    /// Posted receives, by (peer, tag).
    posted: HashMap<(usize, u64), Recv>,
    /// One queue per connection: index peer * streams + stream.
    queues: Vec<VecDeque<Job>>,
    next_stream: Vec<usize>,
    connected: Vec<bool>,
    /// Per member, its receiving threads still running. A connection that ends at a frame
    /// boundary has delivered every frame it carried, so a receive from a member can't
    /// arrive any more only once all of that member's connections have ended.
    readers_left: Vec<usize>,
    /// Per connection (as `queues`), whether its sending thread hit an error.
    writer_ended: Vec<bool>,
    /// Unfinished receives, by id: from which member.
    recv_from: HashMap<i64, usize>,
    /// Operations that can't finish any more (a send whose connection broke, a receive cut
    /// off mid-frame), by id: with which member.
    lost: HashMap<i64, usize>,
    failed: Option<String>,
    closing: bool,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A thread that panicked while holding the lock leaves the state usable for a
        // failure report, which is all that is left to do.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One of `member`'s connections stopped delivering (it closed, or the network dropped
    /// it), after `cut` (a receive it was in the middle of), if any. Only waits that can
    /// no longer be met fail; a peer that finished its part and closed must not fail
    /// operations still arriving on its other connections, or on other members'.
    fn reader_ended(&self, member: usize, cut: Option<i64>) {
        let mut state = self.lock();
        state.readers_left[member] -= 1;
        if let Some(id) = cut {
            state.recv_from.remove(&id);
            state.lost.insert(id, member);
        }
        self.changed.notify_all();
    }

    /// A connection's sending thread hit an error: its send and every one queued behind it
    /// can't finish.
    fn writer_ended(&self, queue: usize, member: usize, id: i64) {
        let mut state = self.lock();
        state.writer_ended[queue] = true;
        state.lost.insert(id, member);
        let stuck: Vec<i64> = state.queues[queue].drain(..).map(|job| job.id).collect();
        for id in stuck {
            state.lost.insert(id, member);
        }
        self.changed.notify_all();
    }

    fn fail(&self, message: String) {
        let mut state = self.lock();
        if state.failed.is_none() && !state.closing {
            state.failed = Some(message);
        }
        self.changed.notify_all();
    }
}

/// Connections accepted from lower-index members, by (member, stream), until connect()
/// takes them.
type Accepted = Arc<(Mutex<HashMap<(usize, usize), TcpStream>>, Condvar)>;

pub struct Worker {
    members: usize,
    index: usize,
    streams: usize,
    address: String,
    shared: Arc<Shared>,
    accepted: Accepted,
    sockets: Vec<TcpStream>,
    /// Every connection's sending and receiving thread: the ones that touch the caller's
    /// memory, so stop() always joins them.
    threads: Vec<JoinHandle<()>>,
    /// The accepting thread, which never touches the caller's memory.
    acceptor: Option<JoinHandle<()>>,
    last_error: CString,
}

fn write_header(socket: &mut TcpStream, words: &[u64]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    socket.write_all(&bytes)
}

fn read_words<const N: usize>(socket: &mut TcpStream) -> std::io::Result<[u64; N]> {
    let mut bytes = vec![0u8; N * 8];
    socket.read_exact(&mut bytes)?;
    let mut words = [0u64; N];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    }
    Ok(words)
}

impl Worker {
    pub fn new(members: usize, index: usize, streams: usize, host: &str) -> Result<Worker, String> {
        if members == 0 || index >= members || streams == 0 {
            return Err(format!(
                "bad worker: member {index} of {members}, {streams} stream(s)"
            ));
        }
        let listener =
            TcpListener::bind((host, 0)).map_err(|e| format!("listen on {host}: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let address = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        let state = State {
            queues: (0..members * streams).map(|_| VecDeque::new()).collect(),
            next_stream: vec![0; members],
            connected: vec![false; members],
            readers_left: vec![0; members],
            writer_ended: vec![false; members * streams],
            ..State::default()
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            changed: Condvar::new(),
        });
        let accepted: Accepted = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
        let mut worker = Worker {
            members,
            index,
            streams,
            address,
            shared,
            accepted,
            sockets: Vec::new(),
            threads: Vec::new(),
            acceptor: None,
            last_error: CString::default(),
        };
        let (shared, accepted) = (worker.shared.clone(), worker.accepted.clone());
        worker.acceptor = Some(
            named("gb-tcp-accept", move || {
                accept_loop(listener, index, streams, shared, accepted)
            })
            .map_err(|e| format!("starting the accepting thread: {e}"))?,
        );
        Ok(worker)
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// Open (lower index) or take the accepted (higher index) connections to `member`,
    /// then start their threads. `timeout` bounds the whole call.
    pub fn connect(
        &mut self,
        member: usize,
        address: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        if member >= self.members || member == self.index {
            return Err(format!(
                "no member {member} to connect to (this is {})",
                self.index
            ));
        }
        if self.shared.lock().connected[member] {
            return Err(format!("member {member} is already connected"));
        }
        let deadline = Instant::now() + timeout;
        let mut sockets = Vec::with_capacity(self.streams);
        if self.index < member {
            let target: SocketAddr = address
                .to_socket_addrs()
                .map_err(|e| format!("member {member}'s address {address}: {e}"))?
                .next()
                .ok_or_else(|| {
                    format!("member {member}'s address {address} resolves to nothing")
                })?;
            for stream in 0..self.streams {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!(
                        "timed out connecting to member {member} at {address}"
                    ));
                }
                let mut socket = TcpStream::connect_timeout(&target, left)
                    .map_err(|e| format!("connect to member {member} at {address}: {e}"))?;
                socket.set_nodelay(true).map_err(|e| e.to_string())?;
                write_header(
                    &mut socket,
                    &[
                        MAGIC as u64 | (PROTOCOL as u64) << 32,
                        self.index as u64,
                        stream as u64,
                    ],
                )
                .map_err(|e| format!("hello to member {member}: {e}"))?;
                sockets.push(socket);
            }
        } else {
            let (lock, arrived) = &*self.accepted;
            let mut map = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(error) = self.shared.lock().failed.clone() {
                    return Err(error);
                }
                if (0..self.streams).all(|s| map.contains_key(&(member, s))) {
                    break;
                }
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!(
                        "timed out waiting for member {member}'s {} connection(s)",
                        self.streams
                    ));
                }
                map = arrived
                    .wait_timeout(map, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            for stream in 0..self.streams {
                sockets.push(map.remove(&(member, stream)).unwrap());
            }
        }
        // Before any receiving thread runs: one may see the member already closed and count
        // itself out at once.
        self.shared.lock().readers_left[member] = self.streams;
        for (stream, socket) in sockets.into_iter().enumerate() {
            let queue = member * self.streams + stream;
            let reader = socket.try_clone().map_err(|e| e.to_string())?;
            let writer = socket.try_clone().map_err(|e| e.to_string())?;
            self.sockets.push(socket); // first, so stop() can always shut it down
            let shared = self.shared.clone();
            let sender = named("gb-tcp-send", move || {
                send_loop(writer, queue, member, shared)
            });
            // Kept even if the next one fails to start: stop() joins what did.
            self.threads
                .push(sender.map_err(|e| format!("starting a sending thread: {e}"))?);
            let shared = self.shared.clone();
            let receiver = named("gb-tcp-recv", move || recv_loop(reader, member, shared));
            self.threads
                .push(receiver.map_err(|e| format!("starting a receiving thread: {e}"))?);
        }
        self.shared.lock().connected[member] = true;
        Ok(())
    }

    fn usable(&self, state: &State, member: usize) -> Result<(), String> {
        if member >= self.members || member == self.index || !state.connected[member] {
            return Err(format!("member {member} is not connected"));
        }
        if let Some(error) = &state.failed {
            return Err(error.clone());
        }
        if state.closing {
            return Err("the worker has stopped".to_string());
        }
        if state.readers_left[member] == 0 {
            return Err(format!("member {member} closed its bridge connections"));
        }
        Ok(())
    }

    /// Queue `len` bytes at `ptr` for `member`. They must stay valid until wait() says the
    /// send finished: written to the socket, so the memory can be reused.
    pub fn send(&self, member: usize, ptr: usize, len: usize, tag: u64) -> Result<i64, String> {
        let mut state = self.shared.lock();
        self.usable(&state, member)?;
        // The next connection to `member`, round-robin, past any whose writer broke.
        let mut stream = state.next_stream[member];
        for _ in 0..self.streams {
            if !state.writer_ended[member * self.streams + stream] {
                break;
            }
            stream = (stream + 1) % self.streams;
        }
        if state.writer_ended[member * self.streams + stream] {
            return Err(format!("every connection to member {member} is broken"));
        }
        state.next_stream[member] = (stream + 1) % self.streams;
        state.next_id += 1;
        let id = state.next_id;
        state.queues[member * self.streams + stream].push_back(Job { id, tag, ptr, len });
        self.shared.changed.notify_all();
        Ok(id)
    }

    /// Post a receive of exactly `len` bytes into `ptr` for the message `tag` from `member`.
    pub fn recv(&self, member: usize, ptr: usize, len: usize, tag: u64) -> Result<i64, String> {
        let mut state = self.shared.lock();
        self.usable(&state, member)?;
        if state.posted.contains_key(&(member, tag)) {
            return Err(format!(
                "a receive for tag {tag:#x} from member {member} is already posted"
            ));
        }
        state.next_id += 1;
        let id = state.next_id;
        state.recv_from.insert(id, member);
        state.posted.insert((member, tag), Recv { id, ptr, len });
        self.shared.changed.notify_all();
        Ok(id)
    }

    /// Wait until every one of `ids` finished: 0, or TIMED_OUT, or PEER_FAILED. Either
    /// failure stops the worker first (see the module docs), and so does any later wait.
    pub fn wait(&mut self, ids: &[i64], timeout: Duration) -> i64 {
        let deadline = Instant::now() + timeout;
        let shared = self.shared.clone();
        let mut state = shared.lock();
        loop {
            if ids.iter().all(|id| state.finished.contains(id)) {
                for id in ids {
                    state.finished.remove(id);
                }
                return 0;
            }
            if let Some(error) = state.failed.clone() {
                drop(state);
                self.set_error(error);
                self.stop(STOP_TIMEOUT);
                return PEER_FAILED;
            }
            if state.closing {
                drop(state);
                self.set_error("the worker has stopped".to_string());
                return PEER_FAILED;
            }
            // A wait that can't be met any more: an operation lost with its connection, or
            // a receive from a member none of whose connections still delivers.
            let lost = ids.iter().find_map(|id| {
                state.lost.get(id).copied().or_else(|| {
                    state
                        .recv_from
                        .get(id)
                        .filter(|m| state.readers_left[**m] == 0)
                        .copied()
                })
            });
            if let Some(member) = lost {
                drop(state);
                self.set_error(format!(
                    "member {member} closed its bridge connection with operations outstanding"
                ));
                self.stop(STOP_TIMEOUT);
                return PEER_FAILED;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let missing = ids.iter().filter(|id| !state.finished.contains(id)).count();
                drop(state);
                self.set_error(format!(
                    "{missing} of {} operation(s) unfinished",
                    ids.len()
                ));
                self.stop(STOP_TIMEOUT);
                return TIMED_OUT;
            }
            state = shared
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn set_error(&mut self, error: String) {
        self.last_error = CString::new(error.replace('\0', " ")).unwrap_or_default();
    }

    /// stop(), then free the worker. False if the accepting thread outlived `timeout`.
    pub fn close(mut self, timeout: Duration) -> bool {
        self.stop(timeout)
    }

    /// Shut every connection down and join every thread that touches the caller's memory:
    /// afterwards nothing does. Those threads are joined without a time limit: shutting
    /// a socket down wakes one blocked reading or writing it, and the condition variable
    /// wakes the others, so none can stay blocked. Only the accepting thread, which never
    /// touches the caller's memory, may be left after `timeout` (false then; it ends at
    /// its next poll, or with the process). Calling it again is harmless.
    pub fn stop(&mut self, timeout: Duration) -> bool {
        {
            let mut state = self.shared.lock();
            state.closing = true;
            self.shared.changed.notify_all();
        }
        self.accepted.1.notify_all();
        for socket in &self.sockets {
            let _ = socket.shutdown(Shutdown::Both);
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
        let deadline = Instant::now() + timeout;
        match self.acceptor.take() {
            Some(handle) => {
                while !handle.is_finished() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(1));
                }
                if handle.is_finished() {
                    let _ = handle.join();
                    true
                } else {
                    false
                }
            }
            None => true,
        }
    }
}

/// Start a thread the system knows by `name` (at most 15 bytes on Linux), so a thread
/// listing or a backtrace says whose it is.
fn named<F>(name: &str, run: F) -> std::io::Result<JoinHandle<()>>
where
    F: FnOnce() + Send + 'static,
{
    thread::Builder::new().name(name.to_string()).spawn(run)
}

/// Files the `index * streams` connections the lower-index members dial. A hello naming
/// this member or a higher one, an unknown connection number, or a connection already
/// filed is dropped, so a stray or repeated dial can't take a real one's place.
fn accept_loop(
    listener: TcpListener,
    index: usize,
    streams: usize,
    shared: Arc<Shared>,
    accepted: Accepted,
) {
    let mut seen = HashSet::new();
    while seen.len() < index * streams {
        if shared.lock().closing {
            return;
        }
        match listener.accept() {
            Ok((mut socket, _)) => {
                let hello = socket
                    .set_nonblocking(false)
                    .and_then(|_| socket.set_read_timeout(Some(Duration::from_secs(5))))
                    .and_then(|_| read_words::<3>(&mut socket))
                    .and_then(|words| socket.set_read_timeout(None).map(|_| words));
                let [first, member, stream] = match hello {
                    Ok(words) => words,
                    Err(_) => continue, // not one of ours (a port scan, a dead dialer)
                };
                let (member, stream) = (member as usize, stream as usize);
                if first as u32 != MAGIC || member >= index || stream >= streams {
                    continue;
                }
                if (first >> 32) as u32 != PROTOCOL {
                    shared.fail(format!(
                        "member {member} speaks protocol {} and this one {PROTOCOL}: rebuild \
                         gpubridge.multitcp from the same gpubridge on every leader",
                        first >> 32
                    ));
                    return;
                }
                if !seen.insert((member, stream)) {
                    continue; // already filed: keep the first
                }
                let _ = socket.set_nodelay(true);
                let (lock, arrived) = &*accepted;
                lock.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert((member, stream), socket);
                arrived.notify_all();
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                shared.fail(format!("accepting bridge connections: {e}"));
                return;
            }
        }
    }
}

fn send_loop(mut socket: TcpStream, queue: usize, member: usize, shared: Arc<Shared>) {
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if state.closing || state.failed.is_some() {
                    return;
                }
                if let Some(job) = state.queues[queue].pop_front() {
                    break job;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        // SAFETY: the caller keeps the bytes valid until wait() reports this send finished.
        // An empty message may come with a null pointer (an empty tensor's): no slice of it.
        let bytes: &[u8] = if job.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(job.ptr as *const u8, job.len) }
        };
        let sent = write_header(&mut socket, &[job.tag, job.len as u64])
            .and_then(|_| socket.write_all(bytes));
        if sent.is_err() {
            shared.writer_ended(queue, member, job.id);
            return;
        }
        let mut state = shared.lock();
        state.finished.insert(job.id);
        shared.changed.notify_all();
    }
}

fn recv_loop(mut socket: TcpStream, member: usize, shared: Arc<Shared>) {
    loop {
        let [tag, len] = match read_words::<2>(&mut socket) {
            Ok(words) => words,
            Err(_) => {
                shared.reader_ended(member, None); // at a frame boundary: nothing cut off
                return;
            }
        };
        let recv = {
            let mut state = shared.lock();
            loop {
                if state.closing || state.failed.is_some() {
                    return;
                }
                if let Some(recv) = state.posted.remove(&(member, tag)) {
                    break recv;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        if recv.len as u64 != len {
            shared.fail(format!(
                "member {member} sent {len} bytes for tag {tag:#x}, posted for {} bytes",
                recv.len
            ));
            return;
        }
        // SAFETY: the caller keeps the buffer valid until wait() reports this receive done.
        let bytes: &mut [u8] = if recv.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(recv.ptr as *mut u8, recv.len) }
        };
        if socket.read_exact(bytes).is_err() {
            shared.reader_ended(member, Some(recv.id));
            return;
        }
        let mut state = shared.lock();
        state.finished.insert(recv.id);
        state.recv_from.remove(&recv.id);
        shared.changed.notify_all();
    }
}

// ---- C interface ---------------------------------------------------------------------

fn write_c(text: &str, out: *mut c_char, size: usize) {
    if out.is_null() || size == 0 {
        return;
    }
    let bytes = text.as_bytes();
    let n = bytes.len().min(size - 1);
    // SAFETY: the caller passes a buffer of `size` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
    }
}

fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// # Safety
/// `handle` must come from gb_tcp_create and not be closed yet.
unsafe fn worker<'a>(handle: *mut c_void) -> &'a mut Worker {
    &mut *(handle as *mut Worker)
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    })
}

#[no_mangle]
pub extern "C" fn gb_tcp_protocol() -> u32 {
    PROTOCOL
}

/// A worker listening on `host` (an address of this machine), or null with the reason
/// in `error`.
///
/// # Safety
/// `host` NUL-terminated; `error` holds `error_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_create(
    members: c_int,
    index: c_int,
    streams: c_int,
    host: *const c_char,
    error: *mut c_char,
    error_size: usize,
) -> *mut c_void {
    guard(std::ptr::null_mut(), || {
        let host = CStr::from_ptr(host).to_string_lossy().into_owned();
        let counts = (
            usize::try_from(members),
            usize::try_from(index),
            usize::try_from(streams),
        );
        let result = match counts {
            (Ok(m), Ok(i), Ok(s)) => Worker::new(m, i, s, &host),
            _ => Err("negative member, index or stream count".to_string()),
        };
        match result {
            Ok(w) => Box::into_raw(Box::new(w)) as *mut c_void,
            Err(e) => {
                write_c(&e, error, error_size);
                std::ptr::null_mut()
            }
        }
    })
}

/// This worker's address ("host:port") into `out`, NUL-terminated.
///
/// # Safety
/// `handle` from gb_tcp_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_address(
    handle: *mut c_void,
    out: *mut c_char,
    size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        if w.address().len() + 1 > size {
            return FAILED as c_int;
        }
        write_c(w.address(), out, size);
        0
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `address` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_connect(
    handle: *mut c_void,
    member: c_int,
    address: *const c_char,
    timeout: f64,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        let address = CStr::from_ptr(address).to_string_lossy().into_owned();
        match w.connect(member.max(0) as usize, &address, seconds(timeout)) {
            Ok(()) => 0,
            Err(e) => {
                w.set_error(e);
                FAILED as c_int
            }
        }
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `data` valid for `size` bytes until the send finishes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_send(
    handle: *mut c_void,
    member: c_int,
    data: *const c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        let w = worker(handle);
        w.send(member.max(0) as usize, data as usize, size, tag)
            .unwrap_or_else(|e| {
                w.set_error(e);
                FAILED
            })
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `data` valid for `size` bytes until the receive finishes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_recv(
    handle: *mut c_void,
    member: c_int,
    data: *mut c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        let w = worker(handle);
        w.recv(member.max(0) as usize, data as usize, size, tag)
            .unwrap_or_else(|e| {
                w.set_error(e);
                FAILED
            })
    })
}

/// 0 once every operation in `ids` finished; TIMED_OUT or PEER_FAILED (see last_error).
///
/// # Safety
/// `handle` from gb_tcp_create; `ids` holds `count` ids.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_wait(
    handle: *mut c_void,
    ids: *const i64,
    count: c_int,
    timeout: f64,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        let ids = if count > 0 {
            std::slice::from_raw_parts(ids, count as usize)
        } else {
            &[]
        };
        w.wait(ids, seconds(timeout)) as c_int
    })
}

/// The last error's text, valid until the next call on this worker.
///
/// # Safety
/// `handle` from gb_tcp_create.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_last_error(handle: *mut c_void) -> *const c_char {
    guard(std::ptr::null(), || worker(handle).last_error.as_ptr())
}

/// Stop the worker without freeing it (see the module docs): 0, or 1 if the accepting
/// thread (which holds none of the caller's memory) didn't stop within `timeout`.
///
/// # Safety
/// `handle` from gb_tcp_create.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_abort(handle: *mut c_void, timeout: f64) -> c_int {
    guard(FAILED as c_int, || {
        if worker(handle).stop(seconds(timeout)) {
            0
        } else {
            1
        }
    })
}

/// Close the worker and free it: 0, or 1 if the accepting thread didn't stop within
/// `timeout` (see gb_tcp_abort).
///
/// # Safety
/// `handle` from gb_tcp_create; not used again afterwards.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_close(handle: *mut c_void, timeout: f64) -> c_int {
    guard(FAILED as c_int, || {
        let w = Box::from_raw(handle as *mut Worker);
        if w.close(seconds(timeout)) {
            0
        } else {
            1
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(streams: usize) -> (Worker, Worker) {
        let mut a = Worker::new(2, 0, streams, "127.0.0.1").unwrap();
        let mut b = Worker::new(2, 1, streams, "127.0.0.1").unwrap();
        let (address_a, address_b) = (a.address().to_string(), b.address().to_string());
        let timeout = Duration::from_secs(10);
        let dialer = thread::spawn(move || {
            a.connect(1, &address_b, timeout).unwrap();
            a
        });
        b.connect(0, &address_a, timeout).unwrap();
        (dialer.join().unwrap(), b)
    }

    #[test]
    fn messages_arrive_intact_across_streams_in_any_order() {
        let (mut a, mut b) = pair(3);
        let sizes = [0usize, 1, 7, 4096, 1 << 20, 3_000_001];
        let data: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| (0..n).map(|j| (j * 31 + i * 7) as u8).collect())
            .collect();
        let mut out: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
        // Receives posted in reverse order, sends forward: matching is by tag.
        let mut ids_b: Vec<i64> = (0..sizes.len())
            .rev()
            .map(|i| {
                b.recv(0, out[i].as_mut_ptr() as usize, sizes[i], i as u64)
                    .unwrap()
            })
            .collect();
        let ids_a: Vec<i64> = (0..sizes.len())
            .map(|i| {
                a.send(1, data[i].as_ptr() as usize, sizes[i], i as u64)
                    .unwrap()
            })
            .collect();
        ids_b.reverse();
        assert_eq!(a.wait(&ids_a, Duration::from_secs(10)), 0);
        assert_eq!(b.wait(&ids_b, Duration::from_secs(10)), 0);
        assert_eq!(out, data);
        assert!(a.close(Duration::from_secs(5)) & b.close(Duration::from_secs(5)));
    }

    #[test]
    fn a_frame_waits_for_its_receive_to_be_posted() {
        let (mut a, mut b) = pair(1);
        let data = vec![5u8; 1000];
        let mut out = vec![0u8; 1000];
        let sent = a.send(1, data.as_ptr() as usize, data.len(), 9).unwrap();
        thread::sleep(Duration::from_millis(50));
        let got = b.recv(0, out.as_mut_ptr() as usize, out.len(), 9).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), 0);
        assert_eq!(a.wait(&[sent], Duration::from_secs(5)), 0);
        assert_eq!(out, data);
    }

    #[test]
    fn a_missing_message_times_out() {
        let (_a, mut b) = pair(2);
        let mut out = [0u8; 8];
        let id = b.recv(0, out.as_mut_ptr() as usize, 8, 1).unwrap();
        assert_eq!(b.wait(&[id], Duration::from_millis(100)), TIMED_OUT);
        assert!(b.last_error.to_str().unwrap().contains("unfinished"));
    }

    #[test]
    fn a_size_mismatch_or_a_closed_peer_fails_the_wait() {
        let (a, mut b) = pair(1);
        let data = [1u8; 16];
        let mut out = [0u8; 8];
        let got = b.recv(0, out.as_mut_ptr() as usize, 8, 3).unwrap();
        a.send(1, data.as_ptr() as usize, 16, 3).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), PEER_FAILED);
        assert!(b.last_error.to_str().unwrap().contains("sent 16 bytes"));

        let (a, mut b) = pair(2);
        let id = b.recv(0, out.as_mut_ptr() as usize, 8, 4).unwrap();
        a.close(Duration::from_secs(5));
        assert_eq!(b.wait(&[id], Duration::from_secs(5)), PEER_FAILED);
        assert!(b
            .last_error
            .to_str()
            .unwrap()
            .contains("closed its bridge connection"));
    }

    #[test]
    fn nothing_touches_the_buffer_once_stop_returns_mid_transfer() {
        let (mut a, mut b) = pair(4);
        let size = 64 << 20;
        let data = vec![7u8; size];
        let mut out = vec![0u8; size];
        let got = b.recv(0, out.as_mut_ptr() as usize, size, 3).unwrap();
        let _sent = a.send(1, data.as_ptr() as usize, size, 3).unwrap();
        thread::sleep(Duration::from_millis(2)); // let the transfer start
        assert!(b.stop(Duration::from_secs(5)));
        let snapshot = out.clone(); // whatever had arrived; nothing may change it now
        thread::sleep(Duration::from_millis(200));
        assert!(out == snapshot);
        assert_eq!(b.wait(&[got], Duration::from_secs(1)), PEER_FAILED);
        drop(out); // the buffer can go: no thread holds it
        a.stop(Duration::from_secs(5));
    }

    #[test]
    fn empty_messages_need_no_memory() {
        let (mut a, mut b) = pair(2);
        let got = b.recv(0, 0, 0, 5).unwrap(); // null pointers, as an empty tensor's
        let sent = a.send(1, 0, 0, 5).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), 0);
        assert_eq!(a.wait(&[sent], Duration::from_secs(5)), 0);
    }

    #[test]
    fn a_timed_out_wait_stops_the_worker_so_late_data_cant_land() {
        let (mut a, mut b) = pair(2);
        let mut out = vec![0u8; 64];
        let id = b.recv(0, out.as_mut_ptr() as usize, out.len(), 6).unwrap();
        assert_eq!(b.wait(&[id], Duration::from_millis(50)), TIMED_OUT);
        // The buffer may be freed now: a frame that comes late must not reach it. (The
        // sender may already see the connection closed: b stopped.)
        let data = [9u8; 64];
        if let Ok(late) = a.send(1, data.as_ptr() as usize, data.len(), 6) {
            a.wait(&[late], Duration::from_secs(2));
        }
        thread::sleep(Duration::from_millis(100));
        assert_eq!(out, vec![0u8; 64]);
        assert!(b.send(0, data.as_ptr() as usize, 1, 7).is_err()); // and it refuses more
        assert_eq!(b.wait(&[id], Duration::from_secs(1)), PEER_FAILED);
    }

    #[test]
    fn stray_or_repeated_dials_dont_take_a_real_connections_place() {
        let mut b = Worker::new(2, 1, 1, "127.0.0.1").unwrap();
        let target: SocketAddr = b.address().parse().unwrap();
        for (member, stream) in [(1u64, 0u64), (5, 0), (0, 9)] {
            let mut stray = TcpStream::connect(target).unwrap();
            write_header(
                &mut stray,
                &[MAGIC as u64 | (PROTOCOL as u64) << 32, member, stream],
            )
            .unwrap();
        }
        let mut a = Worker::new(2, 0, 1, "127.0.0.1").unwrap();
        let address = b.address().to_string();
        let dialer = thread::spawn(move || {
            a.connect(1, &address, Duration::from_secs(5)).unwrap();
            a
        });
        b.connect(0, "unused", Duration::from_secs(5)).unwrap();
        let mut a = dialer.join().unwrap();
        let (data, mut out) = ([3u8; 4], [0u8; 4]);
        let got = b.recv(0, out.as_mut_ptr() as usize, 4, 1).unwrap();
        let sent = a.send(1, data.as_ptr() as usize, 4, 1).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), 0);
        assert_eq!(a.wait(&[sent], Duration::from_secs(5)), 0);
        assert_eq!(out, data);
    }

    #[test]
    fn a_member_that_closes_fails_only_the_waits_on_it() {
        let mut workers: Vec<Worker> = (0..3)
            .map(|i| Worker::new(3, i, 1, "127.0.0.1").unwrap())
            .collect();
        let addresses: Vec<String> = workers.iter().map(|w| w.address().to_string()).collect();
        let handles: Vec<_> = workers
            .drain(..)
            .enumerate()
            .map(|(i, mut w)| {
                let addresses = addresses.clone();
                thread::spawn(move || {
                    for m in (0..3).filter(|&m| m != i) {
                        w.connect(m, &addresses[m], Duration::from_secs(5)).unwrap();
                    }
                    w
                })
            })
            .collect();
        let mut w: Vec<Worker> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let (data, mut from_two, mut from_one) = ([4u8; 8], [0u8; 8], [0u8; 8]);
        let got = w[0].recv(2, from_two.as_mut_ptr() as usize, 8, 1).unwrap();
        let never = w[0].recv(1, from_one.as_mut_ptr() as usize, 8, 2).unwrap();
        let one = w.remove(1);
        assert!(one.close(Duration::from_secs(5))); // member 1 is done and leaves
        thread::sleep(Duration::from_millis(50));
        let sent = w[1].send(0, data.as_ptr() as usize, 8, 1).unwrap(); // member 2
        assert_eq!(w[0].wait(&[got], Duration::from_secs(5)), 0);
        assert_eq!(w[1].wait(&[sent], Duration::from_secs(5)), 0);
        assert_eq!(from_two, data);
        assert_eq!(w[0].wait(&[never], Duration::from_secs(5)), PEER_FAILED);
        assert!(w[0]
            .last_error
            .to_str()
            .unwrap()
            .contains("member 1 closed"));
    }

    #[test]
    fn frames_sent_before_a_member_closed_still_arrive_on_its_other_connections() {
        let (mut a, mut b) = pair(4);
        let data: Vec<[u8; 16]> = (0..8u8).map(|i| [i; 16]).collect();
        let sent: Vec<i64> = (0..8)
            .map(|i| a.send(1, data[i].as_ptr() as usize, 16, i as u64).unwrap())
            .collect();
        assert_eq!(a.wait(&sent, Duration::from_secs(5)), 0);
        assert!(a.close(Duration::from_secs(5))); // done, and gone before b even posts
        thread::sleep(Duration::from_millis(100));
        let mut out = vec![[0u8; 16]; 8];
        let ids: Vec<i64> = (0..8)
            .map(|i| {
                b.recv(0, out[i].as_mut_ptr() as usize, 16, i as u64)
                    .unwrap()
            })
            .collect();
        assert_eq!(b.wait(&ids, Duration::from_secs(5)), 0);
        assert_eq!(out, data);
        // Now every connection of member 0 has ended: a receive from it can't be met.
        thread::sleep(Duration::from_millis(50));
        assert!(b.recv(0, out[0].as_mut_ptr() as usize, 16, 99).is_err());
    }

    #[test]
    fn a_member_that_closed_before_the_connect_finished_is_seen_as_closed() {
        for _ in 0..5 {
            let mut a = Worker::new(2, 0, 4, "127.0.0.1").unwrap();
            let mut b = Worker::new(2, 1, 4, "127.0.0.1").unwrap();
            a.connect(1, b.address(), Duration::from_secs(5)).unwrap(); // dials; b accepts
            assert!(a.close(Duration::from_secs(5))); // and leaves before b takes them
            thread::sleep(Duration::from_millis(20));
            b.connect(0, "unused", Duration::from_secs(5)).unwrap();
            let mut out = [0u8; 8];
            // Refused at once, or failed without waiting out the deadline.
            match b.recv(0, out.as_mut_ptr() as usize, 8, 1) {
                Err(_) => {}
                Ok(id) => assert_eq!(b.wait(&[id], Duration::from_secs(5)), PEER_FAILED),
            }
        }
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(Worker::new(2, 2, 1, "127.0.0.1").is_err());
        assert!(Worker::new(2, 0, 0, "127.0.0.1").is_err());
        let (a, _b) = pair(1);
        assert!(a.send(0, 0, 0, 1).is_err()); // itself
        assert!(a.send(5, 0, 0, 1).is_err());
    }
}
