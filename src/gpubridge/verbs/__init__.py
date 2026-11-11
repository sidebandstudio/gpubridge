"""The Rust library behind ``bridge="verbs"`` (docs/VERBS_TRANSPORT.md): build, load, call.

``crate/`` is ``gpubridge-verbs``: RC queue pairs over libibverbs, through a C shim
compiled against rdma-core's headers, with the bridge's wire protocol and a
completion thread. Nothing is compiled at install time. ``python -m
gpubridge.verbs build`` compiles it with cargo on Linux (its ``ibverbs`` feature
needs rdma-core's headers: the development package, or
``GPUBRIDGE_VERBS_INCLUDE_DIR``; ``GPUBRIDGE_VERBS_LIB_DIR`` says where
``libibverbs.so.1`` is), and :func:`load` does on first use, into
``GPUBRIDGE_VERBS_CACHE_DIR`` (default ``~/.cache/gpubridge``).
``GPUBRIDGE_VERBS_LIB`` points at a library built elsewhere instead. Without
either, :class:`Unavailable` says what is missing; the rest of gpubridge never
needs this module.

**The test fabric.** ``GPUBRIDGE_VERBS_TEST_FABRIC=<directory>`` makes the bridge
use a MOCK fabric over Unix sockets between processes on one machine, so the
collective tests can run ``bridge="verbs"`` without RDMA hardware. It lives only in
a separate test build of the library (``build(test_fabric=True)``, with the
crate's ``test-fabric`` feature and without libibverbs), which only that setting
loads; the production library can't select it, and telemetry says MOCK. It is not
RDMA and proves nothing about a fabric.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CRATE = Path(__file__).with_name("crate")
#: Where the compiled library goes (``~/.cache/gpubridge`` by default). Not
#: ``GPUBRIDGE_VERBS_LIB_DIR``, which the crate's build script and docs/VERBS_TRANSPORT.md
#: give another meaning: the directory of ``libibverbs.so.1`` to link against.
CACHE_DIR_ENV = "GPUBRIDGE_VERBS_CACHE_DIR"
#: Where ``libibverbs.so.1`` is, when not in the linker's default path (read by build.rs).
LINK_DIR_ENV = "GPUBRIDGE_VERBS_LIB_DIR"
#: A library built elsewhere (``python -m gpubridge.verbs build`` prints its path).
LIB_ENV = "GPUBRIDGE_VERBS_LIB"
#: A directory: run on the MOCK test fabric there (tests only; see the module docstring).
TEST_FABRIC_ENV = "GPUBRIDGE_VERBS_TEST_FABRIC"
#: Passed to the build (see docs/VERBS_TRANSPORT.md, "Building").
BUILD_ENVS = ("GPUBRIDGE_VERBS_INCLUDE_DIR", LINK_DIR_ENV, "CC", "AR")

TIMED_OUT, PEER_FAILED = -2, -3

_lock = threading.Lock()
_libraries: dict[bool, ctypes.CDLL] = {}
log = logging.getLogger("gpubridge")


class Unavailable(RuntimeError):
    """The library can't be built or loaded here; the message says why."""


class TransportError(RuntimeError):
    """A send, receive or connection failed or timed out."""

    #: Set from the native return code, never inferred from an error message.
    timed_out = False


def _sources() -> list[Path]:
    return sorted([CRATE / "Cargo.toml", CRATE / "Cargo.lock", CRATE / "build.rs",
                   *(CRATE / "src").rglob("*.rs"), *(CRATE / "shim").glob("*.[ch]")])


def _cargo() -> str | None:
    return os.environ.get("CARGO") or shutil.which("cargo")


def _suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def features(test_fabric: bool) -> list[str]:
    """The crate features a build has: ``ibverbs`` for real use (Linux only), or only
    ``test-fabric`` for the test build."""
    return ["test-fabric"] if test_fabric else ["ibverbs"]


def library_path(*, test_fabric: bool = False) -> Path:
    """Where the library for this source, toolchain, features and machine lives once built
    (anything that could change the result is part of the name, as for multitcp)."""
    cargo = _cargo()
    rustc = shutil.which("rustc")
    toolchain = subprocess.run([rustc, "--version"], capture_output=True, text=True).stdout \
        if rustc else (cargo or "")
    parts = [*(p.read_bytes() for p in _sources()), toolchain.encode(),
             platform.machine().encode(), " ".join(platform.libc_ver()).encode(),
             sys.platform.encode(), ",".join(features(test_fabric)).encode(),
             *(os.environ.get(v, "").encode() for v in BUILD_ENVS)]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    base = Path(os.environ.get(CACHE_DIR_ENV) or Path.home() / ".cache" / "gpubridge")
    kind = "-test" if test_fabric else ""
    return base / f"libgpubridge_verbs{kind}-{key[:16]}{_suffix()}"


def build(*, test_fabric: bool = False, force: bool = False) -> Path:
    """Compile the crate with cargo (once per source, toolchain and features); return the
    library.

    Raises:
        Unavailable: without cargo, off Linux (for the real library), or if compiling
            fails (for example without rdma-core's headers).
    """
    if not test_fabric and not sys.platform.startswith("linux"):
        raise Unavailable(f"the verbs bridge needs Linux with rdma-core (libibverbs); this is "
                          f"{sys.platform}")
    cargo = _cargo()
    if cargo is None:
        raise Unavailable("no cargo (a Rust toolchain, https://rustup.rs) to build the verbs "
                          f"bridge; or set {LIB_ENV} to a library built elsewhere")
    path = library_path(test_fabric=test_fabric)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "CARGO_ENCODED_RUSTFLAGS"}
    env["RUSTFLAGS"] = ""
    with tempfile.TemporaryDirectory(prefix="gpubridge-verbs-", dir=path.parent) as work:
        crate = Path(work) / "crate"
        shutil.copytree(CRATE, crate, ignore=shutil.ignore_patterns("target"))
        command = [cargo, "build", "--release", "--offline", "--quiet", "--manifest-path",
                   str(crate / "Cargo.toml"), "--target-dir", str(Path(work) / "target"),
                   "--features", ",".join(features(test_fabric))]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise Unavailable("building the verbs bridge failed:\n" + " ".join(command)
                              + "\n" + result.stderr[-4000:])
        built = Path(work) / "target" / "release" / f"libgpubridge_verbs{_suffix()}"
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        shutil.copyfile(built, name)
        Path(name).replace(path)
    return path


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.gb_verbs_protocol.argtypes, lib.gb_verbs_protocol.restype = [], c.c_uint32
    lib.gb_verbs_build_info.argtypes = [c.c_char_p, c.c_size_t]
    lib.gb_verbs_create.argtypes = [c.c_int, c.c_int, c.c_char_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_create.restype = c.c_void_p
    lib.gb_verbs_local_info.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_connect.argtypes = [c.c_void_p, c.c_int, c.c_char_p]
    lib.gb_verbs_send.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_verbs_send.restype = c.c_int64
    lib.gb_verbs_recv.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_verbs_recv.restype = c.c_int64
    lib.gb_verbs_wait.argtypes = [c.c_void_p, c.POINTER(c.c_int64), c.c_int, c.c_double]
    lib.gb_verbs_last_error.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_telemetry.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_telemetry.restype = c.c_int64
    lib.gb_verbs_op_outcome.argtypes = [c.c_void_p, c.c_int64, c.c_char_p, c.c_size_t]
    lib.gb_verbs_op_outcome.restype = c.c_int64
    lib.gb_verbs_abort.argtypes = [c.c_void_p, c.c_char_p]
    lib.gb_verbs_close.argtypes = [c.c_void_p, c.c_double, c.c_char_p, c.c_size_t]
    lib.gb_verbs_rndv_bytes.argtypes, lib.gb_verbs_rndv_bytes.restype = [c.c_void_p], c.c_uint64
    lib.gb_verbs_gpu_load.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_gpu_caps.argtypes = [c.c_void_p, c.c_int, c.c_char_p, c.c_size_t]
    lib.gb_verbs_gpu_caps.restype = c.c_int64
    lib.gb_verbs_reg_device.argtypes = [c.c_void_p, c.c_int, c.c_uint64, c.c_uint64,
                                        c.POINTER(c.c_uint64), c.POINTER(c.c_int)]
    lib.gb_verbs_reg_host.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_uint64,
                                      c.POINTER(c.c_uint64)]
    lib.gb_verbs_release.argtypes = [c.c_void_p, c.c_uint64]
    lib.gb_verbs_invalidate_host.argtypes = [c.c_void_p, c.c_uint64]
    for name in ("gb_verbs_send_reg", "gb_verbs_recv_reg"):
        getattr(lib, name).argtypes = [c.c_void_p, c.c_int, c.c_uint64, c.c_uint64, c.c_uint64,
                                       c.c_uint64]
        getattr(lib, name).restype = c.c_int64
    for name in ("gb_verbs_build_info", "gb_verbs_local_info", "gb_verbs_connect",
                 "gb_verbs_wait", "gb_verbs_last_error", "gb_verbs_abort", "gb_verbs_close",
                 "gb_verbs_gpu_load", "gb_verbs_reg_device", "gb_verbs_reg_host",
                 "gb_verbs_release", "gb_verbs_invalidate_host"):
        getattr(lib, name).restype = c.c_int
    return lib


def test_fabric_dir() -> str | None:
    """The test fabric's directory if ``GPUBRIDGE_VERBS_TEST_FABRIC`` is set."""
    return os.environ.get(TEST_FABRIC_ENV, "").strip() or None


def load(*, test_fabric: bool | None = None, build_if_missing: bool = True) -> ctypes.CDLL:
    """The library: for real use ``GPUBRIDGE_VERBS_LIB`` if set, else this source's build
    (made first if needed); the test build when ``test_fabric`` (by default: when
    ``GPUBRIDGE_VERBS_TEST_FABRIC`` is set).

    Raises:
        Unavailable: if it isn't there and can't be built (see :func:`build`).
    """
    if test_fabric is None:
        test_fabric = test_fabric_dir() is not None
    with _lock:
        if test_fabric not in _libraries:
            given = "" if test_fabric else os.environ.get(LIB_ENV, "").strip()
            if given:
                path = Path(given)
            else:
                path = build(test_fabric=test_fabric) if build_if_missing else \
                    library_path(test_fabric=test_fabric)
            if not path.exists():
                raise Unavailable(f"no verbs bridge library at {path}; run python -m "
                                  f"gpubridge.verbs build, or set {LIB_ENV}")
            try:
                _libraries[test_fabric] = _declare(ctypes.CDLL(str(path)))
            except OSError as exc:
                raise Unavailable(f"loading {path} failed: {exc}") from None
        return _libraries[test_fabric]


def protocol(lib: ctypes.CDLL | None = None) -> int:
    """The wire protocol version this library speaks; every leader's must match."""
    return int((lib or load()).gb_verbs_protocol())


def build_info(lib: ctypes.CDLL | None = None) -> str:
    """``gpubridge-verbs <version> providers=...``: what the library was built with."""
    out = ctypes.create_string_buffer(256)
    (lib or load()).gb_verbs_build_info(out, len(out))
    return out.value.decode()


@dataclass(frozen=True)
class Registration:
    """Memory the library registered for the device to read and write in place: a handle
    for :meth:`Endpoint.send_reg` and :meth:`Endpoint.recv_reg`, the registered range
    (``addr``, ``length``: what offsets are counted from) and how (``dmabuf``,
    ``peer_memory`` or ``host``). Give it back with :meth:`Endpoint.release`."""

    handle: int
    addr: int
    length: int
    method: str

    def contains(self, pointer: int, nbytes: int) -> bool:
        return self.addr <= pointer and pointer + nbytes <= self.addr + self.length


def _closed():
    raise TransportError("the verbs bridge is closed")


#: Where Linux lists RDMA devices.
SYSFS_DEVICES = Path("/sys/class/infiniband")


def rdma_devices() -> list[str]:
    """The RDMA devices this machine has (``/sys/class/infiniband``); empty off Linux."""
    try:
        return sorted(entry.name for entry in SYSFS_DEVICES.iterdir())
    except OSError:
        return []


def availability() -> dict:
    """Whether ``bridge="verbs"`` could run in this process, by the bridge's own check:
    what ``bridge="auto"`` asks every island leader (:func:`gpubridge.transport.choose_bridge`).

    The check is the one setting up the bridge makes: load (or build) the library, then
    open an endpoint with this process's ``GPUBRIDGE_VERBS_*`` settings, which selects
    the device, an active port and its address, or fails with the reason. The endpoint
    is closed again. Nothing is built or opened on a machine with no RDMA device, and
    the MOCK test fabric never counts: it is not an RDMA device.

    Returns:
        ``{"ok": bool, "reason": str, "devices": [...]}``, with ``device``, ``port`` and
        ``link_layer`` of the endpoint's choice when ``ok``.
    """
    if test_fabric_dir() is not None:
        return {"ok": False, "devices": [],
                "reason": f"{TEST_FABRIC_ENV} is set: the MOCK test fabric is not an RDMA device"}
    devices = rdma_devices()
    if not devices:
        return {"ok": False, "devices": [],
                "reason": f"no RDMA device ({SYSFS_DEVICES} is empty or missing)"}
    try:
        from gpubridge.verbs.transport import endpoint_settings

        endpoint = Endpoint(2, 0, endpoint_settings())
    except (Unavailable, TransportError, ValueError, ImportError) as exc:
        return {"ok": False, "devices": devices,
                "reason": (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]}
    try:
        told = endpoint.telemetry()
    except TransportError as exc:
        return {"ok": False, "devices": devices, "reason": str(exc)[:300]}
    finally:
        endpoint.close(timeout=5.0)
    if told.get("mock"):
        return {"ok": False, "devices": devices, "reason": "the library is a MOCK provider"}
    return {"ok": True, "reason": "", "devices": devices, "device": told.get("device"),
            "port": told.get("port"), "link_layer": told.get("link_layer")}


class Endpoint:
    """One bridge member's end: an RC queue pair to every other member, made by
    :meth:`__init__` and connected by :meth:`connect`. Thread-safe in the library; a
    failure is remembered here, after which every call raises. Every call holds the
    handle in use while it is in the library, so :meth:`close` never frees it under one."""

    def __init__(self, members: int, index: int, settings: Mapping[str, object]) -> None:
        fabric = test_fabric_dir()
        self._lib = load(test_fabric=fabric is not None)
        lines = {**settings}
        if fabric is not None:
            lines.update(provider="test-fabric", fabric_dir=fabric)
        text = "".join(f"{k}={v}\n" for k, v in lines.items() if v is not None)
        error = ctypes.create_string_buffer(1024)
        handle = self._lib.gb_verbs_create(members, index, text.encode(), error, len(error))
        if not handle:
            raise TransportError(error.value.decode())
        self._handle: int | None = handle
        #: Calls in the library with the handle, under ``_lock`` (see :meth:`close`).
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._inflight = 0
        #: The first error, after which every call raises: a timed-out collective may
        #: have left messages half sent or received.
        self.failed: str | None = None

    # ---- the handle in use -------------------------------------------------------------

    def _enter(self) -> int | None:
        """The handle, in use until :meth:`_leave`; None once closed."""
        with self._lock:
            if self._handle is not None:
                self._inflight += 1
            return self._handle

    def _leave(self) -> None:
        with self._lock:
            self._inflight -= 1
            if not self._inflight and self._handle is None:
                self._idle.notify_all()  # a close is waiting

    def _use(self, call, closed=_closed):
        """``call(handle)`` with the handle in use; ``closed()`` once it is closed."""
        handle = self._enter()
        if handle is None:
            return closed()
        try:
            return call(handle)
        finally:
            self._leave()

    def _error(self, handle: int) -> str:
        out = ctypes.create_string_buffer(2048)
        self._lib.gb_verbs_last_error(handle, out, len(out))
        return out.value.decode()

    def _failure(self, fn, *args) -> str | None:
        """None if ``fn(handle, *args)`` returned 0, else the library's error text."""
        return self._use(lambda h: None if fn(h, *args) == 0 else self._error(h))

    def _checked(self, fn, *args) -> int:
        """``fn(handle, *args)``'s result, with the handle in use; a failure is remembered
        and raised. Refuses once failed or closed. The hot path (every send, receive and
        wait), so :meth:`_enter` and :meth:`_leave` are inlined."""
        if self.failed is not None:
            raise TransportError(f"the verbs bridge already failed ({self.failed})")
        lock = self._lock
        lock.acquire()
        try:
            handle = self._handle
            if handle is not None:
                self._inflight += 1
        finally:
            lock.release()
        if handle is None:
            _closed()
        try:
            result = fn(handle, *args)
            if result >= 0:
                return result
            text = self._error(handle)
        finally:
            lock.acquire()
            try:
                self._inflight -= 1
                if not self._inflight and self._handle is None:
                    self._idle.notify_all()  # a close is waiting
            finally:
                lock.release()
        self.failed = f"verbs bridge {text}" if result == TIMED_OUT else \
            f"verbs bridge failed: {text}"
        error = TransportError(self.failed)
        error.timed_out = result == TIMED_OUT
        raise error

    def _json(self, fn, *args, size: int = 4096, closed=_closed):
        """The JSON ``fn(handle, *args, out, size)`` writes, into a buffer grown until it
        fits; ``closed()`` once closed."""

        def call(handle: int):
            nonlocal size
            while True:
                out = ctypes.create_string_buffer(size)
                n = fn(handle, *args, out, size)
                if n >= 0:
                    return json.loads(out.value.decode())
                if n == -1:  # refused or panicked: JSON is never empty
                    raise TransportError(f"the verbs library's {fn.__name__} failed")
                size = -n + 1

        return self._use(call, closed)

    def _usable(self) -> None:
        if self.failed is not None:
            raise TransportError(f"the verbs bridge already failed ({self.failed})")
        if self._handle is None:
            raise TransportError("the verbs bridge is closed")

    # ---- calls -------------------------------------------------------------------------

    def local_info(self) -> str:
        """This member's handshake line (addresses and queue pair numbers; no keys)."""
        out = ctypes.create_string_buffer(64 * 1024)
        if self._use(lambda h: self._lib.gb_verbs_local_info(h, out, len(out))) != 0:
            raise TransportError("the verbs handshake line didn't fit")
        return out.value.decode()

    def connect(self, member: int, info: str) -> None:
        self._usable()
        if (error := self._failure(self._lib.gb_verbs_connect, member, info.encode())):
            raise TransportError(error)

    def send(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        return self._checked(self._lib.gb_verbs_send, member, pointer, nbytes, tag)

    def recv(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        return self._checked(self._lib.gb_verbs_recv, member, pointer, nbytes, tag)

    def rndv_bytes(self) -> int:
        """Messages of at least this many bytes go by rendezvous (0: none)."""
        return int(self._use(self._lib.gb_verbs_rndv_bytes))

    def load_gpu(self, vendor: str) -> str:
        """Load the ``cuda`` or ``rocm`` driver into the library; its description.

        Raises:
            TransportError: with the reason it can't be used.
        """
        self._usable()
        out = ctypes.create_string_buffer(1024)
        if (error := self._failure(self._lib.gb_verbs_gpu_load, vendor.encode(), out,
                                   len(out))):
            raise TransportError(error)
        return out.value.decode()

    def gpu_capabilities(self, device: int) -> dict:
        """What the direct paths can do on ``device``: ``{"driver", "dmabuf",
        "peer_memory"}``, each ``ok`` (with the module) or the reason."""
        return self._json(self._lib.gb_verbs_gpu_caps, device)

    def register_device(self, device: int, pointer: int, nbytes: int) -> Registration | str:
        """A direct registration of device memory (dma-buf, then peer memory), or the
        reasons there is none (the caller stages instead and says why)."""
        self._usable()
        out, method = (ctypes.c_uint64 * 3)(), ctypes.c_int(0)
        if (error := self._failure(self._lib.gb_verbs_reg_device, device, pointer, nbytes,
                                   out, ctypes.byref(method))):
            return error
        return Registration(out[0], out[1], out[2],
                            "dmabuf" if method.value == 1 else "peer_memory")

    def register_host(self, pointer: int, nbytes: int, owner: int) -> Registration:
        """Register host memory the caller owns and identifies as ``owner``. The caller
        keeps it valid until :meth:`invalidate_host` and :meth:`release`, and every
        operation on it is done, or the endpoint closed."""
        self._usable()
        out = (ctypes.c_uint64 * 3)()
        if (error := self._failure(self._lib.gb_verbs_reg_host, pointer, nbytes, owner, out)):
            raise TransportError(error)
        return Registration(out[0], out[1], out[2], "host")

    def release(self, registration: Registration) -> None:
        self._use(lambda h: self._lib.gb_verbs_release(h, registration.handle), lambda: None)

    def invalidate_host(self, owner: int) -> int:
        """Drop ``owner``'s host registrations; how many were still in use."""
        return int(self._use(lambda h: self._lib.gb_verbs_invalidate_host(h, owner),
                             lambda: 0))

    def send_reg(self, member: int, registration: Registration, pointer: int, nbytes: int,
                 tag: int) -> int:
        return self._checked(self._lib.gb_verbs_send_reg, member, registration.handle,
                             pointer - registration.addr, nbytes, tag)

    def recv_reg(self, member: int, registration: Registration, pointer: int, nbytes: int,
                 tag: int) -> int:
        return self._checked(self._lib.gb_verbs_recv_reg, member, registration.handle,
                             pointer - registration.addr, nbytes, tag)

    def wait(self, ids: Sequence[int], deadline: float) -> None:
        """Wait for operations ``ids`` until ``deadline`` (``time.monotonic()``)."""
        if not ids:
            self._usable()
            return
        array = (ctypes.c_int64 * len(ids))(*ids)
        timeout = max(0.0, deadline - time.monotonic())
        self._checked(self._lib.gb_verbs_wait, array, len(ids), timeout)

    def telemetry(self) -> dict:
        """What the endpoint reports (docs/VERBS_TRANSPORT.md, "Telemetry")."""
        return self._json(self._lib.gb_verbs_telemetry, size=16 * 1024,
                          closed=lambda: {"state": "closed"})

    def operation_outcome(self, op: int) -> dict:
        """Native outcome after a failed wait; valid even after abort, until close."""
        return self._json(self._lib.gb_verbs_op_outcome, op,
                          closed=lambda: {"status": "cancelled"})

    def abort(self, why: str = "aborted") -> None:
        """Fail the endpoint now: once this returns, nothing reads or writes the plain
        memory of any operation posted (the library copies it through its own slots and
        cancels every operation under its lock), and every queue pair is in the error
        state, so no peer can write any more. Registered memory of failed operations stays
        registered until :meth:`close`: keep it allocated until then. Every later call
        raises."""

        def call(handle: int) -> None:
            if self.failed is None:
                self.failed = why
            self._lib.gb_verbs_abort(handle, why.encode())

        self._use(call, lambda: None)

    def close(self, timeout: float = 10.0) -> bool:
        """Close; False if work didn't drain in time. Its memory then stays registered
        (leaked): the endpoint's own slots, and any registered caller memory that work
        still held, which the caller must then never free.

        Calls from here on raise (closed). A call another thread still has in the library
        (an async collective blocked in a wait, say) is aborted first, so it returns at
        once, and the handle is freed only after it has: if it is still there after
        ``timeout``, the endpoint is left open and never freed (leaked, False)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            handle, self._handle = self._handle, None
            busy = self._inflight
        if handle is None:
            return True
        if busy:
            self._lib.gb_verbs_abort(handle, b"closed while another call was in the library")
            with self._lock:
                while self._inflight and (left := deadline - time.monotonic()) > 0:
                    self._idle.wait(left)
                busy = self._inflight
            if busy:
                log.warning("gpubridge: closing the verbs bridge: %d calls still in the "
                            "library after %g s; it stays open with all its memory "
                            "(leaked, never freed)", busy, timeout)
                return False
        error = ctypes.create_string_buffer(1024)
        left = max(0.0, deadline - time.monotonic())
        if self._lib.gb_verbs_close(handle, left, error, len(error)) != 0:
            log.warning("gpubridge: closing the verbs bridge left work undrained after "
                        "%g s; its memory stays registered (leaked, never released "
                        "early): %s", timeout, error.value.decode())
            return False
        return True
