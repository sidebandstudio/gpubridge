"""The rule both Rust crates state: their threads never touch a Python object.

A native thread that needs the GIL while the interpreter finalizes is what aborts a
Gloo job at exit on Python 3.12. The bridges' threads can't need it: the libraries take
no callback, link nothing of Python's, and see only raw byte ranges. Checked here
against the crates' sources and manifests, the built libraries' imported symbols, and
(on Linux) the threads themselves: named, and joined by close.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from _harness import thread_names

from gpubridge import multitcp, verbs

CRATES = {"tcp": multitcp.CRATE, "verbs": verbs.CRATE}
#: Identifiers of CPython's C API, PyO3 or a Python library: none may appear. (The
#: name PyTorch may, in a comment.)
PYTHON = re.compile(r"\bPy(?!Torch\b)[A-Z_]\w*|\bpyo3\b|\bcpython\b|libpython|[Pp]ython3?\.h")
PROC = Path("/proc/self/task").is_dir()


def _sources(crate: Path) -> list[Path]:
    return sorted([*crate.rglob("*.rs"), *crate.rglob("*.c"), *crate.rglob("*.h")])


@pytest.mark.parametrize("name", list(CRATES))
def test_the_rule_is_written_in_the_crates_docs(name):
    docs = (CRATES[name] / "src" / "lib.rs").read_text()
    assert "# Threads never touch Python" in docs
    assert "never calls into Python and never holds" in " ".join(docs.split()).replace("//! ", "")


@pytest.mark.parametrize("name", list(CRATES))
def test_the_crate_depends_on_the_standard_library_only(name):
    manifest = (CRATES[name] / "Cargo.toml").read_text()
    for section in ("dependencies", "build-dependencies", "dev-dependencies"):
        found = re.search(rf"^\[{section}\]\n(.*?)(?=^\[|\Z)", manifest, re.M | re.S)
        entries = [line for line in (found.group(1) if found else "").splitlines()
                   if line.strip() and not line.strip().startswith("#")]
        assert entries == [], f"{name}: [{section}] has {entries}"
    packages = re.findall(r'^name = "(.+)"', (CRATES[name] / "Cargo.lock").read_text(), re.M)
    assert len(packages) == 1, packages  # the crate itself


@pytest.mark.parametrize("name", list(CRATES))
def test_no_c_function_of_the_library_takes_a_callback(name):
    prefix = {"tcp": "gb_tcp_", "verbs": "gb_verbs_"}[name]
    seen = 0
    for path in (CRATES[name] / "src").rglob("*.rs"):
        for found in re.finditer(r'pub (?:unsafe )?extern "C" fn (\w+)\s*\((.*?)\)\s*(?:->|\{)',
                                 path.read_text(), re.S):
            function, parameters = found.groups()
            assert function.startswith(prefix), f"{path.name}: {function}"
            assert "fn(" not in parameters and "fn (" not in parameters, (
                f"{function} takes a function pointer: {' '.join(parameters.split())}")
            seen += 1
    assert seen >= 9  # the whole interface was found, not an empty match


@pytest.mark.parametrize("name", list(CRATES))
def test_the_sources_name_nothing_of_pythons_c_api(name):
    for path in _sources(CRATES[name]):
        if "target" in path.parts:
            continue
        hits = PYTHON.findall(path.read_text(errors="replace"))
        assert hits == [], f"{path}: {sorted(set(hits))}"


def _libraries():
    built = []
    for label, build in (("tcp", multitcp.build), ("verbs test fabric",
                                                   lambda: verbs.build(test_fabric=True)),
                         ("verbs", verbs.build)):
        try:
            built.append((label, build()))
        except (multitcp.Unavailable, verbs.Unavailable):
            continue
    return built


def test_the_built_libraries_import_no_python_symbol():
    nm = shutil.which("nm")
    libraries = _libraries()
    if nm is None or not libraries:
        pytest.skip("needs nm and a built library (cargo)")
    for label, path in libraries:
        flags = ["-u"] if sys.platform == "darwin" else ["-D", "--undefined-only"]
        listing = subprocess.run([nm, *flags, str(path)], capture_output=True, text=True,
                                 check=True).stdout
        assert len(listing.split()) > 5, f"{label}: nm listed nothing"
        python = [line for line in listing.splitlines() if re.search(r"\b_?Py[A-Z_]", line)]
        assert python == [], f"{label} ({path.name}) imports {python}"


@pytest.mark.skipif(not PROC, reason="thread names come from /proc/self/task (Linux)")
def test_the_tcp_bridges_threads_are_named_and_joined_by_close():
    try:
        multitcp.build()
    except multitcp.Unavailable as exc:
        pytest.skip(str(exc).splitlines()[0])
    def ours() -> list[str]:
        return sorted(n for n in thread_names() if n.startswith("gb-tcp-"))

    def settled(want: list[str]) -> list[str]:
        """The threads once they are ``want`` (a new thread names itself as it starts,
        and an accepting thread with nothing left to accept ends), or after 5 s."""
        deadline = time.monotonic() + 5
        while ours() != want and time.monotonic() < deadline:
            time.sleep(0.01)
        return ours()

    before = ours()
    low = multitcp.Worker(2, 0, 3, "127.0.0.1")
    high = multitcp.Worker(2, 1, 3, "127.0.0.1")
    try:
        # The higher index waits for the lower's three connections; the lower index has
        # nobody below it to accept from, so its accepting thread is already done.
        assert settled(before + ["gb-tcp-accept"]) == before + ["gb-tcp-accept"]
        low.connect(1, high.address(), 10.0)   # the lower index dials
        high.connect(0, low.address(), 10.0)   # the higher takes what was accepted
        want = sorted(before + ["gb-tcp-recv"] * 6 + ["gb-tcp-send"] * 6)
        assert settled(want) == want  # three connections, each end a sender and a receiver
    finally:
        low.close(timeout=5.0)
        high.close(timeout=5.0)
    assert ours() == before  # joined by close, not merely asked to stop


@pytest.mark.skipif(not PROC, reason="thread names come from /proc/self/task (Linux)")
def test_the_verbs_completion_thread_is_joined_by_close(monkeypatch):
    try:
        verbs.build(test_fabric=True)
    except verbs.Unavailable as exc:
        pytest.skip(str(exc).splitlines()[0])
    fabric = tempfile.mkdtemp(prefix="gbvf-", dir="/tmp")
    monkeypatch.setenv(verbs.TEST_FABRIC_ENV, fabric)
    try:
        before = thread_names().count("gpubridge-verbs")
        endpoint = verbs.Endpoint(2, 0, {})
        assert thread_names().count("gpubridge-verbs") == before + 1
        assert endpoint.close(timeout=5.0)
        assert thread_names().count("gpubridge-verbs") == before
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
