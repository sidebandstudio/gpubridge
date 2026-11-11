"""What the docs promise about Gloo, parity and teardown is no stronger than what is so.

Three statements were once written more strongly than the code or the evidence
supports. Each is pinned here to the thing it describes, so the wording can't drift
back: the no-Gloo guarantee carries its one qualification (torch's debug level
DETAIL), the parity claim names how often the ring order differs from Gloo's, and the
exit handler is described as best effort with joins that have no time limit.
"""

from __future__ import annotations

import re
from pathlib import Path

import _parity

import gpubridge
from gpubridge.config import ALLOW_DEBUG_GLOO_ENV

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md",
        ROOT / "HARDWARE_VALIDATION.md", *sorted((ROOT / "docs").glob("*.md"))]


def _text(path: Path) -> str:
    return " ".join(path.read_text().split())  # one line: sentences wrap anywhere


def test_no_doc_states_the_no_gloo_guarantee_without_its_qualification():
    # The sentence a review found: unconditional, two lines under the paragraph that
    # described the exception to it.
    for path in DOCS:
        assert "therefore creates no Gloo group" not in _text(path), path.name
    # Wherever the guarantee is stated in so many words, the opt-in or the debug level
    # is named in the same file.
    stated = re.compile(r"creates no Gloo (process )?group|no Gloo group and no Gloo thread|"
                        r"has no Gloo process group|Gloo is no longer created unless asked")
    for path in DOCS:
        text = _text(path)
        if stated.search(text):
            assert "DETAIL" in text and ("allow_debug_gloo" in text
                                         or ALLOW_DEBUG_GLOO_ENV in text), path.name
    inventory = _text(ROOT / "docs" / "RUNTIME_INVENTORY.md")
    for phrase in ("allow_debug_gloo=True", ALLOW_DEBUG_GLOO_ENV, "raised after `init()` is "
                   "outside the guard", "No GPU run has shown"):
        assert phrase in inventory, phrase
    assert ALLOW_DEBUG_GLOO_ENV in (gpubridge.init.__doc__ or "")


def test_the_parity_claim_names_how_often_the_ring_order_differs_from_gloo():
    data = _parity.load()
    agree, of = map(int, re.fullmatch(r"(\d+) of (\d+)",
                                      data["ring_cases_where_gloo_gave_the_ring_bits"]).groups())
    assert (agree, of) == (27, 160) and data["references"] == {"gloo": 1080, "ring": 160}
    differ = of - agree
    changelog = _text(ROOT / "CHANGELOG.md")
    assert f"not bit parity with Gloo everywhere: in {differ} of the {of} order-sensitive " \
           "cases" in changelog
    assert f"{differ} of the fixture's {of} such cases" in _text(ROOT / "README.md")
    # And what the fixture leaves out is said where its size is quoted.
    assert "are not a general numerical qualification" in changelog
    for left_out in ("point-to-point", "more than four leaders", "NaN"):
        assert left_out in changelog[changelog.index("do not cover"):][:700], left_out


def test_the_exit_handler_is_described_as_best_effort_with_unbounded_joins():
    doc = " ".join((gpubridge._at_exit.__doc__ or "").split())
    assert "have limits" not in doc  # the wording that read as a bounded exit
    for phrase in ("not bounded in time", "never detached", "os._exit", "SIGKILL",
                   "no teardown fault was injected on hardware"):
        assert phrase in doc, phrase
    for path in (ROOT / "README.md", ROOT / "CHANGELOG.md"):
        text = _text(path)
        assert "not bounded in time" in text and "never detached" in text, path.name
        assert "SIGKILL" in text, path.name
    # What the wording describes: both crates join those threads with a plain join()
    # and detach none; only the TCP accepting thread is given up on at a deadline.
    tcp = (ROOT / "src/gpubridge/multitcp/crate/src/lib.rs").read_text()
    stop = tcp[tcp.index("pub fn stop(&mut self"):tcp.index("fn named<F>")]
    assert re.search(r"for handle in self\.threads\.drain\(\.\.\) \{\s*let _ = handle\.join\(\);",
                     stop)
    assert "while !handle.is_finished() && Instant::now() < deadline" in stop
    verbs = (ROOT / "src/gpubridge/verbs/crate/src/endpoint.rs").read_text()
    assert re.search(r"s\.stop\.store\(true, Ordering::Release\);\s*let _ = thread\.join\(\);",
                     verbs)
