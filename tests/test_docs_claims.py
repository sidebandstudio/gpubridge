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


#: The no-Gloo guarantee, in the words the docs use for it.
STATED = re.compile(r"creates no Gloo (process )?group|no Gloo group and no Gloo thread|"
                    r"has no Gloo (process )?group|Gloo is no longer created unless asked")
#: Its first condition: the job did not ask for Gloo.
ASKED = re.compile(r"ask(s|ed)? for (Gloo|one)|unless asked|n't ask for Gloo|not ask for Gloo")
#: Its second: it did not accept torch's debug helpers.
DEBUG = re.compile(r"DETAIL|allow_debug_gloo|" + ALLOW_DEBUG_GLOO_ENV)


def _paragraphs(path: Path) -> list[str]:
    """The file's paragraphs (blocks between blank lines), each as one line."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", path.read_text())]


def test_the_no_gloo_guarantee_is_stated_with_both_of_its_conditions():
    # The sentence a review found: unconditional, two lines under the paragraph that
    # described the exception to it.
    for path in DOCS:
        assert "therefore creates no Gloo group" not in _text(path), path.name
    # Wherever the guarantee is stated, the paragraph that states it carries both
    # conditions: the job didn't ask for Gloo, and didn't accept torch's debug
    # helpers. Not merely somewhere in the file.
    found = 0
    for path in DOCS:
        for paragraph in _paragraphs(path):
            if STATED.search(paragraph):
                found += 1
                assert ASKED.search(paragraph), f"{path.name}: no 'asked for Gloo' in: {paragraph}"
                assert DEBUG.search(paragraph), f"{path.name}: no debug level in: {paragraph}"
    assert found >= 6  # README, CHANGELOG, the inventory, the bridges' docs, item 44
    # The inventory's summary sentence itself, the one a reader quotes: both conditions
    # inside the sentence, the policy by name.
    inventory = _text(ROOT / "docs" / "RUNTIME_INVENTORY.md")
    start = inventory.index("So a GPU job has no Gloo group")
    sentence = inventory[start:inventory.index(". ", start)]
    for phrase in ("did not ask for Gloo", "`flat-gloo`", "`GlooTransport` subclass",
                   "did not accept torch's debug helpers with `allow_debug_gloo`"):
        assert phrase in sentence, (phrase, sentence)
    for phrase in ("allow_debug_gloo=True", ALLOW_DEBUG_GLOO_ENV, "raised after `init()` is "
                   "outside the guard", "No GPU run has shown"):
        assert phrase in inventory, phrase
    # The API's own statement of it, read alone: gpubridge.init()'s docstring. The
    # paragraph that states the guarantee names every way of asking for Gloo and the
    # debug opt-in.
    stated = [" ".join(block.split()) for block in re.split(r"\n\s*\n", gpubridge.init.__doc__)
              if re.search(r"has no Gloo group|creates no Gloo group", " ".join(block.split()))]
    assert len(stated) == 1, stated
    for phrase in ("unless it asks for Gloo or accepts torch's own debug helpers",
                   '``bridge="gloo"``', "``GlooTransport`` subclass", "``flat-gloo`` policy",
                   "``auto-tuned`` rule", "CPU simulation", "island of several ranks",
                   "``allow_debug_gloo``", "DETAIL"):
        assert phrase in stated[0], phrase
    assert ALLOW_DEBUG_GLOO_ENV in gpubridge.init.__doc__
    # No sentence of it claims Gloo is never created without those conditions.
    assert "creates no Gloo group for the job" not in " ".join(gpubridge.init.__doc__.split())
    # The module that does discovery says what it does (no group at all), and leaves
    # the guarantee and its exceptions to the inventory.
    intro = " ".join((gpubridge.rendezvous.__doc__ or "").split())
    assert "Nothing here creates a process group" in intro
    assert not STATED.search(intro), intro


def test_the_shared_refusal_is_stated_with_what_it_covers_and_rests_on():
    # "Every rank raises the same reason" holds for the refusals decided from the
    # discovery records, for ranks that are alive and can reach the store, on discovery
    # keys no earlier job wrote. Wherever it is promised, that is said beside it.
    doc = " ".join(gpubridge.init.__doc__.split())
    readme, changelog = _text(ROOT / "README.md"), _text(ROOT / "CHANGELOG.md")
    for where, text in (("init()", doc), ("README", readme), ("CHANGELOG", changelog)):
        for pattern in (r"same ``?`?RuntimeError``?`? text on ranks that read those records",
                        r"keeps its original exception",
                        # The finite hold, where it is promised: rank 0 waits only
                        # until discovery's own deadline, and a reader still delayed
                        # then can get a store error although alive and connected.
                        r"waits for other ranks' read marks only until its original "
                        r"discovery deadline",
                        r"logs the unread ranks when the store permits, then raises the "
                        r"original refusal",
                        r"reader still delayed at that point can instead receive a store "
                        r"error when the owner exits, even if it is alive and connected",
                        r"deadline bounds polling, not synchronous store calls or total "
                        r"init time",
                        r"no earlier job wrote",
                        r"fresh store",
                        r"persistent store under a prefix an earlier job used is not supported",
                        r"ValueError",                   # the malformed rank's own type
                        r"bridge that cannot connect"):  # later failures: not covered
            assert re.search(pattern, text), f"{where}: {pattern}"
        # And nowhere the unconditional version: delivery to every live rank.
        assert "every rank that is alive and can reach the store" not in text, where
        assert "stays until the others have read" not in text, where
    hold = " ".join((gpubridge.topology.hold_for_readers.__doc__ or "").split())
    assert ("The record wait and this hold share discovery's original deadline, not a new "
            "timeout. No additional polling grace is given when the hold starts after that "
            "deadline; synchronous store calls can still run past it.") in hold
    assert "has ended on rank 0 by the time" not in hold
    readers = " ".join((gpubridge.rendezvous.await_readers.__doc__ or "").split())
    for phrase in ("no generation", "such reuse is not supported", "bounds the polling, not "
                   "each call"):
        assert phrase in readers, phrase
    # Nothing promises more: no "none a store error" without those conditions.
    assert "none a store error" not in hold


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
