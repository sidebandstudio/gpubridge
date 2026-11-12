"""The bridge parity fixture: seeded inputs, and the SHA-256 of what the Gloo bridge made of them.

``tests/data/bridge_parity.json`` holds one hash per case, written ONCE by
``python tests/_parity.py generate`` from jobs on ``bridge="gloo"``. A bridge passes
when, in a job that has no Gloo group, every leader reproduces every hash
(``tests/test_parity_fixture.py``): the same bits as Gloo without Gloo in the job.

A case is one collective between ``leaders`` one-rank islands (every rank a bridge
leader, so the bridge alone decides the result):

- ``all_reduce/<dtype>/<op>/<leaders>/<elements>`` for every supported dtype, SUM, MAX,
  MIN and (floating point) AVG, 2, 3 and 4 leaders;
- ``broadcast/<dtype>/<leaders>/<elements>`` from the last leader.

Sizes are a few elements, then one element below, at and one above each limit a
bridge changes behaviour at, in bytes: the verbs bridge's rendezvous size (256 KiB)
and both Rust bridges' chunk (4 MiB); then two chunks and three elements.

**Inputs** come from integer arithmetic only (a SplitMix64 stream per case and
leader, computed with wrapping int64 tensor ops and laid out as little-endian bytes),
so they are the same bits on every platform and torch release, with no float
conversion and no dependence on torch's random generators. Integer dtypes take the
raw bytes: their sums wrap. Floating-point values are built bit by bit: a random
sign, a random full mantissa and an exponent within 2**-4 to 2**3, so every value is
finite, no sum of four overflows even in float16, and almost every sum rounds.

**Order.** With two leaders a reduction is one operation per element and is the same
in any order. With three or four, a floating-point SUM depends on the order the
leaders' values are added in, and the Rust and UCX bridges (a ring: each of
``leaders`` segments starts at a different leader) do not add in Gloo's order. So
for three and four leaders the fixture has two kinds of floating-point SUM and AVG
cases:

- ``all_reduce/...`` on values whose sums are exact (whole numbers of a few bits),
  so every order gives Gloo's bits: Gloo-generated, like all the cases above;
- ``ring/...`` on the rounding values, whose hashes are NOT Gloo's: they are computed
  locally by :func:`ring_reference`, which adds in the documented ring order with
  plain torch ops and no bridge. They pin that order: every ring bridge must
  reproduce them, and Gloo is not asked to.

``generate`` writes which reference each case has, and records for each ``ring``
case whether Gloo happened to agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp

import gpubridge
from gpubridge import ReduceOp
from gpubridge.config import VENDOR_ENV

FIXTURE = Path(__file__).with_name("data") / "bridge_parity.json"
FORMAT = 1
SEED = 20261111
LEADERS = (2, 3, 4)
#: The limits the sizes straddle, in bytes (the bridges' defaults).
RNDV_BYTES = 256 * 2**10
CHUNK_BYTES = 4 * 2**20
TIMEOUT = timedelta(seconds=300)

DTYPES = {
    "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    "float64": torch.float64, "int8": torch.int8, "uint8": torch.uint8,
    "int32": torch.int32, "int64": torch.int64,
}
FLOATS = ("float16", "bfloat16", "float32", "float64")
#: (exponent bits, mantissa bits) of each floating-point dtype's IEEE layout.
LAYOUT = {"float16": (5, 10), "bfloat16": (8, 7), "float32": (8, 23), "float64": (11, 52)}
OPS = {"sum": ReduceOp.SUM, "max": ReduceOp.MAX, "min": ReduceOp.MIN, "avg": ReduceOp.AVG}
#: Island labels for more islands than simulation has vendors.
LABELS = ("nvidia", "amd", "third", "fourth")


def sizes(element_size: int) -> list[int]:
    """Element counts: small ones, then each side of each limit, then several chunks."""
    counts = [1, 3, 7]
    for limit in (RNDV_BYTES, CHUNK_BYTES):
        at = limit // element_size
        counts += [at - 1, at, at + 1]
    counts.append(2 * CHUNK_BYTES // element_size + 3)
    return counts


def cases(leaders: int) -> list[str]:
    """Every case key for ``leaders`` leaders, in the order a job runs them."""
    keys = []
    for name, dtype in DTYPES.items():
        ops = ["sum", "max", "min"] + (["avg"] if name in FLOATS else [])
        for numel in sizes(dtype.itemsize):
            for op in ops:
                keys.append(f"all_reduce/{name}/{op}/{leaders}/{numel}")
                if leaders > 2 and name in FLOATS and op in ("sum", "avg"):
                    keys.append(f"ring/{name}/{op}/{leaders}/{numel}")
            keys.append(f"broadcast/{name}/{leaders}/{numel}")
    return keys


# ---- inputs: integer arithmetic only ------------------------------------------------------

def _signed(value: int) -> int:
    value &= 2**64 - 1
    return value - 2**64 if value >= 2**63 else value


def _shr(z: torch.Tensor, bits: int) -> torch.Tensor:
    """A logical right shift of an int64 tensor."""
    return (z >> bits) & ((1 << (64 - bits)) - 1)


def _stream(key: str, leader: int, count: int) -> torch.Tensor:
    """``count`` SplitMix64 outputs for this case and leader, as int64 bit patterns."""
    digest = hashlib.sha256(f"{SEED}/{key}/{leader}".encode()).digest()
    start = int.from_bytes(digest[:8], "little", signed=True)
    z = torch.arange(1, count + 1, dtype=torch.int64) * _signed(0x9E3779B97F4A7C15) + start
    z = (z ^ _shr(z, 30)) * _signed(0xBF58476D1CE4E5B9)
    z = (z ^ _shr(z, 27)) * _signed(0x94D049BB133111EB)
    return z ^ _shr(z, 31)


def _low_bytes(bits: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    """The low ``width`` bytes of each int64 (little-endian), as ``dtype`` elements."""
    return bits.view(torch.uint8).view(-1, 8)[:, :width].contiguous().view(-1).view(dtype)


def make_input(key: str, leader: int) -> torch.Tensor:
    """Leader ``leader``'s input tensor for case ``key``."""
    kind, name, *rest = key.split("/")
    numel = int(rest[-1])
    dtype = DTYPES[name]
    width = dtype.itemsize
    if name not in FLOATS:
        raw = _stream(key, leader, -(-numel * width // 8)).view(torch.uint8)
        return raw[:numel * width].clone().view(dtype)
    z = _stream(key, leader, numel)
    exponent_bits, mantissa_bits = LAYOUT[name]
    bias = (1 << (exponent_bits - 1)) - 1
    if kind == "all_reduce" and rest[0] in ("sum", "avg") and int(rest[1]) > 2:
        # Whole numbers from -8 to 7: any order of up to four adds exactly, in every
        # floating-point dtype here (see the module docs, "Order").
        return ((z & 15) - 8).to(dtype)
    mantissa = z & ((1 << mantissa_bits) - 1)
    exponent = bias - 4 + (_shr(z, mantissa_bits) & 7)
    sign = _shr(z, 63)
    bits = (sign << (8 * width - 1)) | (exponent << mantissa_bits) | mantissa
    return _low_bytes(bits, width, dtype)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
                          ).hexdigest()


def ring_reference(key: str) -> torch.Tensor:
    """What a ring all_reduce of case ``key`` gives, computed here with no bridge.

    The order :class:`gpubridge.tagged.TaggedBridge` documents: the tensor is cut into
    ``leaders`` near-equal segments; segment ``s`` starts as leader ``s``'s values and
    takes leader ``s+1``'s, ``s+2``'s ... in turn, each added to the running sum (the
    running sum is the right operand: ``mine += incoming``). AVG then divides by the
    number of leaders.
    """
    from gpubridge.tagged import _segments

    _, name, op, leaders, numel = key.split("/")
    leaders, numel = int(leaders), int(numel)
    inputs = [make_input(key, leader) for leader in range(leaders)]
    out = torch.empty(numel, dtype=DTYPES[name])
    for s, (start, length) in enumerate(_segments(numel, leaders)):
        piece = slice(start, start + length)
        total = inputs[s][piece].clone()
        for step in range(1, leaders):
            incoming = total
            total = inputs[(s + step) % leaders][piece].clone()
            total.add_(incoming)
        out[piece] = total
    if op == "avg":
        out.div_(leaders)
    return out


# ---- running the cases in a job -----------------------------------------------------------

def run_case(key: str, topology) -> torch.Tensor:
    """Run case ``key`` on this rank through gpubridge; the result tensor."""
    kind, _, *rest = key.split("/")
    tensor = make_input(key, topology.rank)
    if kind == "broadcast":
        gpubridge.broadcast(tensor, src=topology.world_size - 1)
    else:
        gpubridge.all_reduce(tensor, op=OPS[rest[0]])
    return tensor


def _rank_main(rank: int, leaders: int, bridge: str, init_file: str, out_dir: str,
               env: dict[str, str], check: bool) -> None:
    os.environ[VENDOR_ENV] = "nvidia"
    os.environ.update(env)
    torch.set_num_threads(1)
    labels = list(LABELS[:leaders])
    original = gpubridge.topology.plan_topology
    gpubridge.topology.plan_topology = lambda vendors, ranks=(): original(
        [labels[r] for r in ranks] if ranks else labels, ranks)
    topology = gpubridge.init(init_method=f"file://{init_file}", rank=rank, world_size=leaders,
                              timeout=TIMEOUT, bridge=bridge,
                              policy="reduce-bridge-broadcast")
    try:
        assert topology.is_leader and len(topology.layout.islands) == leaders
        hashes = {key: digest(run_case(key, topology)) for key in cases(leaders)}
        if check:
            from _harness import assert_no_gloo

            assert topology.gloo_free
            assert_no_gloo()
        # The last case is a large broadcast, which a leader finishes when its own copy
        # has arrived. No leader closes its bridge while another is still inside it: on
        # the UCX bridge a peer that leaves mid-collective fails the ones still waiting
        # (16 of 40 jobs of this shape without this line; the Rust bridges 0 of 300).
        # Through the store on a job without Gloo, so it adds no group.
        gpubridge.barrier()
    finally:
        gpubridge.destroy()
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(hashes))


def run_job(leaders: int, bridge: str, work: Path, env: dict[str, str] | None = None, *,
            no_gloo: bool = False) -> dict[str, str]:
    """Every case for ``leaders`` leaders on ``bridge``, in a job of that many one-rank
    islands; the hash each case gave, after checking that every leader got the same.
    ``no_gloo``: each rank also asserts, after its last collective, that the job has no
    Gloo group and (on Linux) no Gloo thread."""
    work.mkdir(parents=True, exist_ok=True)
    mp.spawn(_rank_main, args=(leaders, bridge, str(work / "rendezvous"), str(work),
                               env or {}, no_gloo), nprocs=leaders)
    per_rank = [json.loads((work / f"rank{r}.json").read_text()) for r in range(leaders)]
    for key in cases(leaders):
        got = {rank[key] for rank in per_rank}
        if len(got) != 1:
            raise AssertionError(f"{bridge}: the leaders disagree on {key}: {sorted(got)}")
    return per_rank[0]


def load() -> dict:
    data = json.loads(FIXTURE.read_text())
    if data.get("format") != FORMAT or data.get("seed") != SEED:
        raise AssertionError(f"{FIXTURE} is format {data.get('format')}, seed "
                             f"{data.get('seed')}; this code reads format {FORMAT}, seed {SEED}")
    return data


def compare(got: dict[str, str], leaders: int, *, ring: bool) -> list[str]:
    """The cases of ``leaders`` leaders whose hash differs from the fixture's. ``ring``:
    whether the bridge adds in ring order, so the ``ring/`` cases apply to it."""
    expected = load()["cases"]
    return [key for key in cases(leaders)
            if (ring or not key.startswith("ring/")) and got[key] != expected[key]]


# ---- generating the fixture ---------------------------------------------------------------

def _commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                         "--", "src"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return head.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generate(out: Path, commit: str | None) -> None:
    hashes: dict[str, str] = {}
    reference: dict[str, str] = {}
    gloo_agrees: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gpubridge-parity-") as work:
        for leaders in LEADERS:
            got = run_job(leaders, "gloo", Path(work) / f"gloo{leaders}")
            for key in cases(leaders):
                if key.startswith("ring/"):
                    hashes[key] = digest(ring_reference(key))
                    reference[key] = "ring"
                    gloo_agrees[key] = got[key] == hashes[key]
                else:
                    hashes[key] = got[key]
                    reference[key] = "gloo"
            print(f"{leaders} leaders: {len(cases(leaders))} cases", flush=True)
    agree = sum(gloo_agrees.values())
    samples = {}
    for name, dtype in DTYPES.items():  # a few inputs' own hashes: the generator, pinned
        keys = [f"all_reduce/{name}/sum/2/7", f"all_reduce/{name}/sum/3/7",
                f"broadcast/{name}/4/{RNDV_BYTES // dtype.itemsize + 1}"]
        keys += [f"ring/{name}/sum/3/7"] if name in FLOATS else []
        for key in keys:
            for leader in (0, 1):
                samples[f"{key}@{leader}"] = digest(make_input(key, leader))
    data = {
        "format": FORMAT,
        "seed": SEED,
        "what": "SHA-256 of each case's result bytes (little-endian), see tests/_parity.py",
        "generated": {
            "bridge": "gloo",
            "gpubridge_commit": commit or _commit(),
            "gpubridge_version": gpubridge.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "limits": {"rndv_bytes": RNDV_BYTES, "chunk_bytes": CHUNK_BYTES},
        "references": {"gloo": sum(r == "gloo" for r in reference.values()),
                       "ring": sum(r == "ring" for r in reference.values())},
        "ring_cases_where_gloo_gave_the_ring_bits": f"{agree} of {len(gloo_agrees)}",
        "ring": sorted(key for key, r in reference.items() if r == "ring"),
        "inputs": samples,
        "cases": hashes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}: {len(hashes)} cases, {data['references']}; Gloo gave the ring's "
          f"bits in {agree} of {len(gloo_agrees)} ring cases")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write the fixture from bridge=\"gloo\" jobs")
    gen.add_argument("--out", type=Path, default=FIXTURE)
    gen.add_argument("--commit", help="the commit to record (default: git rev-parse HEAD)")
    check = sub.add_parser("check", help="run one bridge against the fixture")
    check.add_argument("--bridge", required=True)
    check.add_argument("--leaders", type=int, nargs="*", default=list(LEADERS))
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.out, args.commit)
        return 0
    loopback = next(name for _, name in socket.if_nameindex() if name in ("lo", "lo0"))
    env = {"GPUBRIDGE_MULTITCP_IFNAME": loopback}
    bad = 0
    with tempfile.TemporaryDirectory(prefix="gpubridge-parity-") as work:
        for leaders in args.leaders:
            got = run_job(leaders, args.bridge, Path(work) / f"{args.bridge}{leaders}", env)
            wrong = compare(got, leaders, ring=args.bridge != "gloo")
            print(f"{args.bridge}, {leaders} leaders: {len(cases(leaders)) - len(wrong)} of "
                  f"{len(cases(leaders))} cases match" + (f"; differ: {wrong[:8]}" if wrong else ""))
            bad += len(wrong)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
