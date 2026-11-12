# Contributing to gpubridge

Thanks for helping. gpubridge is an alpha: everything is tested on CPU, and
GPU runs so far are few and narrow (the [CHANGELOG](CHANGELOG.md) lists each
one, [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) what it showed and what is
still open). So the most useful contributions are often
results from real hardware (see [GPU validation](#gpu-validation) below).

## Running the CPU test suite

No GPU is needed. The tests start real multi-process jobs on CPU in
simulation mode (`GPUBRIDGE_VENDOR=nvidia|amd` fakes each process's vendor),
covering mixed, uneven and single-vendor layouts.

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # or any torch >= 2.8
uv pip install -e ".[test]"
pytest
```

- **cargo is needed.** The default bridge is a Rust library, built on first use
  (`python -m gpubridge.multitcp build`; a Rust toolchain from
  https://rustup.rs, no network). Without it, every test of a mixed layout that
  names no bridge fails at `init()` saying so.
- **Both defaults.** `pytest` runs the suite on the default bridge (`auto`: the
  TCP bridge on a machine without RDMA). `GPUBRIDGE_BRIDGE=gloo pytest` runs it
  again with every job that names no bridge on the Gloo bridge, the path the
  GPU results were measured on. Both must pass.

- **What CI runs:** Python 3.10 with torch 2.8.0 and Python 3.12 with the
  newest torch, both with Triton installed, and the 3.12 suite a second time
  with `GPUBRIDGE_BRIDGE=gloo`. `shellcheck` covers the job
  scripts, and a separate job runs `tests/test_ucx.py` over UCX's TCP
  transport.
- **Triton is optional.** Without it, the compression kernel tests skip. Set
  `GPUBRIDGE_REQUIRE_TRITON=1` to make them fail instead, as CI does.
- **UCX tests** need UCX and a C++ compiler: `python -m gpubridge.ucx build`,
  then `UCX_TLS=tcp,self pytest tests/test_ucx.py`.
- **GPU-only tests** are marked `gpu` and skip without a GPU. On a GPU node,
  run `pytest -m gpu`.
- **A quick look:** `python examples/simulate_two_islands.py` runs two NVIDIA
  and two AMD ranks on one CPU.

## Branches and pull requests

- One branch per change, from the latest `main`. Never push to `main`.
- Open a pull request. It merges with a merge commit once CI is green.
- Keep a pull request to one change, with tests for it. A change to how data
  travels must keep every collective bit for bit equal to `flat-gloo`, the
  reference policy, unless it is opt-in and documented as changing results
  (as bridge compression is). A bridge transport must also reproduce
  `tests/data/bridge_parity.json`, the hashes the Gloo bridge generated, in a
  job with no Gloo group (`tests/test_parity_fixture.py`). That file is
  written once: regenerating it (`python tests/_parity.py generate`) is a
  change to the reference and needs its own reason.
- A job that doesn't ask for Gloo must not create a Gloo group (`bridge="gloo"`,
  the `flat-gloo` policy and simulation's islands of several ranks are the
  only askers; torch's debug level DETAIL would add Gloo helper groups, so it is
  refused unless the job says `allow_debug_gloo`: `tests/test_debug_gloo.py`), and a native thread must never touch a Python object
  (`tests/test_no_gloo.py`, `tests/test_native_threads.py`).
- New behavior that GPUs could change (speed, stream ordering, RDMA) gets a
  numbered item in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md), and the
  docs say "not validated on GPUs" until a run shows it.
- Add a line to the `[Unreleased]` section at the top of
  [CHANGELOG.md](CHANGELOG.md), with the pull request number. Start the
  section if a release just closed it.
- Never commit site-specific values such as cluster project paths or account
  names. The GPU kit reads them from settings, for example the gitignored
  `scripts/gpu/explorer/local.env`.

## GPU validation

Claims that only GPUs can settle are numbered items in
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md), each with what would
confirm or refute it. The validation kit runs them:

- [docs/GPU_VALIDATION_RUNBOOK.md](docs/GPU_VALIDATION_RUNBOOK.md) maps each
  item to a step: a Slurm cluster, a single AMD machine, and a real mixed
  NVIDIA and AMD job.
- Every step is a script under [`scripts/gpu/`](scripts/gpu) with a
  `--dry-run` flag. Each one runs end to end on CPU against a fake Slurm in
  the test suite.
- Each run writes JSON records with the job's metadata, keeping only an
  allowlist of environment variables. `scripts/gpu/validation_tables.py`
  turns them into the tables in HARDWARE_VALIDATION.md.
- `GPUBRIDGE_SPLIT_TEST=half` runs two islands on a single-vendor cluster.
  Such runs are labeled `split-test` everywhere and never count as
  mixed-vendor results.

If you run any of it on real GPUs, please open an issue with the records,
even when something fails.
