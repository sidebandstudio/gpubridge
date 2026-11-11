#!/usr/bin/env bash
# The CPU baseline: build and test everything that runs without GPUs or RDMA hardware,
# the way CI does, and record what it ran on (scripts/baseline_identity.py).
#
#   scripts/baseline.sh OUT_DIR
#
# Run from the repository root with the venv to test active (or PYTHON set) and cargo
# on PATH. Every step runs, in order, even after one fails: OUT_DIR/steps.jsonl gets a
# line per step with its exit code, or why it was skipped; OUT_DIR/<step>.log its
# output. Exits 1 if any step failed, else 0.
#
# A pass is not GPU or RDMA validation: GPU tests skip, the verbs bridge runs on its
# MOCK test fabric, and the libibverbs build and its tests run only on Linux with
# rdma-core's headers. The suite runs twice: on the default bridge, then on the Gloo
# bridge (GPUBRIDGE_BRIDGE=gloo). Settings:
#   PYTHON          the interpreter (default: python)
#   PYTEST_ARGS     what pytest runs (default: the whole suite)
#   GPUBRIDGE_MULTITCP_LIB_DIR, GPUBRIDGE_VERBS_CACHE_DIR  where built libraries go
#                   (default ~/.cache/gpubridge), e.g. OUT_DIR/lib to keep runs apart
set -u

out=${1:?usage: scripts/baseline.sh OUT_DIR}
python=${PYTHON:-python}
mkdir -p "$out"
: >"$out/steps.jsonl"
failed=0
verbs=src/gpubridge/verbs/crate/Cargo.toml
multitcp=src/gpubridge/multitcp/crate/Cargo.toml

record() { # name rc seconds [skip reason]
    if [ -n "${4:-}" ]; then
        printf '{"step": "%s", "rc": null, "skipped": "%s"}\n' "$1" "$4" >>"$out/steps.jsonl"
    else
        printf '{"step": "%s", "rc": %d, "seconds": %d}\n' "$1" "$2" "$3" >>"$out/steps.jsonl"
    fi
}

step() { # name command...
    local name=$1 start rc
    shift
    start=$(date +%s)
    echo "== $name: $*"
    "$@" >"$out/$name.log" 2>&1
    rc=$?
    record "$name" "$rc" "$(($(date +%s) - start))"
    if [ "$rc" -ne 0 ]; then
        failed=1
        echo "   FAILED ($rc), see $out/$name.log"
    fi
}

skip() { # name reason
    echo "== $1: skipped, $2"
    record "$1" 0 0 "$2"
}

step identity "$python" scripts/baseline_identity.py "$out/identity.json"

step verbs_fmt cargo fmt --check --manifest-path "$verbs"
step verbs_clippy cargo clippy --offline --manifest-path "$verbs" --all-targets -- -D warnings
step verbs_test cargo test --offline --manifest-path "$verbs"
step verbs_clippy_test_fabric cargo clippy --offline --manifest-path "$verbs" --all-targets \
    --features test-fabric -- -D warnings
step verbs_test_test_fabric cargo test --offline --manifest-path "$verbs" --features test-fabric
if [ "$(uname -s)" = Linux ]; then
    step verbs_clippy_ibverbs cargo clippy --offline --manifest-path "$verbs" --all-targets \
        --features ibverbs -- -D warnings
    step verbs_test_ibverbs cargo test --offline --manifest-path "$verbs" --features ibverbs
else
    skip verbs_ibverbs "the ibverbs feature needs Linux with rdma-core's headers"
fi

step multitcp_fmt cargo fmt --check --manifest-path "$multitcp"
step multitcp_clippy cargo clippy --offline --manifest-path "$multitcp" --all-targets -- -D warnings
step multitcp_test cargo test --offline --manifest-path "$multitcp"

step multitcp_build "$python" -m gpubridge.multitcp build
step multitcp_info "$python" -m gpubridge.multitcp info
step verbs_test_fabric_build "$python" -c \
    "from gpubridge import verbs; print(verbs.build(test_fabric=True))"
if [ "$(uname -s)" = Linux ]; then
    step verbs_build "$python" -m gpubridge.verbs build
    step verbs_info "$python" -m gpubridge.verbs info
else
    skip verbs_build "the libibverbs library needs Linux with rdma-core"
fi

# The built libraries' hashes, now that they exist.
step identity_built "$python" scripts/baseline_identity.py "$out/identity.json"

# shellcheck disable=SC2086 # PYTEST_ARGS is a list of words on purpose
step pytest "$python" -m pytest -rs --durations=30 -p no:cacheprovider \
    --junitxml="$out/junit.xml" ${PYTEST_ARGS:-}
# The suite again with every job that names no bridge on the Gloo bridge (opt-in since
# the default became bridge="auto"), as CI's second 3.12 leg does.
# shellcheck disable=SC2086
step pytest_gloo env GPUBRIDGE_BRIDGE=gloo "$python" -m pytest -rs --durations=30 \
    -p no:cacheprovider --junitxml="$out/junit-gloo.xml" ${PYTEST_ARGS:-}
step simulate "$python" examples/simulate_two_islands.py

printf '{"completed": true, "failed": %d}\n' "$failed" >>"$out/steps.jsonl"
echo "steps: $out/steps.jsonl"
exit "$failed"
