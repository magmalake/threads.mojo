#!/bin/sh
# Build and run tests/stress_threads.mojo, optionally under a sanitizer.
#
#   tests/run_stress.sh plain    # no sanitizer, many rounds
#   tests/run_stress.sh tsan     # --sanitize thread, zero tolerance
#   tests/run_stress.sh asan     # --sanitize address (+ LeakSanitizer)
#
# Which sanitizer runs where is a property of the toolchain, not a choice:
# with Mojo 1.0.0, `--sanitize thread` works on osx-arm64 and `--sanitize
# address` does not link there at all; on linux-64 it is the other way round —
# ASan runs, and a TSan binary links but aborts before `main` because the Mojo
# runtime's bundled TCMalloc cannot get its 1 GiB-aligned mmap inside the
# address space TSan has reserved. CI runs each where it works.
#
# Three things this does that a bare `mojo build && ./binary` does not:
#
#  1. **A watchdog.** A bug in a join path hangs rather than fails, and a hung
#     job burns a runner until the six-hour cap. macOS ships no `timeout(1)`
#     and the pixi env does not provide coreutils, so the watchdog is written
#     here in POSIX sh and works the same on both platforms. It self-tests
#     against `sleep` on every invocation — a watchdog that has silently
#     stopped firing is worse than none.
#
#  2. **A verdict that does not trust the exit code.** A sanitizer's exit
#     status has varied across versions — TSan's is 0 here unless
#     `halt_on_error` is set — so the gate is a grep for
#     `WARNING: ThreadSanitizer` / `ERROR: AddressSanitizer` /
#     `ERROR: LeakSanitizer` over the captured output. Any hit fails the task.
#     `halt_on_error=1` means the first race also stops the run.
#
#  3. **Symbols.** Without `LLVM_SYMBOLIZER_PATH`, TSan prints
#     "Stack dump without symbol names" and the report names an address. The
#     `llvm-symbolizer` in the pixi env turns that into Mojo frames, which is
#     the difference between a report you can act on and one you cannot.
#
# Overridable: STRESS_ROUNDS, STRESS_TASKS, STRESS_TIMEOUT (seconds),
# TSAN_OPTIONS, ASAN_OPTIONS.
set -eu

mode="${1:-plain}"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
mkdir -p build

marker="build/.stress-timed-out"

# run_with_timeout SECONDS COMMAND... — returns the command's status, or fails
# with the marker file present if the watchdog had to kill it.
run_with_timeout() {
    secs=$1
    shift
    rm -f "$marker"
    "$@" &
    job=$!
    # One-second ticks rather than one long `sleep`, so the watchdog notices
    # the job finishing and exits on its own — a `sleep 900` left orphaned with
    # the step's stdout still open is its own kind of hung runner. Its output
    # goes to /dev/null for the same reason.
    (
        waited=0
        while [ "$waited" -lt "$secs" ]; do
            sleep 1
            waited=$((waited + 1))
            kill -0 "$job" 2>/dev/null || exit 0
        done
        : >"$marker"
        kill -9 "$job" 2>/dev/null || true
    ) >/dev/null 2>&1 &
    dog=$!
    status=0
    wait "$job" || status=$?
    kill "$dog" 2>/dev/null || true
    wait "$dog" 2>/dev/null || true
    return "$status"
}

# The watchdog is only worth having if it fires, so prove it every run: one
# second against a `sleep` that would otherwise take thirty.
selftest_watchdog() {
    if run_with_timeout 1 sleep 30 >/dev/null 2>&1; then
        echo "run_stress: watchdog self-test FAILED — sleep 30 was not killed" >&2
        exit 1
    fi
    if [ ! -f "$marker" ]; then
        echo "run_stress: watchdog self-test FAILED — no timeout marker" >&2
        exit 1
    fi
    rm -f "$marker"
    echo "run_stress: watchdog ok (killed a 30s sleep after 1s)"
}

# Without this a sanitizer prints "Stack dump without symbol names" and the
# report names an address rather than a Mojo frame. llvm-symbolizer ships in
# the same conda package as mojo.
export_symbolizer() {
    if [ -z "${LLVM_SYMBOLIZER_PATH:-}" ]; then
        symbolizer=$(command -v llvm-symbolizer || true)
        if [ -n "$symbolizer" ]; then
            LLVM_SYMBOLIZER_PATH="$symbolizer"
            export LLVM_SYMBOLIZER_PATH
        fi
    fi
}

selftest_watchdog

tasks="${STRESS_TASKS:-4096}"

case "$mode" in
plain)
    rounds="${STRESS_ROUNDS:-300}"
    timeout_s="${STRESS_TIMEOUT:-600}"
    binary=build/stress-threads
    log=build/stress.log
    verdict=
    echo "run_stress: building $binary"
    mojo build tests/stress_threads.mojo -I src -o "$binary"
    ;;
tsan)
    rounds="${STRESS_ROUNDS:-100}"
    timeout_s="${STRESS_TIMEOUT:-900}"
    binary=build/stress-threads-tsan
    log=build/stress-tsan.log
    verdict="WARNING: ThreadSanitizer"
    echo "run_stress: building $binary with --sanitize thread"
    mojo build --sanitize thread tests/stress_threads.mojo -I src -o "$binary"
    TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1}"
    export TSAN_OPTIONS
    export_symbolizer
    echo "run_stress: TSAN_OPTIONS=$TSAN_OPTIONS"
    echo "run_stress: LLVM_SYMBOLIZER_PATH=${LLVM_SYMBOLIZER_PATH:-<unset>}"
    ;;
asan)
    rounds="${STRESS_ROUNDS:-100}"
    timeout_s="${STRESS_TIMEOUT:-900}"
    binary=build/stress-threads-asan
    log=build/stress-asan.log
    verdict="ERROR: (Address|Leak)Sanitizer"
    echo "run_stress: building $binary with --sanitize address"
    mojo build --sanitize address tests/stress_threads.mojo -I src -o "$binary"
    # LeakSanitizer rides along with ASan on Linux and is the reason this leg
    # earns its keep: every phase here allocates a worker block or a pool
    # header and frees it after the joins, so a leak is a real finding.
    ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1}"
    export ASAN_OPTIONS
    export_symbolizer
    echo "run_stress: ASAN_OPTIONS=$ASAN_OPTIONS"
    echo "run_stress: LLVM_SYMBOLIZER_PATH=${LLVM_SYMBOLIZER_PATH:-<unset>}"
    ;;
*)
    echo "usage: $0 [plain|tsan|asan]" >&2
    exit 2
    ;;
esac

echo "run_stress: running $binary --rounds $rounds --tasks $tasks (watchdog ${timeout_s}s)"
status=0
run_with_timeout "$timeout_s" "$binary" --rounds "$rounds" --tasks "$tasks" \
    >"$log" 2>&1 || status=$?
cat "$log"

if [ -f "$marker" ]; then
    rm -f "$marker"
    echo "run_stress: FAILED — killed by the watchdog after ${timeout_s}s (a hung join?)" >&2
    exit 1
fi

# Not a race: the sanitizer never got as far as `main`. Worth saying plainly,
# because the abort looks like a crash in the library and is not one.
if grep -q "TCMalloc assumes a 48-bit virtual address space" "$log"; then
    echo "run_stress: FAILED — the Mojo runtime's TCMalloc aborted during startup." >&2
    echo "run_stress: this is the known linux-64 + --sanitize thread limitation in Mojo" >&2
    echo "run_stress: 1.0.0: the binary links, but so does a hello-world, and neither" >&2
    echo "run_stress: can start. Run the ThreadSanitizer leg on osx-arm64." >&2
    exit 1
fi

if [ -n "$verdict" ] && grep -qE "$verdict" "$log"; then
    echo "run_stress: FAILED — the sanitizer reported a finding (see above)" >&2
    exit 1
fi

if [ "$status" -ne 0 ]; then
    echo "run_stress: FAILED — $binary exited $status" >&2
    exit "$status"
fi

if [ -n "$verdict" ]; then
    echo "run_stress: ok — no sanitizer findings"
else
    echo "run_stress: ok"
fi
