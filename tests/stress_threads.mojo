"""Sanitizer stress harness for threads.mojo.

`tests/test_threads.mojo` proves the primitives compute the right numbers.
This program exists for the other half: to be run under
`mojo build --sanitize thread`, so that a data race anywhere in the library —
one that happens to produce the right answer on this machine, this run — is a
build-and-run failure rather than something someone notices in six months.

ThreadSanitizer only reasons about code that actually executed, so the shape
here is a **sweep** rather than a happy path:

- `parallel_for` (typed and opaque) at 1, 2, 4, 10 and an oversubscribed 64
  workers, because the scheduler's counter, its clamping and its join loop take
  different paths at each;
- `TypedPool` and `WorkerPool` started and stopped 50 times each, alternating
  `shutdown()` with dropping the pool so the destructor's own stop/join path is
  covered as often as the explicit one;
- `AtomicCounter` hammered by every worker at once on one cache line.

Every phase also **asserts its arithmetic**. A lost update fails the run even
on a run where the sanitizer saw nothing, which matters because TSan reports
races it observed, not races that exist.

```console
$ pixi run -e stable stress          # no sanitizer, many rounds
$ pixi run -e stable stress-tsan     # --sanitize thread (osx-arm64)
$ pixi run -e stable stress-asan     # --sanitize address (linux-64)
```

`--rounds N` repeats the whole sweep; `tests/run_stress.sh` picks the count and
puts a watchdog around the run so a deadlock in a join fails loudly instead of
hanging.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import argv
from std.testing import assert_equal, assert_true
from std.time import perf_counter_ns

from threads import (
    AtomicCounter,
    AtomicFlag,
    OpaquePtr,
    TypedPool,
    WorkerPool,
    i64_ptr,
    num_cpus,
    opaque_ptr,
    parallel_for,
    yield_now,
)

comptime USAGE = """usage: stress-threads [--rounds N] [--tasks N]

Runs every threading primitive in the tin under contention and checks the
arithmetic. Built with `--sanitize thread`, it is also a race detector.

options:
  --rounds N   repeat the whole sweep N times (default: 1)
  --tasks N    indices handed out per parallel_for sweep (default: 4096)
"""


def worker_sweep() -> List[Int]:
    """The worker counts every `parallel_for` phase is run at.

    1 exercises the degenerate single-worker path, 10 is a full machine here,
    and 64 is deliberately oversubscribed: more threads than cores is where a
    spawn/join bug shows up. A function rather than a tuple because a tuple
    only takes a comptime index, and this list is walked at runtime.
    """
    return [1, 2, 4, 10, 64]


# Pools started and stopped back to back, per round.
comptime POOL_CYCLES: Int = 50

# Workers in each of those pools.
comptime POOL_WORKERS: Int = 4

# `fetch_add`s per task in the contention phase.
comptime BUMPS_PER_TASK: Int = 20_000

# Tasks in the contention phase.
comptime CONTENDED_TASKS: Int = 64

# How long a phase may wait for workers to make progress before giving up.
comptime PROGRESS_TIMEOUT_NS: Int = 10_000_000_000


# ── parallel_for, typed ──────────────────────────────────────────────────────


struct _Sum(Movable):
    """Shared state for the typed sweep.

    `total` is only ever touched through `AtomicCounter`, so it is the atomic
    under test. `touched` has one slot per index, written by the single task
    that drew that index, so a duplicated or dropped index is a wrong number
    rather than a coin flip — and the slots are adjacent 8-byte cells, which is
    exactly the layout a sanitizer with per-byte shadow state should *not*
    complain about.
    """

    var total: Int64
    var touched: List[Int64]

    def __init__(out self, n: Int):
        self.total = 0
        self.touched = List[Int64](length=n, fill=0)


def _add_index(i: Int, mut s: _Sum) -> None:
    """Fold `i` into the shared counter and claim slot `i`."""
    _ = AtomicCounter.at(Int(Pointer(to=s.total))).fetch_add(Int64(i))
    s.touched[i] = s.touched[i] + 1


def _typed_sweep(n_tasks: Int) raises:
    """`parallel_for` (typed) at every worker count in the sweep."""
    var sweep = worker_sweep()
    for w in range(len(sweep)):
        var workers = sweep[w]
        var s = _Sum(n_tasks)
        parallel_for[_add_index](n_tasks, s, num_workers=workers)
        var want = (n_tasks * (n_tasks - 1)) // 2
        assert_equal(
            Int(s.total),
            want,
            String(
                "typed parallel_for with ",
                workers,
                " workers summed to ",
                Int(s.total),
                ", want ",
                want,
            ),
        )
        for i in range(n_tasks):
            assert_equal(
                Int(s.touched[i]),
                1,
                String(
                    "index ",
                    i,
                    " ran ",
                    Int(s.touched[i]),
                    "x with ",
                    workers,
                    " workers",
                ),
            )


# ── parallel_for, opaque ─────────────────────────────────────────────────────

# Context layout for the opaque sweep, in 64-bit cells: [0] the shared total,
# then one slot per index.
comptime _O_TOTAL: Int = 0
comptime _O_SLOTS: Int = 1


def _add_index_opaque(i: Int, ctx: OpaquePtr) -> None:
    """The opaque twin of `_add_index`, over a hand-laid-out block of cells."""
    var cells = i64_ptr(Int(ctx))
    _ = AtomicCounter.at(Int(ctx) + _O_TOTAL * 8).fetch_add(Int64(i))
    cells[unsafe_offset=_O_SLOTS + i] = cells[unsafe_offset=_O_SLOTS + i] + 1


def _opaque_sweep(n_tasks: Int) raises:
    """`parallel_for` (opaque) at every worker count in the sweep."""
    var block = unsafe_alloc[Int64](_O_SLOTS + n_tasks)
    var ctx = opaque_ptr(Int(block))
    var sweep = worker_sweep()
    for w in range(len(sweep)):
        var workers = sweep[w]
        for i in range(_O_SLOTS + n_tasks):
            block[unsafe_offset=i] = 0
        parallel_for[_add_index_opaque](n_tasks, ctx, num_workers=workers)
        var want = (n_tasks * (n_tasks - 1)) // 2
        var got = Int(block[unsafe_offset=_O_TOTAL])
        if got != want:
            block.unsafe_free()
            raise Error(
                "opaque parallel_for with ",
                workers,
                " workers summed to ",
                got,
                ", want ",
                want,
            )
        for i in range(n_tasks):
            if Int(block[unsafe_offset=_O_SLOTS + i]) != 1:
                var seen = Int(block[unsafe_offset=_O_SLOTS + i])
                block.unsafe_free()
                raise Error(
                    "opaque index ",
                    i,
                    " ran ",
                    seen,
                    "x with ",
                    workers,
                    " workers",
                )
    block.unsafe_free()


# ── AtomicCounter under contention ───────────────────────────────────────────


struct _Contended(Movable):
    """One cell, hammered by every worker at once."""

    var cell: Int64

    def __init__(out self):
        self.cell = 0


def _bump_many(i: Int, mut c: _Contended) -> None:
    """`BUMPS_PER_TASK` read-modify-writes of one shared cache line."""
    var counter = AtomicCounter.at(Int(Pointer(to=c.cell)))
    for _ in range(BUMPS_PER_TASK):
        _ = counter.fetch_add(1)


def _contention_phase() raises:
    """Every worker on one counter. A non-atomic RMW loses updates here."""
    var c = _Contended()
    parallel_for[_bump_many](CONTENDED_TASKS, c, num_workers=num_cpus())
    var want = CONTENDED_TASKS * BUMPS_PER_TASK
    assert_equal(
        Int(c.cell),
        want,
        String("contended counter reached ", Int(c.cell), ", want ", want),
    )


# ── TypedPool lifecycle ──────────────────────────────────────────────────────


struct _PoolTotals(Movable):
    """Pool state: one hammered counter, one slot per worker."""

    var ticks: Int64
    var slots: List[Int64]

    def __init__(out self, n: Int):
        self.ticks = 0
        self.slots = List[Int64](length=n, fill=0)


def _tick_typed(worker: Int, mut t: _PoolTotals, stop: AtomicFlag) -> None:
    """Tick until stopped, then claim this worker's slot."""
    var ticks = AtomicCounter.at(Int(Pointer(to=t.ticks)))
    while not stop.is_set():
        _ = ticks.fetch_add(1)
    # Plain store to a slot only this worker touches; the join publishes it.
    t.slots[worker] = t.slots[worker] + 1


def _wait_for_ticks(mut t: _PoolTotals, target: Int) -> Int:
    """Spin politely until the tick count reaches `target` or time runs out."""
    var ticks = AtomicCounter.at(Int(Pointer(to=t.ticks)))
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < PROGRESS_TIMEOUT_NS:
        if Int(ticks.load()) >= target:
            break
        yield_now()
    return Int(ticks.load())


def _check_pool_round(mut t: _PoolTotals, n: Int, cycle: Int) raises:
    """Every worker ran exactly once and the counter moved."""
    assert_true(
        Int(t.ticks) > 0,
        String("typed pool cycle ", cycle, " never ticked"),
    )
    for i in range(n):
        assert_equal(
            Int(t.slots[i]),
            1,
            String(
                "typed pool cycle ",
                cycle,
                ": slot ",
                i,
                " claimed ",
                Int(t.slots[i]),
                "x",
            ),
        )


def _typed_pool_cycles() raises:
    """`POOL_CYCLES` pools, alternating `shutdown()` with the drop path.

    The odd cycles never call `shutdown()`: the pool's last use is the wait, so
    it is destroyed there and the destructor is what sets the flag, joins and
    frees the header. That path carries the same ordering obligations as the
    explicit one and is the one a caller gets by accident.
    """
    for cycle in range(POOL_CYCLES):
        var t = _PoolTotals(POOL_WORKERS)
        var pool = TypedPool.start[_tick_typed](POOL_WORKERS, t)
        var observed = _wait_for_ticks(t, 64)
        assert_true(
            observed >= 64,
            String("typed pool cycle ", cycle, " reached ", observed, " ticks"),
        )
        if cycle % 2 == 0:
            pool.shutdown()
        else:
            _ = pool^
        _check_pool_round(t, POOL_WORKERS, cycle)


# ── WorkerPool lifecycle (the opaque form) ───────────────────────────────────

# Context layout: [0] tick counter, then one slot per worker.
comptime _P_TICKS: Int = 0
comptime _P_SLOTS: Int = 1


def _tick_opaque(worker: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    """The opaque twin of `_tick_typed`."""
    var cells = i64_ptr(Int(ctx))
    var ticks = AtomicCounter.at(Int(ctx) + _P_TICKS * 8)
    while not stop.is_set():
        _ = ticks.fetch_add(1)
    cells[unsafe_offset=_P_SLOTS + worker] = (
        cells[unsafe_offset=_P_SLOTS + worker] + 1
    )


def _worker_pool_cycles() raises:
    """`POOL_CYCLES` opaque pools, alternating `shutdown()` with the drop path.
    """
    var n = POOL_WORKERS
    var block = unsafe_alloc[Int64](_P_SLOTS + n)
    var ctx = opaque_ptr(Int(block))
    var ticks = AtomicCounter.at(Int(block) + _P_TICKS * 8)
    for cycle in range(POOL_CYCLES):
        for i in range(_P_SLOTS + n):
            block[unsafe_offset=i] = 0
        var pool = WorkerPool.start[_tick_opaque](n, ctx)
        var t0 = perf_counter_ns()
        while perf_counter_ns() - t0 < PROGRESS_TIMEOUT_NS:
            if Int(ticks.load()) >= 64:
                break
            yield_now()
        if cycle % 2 == 0:
            pool.shutdown()
        else:
            _ = pool^
        # `ctx` outlives every pool: it is freed after the loop, and each
        # pool's destructor joined before the next cycle zeroed the block.
        for i in range(n):
            if Int(block[unsafe_offset=_P_SLOTS + i]) != 1:
                var seen = Int(block[unsafe_offset=_P_SLOTS + i])
                block.unsafe_free()
                raise Error(
                    "worker pool cycle ",
                    cycle,
                    ": slot ",
                    i,
                    " claimed ",
                    seen,
                    "x",
                )
    block.unsafe_free()


# ── driver ───────────────────────────────────────────────────────────────────


def _round(n_tasks: Int) raises:
    """One full sweep of every phase."""
    _typed_sweep(n_tasks)
    _opaque_sweep(n_tasks)
    _contention_phase()
    _typed_pool_cycles()
    _worker_pool_cycles()


def main() raises:
    var args = argv()
    var rounds = 1
    var n_tasks = 4096
    var i = 1
    while i < len(args):
        var flag = String(args[i])
        if flag == "--rounds" and i + 1 < len(args):
            rounds = Int(String(args[i + 1]))
            i += 2
        elif flag == "--tasks" and i + 1 < len(args):
            n_tasks = Int(String(args[i + 1]))
            i += 2
        else:
            print(USAGE)
            raise Error("unknown option: ", flag)

    if rounds < 1:
        raise Error("--rounds must be at least 1")
    if n_tasks < 1:
        raise Error("--tasks must be at least 1")

    print(
        String(
            "stress: ",
            rounds,
            " round(s), ",
            n_tasks,
            " tasks/sweep, ",
            POOL_CYCLES,
            " pool cycles, ",
            num_cpus(),
            " logical CPUs",
        )
    )
    # One line per round is unreadable at 300 rounds and invisible at 1, so
    # report roughly ten times whatever the count is, plus the last round.
    var every = rounds // 10
    if every < 1:
        every = 1
    var t0 = perf_counter_ns()
    for r in range(rounds):
        var r0 = perf_counter_ns()
        _round(n_tasks)
        if (r + 1) % every == 0 or r + 1 == rounds:
            print(
                String(
                    "  round ",
                    r + 1,
                    "/",
                    rounds,
                    " ok (",
                    (perf_counter_ns() - r0) // 1_000_000,
                    " ms)",
                )
            )
    print(
        String(
            "stress: all rounds ok in ",
            (perf_counter_ns() - t0) // 1_000_000,
            " ms",
        )
    )
