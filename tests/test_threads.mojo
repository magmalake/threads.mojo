"""Unit tests for threads.mojo.

These are not smoke tests. The interesting ones are the races: an atomic
counter hammered by 8 threads, a plain integer guarded only by the mutex, a
release/acquire handoff whose payload is a plain write, and a loop that
allocates and frees Mojo `List`s on every worker at once. Each is written so
that a broken primitive produces a *wrong number*, not a flaky hang.

Every context here is a heap block of 64-bit cells laid out by hand, because
that is the only thing a thread start routine can be handed. The cell indices
are named at the top of each test.
"""

from std.memory.alloc import unsafe_alloc
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns

from threads import (
    AtomicCounter,
    AtomicFlag,
    CondVar,
    CondVarRef,
    Mutex,
    MutexRef,
    OpaquePtr,
    ThreadGroup,
    ThreadHandle,
    TypedPool,
    WorkerPool,
    atomic_fetch_add,
    current_thread_id,
    i64_ptr,
    join_all,
    num_cpus,
    opaque_ptr,
    parallel_for,
    spawn_n,
    yield_now,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _cells(n: Int) -> OpaquePtr:
    """Allocate `n` zeroed 64-bit cells and return them as an opaque block."""
    var block = unsafe_alloc[Int64](n)
    for i in range(n):
        block[unsafe_offset=i] = 0
    return opaque_ptr(Int(block))


def _free_cells(ctx: OpaquePtr):
    i64_ptr(Int(ctx)).unsafe_free()


def _spin_ns(duration: Int):
    """Burn `duration` nanoseconds on this thread without sleeping.

    Deliberately not `usleep`: a libc sleep from a Mojo thread has been
    observed to overshoot by orders of magnitude, which would turn the timing
    test below into a coin flip.
    """
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < duration:
        pass


# ── thread lifecycle ─────────────────────────────────────────────────────────


def _write_marker(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] marker, [1] thread id observed inside the thread."""
    var cells = i64_ptr(Int(arg))
    cells[unsafe_offset=0] = 0xC0FFEE
    cells[unsafe_offset=1] = Int64(Int(current_thread_id()))
    return arg


def test_spawn_join_round_trip() raises:
    """A spawned thread runs, writes through the arg pointer, and joins."""
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    var t = ThreadHandle.spawn[_write_marker](ctx)
    assert_true(t.is_joinable())
    t.join()
    assert_equal(Int(cells[unsafe_offset=0]), 0xC0FFEE)
    _free_cells(ctx)


def test_thread_really_is_another_thread() raises:
    """The id observed inside the thread differs from the caller's."""
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    var t = ThreadHandle.spawn[_write_marker](ctx)
    t.join()
    assert_true(Int(cells[unsafe_offset=1]) != Int(current_thread_id()))
    _free_cells(ctx)


def test_double_join_is_a_no_op() raises:
    """A second join on the same handle short-circuits instead of calling
    pthread_join on a stale id (which is undefined, not an error)."""
    var ctx = _cells(2)
    var t = ThreadHandle.spawn[_write_marker](ctx)
    t.join()
    assert_true(not t.is_joinable())
    t.join()
    t.join()
    assert_true(not t.is_joinable())
    _free_cells(ctx)


def _bump_own_slot(arg: OpaquePtr) -> OpaquePtr:
    """Each thread gets its own cell via `spawn_n`'s stride."""
    var cells = i64_ptr(Int(arg))
    cells[unsafe_offset=0] = cells[unsafe_offset=0] + 1
    return arg


def test_sixteen_threads_each_get_their_own_slot() raises:
    """16 threads, one 8-byte slice of the context each — the `spawn_n` stride
    path."""
    var n = 16
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    var group = spawn_n[_bump_own_slot](n, ctx, stride=8)
    assert_equal(len(group), n)
    join_all(group)
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=i]), 1)
    _free_cells(ctx)


def test_num_cpus_is_at_least_one() raises:
    assert_true(num_cpus() >= 1)


def test_pin_to_cpu() raises:
    """Real affinity on Linux, a documented no-op on macOS. Either way it must
    not raise for cpu 0, and the thread must still finish."""
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    var t = ThreadHandle.spawn[_write_marker](ctx)
    t.pin_to_cpu(0)
    t.join()
    assert_equal(Int(cells[unsafe_offset=0]), 0xC0FFEE)
    _free_cells(ctx)


def test_pin_all_in_a_group() raises:
    var n = 4
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    var group = spawn_n[_bump_own_slot](n, ctx, stride=8)
    group.pin_all()
    group.join_all()
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=i]), 1)
    _free_cells(ctx)


def test_detached_thread_is_not_joinable() raises:
    """A detached handle goes inert; the thread's context is leaked on purpose
    so the detached thread cannot outlive it."""
    var ctx = _cells(2)
    var t = ThreadHandle.spawn[_write_marker](ctx)
    t.detach()
    assert_true(not t.is_joinable())
    t.join()
    # `ctx` is intentionally not freed: nothing tells us when a detached
    # thread stopped reading it. This is the leak the docstring warns about.


# ── atomics ──────────────────────────────────────────────────────────────────


def _hammer_counter(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] the counter, [1] iterations."""
    var cells = i64_ptr(Int(arg))
    var counter = AtomicCounter.at(Int(arg))
    var iterations = Int(cells[unsafe_offset=1])
    for _ in range(iterations):
        _ = counter.fetch_add(1)
    return arg


def test_atomic_counter_under_eight_threads() raises:
    """8 threads x 100_000 increments == 800_000, exactly.

    This is the real atomicity proof: a non-atomic read-modify-write loses
    updates here every single run.
    """
    var threads = 8
    var iterations = 100_000
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    cells[unsafe_offset=1] = Int64(iterations)
    var group = spawn_n[_hammer_counter](threads, ctx)
    group.join_all()
    assert_equal(Int(cells[unsafe_offset=0]), threads * iterations)
    _free_cells(ctx)


def _publish_then_flag(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] flag, [1] payload, [2] observers that saw the payload,
    [3] observers that woke at all."""
    var cells = i64_ptr(Int(arg))
    var flag = AtomicFlag.at(Int(arg))
    # A plain, non-atomic write. Only the release store below makes it visible.
    cells[unsafe_offset=1] = 0x5EEDBEEF
    flag.set()
    return arg


def _observe_flag(arg: OpaquePtr) -> OpaquePtr:
    var cells = i64_ptr(Int(arg))
    var flag = AtomicFlag.at(Int(arg))
    var spins = 0
    while not flag.is_set():
        yield_now()
        spins += 1
        if spins > 100_000_000:
            return arg  # bail rather than hang the suite
    _ = atomic_fetch_add(i64_ptr(Int(arg) + 3 * 8), 1)
    if cells[unsafe_offset=1] == 0x5EEDBEEF:
        _ = atomic_fetch_add(i64_ptr(Int(arg) + 2 * 8), 1)
    return arg


def test_atomic_flag_release_acquire_visibility() raises:
    """The writer's plain write is visible to every reader that observes the
    flag. Readers spin on `is_set()` (acquire) and then read the payload."""
    var readers = 4
    var ctx = _cells(4)
    var cells = i64_ptr(Int(ctx))
    var observers = spawn_n[_observe_flag](readers, ctx)
    var writer = ThreadHandle.spawn[_publish_then_flag](ctx)
    writer.join()
    observers.join_all()
    assert_equal(Int(cells[unsafe_offset=3]), readers)
    assert_equal(Int(cells[unsafe_offset=2]), readers)
    _free_cells(ctx)


def test_atomic_flag_and_counter_standalone_cells() raises:
    """The `alloc` / `unsafe_free` convenience shape."""
    var flag = AtomicFlag.alloc()
    assert_true(not flag.is_set())
    flag.set()
    assert_true(flag.is_set())
    flag.clear()
    assert_true(not flag.is_set())
    flag.set_value(7)
    assert_equal(Int(flag.raw()), 7)
    flag.unsafe_free()

    var counter = AtomicCounter.alloc(10)
    assert_equal(Int(counter.load()), 10)
    assert_equal(Int(counter.fetch_add(5)), 10)
    assert_equal(Int(counter.load()), 15)
    counter.store(0)
    assert_equal(Int(counter.load()), 0)
    counter.unsafe_free()


# ── mutex ────────────────────────────────────────────────────────────────────


def _guarded_increment(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] mutex address, [1] a plain Int, [2] iterations."""
    var cells = i64_ptr(Int(arg))
    var mutex = MutexRef.at(Int(cells[unsafe_offset=0]))
    var iterations = Int(cells[unsafe_offset=2])
    for _ in range(iterations):
        mutex.lock()
        # Deliberately a plain, non-atomic read-modify-write. The mutex is the
        # only thing making this correct.
        cells[unsafe_offset=1] = cells[unsafe_offset=1] + 1
        mutex.unlock()
    return arg


def test_mutex_gives_mutual_exclusion() raises:
    """8 threads x 10_000 unguarded-looking increments == 80_000, exactly."""
    var threads = 8
    var iterations = 10_000
    var mutex = Mutex()
    var ctx = _cells(3)
    var cells = i64_ptr(Int(ctx))
    cells[unsafe_offset=0] = Int64(mutex.address())
    cells[unsafe_offset=2] = Int64(iterations)
    var group = spawn_n[_guarded_increment](threads, ctx)
    group.join_all()
    assert_equal(Int(cells[unsafe_offset=1]), threads * iterations)
    _free_cells(ctx)
    # Mention the mutex after the join so last-use destruction cannot free it
    # while a worker still holds a MutexRef to it.
    assert_true(mutex.address() != 0)


def _time_a_blocked_lock(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] mutex address, [1] nanoseconds spent blocked, [2] ready flag.
    """
    var cells = i64_ptr(Int(arg))
    var mutex = MutexRef.at(Int(cells[unsafe_offset=0]))
    var ready = AtomicFlag.at(Int(arg) + 2 * 8)
    ready.set()
    var t0 = perf_counter_ns()
    mutex.lock()
    var t1 = perf_counter_ns()
    mutex.unlock()
    cells[unsafe_offset=1] = Int64(t1 - t0)
    return arg


def test_mutex_actually_blocks() raises:
    """A thread that wants a held mutex waits for it.

    The margin is deliberately huge — hold for 50 ms, assert the waiter was
    blocked for more than 10 ms — so this measures "did it block at all",
    not scheduler latency.
    """
    var hold_ns = 50_000_000
    var mutex = Mutex()
    var ctx = _cells(3)
    var cells = i64_ptr(Int(ctx))
    cells[unsafe_offset=0] = Int64(mutex.address())
    var ready = AtomicFlag.at(Int(ctx) + 2 * 8)

    mutex.lock()
    var t = ThreadHandle.spawn[_time_a_blocked_lock](ctx)
    # Wait until the worker is definitely about to block, then hold.
    while not ready.is_set():
        yield_now()
    _spin_ns(hold_ns)
    mutex.unlock()
    t.join()

    assert_true(
        Int(cells[unsafe_offset=1]) > 10_000_000,
        String("waiter was blocked only ", Int(cells[unsafe_offset=1]), " ns"),
    )
    _free_cells(ctx)
    assert_true(mutex.address() != 0)


def test_mutex_try_lock() raises:
    var mutex = Mutex()
    assert_true(mutex.try_lock())
    mutex.unlock()
    mutex.lock()
    mutex.unlock()
    assert_true(mutex.address() != 0)


def _set_to_five(ctx: OpaquePtr) -> None:
    i64_ptr(Int(ctx))[unsafe_offset=0] = 5


def test_mutex_with_lock() raises:
    var mutex = Mutex()
    var ctx = _cells(1)
    mutex.with_lock[_set_to_five](ctx)
    assert_equal(Int(i64_ptr(Int(ctx))[unsafe_offset=0]), 5)
    # The mutex must be free again afterwards.
    assert_true(mutex.try_lock())
    mutex.unlock()
    _free_cells(ctx)


# ── condition variable ───────────────────────────────────────────────────────


def _wait_for_predicate(arg: OpaquePtr) -> OpaquePtr:
    """cells: [0] mutex address, [1] cond address, [2] predicate, [3] result."""
    var cells = i64_ptr(Int(arg))
    var mutex = MutexRef.at(Int(cells[unsafe_offset=0]))
    var cond = CondVarRef.at(Int(cells[unsafe_offset=1]))
    mutex.lock()
    while cells[unsafe_offset=2] == 0:
        cond.wait(mutex)
    cells[unsafe_offset=3] = cells[unsafe_offset=2] * 2
    mutex.unlock()
    return arg


def test_condvar_wait_and_signal() raises:
    var mutex = Mutex()
    var cond = CondVar()
    var ctx = _cells(4)
    var cells = i64_ptr(Int(ctx))
    cells[unsafe_offset=0] = Int64(mutex.address())
    cells[unsafe_offset=1] = Int64(cond.address())
    var t = ThreadHandle.spawn[_wait_for_predicate](ctx)
    _spin_ns(2_000_000)
    mutex.lock()
    cells[unsafe_offset=2] = 21
    cond.signal()
    mutex.unlock()
    t.join()
    assert_equal(Int(cells[unsafe_offset=3]), 42)
    _free_cells(ctx)
    assert_true(mutex.address() != 0 and cond.address() != 0)


# ── parallel_for ─────────────────────────────────────────────────────────────


def _square_into_slot(i: Int, ctx: OpaquePtr) -> None:
    """cells[i] = i*i. One cell per task, so no synchronisation is needed."""
    i64_ptr(Int(ctx))[unsafe_offset=i] = Int64(i * i)


def test_parallel_for_covers_every_index_exactly_once() raises:
    """4096 tasks over the default worker count; every slot written once."""
    var n = 4096
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_square_into_slot](n, ctx)
    var total = 0
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=i]), i * i)
        total += i * i
    assert_true(total > 0)
    _free_cells(ctx)


def _bump_shared_counter(i: Int, ctx: OpaquePtr) -> None:
    """cells[0] is an atomic counter; cells[1] accumulates the index sum."""
    _ = AtomicCounter.at(Int(ctx)).fetch_add(1)
    _ = atomic_fetch_add(i64_ptr(Int(ctx) + 8), Int64(i))


def test_parallel_for_task_count_and_index_sum() raises:
    """Each index is handed out exactly once, checked two ways at once: the
    task count and the sum of the indices."""
    var n = 10_000
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_bump_shared_counter](n, ctx, num_workers=8)
    assert_equal(Int(cells[unsafe_offset=0]), n)
    assert_equal(Int(cells[unsafe_offset=1]), (n - 1) * n // 2)
    _free_cells(ctx)


def test_parallel_for_with_zero_tasks_does_nothing() raises:
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_bump_shared_counter](0, ctx)
    parallel_for[_bump_shared_counter](-5, ctx)
    assert_equal(Int(cells[unsafe_offset=0]), 0)
    _free_cells(ctx)


def test_parallel_for_with_fewer_tasks_than_workers() raises:
    """Workers are clamped to the task count — three tasks never start 64
    threads."""
    var ctx = _cells(8)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_square_into_slot](3, ctx, num_workers=64)
    assert_equal(Int(cells[unsafe_offset=0]), 0)
    assert_equal(Int(cells[unsafe_offset=1]), 1)
    assert_equal(Int(cells[unsafe_offset=2]), 4)
    assert_equal(Int(cells[unsafe_offset=3]), 0)
    _free_cells(ctx)


def test_parallel_for_single_worker() raises:
    var n = 100
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_square_into_slot](n, ctx, num_workers=1)
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=i]), i * i)
    _free_cells(ctx)


def _touch_ctx_late(i: Int, ctx: OpaquePtr) -> None:
    """Spend a while, *then* write. If parallel_for freed its worker block or
    the caller's context before joining, this is where it would crash."""
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < 1_000_000:
        pass
    i64_ptr(Int(ctx))[unsafe_offset=i] = Int64(i + 1)


def test_parallel_for_context_outlives_the_join() raises:
    """The context is still valid at the *end* of every task, not merely at
    the start — the ordering guarantee the whole send contract rests on."""
    var n = 32
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_touch_ctx_late](n, ctx, num_workers=8)
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=i]), i + 1)
    _free_cells(ctx)


def _fail_on_odd(i: Int, ctx: OpaquePtr) -> None:
    """The documented error-cell pattern: cells[0] is a flag, cells[1+i] is a
    per-task error code."""
    if i % 2 == 1:
        i64_ptr(Int(ctx))[unsafe_offset=1 + i] = Int64(100 + i)
        AtomicFlag.at(Int(ctx)).set()


def test_parallel_for_error_cell_pattern() raises:
    """A task cannot raise, so it publishes a code and sets a flag."""
    var n = 8
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_fail_on_odd](n, ctx, num_workers=4)
    assert_true(AtomicFlag.at(Int(ctx)).is_set())
    var failures = 0
    for i in range(n):
        if cells[unsafe_offset=1 + i] != 0:
            assert_equal(Int(cells[unsafe_offset=1 + i]), 100 + i)
            failures += 1
    assert_equal(failures, 4)
    _free_cells(ctx)


# ── parallel_for, typed form ─────────────────────────────────────────────────


@fieldwise_init
struct _Totals(Copyable, Movable):
    """Shared state for the typed form: one atomically bumped cell."""

    var count: Int64
    var sum: Int64


def _bump_totals(i: Int, mut t: _Totals) -> None:
    _ = AtomicCounter.at(Int(Pointer(to=t.count))).fetch_add(1)
    _ = atomic_fetch_add(i64_ptr(Int(Pointer(to=t.sum))), Int64(i))


def test_parallel_for_typed_task_count_and_index_sum() raises:
    """The typed form hands every index out exactly once, and the state the
    caller passed by `ref` is the one the tasks wrote."""
    var n = 10_000
    var totals = _Totals(0, 0)
    parallel_for[_bump_totals](n, totals, num_workers=8)
    assert_equal(Int(totals.count), n)
    assert_equal(Int(totals.sum), (n - 1) * n // 2)


def test_parallel_for_typed_with_zero_tasks_does_nothing() raises:
    var totals = _Totals(0, 0)
    parallel_for[_bump_totals](0, totals)
    parallel_for[_bump_totals](-5, totals)
    assert_equal(Int(totals.count), 0)


struct _Slots(Movable):
    """Shared state that owns heap memory, so a lifetime mistake would be a
    write into freed memory rather than into a dead stack slot."""

    var cells: List[Int64]

    def __init__(out self, n: Int):
        self.cells = List[Int64](length=n, fill=0)


def _square_into_own_slot(i: Int, mut s: _Slots) -> None:
    """One slot per task; no synchronisation needed."""
    s.cells[i] = Int64(i * i)


def test_parallel_for_typed_state_lives_across_the_join() raises:
    """`slots` is never mentioned between the call and the asserts in a way
    the opaque form would count as a use — with the typed form it does not
    have to be: as an argument it is alive for the whole call."""
    var n = 4096
    var slots = _Slots(n)
    parallel_for[_square_into_own_slot](n, slots)
    for i in range(n):
        assert_equal(Int(slots.cells[i]), i * i)


# ── allocator under threads ──────────────────────────────────────────────────


def _allocate_and_free(i: Int, ctx: OpaquePtr) -> None:
    """Build and drop Mojo `List`s and `String`s on a worker thread.

    This is the load-bearing assumption behind every worker in this library
    and in anything built on it: that the Mojo runtime allocator is safe to
    use concurrently. If it is not, this test is where it shows up.
    """
    var total = 0
    for round in range(20):
        var items = List[Int]()
        for k in range(200):
            items.append(k + round)
        var s = String()
        for k in range(8):
            s += String(items[k])
        total += len(items) + s.byte_length()
    i64_ptr(Int(ctx))[unsafe_offset=i] = Int64(total)


def test_allocator_is_usable_from_many_threads_at_once() raises:
    """8 workers allocating and freeing concurrently for 160 rounds each."""
    var n = 64
    var ctx = _cells(n)
    var cells = i64_ptr(Int(ctx))
    parallel_for[_allocate_and_free](n, ctx, num_workers=8)
    for i in range(n):
        assert_true(
            Int(cells[unsafe_offset=i]) > 0,
            String("task ", i, " produced no allocation work"),
        )
        assert_equal(Int(cells[unsafe_offset=i]), Int(cells[unsafe_offset=0]))
    _free_cells(ctx)


# ── WorkerPool ───────────────────────────────────────────────────────────────
#
# Context layout for every pool test below, in 64-bit cells:
#   [0]              shared tick counter, hammered by all workers
#   [1 .. 1 + n)     one slot per worker, written once as it returns
#
# The per-worker slot is what makes "the index each worker receives is unique
# and covers 0..n-1" a *number* rather than a hope: a duplicate index leaves one
# slot at 2 and another at 0.

comptime _P_TICKS: Int = 0
comptime _P_SLOTS: Int = 1


def _tick_until_stopped(worker: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    """Bump the shared counter until the stop flag is set, then claim a slot."""
    var ticks = AtomicCounter.at(Int(ctx) + _P_TICKS * 8)
    var cells = i64_ptr(Int(ctx))
    while not stop.is_set():
        _ = ticks.fetch_add(1)
    # Plain write to a slot only this worker touches; the join publishes it.
    cells[unsafe_offset=_P_SLOTS + worker] = (
        cells[unsafe_offset=_P_SLOTS + worker] + 1
    )


def _wait_for_ticks(ctx: OpaquePtr, target: Int) -> Int:
    """Spin (politely) until the tick counter reaches `target` or 5s elapse.

    Returns the counter value observed, so the caller can assert on it rather
    than on a timeout that may have expired for an unrelated reason.
    """
    var ticks = AtomicCounter.at(Int(ctx) + _P_TICKS * 8)
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < 5_000_000_000:
        if Int(ticks.load()) >= target:
            break
        yield_now()
    return Int(ticks.load())


def test_worker_pool_runs_until_stopped_then_joins() raises:
    """Four workers hammer a shared counter; `shutdown()` terminates."""
    var n = 4
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
    assert_equal(pool.num_workers(), n)
    assert_true(not pool.is_stopping())
    # Insist on real progress first, so "the pool stopped" is not vacuous.
    var observed = _wait_for_ticks(ctx, 10000)
    assert_true(
        observed >= 10000,
        String("workers only reached ", observed, " ticks in 5s"),
    )
    pool.shutdown()
    assert_true(pool.is_stopping())
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 1)
    _free_cells(ctx)


def test_worker_pool_stop_before_any_work_still_joins_cleanly() raises:
    """`request_stop()` racing the spawns: every worker still returns exactly
    once, whether it observed the flag on its first check or its millionth."""
    var n = 8
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
    pool.request_stop()
    pool.join()
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 1)
    _free_cells(ctx)


def test_worker_pool_indices_are_unique_and_cover_the_range() raises:
    """16 workers, 16 slots, each written exactly once."""
    var n = 16
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
    pool.shutdown()
    var total = 0
    for i in range(n):
        assert_equal(
            Int(cells[unsafe_offset=_P_SLOTS + i]),
            1,
            String(
                "slot ",
                i,
                " was claimed ",
                Int(cells[unsafe_offset=_P_SLOTS + i]),
                "x",
            ),
        )
        total += Int(cells[unsafe_offset=_P_SLOTS + i])
    assert_equal(total, n)
    _free_cells(ctx)


def test_worker_pool_single_worker() raises:
    """A one-worker pool is the degenerate case: it still starts, ticks, stops, and joins.
    """
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](1, ctx)
    assert_equal(pool.num_workers(), 1)
    var observed = _wait_for_ticks(ctx, 1000)
    assert_true(observed >= 1000, String("one worker managed ", observed))
    pool.shutdown()
    assert_equal(Int(cells[unsafe_offset=_P_SLOTS]), 1)
    _free_cells(ctx)


def test_worker_pool_rejects_a_worker_count_below_one() raises:
    """A pool with no workers is a silent no-op waiting to happen — refuse it
    rather than hand back something that looks like it is doing work."""
    var ctx = _cells(2)
    with assert_raises():
        var pool = WorkerPool.start[_tick_until_stopped](0, ctx)
        _ = pool^
    with assert_raises():
        var pool = WorkerPool.start[_tick_until_stopped](-3, ctx)
        _ = pool^
    _free_cells(ctx)


def _always_raises(x: Int) raises -> Int:
    raise Error("boom in worker ", x)


def _fail_and_record(worker: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    """A worker whose body hits an error and returns without ever consulting
    the stop flag. `WorkerFn` is non-raising — pthread has no exception
    channel — so publishing to a cell is the only honest report."""
    var cells = i64_ptr(Int(ctx))
    try:
        _ = _always_raises(worker)
    except:
        cells[unsafe_offset=_P_SLOTS + worker] = 0xBAD
        return
    cells[unsafe_offset=_P_SLOTS + worker] = 1


def test_worker_pool_worker_that_fails_does_not_wedge_the_join() raises:
    """Every worker exits early on an error; `shutdown()` must still return."""
    var n = 4
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_fail_and_record](n, ctx)
    pool.shutdown()
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 0xBAD)
    _free_cells(ctx)


def test_worker_pool_dropped_without_shutdown_still_stops_and_joins() raises:
    """The destructor is the safety net: it sets the flag, joins, then frees
    the header. Without that, freeing the header under live workers would be a
    use-after-free — and Mojo's last-use destruction makes it easy to reach."""
    var n = 4
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
    var observed = _wait_for_ticks(ctx, 1000)
    assert_true(observed >= 1000, String("workers managed ", observed))
    _ = pool^  # destroy here, deterministically — no shutdown() call
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 1)
    _free_cells(ctx)


def test_worker_pool_stop_flag_can_be_set_by_a_third_party() raises:
    """`stop_flag()` hands out a view, so anything holding the address — a
    signal handler, another thread, an FFI callback — can end the pool."""
    var n = 3
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
    var observed = _wait_for_ticks(ctx, 1000)
    assert_true(observed >= 1000, String("workers managed ", observed))
    # Rebuilt from a bare address, exactly as a stranger would.
    var elsewhere = AtomicFlag.at(pool.stop_flag().address())
    assert_true(not elsewhere.is_set())
    elsewhere.set()
    pool.join()  # no request_stop() here — the third party already asked
    for i in range(n):
        assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 1)
    _free_cells(ctx)


# ── TypedPool ────────────────────────────────────────────────────────────────
#
# The typed pool's whole claim is that the state it was started over cannot be
# destroyed before the last worker has stopped reading it, because the pool's
# type carries the state's origin. The first three tests below are the typed
# spellings of the `WorkerPool` tests above; the last one is the claim itself.


struct _PoolTotals(Movable):
    """Shared state for the typed pool: one hammered counter, one slot per
    worker so a duplicated index shows up as a wrong number."""

    var ticks: Int64
    var slots: List[Int64]

    def __init__(out self, n: Int):
        self.ticks = 0
        self.slots = List[Int64](length=n, fill=0)


def _tick_typed(worker: Int, mut t: _PoolTotals, stop: AtomicFlag) -> None:
    """The typed twin of `_tick_until_stopped`."""
    var ticks = AtomicCounter.at(Int(Pointer(to=t.ticks)))
    while not stop.is_set():
        _ = ticks.fetch_add(1)
    # Plain write to a slot only this worker touches; the join publishes it.
    t.slots[worker] = t.slots[worker] + 1


def _wait_for_typed_ticks(mut t: _PoolTotals, target: Int) -> Int:
    """Spin (politely) until the tick counter reaches `target` or 5s elapse."""
    var ticks = AtomicCounter.at(Int(Pointer(to=t.ticks)))
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < 5_000_000_000:
        if Int(ticks.load()) >= target:
            break
        yield_now()
    return Int(ticks.load())


def test_typed_pool_runs_until_stopped_then_joins() raises:
    """Four workers hammer a counter inside the caller's struct; `shutdown()`
    terminates and the state is still readable afterwards."""
    var n = 4
    var totals = _PoolTotals(n)
    var pool = TypedPool.start[_tick_typed](n, totals)
    assert_equal(pool.num_workers(), n)
    assert_true(not pool.is_stopping())
    var observed = _wait_for_typed_ticks(totals, 10000)
    assert_true(
        observed >= 10000,
        String("workers only reached ", observed, " ticks in 5s"),
    )
    pool.shutdown()
    assert_true(pool.is_stopping())
    # Read back through the pool as well as through the local: both are the
    # same object, and `state()` is what a caller who never named the local
    # again would use.
    assert_true(Int(pool.state().ticks) >= 10000)
    assert_equal(Int(totals.ticks), Int(pool.state().ticks))
    for i in range(n):
        assert_equal(Int(totals.slots[i]), 1)


def test_typed_pool_indices_are_unique_and_cover_the_range() raises:
    """16 workers, 16 slots, each written exactly once."""
    var n = 16
    var totals = _PoolTotals(n)
    var pool = TypedPool.start[_tick_typed](n, totals)
    pool.shutdown()
    var total = 0
    for i in range(n):
        assert_equal(
            Int(totals.slots[i]),
            1,
            String("slot ", i, " was claimed ", Int(totals.slots[i]), "x"),
        )
        total += Int(totals.slots[i])
    assert_equal(total, n)


def test_typed_pool_single_worker() raises:
    """A one-worker typed pool still starts, ticks, stops, and joins."""
    var totals = _PoolTotals(1)
    var pool = TypedPool.start[_tick_typed](1, totals)
    assert_equal(pool.num_workers(), 1)
    var observed = _wait_for_typed_ticks(totals, 1000)
    assert_true(observed >= 1000, String("one worker managed ", observed))
    pool.shutdown()
    assert_equal(Int(totals.slots[0]), 1)


struct _Poisoned(Movable):
    """Owns one heap cell and poisons it on destruction.

    `__deinit__` writes -1 and *leaks* the block rather than freeing it: a
    write into a freed block is undefined behaviour that may or may not show
    up, while a write into a poisoned, leaked block is a number a worker can
    read and report. The leak is deliberate and bounded — one cell per test.

    `seen_poison` is bumped by any worker that reads -1 out of the cell, which
    is precisely the event the origin on `TypedPool` is supposed to make
    impossible.
    """

    var cell: Pointer[Int64, MutUntrackedOrigin]
    var seen_poison: Int64
    var reads: Int64

    def __init__(out self):
        self.cell = Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=Int(unsafe_alloc[Int64](1))
        )
        self.cell[] = 0xC0FFEE
        self.seen_poison = 0
        self.reads = 0

    def value(self) -> Int64:
        """The cell's contents, read through `self`.

        A method rather than `p.cell[]` at the call site on purpose: copying
        the untracked pointer field out is the struct's last *use*, so the
        drop lands between the copy and the deref and the caller reads the
        poison for reasons that have nothing to do with threads. Borrowing
        through `self` keeps the struct alive across the read.
        """
        return self.cell[]

    def __deinit__(deinit self):
        self.cell[] = -1


def _read_until_stopped(
    worker: Int, mut p: _Poisoned, stop: AtomicFlag
) -> None:
    """Read the heap cell for as long as the pool runs, counting poison.

    Every read goes through `p`, so if `p` were destroyed early this loop
    would be reading a poisoned cell for the rest of its life.
    """
    var reads = AtomicCounter.at(Int(Pointer(to=p.reads)))
    var poison = AtomicCounter.at(Int(Pointer(to=p.seen_poison)))
    while not stop.is_set():
        if p.value() == -1:
            _ = poison.fetch_add(1)
        _ = reads.fetch_add(1)
    # One last read after the flag, which is the closest a worker gets to the
    # join: if the state dies at the caller's last mention, this is where it
    # would already be poisoned.
    if p.value() == -1:
        _ = poison.fetch_add(1)


def test_typed_pool_state_outlives_the_pool() raises:
    """The point of the design, as a number.

    `poisoned` is never mentioned after `TypedPool.start` — no later use props
    it up. With an `OpaquePtr` the compiler would destroy it on the `start`
    line and the workers would spend their whole lives reading -1. Because the
    pool's type carries the origin, the destruction cannot happen before the
    pool value is destroyed, and the pool's destructor joins first.
    """
    var poisoned = _Poisoned()
    var pool = TypedPool.start[_read_until_stopped](4, poisoned)
    # From here on, only `pool` — deliberately.
    var seen = AtomicCounter.at(Int(Pointer(to=pool.state().reads)))
    var t0 = perf_counter_ns()
    while perf_counter_ns() - t0 < 5_000_000_000:
        if Int(seen.load()) > 100_000:
            break
        yield_now()
    pool.shutdown()
    var reads = Int(pool.state().reads)
    var poison = Int(pool.state().seen_poison)
    var cell = Int(pool.state().value())
    assert_true(
        reads > 100_000, String("workers only managed ", reads, " reads")
    )
    assert_equal(poison, 0, String(poison, " reads saw the poisoned cell"))
    assert_equal(cell, 0xC0FFEE, "the cell was already poisoned at shutdown")


def test_typed_pool_dropped_without_shutdown_still_stops_and_joins() raises:
    """Same guarantee on the drop path: the pool is destroyed rather than shut
    down, and the state is still live when the last worker reads it."""
    var totals = _PoolTotals(4)
    var pool = TypedPool.start[_tick_typed](4, totals)
    var observed = _wait_for_typed_ticks(totals, 1000)
    assert_true(observed >= 1000, String("workers managed ", observed))
    _ = pool^  # destroy here, deterministically — no shutdown() call
    for i in range(4):
        assert_equal(Int(totals.slots[i]), 1)


def test_typed_pool_rejects_a_worker_count_below_one() raises:
    """The `n >= 1` check is `WorkerPool`'s and still fires through the wrapper.
    """
    var totals = _PoolTotals(1)
    with assert_raises():
        var pool = TypedPool.start[_tick_typed](0, totals)
        _ = pool^


# ── stress ───────────────────────────────────────────────────────────────────


def test_stress_fifty_worker_pools() raises:
    """50 pools of 4, started and shut down back to back. Catches a leaked
    header, a leaked pthread handle, or a join that only works the first time.
    """
    var n = 4
    var ctx = _cells(1 + n)
    var cells = i64_ptr(Int(ctx))
    for _ in range(50):
        for i in range(1 + n):
            cells[unsafe_offset=i] = 0
        var pool = WorkerPool.start[_tick_until_stopped](n, ctx)
        pool.shutdown()
        for i in range(n):
            assert_equal(Int(cells[unsafe_offset=_P_SLOTS + i]), 1)
    _free_cells(ctx)


def test_stress_five_hundred_sequential_threads() raises:
    """Spawn and join 500 threads one after another. Catches leaked pthread
    handles and any per-thread resource the wrapper forgets to release."""
    var ctx = _cells(2)
    var cells = i64_ptr(Int(ctx))
    for _ in range(500):
        cells[unsafe_offset=0] = 0
        var t = ThreadHandle.spawn[_write_marker](ctx)
        t.join()
        assert_equal(Int(cells[unsafe_offset=0]), 0xC0FFEE)
    _free_cells(ctx)


def test_stress_fifty_parallel_for_rounds() raises:
    """50 back-to-back `parallel_for` calls, each allocating and freeing its
    own worker block."""
    var n = 256
    var ctx = _cells(n + 2)
    var cells = i64_ptr(Int(ctx))
    for _ in range(50):
        for i in range(n + 2):
            cells[unsafe_offset=i] = 0
        parallel_for[_square_into_slot](n, ctx, num_workers=4)
        assert_equal(Int(cells[unsafe_offset=n - 1]), (n - 1) * (n - 1))
    _free_cells(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
