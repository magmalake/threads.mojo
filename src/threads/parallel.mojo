"""`parallel_for` — the reason this tin exists — plus the `spawn_n` / `join_all`
layer underneath it.

```mojo
from threads import parallel_for, OpaquePtr, i64_ptr

def square_chunk(i: Int, ctx: OpaquePtr) -> None:
    var cells = i64_ptr(Int(ctx))
    cells[i] = cells[i] * cells[i]

def main() raises:
    ...
    parallel_for[square_chunk](n_tasks=64, ctx=my_ctx)
```

`parallel_for` starts `min(n_tasks, num_workers or num_cpus())` kernel threads.
Each one loops on `AtomicCounter.fetch_add(1)` and runs `work(i, ctx)` for every
index it draws, stopping when the counter passes `n_tasks`. Every thread is
joined before the call returns, so by construction nothing the caller owns can
be destroyed while a worker still reads it.

## The send contract, restated

- **`work` is thin and non-raising.** It is called through a C function
  pointer, so it cannot capture, and pthread has no exception channel.
- **All shared state travels through `ctx`.** One pointer, for every task.
- **The caller guarantees `ctx` outlives the call.** `parallel_for` joins
  before returning, so "outlives the call" is all that is needed — but see
  `threads.thread` on Mojo's last-use destruction if you build the context out
  of a local value.
- **No allocator assumption beyond what the tests prove.** Workers *may*
  allocate and free (`List`, `String`, `alloc`) concurrently; the
  `allocator under threads` test in `tests/test_threads.mojo` is the evidence,
  and it is run on all four CI legs.

## Scheduling

The queue is a single shared counter, so it is dynamically load balanced — a
slow task does not stall the others — at a cost of one atomic RMW per task.
Make tasks chunky: prefer `n_tasks` in the tens-to-thousands over one task per
element. `n_tasks` may exceed the worker count freely; that is the normal case.

## Reporting failures out of a task

`work` cannot raise. The pattern that fits the model is an error cell in the
context: an `AtomicFlag` plus a fixed-size array of error codes, one slot per
worker or per task.

```mojo
# ctx layout, in 64-bit cells:
#   [0]        error flag (0 = clean)
#   [1..n+1]   per-task error code
def task(i: Int, ctx: OpaquePtr) -> None:
    var flag = AtomicFlag.at(ctx, 0)
    var cells = i64_ptr(Int(ctx))
    if something_went_wrong:
        cells[1 + i] = MY_ERROR_CODE   # plain write, this task owns the slot
        flag.set()                      # release: publishes the write above
        return

# after parallel_for returns:
if flag.is_set():                       # acquire
    raise Error("task ", first_bad, " failed with ", cells[1 + first_bad])
```

The flag's release/acquire pairing is what makes the plain per-slot writes
visible to the caller; the join at the end of `parallel_for` also gives you
that, so the flag is really there to let you skip the scan when nothing failed.
"""

from std.memory import alloc

from .atomic import AtomicCounter, atomic_store_relaxed
from .ffi import OpaquePtr, I64Ptr, i64_ptr, opaque_ptr
from .thread import StartFn, ThreadHandle, num_cpus


comptime WorkFn = def (Int, OpaquePtr) thin -> None
"""A `parallel_for` task body: takes its task index and the shared context."""


# Context-block layout for the workers, in 64-bit cells.
comptime _SLOT_COUNTER: Int = 0
comptime _SLOT_N_TASKS: Int = 1
comptime _SLOT_USER_CTX: Int = 2
comptime _PAR_CELLS: Int = 3


# ── ThreadGroup ──────────────────────────────────────────────────────────────


struct ThreadGroup(Movable, Sized):
    """A batch of threads spawned together and joined together.

    Move-only for the same reason `ThreadHandle` is: the group owns the threads
    and there must be exactly one join.
    """

    var _handles: List[ThreadHandle]
    """One handle per live thread; each is zeroed as it is joined."""

    def __init__(out self):
        """An empty group."""
        self._handles = List[ThreadHandle]()

    @always_inline
    def __len__(self) -> Int:
        """The number of threads in the group.

        Returns:
            The thread count, including already-joined ones.
        """
        return len(self._handles)

    def append(mut self, var handle: ThreadHandle):
        """Take ownership of one more thread.

        Args:
            handle: The handle to adopt.
        """
        self._handles.append(handle^)

    def join_all(mut self) raises:
        """Join every thread in the group.

        Joins are attempted in spawn order. If one fails, the remaining threads
        are still joined before the error is re-raised — leaving live threads
        behind while the caller unwinds and frees their context would be far
        worse than a late error.

        Raises:
            Error: The first `pthread_join` failure, after all joins have been
                attempted.
        """
        var first_error = String()
        var failed = False
        # Indexed, not `for ref h in ...`: `List` iteration requires a
        # `Copyable` element on both toolchains, and `ThreadHandle` is
        # deliberately move-only.
        for i in range(len(self._handles)):
            try:
                self._handles[i].join()
            except e:
                if not failed:
                    failed = True
                    first_error = String(e)
        if failed:
            raise Error(first_error)

    def pin_all(self) raises:
        """Pin thread `i` to CPU `i % num_cpus()`.

        A no-op on macOS (see `ThreadHandle.pin_to_cpu`).

        Raises:
            Error: On Linux if the affinity call fails.
        """
        var ncpu = num_cpus()
        for i in range(len(self._handles)):
            self._handles[i].pin_to_cpu(i % ncpu)


def join_all(mut group: ThreadGroup) raises:
    """Free-function spelling of `ThreadGroup.join_all`.

    Args:
        group: The group to join.

    Raises:
        Error: The first join failure.
    """
    group.join_all()


def spawn_n[
    start: StartFn
](n: Int, ctx: OpaquePtr, stride: Int = 0) raises -> ThreadGroup:
    """Start `n` threads running `start`, thread `i` receiving
    `ctx + i * stride`.

    With the default `stride` of 0 every thread gets the same pointer, which is
    what you want for a shared context. A non-zero stride carves the context
    into equal per-thread slices without a second allocation.

    If a spawn fails partway through, the threads already started are joined
    before the error propagates, so the caller never has to reason about a
    half-started group.

    Parameters:
        start: The thread body — thin and non-raising.

    Args:
        n: How many threads to start. `n <= 0` yields an empty group.
        ctx: Base context pointer.
        stride: Bytes between consecutive threads' context pointers.

    Returns:
        A `ThreadGroup` the caller must `join_all()`.

    Raises:
        Error: If any `pthread_create` fails.
    """
    var group = ThreadGroup()
    if n <= 0:
        return group^
    for i in range(n):
        try:
            group.append(
                ThreadHandle.spawn[start](opaque_ptr(Int(ctx) + i * stride))
            )
        except e:
            # Never leave threads running against a context the caller is
            # about to reclaim.
            group.join_all()
            raise Error("spawn_n failed at thread ", i, ": ", String(e))
    return group^


# ── parallel_for ─────────────────────────────────────────────────────────────


def _parallel_worker[work: WorkFn](arg: OpaquePtr) -> OpaquePtr:
    """Pull indices off the shared counter until the queue is drained."""
    var cells = i64_ptr(Int(arg))
    var counter = AtomicCounter.at(Int(arg) + _SLOT_COUNTER * 8)
    # Plain reads: these cells were written before pthread_create, and thread
    # creation is a happens-before edge, so no atomic is needed here.
    var n_tasks = Int(cells[_SLOT_N_TASKS])
    var user_ctx = opaque_ptr(Int(cells[_SLOT_USER_CTX]))
    while True:
        var i = Int(counter.fetch_add(1))
        if i >= n_tasks:
            break
        work(i, user_ctx)
    return arg


def parallel_for[
    work: WorkFn
](n_tasks: Int, ctx: OpaquePtr, num_workers: Int = 0) raises:
    """Run `work(i, ctx)` for every `i` in `[0, n_tasks)`, across threads.

    Starts `min(n_tasks, num_workers or num_cpus())` threads, hands out indices
    from one shared atomic counter, and joins every thread before returning.
    The calling thread does not run tasks; it blocks in the joins.

    Parameters:
        work: The task body. Thin (non-capturing) and non-raising — see the
            module docstring for reporting failures out of a task.

    Args:
        n_tasks: How many indices to hand out. Zero or negative does nothing.
        ctx: The shared context pointer given to every task. Must outlive the
            call; since the call joins before returning, that is automatic for
            anything the caller keeps alive across it.
        num_workers: Thread count. `0` (the default) means `num_cpus()`. Values
            above `n_tasks` are clamped down, because a thread with no index to
            draw is pure overhead.

    Raises:
        Error: If a thread fails to start or to join. The worker context block
            is freed on every path *after* the joins, never before.
    """
    if n_tasks <= 0:
        return

    var workers = num_workers if num_workers > 0 else num_cpus()
    if workers > n_tasks:
        workers = n_tasks
    if workers < 1:
        workers = 1

    # One shared block of 64-bit cells, heap allocated so its address survives
    # any move of this frame's locals, and freed only after every join.
    var block = alloc[Int64](_PAR_CELLS)
    var cells = i64_ptr(Int(block))
    atomic_store_relaxed(i64_ptr(Int(block) + _SLOT_COUNTER * 8), 0)
    cells[_SLOT_N_TASKS] = Int64(n_tasks)
    cells[_SLOT_USER_CTX] = Int64(Int(ctx))
    var shared = opaque_ptr(Int(block))

    var group: ThreadGroup
    try:
        group = spawn_n[_parallel_worker[work]](workers, shared)
    except e:
        block.unsafe_free()
        raise Error("parallel_for could not start workers: ", String(e))

    # The joins must happen before the block is freed — every worker is still
    # reading `n_tasks` and the counter out of it.
    try:
        group.join_all()
    except e:
        block.unsafe_free()
        raise Error("parallel_for could not join workers: ", String(e))
    block.unsafe_free()
