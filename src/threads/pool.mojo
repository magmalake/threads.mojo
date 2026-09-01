"""`WorkerPool` — N long-lived threads that run until you tell them to stop.

`parallel_for` is the right shape when the work is a *bounded set of indices*:
it hands out tasks, drains the queue, and joins. A server is the other shape —
the work never runs out, each worker loops on its own until something external
says stop. That pattern is already expressible with `spawn_n` + `AtomicFlag` +
`ThreadGroup.join_all()`, and this struct is exactly those three bundled so
every caller does not re-derive the header layout and the free-after-join rule.

It is deliberately thin. There is no queue here, no scheduler, no task type: a
`WorkerPool` starts `n` symmetric threads, gives each one an index and a shared
stop flag, and joins them. Whatever the workers pull from — a channel, a
socket, an FFI `next()` call — is the caller's business and travels through the
`ctx` pointer.

```mojo
from threads import AtomicCounter, AtomicFlag, OpaquePtr, WorkerPool

def tally(worker: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    var counter = AtomicCounter.at(Int(ctx))
    while not stop.is_set():
        _ = counter.fetch_add(1)

def main() raises:
    var pool = WorkerPool.start[tally](n=4, ctx=my_ctx)
    ...
    pool.shutdown()          # request_stop + join
```

## The stop flag is cooperative

`request_stop()` is a release-store to one cell. Nothing preempts a worker:
it observes the flag the next time it looks. Two consequences worth stating
plainly, because both have surprised people:

- **A worker blocked in a syscall will not observe the flag until that syscall
  returns.** A thread parked in `recv`, `accept`, or an FFI call that blocks
  on a channel stays parked; setting the flag does not interrupt it. If your
  workers block, you need a *wake* mechanism on the thing they block on — a
  sentinel message, a closed channel, a timeout on the blocking call — and
  `request_stop()` is only the second half of the handshake. `restate.mojo`'s
  served mode is the worked example: its shim grew an `rst_stop` that makes the
  blocking `rst_next` return, precisely because the flag alone could not reach
  a thread inside the Rust receiver.
- **A worker that never checks the flag never stops**, and `join()` therefore
  never returns. Loop on `not stop.is_set()`.

## Lifetimes

The pool allocates one small header block on the heap — the index counter, the
stop cell, and the user context address — and hands *that* to every thread.
The header is freed only after every worker has been joined. The pool's
destructor enforces this: dropping a `WorkerPool` requests a stop, joins, and
only then frees. That is not politeness, it is the "Mojo destroys a value at
its last *use*" trap (`.unsafe_ptr()` counts as a use) with the sharp edge
filed off — a pool whose last mention is `pool.request_stop()` is destroyed on
that line, and the destructor's join is what keeps the header alive under the
still-running workers.

The `ctx` you pass in is *not* owned by the pool. It must outlive the pool, and
because the destructor joins, "outlives the pool value" is sufficient.
"""

from std.memory import alloc

from .atomic import AtomicCounter, AtomicFlag, atomic_store_relaxed
from .ffi import I64Ptr, OpaquePtr, i64_ptr, opaque_ptr
from .parallel import ThreadGroup, spawn_n
from .thread import num_cpus


comptime WorkerFn = def(Int, OpaquePtr, AtomicFlag) thin -> None
"""A worker body: its index in `[0, n)`, the shared user context, and the
pool's stop flag.

Thin (non-capturing) and non-raising, for the same reasons `parallel.WorkFn`
is: it is reached through a C function pointer, and pthread has no exception
channel. Report failures through a cell in `ctx` — see `threads.parallel`'s
module docstring for the error-cell pattern."""


# Header layout, in 64-bit cells. One allocation, not two: the stop flag lives
# in the header rather than in a cell of its own, so there is exactly one
# lifetime to reason about and one `unsafe_free` on exactly one path.
comptime _SLOT_NEXT_INDEX: Int = 0
"""Ticket counter — each worker does one `fetch_add(1)` to learn its index."""
comptime _SLOT_STOP: Int = 1
"""The stop flag's cell."""
comptime _SLOT_USER_CTX: Int = 2
"""Address of the caller's context, read once per worker."""
comptime _POOL_CELLS: Int = 3


def _pool_worker[work: WorkerFn](arg: OpaquePtr) -> OpaquePtr:
    """Thread body: claim an index, rebuild the shared pieces, run `work`."""
    var cells = i64_ptr(Int(arg))
    # One RMW, once per worker: workers are symmetric, so the only thing an
    # index is for is letting a worker address its own slot in the caller's
    # context.
    var index = Int(
        AtomicCounter.at(Int(arg) + _SLOT_NEXT_INDEX * 8).fetch_add(1)
    )
    var stop = AtomicFlag.at(Int(arg) + _SLOT_STOP * 8)
    # Plain read: this cell was written before `pthread_create`, and thread
    # creation is a happens-before edge.
    var user_ctx = opaque_ptr(Int(cells[_SLOT_USER_CTX]))
    work(index, user_ctx, stop)
    return arg


struct WorkerPool(Movable):
    """`n` symmetric threads running the same body until asked to stop.

    Move-only, like everything else in this library that owns threads: the
    pool owns the header block and the `ThreadGroup`, and there must be
    exactly one join and exactly one free.
    """

    var _group: ThreadGroup
    """The threads, in spawn order."""

    var _header: I64Ptr
    """The shared header block — freed only after `_group` is fully joined."""

    var _n: Int
    """How many threads were started."""

    def __init__(out self, var group: ThreadGroup, header: I64Ptr, n: Int):
        """Adopt an already-spawned group. Use `WorkerPool.start` instead.

        Args:
            group: The spawned threads.
            header: The heap header block backing them.
            n: The worker count.
        """
        self._group = group^
        self._header = header
        self._n = n

    @staticmethod
    def start[work: WorkerFn](n: Int, ctx: OpaquePtr) raises -> WorkerPool:
        """Start `n` threads, each running `work(index, ctx, stop_flag)`.

        Every worker gets the same `ctx` and the same stop flag; the index is
        the only thing that distinguishes them, and it is drawn from a counter
        rather than baked in, so the workers are genuinely symmetric.

        Parameters:
            work: The worker body — thin and non-raising.

        Args:
            n: How many threads to start. Must be at least 1.
            ctx: The shared context handed to every worker. Must outlive the
                pool; since the destructor joins, outliving the pool value is
                enough.

        Returns:
            A running pool. Call `shutdown()` when done — or just drop it,
            which does the same thing.

        Raises:
            Error: If `n < 1`, or if any `pthread_create` fails. On a failed
                spawn the threads that did start are joined before the error
                propagates, and the header is freed.
        """
        if n < 1:
            raise Error("WorkerPool.start needs at least one worker, got ", n)

        # Heap allocated so its address survives any move of this frame's
        # locals, and so the workers can keep reading it after `start` returns.
        var header = alloc[Int64](_POOL_CELLS)
        atomic_store_relaxed(
            i64_ptr(Int(header) + _SLOT_NEXT_INDEX * 8), Int64(0)
        )
        atomic_store_relaxed(i64_ptr(Int(header) + _SLOT_STOP * 8), Int64(0))
        header[_SLOT_USER_CTX] = Int64(Int(ctx))

        var group: ThreadGroup
        try:
            group = spawn_n[_pool_worker[work]](n, opaque_ptr(Int(header)))
        except e:
            # `spawn_n` joins whatever it started before raising, so nothing
            # can still be reading the header here.
            header.unsafe_free()
            raise Error("WorkerPool.start could not start workers: ", String(e))
        return WorkerPool(group^, header, n)

    @staticmethod
    def start[work: WorkerFn](ctx: OpaquePtr) raises -> WorkerPool:
        """Start one thread per logical CPU.

        Parameters:
            work: The worker body.

        Args:
            ctx: The shared context.

        Returns:
            A running pool of `num_cpus()` workers.

        Raises:
            Error: If any `pthread_create` fails.
        """
        return WorkerPool.start[work](num_cpus(), ctx)

    @always_inline
    def num_workers(self) -> Int:
        """How many threads this pool started.

        Returns:
            The worker count, whether or not they have been joined.
        """
        return self._n

    @always_inline
    def stop_flag(self) -> AtomicFlag:
        """The shared stop flag, as a view.

        Useful when something other than the pool's owner needs to be able to
        ask for a stop — store `flag.address()` in your own context and rebuild
        it with `AtomicFlag.at`.

        Returns:
            A view of the flag cell. Valid only while the pool is alive.
        """
        return AtomicFlag.at(Int(self._header) + _SLOT_STOP * 8)

    @always_inline
    def request_stop(self):
        """Ask every worker to return: release-store 1 to the stop flag.

        Returns immediately. Cooperative — see the module docstring: a worker
        blocked in a syscall does not observe this until it returns.
        """
        self.stop_flag().set()

    def is_stopping(self) -> Bool:
        """Whether a stop has been requested.

        Returns:
            True after `request_stop()` (or `shutdown()`) has been called.
        """
        return self.stop_flag().is_set()

    def join(mut self) raises:
        """Wait for every worker to return.

        Does **not** set the stop flag — call `request_stop()` first, or use
        `shutdown()`. Joining a pool whose workers loop on the flag without
        having requested a stop hangs forever, which is a bug in the caller,
        not something this method can rescue.

        Idempotent: `ThreadHandle.join` zeroes its id, so a second call is a
        no-op.

        Raises:
            Error: The first `pthread_join` failure, after every thread has
                had a join attempted.
        """
        self._group.join_all()

    def shutdown(mut self) raises:
        """`request_stop()` then `join()` — the ordinary way to end a pool.

        Raises:
            Error: If a join fails. The stop has already been requested by
                then, so the workers are on their way out regardless.
        """
        self.request_stop()
        self._group.join_all()

    def pin_all(self) raises:
        """Pin worker `i` to CPU `i % num_cpus()` (a no-op on macOS).

        Raises:
            Error: On Linux if the affinity call fails.
        """
        self._group.pin_all()

    def __del__(deinit self):
        """Stop, join, then free the header — in that order, always.

        A pool that is simply dropped still shuts down cleanly. Freeing the
        header while a worker could still read it would be a use-after-free,
        and Mojo's last-use destruction makes that easy to trigger by accident,
        so the destructor pays for the join rather than trusting the caller to
        have done it. If `shutdown()` was already called, the joins are no-ops
        and this costs nothing.

        A join failure here cannot be reported — a destructor cannot raise —
        so it is swallowed. Call `shutdown()` explicitly if you want to see it.
        """
        self.stop_flag().set()
        try:
            self._group.join_all()
        except:
            pass
        self._header.unsafe_free()
