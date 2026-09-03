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
from threads import AtomicCounter, AtomicFlag, TypedPool

@fieldwise_init
struct Totals(Copyable, Movable):
    var served: Int64

def serve(worker: Int, mut totals: Totals, stop: AtomicFlag) -> None:
    var served = AtomicCounter.at(Int(Pointer(to=totals.served)))
    while not stop.is_set():
        _ = served.fetch_add(1)

def main() raises:
    var totals = Totals(0)
    var pool = TypedPool.start[serve](n=4, state=totals)
    ...
    pool.shutdown()          # request_stop + join
    print(pool.state().served)
```

That is the **typed** form. The state is a value the caller owns, the workers
receive it by `mut`, and the `void *` erasure pthread demands happens twice
inside this file and nowhere else. See "Lifetimes" below for why the pool's
*type* mentions the state's origin.

The **untyped** form is the one underneath, for when the context is a
hand-laid-out block of cells rather than a struct, or a pointer the compiler
could not track anyway:

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
because the destructor joins, "outlives the pool value" is sufficient. With
`WorkerPool` that sentence is a *rule for the caller*: an `OpaquePtr` carries no
origin, so nothing checks it.

`TypedPool` turns the same sentence into something the compiler enforces, and
the mechanism is worth stating because a pool is not a `parallel_for`. In
`parallel_for` the typed overload takes `ref[origin] state` and that is the
whole story: the state is an *argument*, so it is alive for the duration of the
call, and the call joins before it returns. A pool outlives its `start` call, so
a `ref` argument on `start` alone buys nothing — the borrow ends when `start`
returns and the workers are still running.

So the origin travels out of `start` on the returned value: `TypedPool[T,
origin]` names it in its type and holds a `Pointer[T, origin]` in a field. A
live `TypedPool` value is therefore a live use of the origin it was started
over, and Mojo will not destroy `state` while the pool value exists. The pool's
own destructor sets the flag, joins every worker, and only then frees its
header — so the last worker's last read of `state` happens strictly before
`state` is destroyed. "Outlives the pool value" is still exactly the guarantee
needed; it is now a guarantee rather than a request.

The failure this rules out is not hypothetical, and it is silent: Mojo destroys
a value at its **last use**, and an untracked pointer is not a use. Build a
context out of a local, hand its address to an untyped pool, and never mention
the local again, and the local is destroyed at the line where you took its
address — with the workers still reading it, and no diagnostic. The
`typed pool state outlives the pool` test in `tests/test_threads.mojo` pins the
opposite: it poisons the state in its destructor and asserts no worker ever
saw the poison.
"""

from std.memory.alloc import unsafe_alloc

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

comptime PoolTaskFn[T: AnyType] = def(Int, mut T, AtomicFlag) thin -> None
"""A worker body, typed form: its index in `[0, n)`, mutable access to the
shared state, and the pool's stop flag.

Every worker gets the same `T`, at the same time, for as long as the pool runs.
Thin and non-raising, exactly like `WorkerFn` — the typing changes what the
worker is handed, not how it is reached."""


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
    var user_ctx = opaque_ptr(Int(cells[unsafe_offset=_SLOT_USER_CTX]))
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
        var header = unsafe_alloc[Int64](_POOL_CELLS)
        atomic_store_relaxed(
            i64_ptr(Int(header) + _SLOT_NEXT_INDEX * 8), Int64(0)
        )
        atomic_store_relaxed(i64_ptr(Int(header) + _SLOT_STOP * 8), Int64(0))
        header[unsafe_offset=_SLOT_USER_CTX] = Int64(Int(ctx))

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

    def __deinit__(deinit self):
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


# ── TypedPool ────────────────────────────────────────────────────────────────


def _typed_pool_worker[
    T: AnyType, task: PoolTaskFn[T]
](index: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    """Turn the opaque pointer back into a `T` and call the typed worker body.

    The untracked origin here is the erasure `WorkerFn` forces, and it is sound
    only because `TypedPool` holds the real origin on the value that owns these
    threads — see the module docstring.
    """
    task(
        index,
        Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(ctx))[],
        stop,
    )


struct TypedPool[T: AnyType, origin: MutOrigin](Movable):
    """A `WorkerPool` whose workers all share one `T`, held alive by `origin`.

    `WorkerPool` takes an `OpaquePtr` and asks the caller to keep the pointee
    alive; this wraps it and makes the compiler do the keeping. The `origin`
    parameter is the whole mechanism: a `TypedPool[T, origin]` value is a use of
    the origin it was started over, so `state` cannot be destroyed while the
    pool value is alive, and the pool's destructor joins every worker before it
    returns. A pool outlives its `start` call, which is why the origin has to
    ride on the *type* rather than only on `start`'s `ref` argument — see the
    module docstring's "Lifetimes" section for the version with the reasoning.

    Move-only, like the `WorkerPool` it owns: exactly one join, exactly one
    free. Everything else is a straight delegation, with the same names and
    signatures, so switching between the two forms is a change of construction
    and nothing more.

    Parameters:
        T: The shared state type.
        origin: The origin of the state the pool was started over.
    """

    var _pool: WorkerPool
    """The untyped pool doing the actual work."""

    var _state: Pointer[Self.T, Self.origin]
    """The state, as an origin-carrying pointer.

    Read for its address when starting, and dereferenced by `state()`. Its real
    job is being a field of this type at all: it is what ties the pool value's
    lifetime to the caller's `state`.
    """

    def __init__(
        out self, var pool: WorkerPool, state: Pointer[Self.T, Self.origin]
    ):
        """Adopt an already-started pool. Use `TypedPool.start` instead.

        Args:
            pool: The running untyped pool.
            state: A pointer to the state its workers were given.
        """
        self._pool = pool^
        self._state = state

    @staticmethod
    def start[
        task: PoolTaskFn[Self.T]
    ](n: Int, ref[Self.origin] state: Self.T) raises -> Self:
        """Start `n` threads, each running `task(index, state, stop_flag)`.

        `T` and `origin` are inferred from `state`, so the call site is
        `TypedPool.start[task](n, state)` with no parameters written out.
        Because `state` is taken by mutable `ref`, an immutable binding or a
        temporary is refused here rather than dangling later.

        Parameters:
            task: The worker body — thin and non-raising.

        Args:
            n: How many threads to start. Must be at least 1.
            state: The value every worker receives by `mut`. Guarding
                concurrent access to it is the worker's job — atomics, a
                `Mutex`, or per-worker slots. The returned pool carries its
                origin, so it is kept alive until the pool is destroyed.

        Returns:
            A running pool. Call `shutdown()` when done — or just drop it,
            which does the same thing.

        Raises:
            Error: If `n < 1`, or if any `pthread_create` fails.
        """
        var ptr = Pointer(to=state)
        var pool = WorkerPool.start[_typed_pool_worker[Self.T, task]](
            n, opaque_ptr(Int(ptr))
        )
        return Self(pool^, ptr)

    @staticmethod
    def start[
        task: PoolTaskFn[Self.T]
    ](ref[Self.origin] state: Self.T) raises -> Self:
        """Start one thread per logical CPU.

        Parameters:
            task: The worker body.

        Args:
            state: The shared state.

        Returns:
            A running pool of `num_cpus()` workers.

        Raises:
            Error: If any `pthread_create` fails.
        """
        return Self.start[task](num_cpus(), state)

    def state(self) -> ref[Self.origin] Self.T:
        """The state this pool was started over.

        Reading it through the pool is how the starter gets at totals after a
        `shutdown()` without having to name the local again — which, with the
        untyped form, is the mention that would have been load-bearing.
        Concurrent access rules still apply while workers are running.

        Returns:
            A mutable reference to the shared state.
        """
        return self._state[]

    @always_inline
    def num_workers(self) -> Int:
        """How many threads this pool started.

        Returns:
            The worker count, whether or not they have been joined.
        """
        return self._pool.num_workers()

    @always_inline
    def stop_flag(self) -> AtomicFlag:
        """The shared stop flag, as a view.

        Returns:
            A view of the flag cell. Valid only while the pool is alive.
        """
        return self._pool.stop_flag()

    @always_inline
    def request_stop(self):
        """Ask every worker to return: release-store 1 to the stop flag.

        Cooperative, with the same two surprises as `WorkerPool` — see the
        module docstring.
        """
        self._pool.request_stop()

    def is_stopping(self) -> Bool:
        """Whether a stop has been requested.

        Returns:
            True after `request_stop()` (or `shutdown()`) has been called.
        """
        return self._pool.is_stopping()

    def join(mut self) raises:
        """Wait for every worker to return.

        Does **not** set the stop flag — call `request_stop()` first, or use
        `shutdown()`.

        Raises:
            Error: The first `pthread_join` failure.
        """
        self._pool.join()

    def shutdown(mut self) raises:
        """`request_stop()` then `join()` — the ordinary way to end a pool.

        Raises:
            Error: If a join fails.
        """
        self._pool.shutdown()

    def pin_all(self) raises:
        """Pin worker `i` to CPU `i % num_cpus()` (a no-op on macOS).

        Raises:
            Error: On Linux if the affinity call fails.
        """
        self._pool.pin_all()
