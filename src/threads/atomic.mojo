"""Atomics that compile on Mojo 1.0.0 *and* on current nightly.

Everything here operates on a **naturally aligned 64-bit cell at an address you
own**. That is on purpose: atomics are only interesting between threads, and
threads in this library share state through a raw context pointer, so the
useful shape is a view over somebody else's memory rather than a value you hold.
`AtomicCounter` and `AtomicFlag` are both such views — copying one copies the
view, not the cell. The seven free functions below are that view expressed as
`std.atomic` calls against a `Cell` the caller's pointer is bitcast to; a `Cell`
is laid out exactly like the `Int64` it aliases, which the `comptime assert`s in
`_cell` pin down so that a future stdlib change to `Atomic` fails the build
rather than silently scribbling past the cell.

Orderings are spelled into the function names (`_relaxed`, `_acquire`,
`_release`) rather than passed as a parameter, because the ordering has to be
selected at compile time anyway and a named function is both clearer at the call
site and cheaper to read than a `comptime if` ladder. `fetch_add` is
sequentially consistent; that is what a work-queue counter wants.

## Why `Cell` comes from outside this file

`std.atomic.Atomic` changed its parameter *kind* between the two toolchains this
tin targets:

| toolchain | declaration |
|---|---|
| Mojo 1.0.0 | `struct Atomic[dtype: DType, *, scope: StaticString = ""]` |
| nightly | `struct Atomic[T: Deinitable & Movable, *, scope: ...]` |

so `Atomic[DType.int64]` is the only spelling stable accepts and `Atomic[Int64]`
is the only spelling nightly accepts. Everything else is identical — the same
`fetch_add[ordering=…]`, `load[ordering=…]`, `store[ordering=…]` and `fence`
calls compile unchanged against either. The divergence is one type expression
and nothing more.

Mojo offers no way to pick a *type* at module scope from a compiler-version
predicate: a `comptime if` around a `comptime Cell = …` does not parse on either
toolchain, and `-D` defines (`get_defined_bool` plus a `comptime if`) can only
branch inside a function body. So the choice is made by the include path
instead. `compat/stable/threads_compat.mojo` and
`compat/nightly/threads_compat.mojo` each declare `Cell` and nothing else, and a
build says which one it means:

```sh
mojo build … -I src -I compat/stable    # Mojo 1.0.0
mojo build … -I src -I compat/nightly   # nightly
```

This used to be solved one level *down* instead, by reaching past `std.atomic`
to the `pop.atomic.rmw` / `pop.load` / `pop.store` intrinsics it is itself
written on top of, on the reasoning that a compiler primitive would be steadier
than a stdlib signature. That held until it did not: nightlies from
`26.6.0.dev2026090105` onward reject the `__mlir_attr` syntax those calls need
(`error: invalid MLIR attribute: expected '<'`, filed as modular/modular#7094),
and because this was the only file in the whole stack touching raw MLIR it was
single-handedly capping every downstream tin below that nightly. Two spellings
of one alias, in a directory the caller selects, is a far smaller and more
legible surface than a private compiler ABI — and when it breaks it breaks in
Mojo the language rather than in an attribute grammar nobody documents.

Expect `compat/` to grow. The 1.0 and nightly toolchains are diverging rather
than converging, and `async` is the next thing this tin will want that they
spell differently. The pattern to follow is the one here: the divergent file
holds the declaration that differs and nothing whatsoever besides, so that a
reader can see the entire delta between the two toolchains at a glance and every
line that actually does work stays single-sourced.
"""

from std.atomic import Ordering, fence
from std.memory.alloc import unsafe_alloc
from std.sys.info import align_of, size_of

from threads_compat import Cell

from .ffi import I64Ptr, i64_ptr, OpaquePtr


comptime CellPtr = Pointer[Cell, MutUntrackedOrigin]
"""A pointer to a shared atomic cell — what every primitive below actually
operates on, obtained by bitcasting the caller's `I64Ptr`."""


# ── The cell view ────────────────────────────────────────────────────────────


@always_inline
def _cell(ptr: I64Ptr) -> CellPtr:
    """Reinterpret a caller's 64-bit cell as the atomic it is being used as.

    This bitcast is the load-bearing step of the whole module. Callers hand us
    an address out of a shared context block, not an `Atomic` they constructed,
    so `Atomic`'s own constructor never runs over this memory and the only thing
    making that sound is that an `Atomic` over a 64-bit integer *is* a 64-bit
    integer in layout. The two assertions below state that requirement to the
    compiler rather than leaving it as folklore; both hold on both toolchains
    today, and if the stdlib ever adds a field the build stops here.

    Args:
        ptr: The cell, 8-byte aligned.

    Returns:
        The same address, typed as the atomic.
    """
    comptime assert size_of[Cell]() == size_of[Int64](), (
        "std.atomic.Atomic over a 64-bit integer must be 8 bytes for these"
        " views to be sound"
    )
    comptime assert align_of[Cell]() == align_of[Int64](), (
        "std.atomic.Atomic over a 64-bit integer must be 8-byte aligned for"
        " these views to be sound"
    )
    return ptr.unsafe_bitcast[Cell]()


# ── Primitives ───────────────────────────────────────────────────────────────


@always_inline
def atomic_fetch_add(ptr: I64Ptr, delta: Int64) -> Int64:
    """Atomically add `delta` to the cell, sequentially consistent.

    Args:
        ptr: The cell.
        delta: The amount to add.

    Returns:
        The value *before* the addition, so `fetch_add(1)` hands out a unique
        ticket to every caller.
    """
    return _cell(ptr)[].fetch_add[ordering=Ordering.SEQUENTIAL](delta)


@always_inline
def atomic_load_relaxed(ptr: I64Ptr) -> Int64:
    """Atomically read the cell with relaxed ordering.

    Args:
        ptr: The cell.

    Returns:
        The current value. No ordering guarantee is made about any other
        memory — use `atomic_load_acquire` if you are about to read data that
        another thread published.
    """
    return _cell(ptr)[].load[ordering=Ordering.RELAXED]()


@always_inline
def atomic_load_acquire(ptr: I64Ptr) -> Int64:
    """Atomically read the cell with acquire ordering.

    Args:
        ptr: The cell.

    Returns:
        The current value. Everything the releasing thread wrote before its
        matching `atomic_store_release` is visible afterwards.
    """
    return _cell(ptr)[].load[ordering=Ordering.ACQUIRE]()


@always_inline
def atomic_store_relaxed(ptr: I64Ptr, value: Int64):
    """Atomically write the cell with relaxed ordering.

    Args:
        ptr: The cell.
        value: The value to write.
    """
    _cell(ptr)[].store[ordering=Ordering.RELAXED](value)


@always_inline
def atomic_store_release(ptr: I64Ptr, value: Int64):
    """Atomically write the cell with release ordering.

    Everything this thread wrote before the call becomes visible to any thread
    that observes this write through `atomic_load_acquire`.

    Args:
        ptr: The cell.
        value: The value to write.
    """
    _cell(ptr)[].store[ordering=Ordering.RELEASE](value)


@always_inline
def atomic_fence_acquire():
    """A standalone acquire fence.

    `std.atomic.fence` covers standalone fences on both toolchains with the same
    `Ordering` the loads and stores take, so these two moved across with the
    rest and no intrinsic is left in this file.
    """
    fence[ordering=Ordering.ACQUIRE]()


@always_inline
def atomic_fence_release():
    """A standalone release fence. See `atomic_fence_acquire`."""
    fence[ordering=Ordering.RELEASE]()


# ── AtomicCounter ────────────────────────────────────────────────────────────


@fieldwise_init
struct AtomicCounter(Copyable, ImplicitlyCopyable, Movable):
    """A shared 64-bit counter: a *view* over a cell somebody else owns.

    The canonical use is a work queue. Every worker calls `fetch_add(1)` and
    treats the returned value as its task index; the first worker to see an
    index at or past the task count stops.

    ```mojo
    while True:
        var i = Int(counter.fetch_add(1))
        if i >= n_tasks:
            break
        do_work(i)
    ```

    Copying an `AtomicCounter` copies the view, never the cell — two copies
    address the same counter, which is exactly what you want when you hand one
    to each worker.

    The cell must be 8-byte aligned and must outlive every view of it. `alloc`
    satisfies the alignment; the lifetime is on you.
    """

    var _ptr: I64Ptr
    """The counter cell."""

    @staticmethod
    @always_inline
    def at(address: Int) -> Self:
        """View the counter cell at a raw address.

        Args:
            address: Address of an 8-byte-aligned 64-bit cell.

        Returns:
            A view of that cell.
        """
        return Self(i64_ptr(address))

    @staticmethod
    @always_inline
    def at(ctx: OpaquePtr, slot: Int) -> Self:
        """View the `slot`-th 64-bit cell of a context block.

        Args:
            ctx: Base of a block of 64-bit cells, itself 8-byte aligned.
            slot: Which cell, counted in 8-byte units.

        Returns:
            A view of that cell.
        """
        return Self(i64_ptr(Int(ctx) + slot * 8))

    @staticmethod
    def alloc(initial: Int64 = 0) -> Self:
        """Allocate a fresh counter cell on the heap.

        The view returned owns nothing — call `unsafe_free` on exactly one copy
        once every thread that could touch it has been joined.

        Args:
            initial: Starting value.

        Returns:
            A view of the new cell.
        """
        var cell = unsafe_alloc[Int64](1)
        var view = Self(i64_ptr(Int(cell)))
        atomic_store_relaxed(view._ptr, initial)
        return view

    @always_inline
    def unsafe_free(self):
        """Free a cell obtained from `alloc`. Never call this while a thread
        that can see the cell is still running."""
        self._ptr.unsafe_free()

    @always_inline
    def address(self) -> Int:
        """Return the raw address of the cell.

        Returns:
            The address, suitable for storing in a context block so a worker
            can rebuild the view.
        """
        return Int(self._ptr)

    @always_inline
    def fetch_add(self, delta: Int64 = 1) -> Int64:
        """Atomically add and return the previous value.

        Args:
            delta: The amount to add.

        Returns:
            The value before the addition.
        """
        return atomic_fetch_add(self._ptr, delta)

    @always_inline
    def load(self) -> Int64:
        """Atomically read the counter (acquire).

        Returns:
            The current value.
        """
        return atomic_load_acquire(self._ptr)

    @always_inline
    def store(self, value: Int64):
        """Atomically write the counter (release).

        Args:
            value: The value to write.
        """
        atomic_store_release(self._ptr, value)


# ── AtomicFlag ───────────────────────────────────────────────────────────────


@fieldwise_init
struct AtomicFlag(Copyable, ImplicitlyCopyable, Movable):
    """A one-way (or resettable) publish/observe flag over a shared cell.

    `set` is a *release* store and `is_set` is an *acquire* load, so the pair
    carries data with it:

    ```mojo
    # publisher
    payload[0] = 42        # plain write
    flag.set()             # release

    # observer
    while not flag.is_set():   # acquire
        spin_hint()
    assert payload[0] == 42     # guaranteed visible
    ```

    That release/acquire pairing is the whole point — a plain `Bool` written
    from one thread and read from another has no such guarantee, and the
    compiler is free to hoist the read out of the spin loop entirely.

    Like `AtomicCounter`, this is a view: copying it does not copy the cell.
    """

    var _ptr: I64Ptr
    """The flag cell."""

    @staticmethod
    @always_inline
    def at(address: Int) -> Self:
        """View the flag cell at a raw address.

        Args:
            address: Address of an 8-byte-aligned 64-bit cell.

        Returns:
            A view of that cell.
        """
        return Self(i64_ptr(address))

    @staticmethod
    @always_inline
    def at(ctx: OpaquePtr, slot: Int) -> Self:
        """View the `slot`-th 64-bit cell of a context block.

        Args:
            ctx: Base of a block of 64-bit cells, itself 8-byte aligned.
            slot: Which cell, counted in 8-byte units.

        Returns:
            A view of that cell.
        """
        return Self(i64_ptr(Int(ctx) + slot * 8))

    @staticmethod
    def alloc() -> Self:
        """Allocate a fresh, cleared flag cell on the heap.

        Returns:
            A view of the new cell. Free it with `unsafe_free` after joining.
        """
        var cell = unsafe_alloc[Int64](1)
        var view = Self(i64_ptr(Int(cell)))
        atomic_store_relaxed(view._ptr, 0)
        return view

    @always_inline
    def unsafe_free(self):
        """Free a cell obtained from `alloc`. Never call this while a thread
        that can see the cell is still running."""
        self._ptr.unsafe_free()

    @always_inline
    def address(self) -> Int:
        """Return the raw address of the cell.

        Returns:
            The address, suitable for storing in a context block.
        """
        return Int(self._ptr)

    @always_inline
    def set(self):
        """Publish: release-store 1. Everything written before this call is
        visible to a thread that later observes `is_set()`."""
        atomic_store_release(self._ptr, 1)

    @always_inline
    def clear(self):
        """Reset the flag to 0 with a release store."""
        atomic_store_release(self._ptr, 0)

    @always_inline
    def is_set(self) -> Bool:
        """Observe: acquire-load.

        Returns:
            True once the publisher has called `set`, at which point the
            publisher's earlier writes are visible.
        """
        return atomic_load_acquire(self._ptr) != 0

    @always_inline
    def raw(self) -> Int64:
        """Return the cell's value as an integer (acquire load) — useful when a
        flag doubles as a small error code.

        Returns:
            The current value.
        """
        return atomic_load_acquire(self._ptr)

    @always_inline
    def set_value(self, value: Int64):
        """Release-store an arbitrary non-zero value, for the flag-as-error-code
        pattern.

        Args:
            value: The value to publish.
        """
        atomic_store_release(self._ptr, value)
