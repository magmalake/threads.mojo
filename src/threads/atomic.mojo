"""Atomics that compile on Mojo 1.0.0 *and* on current nightly.

Why this module exists at all: the stdlib's `std.atomic.Atomic` changed its
parameter kind between the two toolchains this tin targets.

| toolchain | declaration |
|---|---|
| Mojo 1.0.0 | `struct Atomic[dtype: DType, *, scope: StaticString = ""]` |
| nightly 1.1.0.dev2026083005 | `struct Atomic[T: Deinitable & Movable, *, scope: ...]` |

So `Atomic[DType.int64]` is the only spelling stable accepts and
`Atomic[Int64]` is the only spelling nightly accepts, and there is no
compiler-version constant to branch on. Rather than fork the source, this
module goes one level down to the `pop.atomic.rmw` / `pop.load` / `pop.store`
compiler intrinsics that `std.atomic` itself is written on top of. Those have
been stable across both toolchains, and going through them means a consumer of
this tin writes one spelling that works on either compiler.

Everything here operates on a **naturally aligned 64-bit cell at an address you
own**. That is on purpose: atomics are only interesting between threads, and
threads in this library share state through a raw context pointer, so the
useful shape is a view over somebody else's memory rather than a value you hold.
`AtomicCounter` and `AtomicFlag` are both such views — copying one copies the
view, not the cell.

Orderings are spelled into the function names (`_relaxed`, `_acquire`,
`_release`, `_seq_cst`) rather than passed as a parameter, because the MLIR
ordering attribute has to be selected at compile time and a named function is
both clearer at the call site and cheaper to read than a `comptime if` ladder.
`fetch_add` is sequentially consistent; that is what a work-queue counter wants.
"""

from std.memory import alloc

from .ffi import I64Ptr, i64_ptr, OpaquePtr


# ── Raw intrinsics ───────────────────────────────────────────────────────────


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
    var res = __mlir_op.`pop.atomic.rmw`[
        bin_op=__mlir_attr.`#pop<bin_op add>`,
        ordering=__mlir_attr.`#pop<atomic_ordering seq_cst>`,
        _type=Int64._mlir_type,
    ](
        ptr.unsafe_bitcast[Int64._mlir_type]()._get_kgen_pointer(),
        delta._mlir_value,
    )
    return Int64(mlir_value=res)


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
    var res = __mlir_op.`pop.load`[
        ordering=__mlir_attr.`#pop<atomic_ordering monotonic>`,
        isVolatile=False.__mlir_i1__(),
        isInvariant=False.__mlir_i1__(),
        isNonTemporal=False.__mlir_i1__(),
    ](ptr.unsafe_bitcast[Int64._mlir_type]()._get_kgen_pointer())
    return Int64(mlir_value=res)


@always_inline
def atomic_load_acquire(ptr: I64Ptr) -> Int64:
    """Atomically read the cell with acquire ordering.

    Args:
        ptr: The cell.

    Returns:
        The current value. Everything the releasing thread wrote before its
        matching `atomic_store_release` is visible afterwards.
    """
    var res = __mlir_op.`pop.load`[
        ordering=__mlir_attr.`#pop<atomic_ordering acquire>`,
        isVolatile=False.__mlir_i1__(),
        isInvariant=False.__mlir_i1__(),
        isNonTemporal=False.__mlir_i1__(),
    ](ptr.unsafe_bitcast[Int64._mlir_type]()._get_kgen_pointer())
    return Int64(mlir_value=res)


@always_inline
def atomic_store_relaxed(ptr: I64Ptr, value: Int64):
    """Atomically write the cell with relaxed ordering.

    Args:
        ptr: The cell.
        value: The value to write.
    """
    __mlir_op.`pop.store`[
        ordering=__mlir_attr.`#pop<atomic_ordering monotonic>`,
        isVolatile=False.__mlir_i1__(),
        isNonTemporal=False.__mlir_i1__(),
    ](
        value._mlir_value,
        ptr.unsafe_bitcast[Int64._mlir_type]()._get_kgen_pointer(),
    )


@always_inline
def atomic_store_release(ptr: I64Ptr, value: Int64):
    """Atomically write the cell with release ordering.

    Everything this thread wrote before the call becomes visible to any thread
    that observes this write through `atomic_load_acquire`.

    Args:
        ptr: The cell.
        value: The value to write.
    """
    __mlir_op.`pop.store`[
        ordering=__mlir_attr.`#pop<atomic_ordering release>`,
        isVolatile=False.__mlir_i1__(),
        isNonTemporal=False.__mlir_i1__(),
    ](
        value._mlir_value,
        ptr.unsafe_bitcast[Int64._mlir_type]()._get_kgen_pointer(),
    )


@always_inline
def atomic_fence_acquire():
    """A standalone acquire fence."""
    __mlir_op.`pop.fence`[
        ordering=__mlir_attr.`#pop<atomic_ordering acquire>`,
        _type=None,
    ]()


@always_inline
def atomic_fence_release():
    """A standalone release fence."""
    __mlir_op.`pop.fence`[
        ordering=__mlir_attr.`#pop<atomic_ordering release>`,
        _type=None,
    ]()


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
        var cell = alloc[Int64](1)
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
        var cell = alloc[Int64](1)
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
