"""Shared FFI vocabulary: the opaque pointer type every `threads` entry point
speaks, and the C NULL convention.

`pthread_create` hands the start routine a `void *`, so every callable that
crosses a thread boundary in this library has the shape

```mojo
def (Pointer[UInt8, MutUntrackedOrigin]) thin -> Pointer[UInt8, MutUntrackedOrigin]
```

`OpaquePtr` is that pointer type. It is deliberately a byte pointer rather than
a `void`-like: Mojo has no `void`, and a byte pointer is what you want anyway
when the context is a hand-laid-out block of cells.

NULL is passed as a plain `Int` `0` in every `external_call` that needs it, not
as a pointer. `Pointer` is non-nullable on both Mojo 1.0.0 and current
nightly (`constraint failed: Pointer is non-nullable`), so a null pointer value
cannot be constructed at all. An `Int` argument occupies the same register as a
pointer under both the SysV and AAPCS64 C ABIs, so the call is identical.
"""


comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]
"""The `void *` of this library: what a thread start routine takes and returns,
and what `parallel_for` passes to every task as its shared context."""

comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
"""Pointer to a naturally aligned 64-bit cell — the unit every atomic in
`threads.atomic` operates on."""

comptime NULL: Int = 0
"""C `NULL`, passed as an integer. See the module docstring for why."""


@always_inline
def opaque_ptr(address: Int) -> OpaquePtr:
    """Rebuild an `OpaquePtr` from a raw address.

    Args:
        address: The address, e.g. from `Int(some_pointer)` or a cell that a
            worker reads out of its context block.

    Returns:
        A pointer to that address. No validity check is performed.
    """
    return OpaquePtr(unsafe_from_address=address)


@always_inline
def i64_ptr(address: Int) -> I64Ptr:
    """Rebuild an `I64Ptr` from a raw address.

    Args:
        address: The address of a naturally aligned 64-bit cell.

    Returns:
        A pointer to that cell. No validity or alignment check is performed.
    """
    return I64Ptr(unsafe_from_address=address)
