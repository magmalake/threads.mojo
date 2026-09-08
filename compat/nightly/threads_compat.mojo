"""The nightly half of the toolchain split. Selected with `-I compat/nightly`.

This file exists to hold one line, and it should never hold more than that.
`threads.atomic` explains the split in full; the short version is that
`std.atomic.Atomic` takes a `DType` on Mojo 1.0.0 and a type on nightly, every
other spelling in this tin is identical across the two, and a type cannot be
chosen by a `comptime if` at module scope — so the choice is made by which
directory is on the include path.
"""

from std.atomic import Atomic


comptime Cell = Atomic[Int64]
"""A 64-bit atomic cell, spelled the way nightly declares it:
`struct Atomic[T: Deinitable & Movable, *, scope: StaticString = ""]`. Passing
`DType.int64` here is rejected by this toolchain, which is the whole reason for
the split."""
