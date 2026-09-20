"""The toolchain-dependent half of this tin. Selected with `-I compat`.

This file exists to hold the declarations that two supported Mojo toolchains
spell differently, and nothing else. `threads.atomic` explains the mechanism in
full; the short version is that a type cannot be chosen by a `comptime if` at
module scope, so the choice is made by which directory is on the include path.

Right now there is only one such directory, because Mojo 1.1.0 and the 26.7
nightlies agree on every spelling here. While Mojo 1.0.0 was still supported
there were two — `compat/stable` and `compat/nightly` — differing in the one
line below, and the moment the toolchains diverge again they split back apart.
"""

from std.atomic import Atomic


comptime Cell = Atomic[Int64]
"""A 64-bit atomic cell. `std.atomic.Atomic` is parameterised on a type —
`struct Atomic[T: Deinitable & Movable, *, scope: StaticString = ""]`. Mojo
1.0.0 parameterised it on a `DType` instead and wanted `Atomic[DType.int64]`
here, which is the divergence this directory was created for."""
