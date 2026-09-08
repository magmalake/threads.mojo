# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before this file are in the commit log (each release is one commit
whose subject begins with its version).

## [Unreleased]

## [0.5.1] - 2026-09-08

Documentation only. No code, no signature and no build input changed — the
artifact is equivalent to 0.5.0, and there is no behavioural reason to upgrade.

### Changed
- `threads.atomic`'s module docstring now says why the module *stays* on
  `std.atomic` now that the nightly regression that first forced the move
  ([modular/modular#7094](https://github.com/modular/modular/issues/7094)) has
  been fixed upstream. The intrinsics would compile again today; the module is
  not going back to them, because `std.atomic` is the interface carrying a
  compatibility promise and `pop.*` is an implementation detail of it. Worth
  writing down, since the next reader will otherwise find a workaround whose
  stated reason no longer holds and conclude it can be reverted.

## [0.5.0] - 2026-09-08

`threads.atomic` is built on `std.atomic` instead of the `pop.*` compiler
intrinsics. The two toolchains spell the type differently — `Atomic[DType.int64]`
on Mojo 1.0.0, `Atomic[Int64]` on nightly — and that is the whole of the
difference: every call site is identical once an alias hides it. So the divergent
code is **one line per toolchain**, in `compat/stable/` and `compat/nightly/`,
picked by the `-I` path; the other 400 lines are single-sourced.

**Consumers of the published tin need no change** — it ships precompiled.
Consumers building from **source paths** must add the compat directory beside
`-I ../threads.mojo/src`; this repository sets `$THREADS_COMPAT` per feature and
its own tasks show the shape.

### Changed
- **`threads.atomic` is written on `std.atomic` instead of raw MLIR
  intrinsics.** All seven primitives — `atomic_fetch_add`,
  `atomic_load_relaxed`, `atomic_load_acquire`, `atomic_store_relaxed`,
  `atomic_store_release`, `atomic_fence_acquire`, `atomic_fence_release` —
  now go through `Atomic.fetch_add` / `.load` / `.store` and `std.atomic.fence`
  with an explicit `Ordering`. `AtomicCounter` and `AtomicFlag` are unchanged
  and no public signature moved, so **no consumer's Mojo source changes**.

  This fixes a hard build break rather than being a cleanup. Nightlies from
  `26.6.0.dev2026090105` onward reject the `__mlir_attr` spelling the old
  intrinsic calls needed (`error: invalid MLIR attribute: expected '<'`, filed
  as [modular/modular#7094](https://github.com/modular/modular/issues/7094)),
  and because `atomic.mojo` was the only file in all of magmalake touching raw
  MLIR, it alone was capping every tin downstream of it below that nightly.
  There is no raw MLIR left in this repo.

- **Consumers building from source paths need one more `-I`.** The reason the
  old code reached for intrinsics in the first place has not gone away —
  `std.atomic.Atomic` takes a `DType` on Mojo 1.0.0 and a type on nightly, and
  Mojo cannot pick a *type* from a `comptime if` at module scope. So that one
  declaration, and nothing else, now lives in a file the include path selects:

  ```sh
  mojo build … -I ../threads.mojo/src -I ../threads.mojo/compat/stable    # Mojo 1.0.0
  mojo build … -I ../threads.mojo/src -I ../threads.mojo/compat/nightly   # nightly
  ```

  Each of `compat/stable/threads_compat.mojo` and
  `compat/nightly/threads_compat.mojo` is a docstring plus one
  `comptime Cell = Atomic[…]`. Consumers taking the **published tin** need no
  change at all — the conda package is precompiled here, against the stable
  spelling, via the build backend's `extra-args`.

  Expect `compat/` to grow: 1.0 and nightly are diverging rather than
  converging, and `async` is the next thing this tin will want that they spell
  differently.

- **The nightly CI leg is floored at `26.6.0.dev2026090705`**, the first
  nightly this repo pins that the old code could not compile on. A lower floor
  would let the leg pass on a toolchain that never had the problem, which would
  make the fix untestable.
