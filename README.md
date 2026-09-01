# threads.mojo

[![mojoshelf](https://mojoshelf.org/badge/threads-mojo.svg)](https://mojoshelf.org/tins/threads-mojo) [![mojo nightly](https://mojoshelf.org/badge/threads-mojo/nightly.svg)](https://mojoshelf.org/tins/threads-mojo)

[![CI](https://github.com/magmalake/threads.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/magmalake/threads.mojo/actions/workflows/ci.yml)

Part of [**magmalake**](https://magmalake.org) — data lake building blocks in Mojo.

Minimal OS threads for [Mojo](https://www.modular.com/mojo): spawn and join
pthreads, share state through atomics and a mutex, fan a loop out over cores
with `parallel_for`, and keep long-lived workers running with `WorkerPool`. No
dependencies — the Mojo standard library and the pthread symbols every libc
already exports, nothing else. No `dlopen`, no C shim, no build step.

It exists because Mojo currently ships no way to use more than one core from
Mojo code: `parallelize` was removed from `std.algorithm`, and no thread pool
is reachable from `std`. If you want a second core, you call `pthread_create`
yourself. This tin is that call, done once, carefully, and tested.

Works on **Mojo 1.0.0 (stable) and current nightly**, on **linux-64** and
**osx-arm64** — all four combinations in CI.

## Install

```sh
pixi shelf add threads-mojo
```

Working with a coding agent? `npx skills add mojoshelf/mojoshelf --skill mojoshelf-consume --yes` teaches it to find and install tins itself — it installs the `shelf` CLI too.

That resolves the tin from [mojoshelf](https://mojoshelf.org) and adds it — along with the tins it depends on — as **pixi git source dependencies**. magmalake tins are not published to a conda channel, so `pixi add threads-mojo` will not find them.

As a dependency declaration, or for a nightly consumer:

```toml
# pixi.toml
[dependencies]
threads-mojo = { git = "https://github.com/magmalake/threads.mojo" }
```

or, like the rest of magmalake, by source path: check the repo out next to
yours and add `-I ../threads.mojo/src` to your `mojo build`.

## Quick start

```mojo
from std.memory import alloc
from threads import OpaquePtr, i64_ptr, opaque_ptr, parallel_for


def square(i: Int, ctx: OpaquePtr) -> None:
    var cells = i64_ptr(Int(ctx))
    cells[i] = cells[i] * cells[i]


def main() raises:
    var n = 1_000_000
    var data = alloc[Int64](n)
    for i in range(n):
        data[i] = Int64(i)

    parallel_for[square](n_tasks=n, ctx=opaque_ptr(Int(data)))

    print(data[999])   # 998001
    data.unsafe_free()
```

## The send contract

`pthread_create` takes a C function pointer and one `void *`. That is the whole
channel, and this library does not pretend otherwise. Three rules, and they are
the same three everywhere in the API:

1. **Thin functions only.** A thread body is
   `def(Int, OpaquePtr) thin -> None` (or `def(OpaquePtr) thin -> OpaquePtr`
   for a raw thread). It cannot capture and it cannot raise — pthread has no
   exception channel.
2. **Shared state travels through the `ctx` pointer.** One pointer, for every
   task. Lay out a block of 64-bit cells and index it; that is what the tests
   and the demo do.
3. **The caller guarantees `ctx` outlives the call.** `parallel_for` joins
   every worker before it returns, so "outlives the call" is all you need — but
   Mojo destroys a value at its **last use**, not at the end of the scope, so a
   context you stop mentioning after the spawn can be freed underneath a
   running thread. Mention it after the join, or heap-allocate it and free it
   after the join. `test_parallel_for_context_outlives_the_join` and
   `test_mutex_gives_mutual_exclusion` both exist to catch a regression here.

A task cannot raise, so failures come back through the context. The pattern —
an `AtomicFlag` plus one error-code cell per task — is documented in
`threads/parallel.mojo` and exercised by `test_parallel_for_error_cell_pattern`.

## API

```mojo
from threads import (
    ThreadHandle, ThreadGroup, spawn_n, join_all,
    num_cpus, current_thread_id, yield_now,
    AtomicCounter, AtomicFlag,
    Mutex, MutexRef, CondVar, CondVarRef,
    parallel_for, WorkerPool,
    OpaquePtr, i64_ptr, opaque_ptr,
)
```

### `threads.pool`

`parallel_for` fits a **bounded** set of indices: hand out tasks, drain the
queue, join. A server is the other shape — the work never runs out, and each
worker loops on its own until something says stop. `WorkerPool` is that shape:
`spawn_n` + `AtomicFlag` + `ThreadGroup.join_all()` bundled so callers stop
re-deriving the header layout and the free-after-join rule. It is deliberately
thin — no queue, no task type, no scheduler.

```mojo
from threads import AtomicCounter, AtomicFlag, OpaquePtr, WorkerPool

def serve(worker: Int, ctx: OpaquePtr, stop: AtomicFlag) -> None:
    while not stop.is_set():
        handle_one_request(ctx)      # whatever `ctx` addresses

def main() raises:
    var pool = WorkerPool.start[serve](n=8, ctx=my_ctx)
    wait_for_sigint()
    pool.shutdown()                  # request_stop + join
```

| item | signature | notes |
|---|---|---|
| `WorkerFn` | `def(Int, OpaquePtr, AtomicFlag) thin -> None` | worker index, shared context, stop flag |
| `WorkerPool.start` | `start[work: WorkerFn](n, ctx) raises -> WorkerPool` | `n >= 1` (a zero-worker pool is refused, not silently created); an `n`-less overload uses `num_cpus()` |
| `.request_stop()` | `-> None` | release-store to the flag; returns immediately |
| `.join()` | `(mut self) raises` | joins every worker; does **not** set the flag |
| `.shutdown()` | `(mut self) raises` | `request_stop` then `join` |
| `.num_workers()` / `.is_stopping()` / `.stop_flag()` | | `stop_flag()` is a view, so a third party holding its `address()` can end the pool |
| `.pin_all()` | `raises` | worker `i` to CPU `i % num_cpus()` (a no-op on macOS) |

Workers are **symmetric**: every one runs the same body, and the index exists
only so a worker can address its own slot in your context. It is drawn from an
atomic counter at thread start, not baked in.

Two things about the stop flag that have surprised people, both of them
consequences of it being cooperative rather than preemptive:

- **A worker blocked in a syscall does not observe the flag until that syscall
  returns.** A thread parked in `recv`, `accept`, or a blocking FFI call stays
  parked; `request_stop()` does not interrupt it. If your workers block, the
  thing they block on needs its own wake mechanism — a sentinel, a closed
  channel, a timeout — and the flag is only the second half of the handshake.
  ([restate.mojo](https://github.com/winding-lines/restate.mojo)'s served mode
  is the worked example: its Rust shim grew an `rst_stop` for exactly this.)
- **A worker that never checks the flag never stops**, so `join()` never
  returns. Loop on `not stop.is_set()`.

Lifetimes are handled rather than documented-at: the pool heap-allocates one
small header (index counter, stop cell, user-context address), and its
destructor sets the flag, joins, and only then frees. Dropping a pool without
calling `shutdown()` is therefore safe — which matters, because Mojo destroys a
value at its **last use**, so a pool whose last mention is `request_stop()` is
destroyed on that line and the destructor's join is what keeps the header alive
under the still-running workers. Your `ctx` is not owned by the pool and must
outlive it; since the destructor joins, outliving the pool *value* is enough.

### `threads.parallel`

| item | signature | notes |
|---|---|---|
| `parallel_for` | `parallel_for[work: def(Int, OpaquePtr) thin -> None](n_tasks, ctx, num_workers=0) raises` | starts `min(n_tasks, num_workers or num_cpus())` threads that pull indices off one shared atomic counter; joins all before returning |
| `spawn_n` | `spawn_n[start](n, ctx, stride=0) raises -> ThreadGroup` | thread `i` gets `ctx + i * stride`; a partial failure joins what already started |
| `ThreadGroup` | `.join_all()`, `.pin_all()`, `len()` | move-only; `join_all` attempts every join even if one fails |

The work queue is one shared counter, so scheduling is dynamic — a slow task
does not stall the others — at a cost of one atomic RMW per task. Make tasks
chunky: tens to thousands of tasks, not one per element.

### `threads.thread`

| item | signature | notes |
|---|---|---|
| `ThreadHandle.spawn` | `spawn[start: def(OpaquePtr) thin -> OpaquePtr](arg) raises -> ThreadHandle` | move-only handle over `pthread_t` |
| `ThreadHandle.join` | `join(mut self) raises` | idempotent — zeroes the id, so a second call is a no-op instead of a second `pthread_join` on a stale id (which is UB, not an error) |
| `ThreadHandle.detach` | `detach(mut self) raises` | leaves the handle inert |
| `ThreadHandle.pin_to_cpu` | `pin_to_cpu(cpu) raises` | **real on Linux** (`pthread_setaffinity_np`); **a documented no-op on macOS** — Darwin exposes only an affinity *hint*, never a hard pin |
| `num_cpus` | `-> Int` | `sysconf(_SC_NPROCESSORS_ONLN)`, at least 1 |
| `current_thread_id` | `-> UInt64` | `pthread_self` |
| `yield_now` | `-> None` | `sched_yield`, for spin loops |

### `threads.atomic`

Everything operates on a naturally aligned 64-bit cell **at an address you
own** — atomics are only interesting between threads, and threads only get a
raw pointer, so the useful shape is a view over shared memory rather than a
value you hold. `AtomicCounter` and `AtomicFlag` are both views: copying one
copies the view, not the cell.

| item | notes |
|---|---|
| `AtomicCounter` | `.fetch_add(delta=1)`, `.load()`, `.store(v)`, `.at(address)`, `.at(ctx, slot)`, `.alloc(initial)` / `.unsafe_free()` |
| `AtomicFlag` | `.set()` (release), `.is_set()` (acquire), `.clear()`, `.set_value(v)` / `.raw()` for the flag-as-error-code pattern |
| free functions | `atomic_fetch_add`, `atomic_load_relaxed`, `atomic_load_acquire`, `atomic_store_relaxed`, `atomic_store_release`, `atomic_fence_acquire`, `atomic_fence_release` |

### `threads.mutex`

| item | notes |
|---|---|
| `Mutex` | owns a 64-byte `pthread_mutex_t` blob; move-only; `.lock()`, `.unlock()`, `.try_lock()`, `.with_lock[body](ctx)`, `.address()`, `.ref()` |
| `MutexRef` | copyable non-owning view; what a worker rebuilds from an address in its context |
| `CondVar` / `CondVarRef` | the same split over `pthread_cond_t`: `.wait(mutex)`, `.signal()`, `.broadcast()` |

There is deliberately **no RAII lock guard**. Mojo destroys a value at its last
*use*, so a guard whose only job is to unlock in its destructor would unlock at
the point you stop mentioning it — usually the first line of the critical
section. That is a silent correctness bug rather than a compile error. Use
`lock()`/`unlock()`, or `with_lock`, which brackets a thin function properly.

## Why this does not just re-export `std.atomic`

Because the two toolchains this tin targets disagree about what `Atomic` is:

| toolchain | declaration | the only spelling it accepts |
|---|---|---|
| Mojo 1.0.0 | `struct Atomic[dtype: DType, ...]` | `Atomic[DType.int64]` |
| nightly 1.1.0.dev2026083005 | `struct Atomic[T: Deinitable & Movable, ...]` | `Atomic[Int64]` |

Neither spelling compiles on the other compiler, and Mojo exposes no
compiler-version constant to branch on. So `threads.atomic` goes one level
down, to the `pop.atomic.rmw` / `pop.load` / `pop.store` / `pop.fence`
intrinsics that `std.atomic` is itself written on top of — those are identical
on both. Bridging that gap is half of what this tin is for: your code writes
one spelling and keeps compiling across the split.

Two smaller portability notes baked in for the same reason: `UnsafePointer` is
non-nullable on both toolchains now (`constraint failed: Pointer is
non-nullable`), so C `NULL` is passed as a plain `Int` — identical in the
register under both the SysV and AAPCS64 ABIs. And `List` iteration requires a
`Copyable` element on both, so `ThreadGroup` iterates its move-only handles by
index.

## What it costs, and where it stops

`threads-mojo bench`, on an Apple M4 (10 logical CPUs, Mojo 1.0.0):

```
  parallel sum over 100000000 Int64        parallel memcpy of 762 MiB
  workers      ms   speedup                workers      ms   speedup    GB/s
        1   32.03     1.00x                      1   75.96     1.00x   21.06
        2   19.21     1.67x                      2   25.25     3.01x   63.36
        4   13.50     2.37x                      4   18.95     4.01x   84.45
        8    8.82     3.63x                      8   18.46     4.12x   86.68
       10    8.66     3.70x                     10   18.25     4.16x   87.68
```

The memcpy row is in the demo on purpose. It is bandwidth bound, so it stops
scaling at four workers no matter how many cores you throw at it — about 4× is
the ceiling for anything memory-bound on this machine. A parallel-for demo that
only shows a straight line is measuring the wrong thing.

Overheads, measured on the same machine: **~14 µs** per spawn+join round trip
(500 sequential threads in 6.9 ms), **~2.4 ns** per uncontended `fetch_add`,
and **14–21 ns** per `fetch_add` under 8-way contention on one cache line
(800,000 of them in 11–17 ms across repeated runs, on both toolchains). That
last spread is why `parallel_for` wants chunky tasks: the counter is a shared
cache line, and every task costs one trip through it.

## What this enables: parallel Parquet decode

The point of the tin is other people's inner loops, so here is one. Using an
**unmodified** [parquet.mojo](https://github.com/magmalake/parquet.mojo)
checkout on the include path, decoding a 1M-row / 4-column Parquet file one
row group per `parallel_for` task:

| file | workers | ms | speedup | rows/s |
|---|---|---|---|---|
| 1M rows, 4 row groups | 1 | 5.02 | 1.00× | 199 M |
| | 2 | 2.84 | 1.77× | 352 M |
| | 4 | 2.07 | **2.43×** | 484 M |
| 1M rows, 28 row groups | 1 | 4.49 | 1.00× | 223 M |
| | 2 | 2.54 | 1.77× | 394 M |
| | 4 | 1.82 | 2.47× | 549 M |
| | 8 | 1.52 | **2.96×** | 660 M |
| | 16 | 1.59 | 2.83× | 629 M |

**Method, so you can weigh it.** Apple M4, 10 logical CPUs, Mojo 1.0.0, best of
5 runs. The file is parquet.mojo's own `build/bench-wide.parquet`
(1,000,000 rows; `int64`, `double`, dictionary-encoded `int64`, dictionary-
encoded `string`; UNCOMPRESSED; 4 row groups of 250k). The 28-row-group variant
is the same data rewritten through parquet.mojo's writer with
`row_group_size = 62500`, also UNCOMPRESSED, in a scratch directory. One
`ParquetReader` per row group is built **outside** the timed region, each over
its own copy of the file bytes and restricted with `select_row_groups`; the
timed region is `parallel_for` over `read_table()`, and every run checks that
the decoded row count still sums to 1,000,000. Both the 1-worker and the
N-worker rows go through the same `parallel_for` path, so the comparison is
worker count and nothing else.

**Read it honestly.** 2.96× is not 8×, and it should not be: parquet.mojo's
decoder already runs at 232 M rows/s on one core and is largely moving bytes,
so it hits the same memory ceiling the memcpy column above shows at ~4×. Going
past 8 workers gets slower — 28 row groups over 16 threads is more contention
than parallelism. The experiment lives in a scratch directory and changes
nothing in parquet.mojo; it is evidence, not a feature of that repo.

## Tests

```console
$ pixi run test              # nightly
$ pixi run -e stable test    # Mojo 1.0.0
```

35 tests on each environment, on both platforms. The load-bearing ones are the
races, each written so a broken primitive produces a *wrong number* rather than
a flake:

- **`AtomicCounter`** — 8 threads × 100,000 `fetch_add` must equal exactly
  800,000. A non-atomic read-modify-write loses updates here every run.
- **`AtomicFlag`** — a writer does a *plain* write and then a release store;
  four spinning readers acquire-load the flag and must all see the payload.
- **`Mutex`** — 8 threads × 10,000 unguarded-looking increments of a plain
  `Int` must equal exactly 80,000; and a separate timing test that the lock
  really blocks (hold for 50 ms, assert the waiter waited > 10 ms).
- **Allocator under threads** — 8 workers building and dropping Mojo `List`s
  and `String`s concurrently, 64 tasks × 20 rounds each. Every worker in this
  library rests on the Mojo runtime allocator being usable from several threads
  at once; this test is the evidence, and it passes on all four CI legs.
- **`WorkerPool`** — nine tests: workers that tick a shared counter until
  stopped and then join; `request_stop()` racing the spawns; the 16 indices
  each claimed exactly once (a duplicate leaves one slot at 2 and another at
  0); `n = 1`; `n < 1` refused; a worker that fails and returns without ever
  consulting the flag still joins; a pool destroyed without `shutdown()` still
  stops, joins, and frees in that order; and a third party setting the flag
  through `stop_flag().address()`.
- **Stress** — 500 sequential spawn/joins, 50 back-to-back `parallel_for`
  rounds, 50 `WorkerPool`s started and shut down back to back.

## Credits and license

The threading primitives here are distilled from
[ehsanmok/flare](https://github.com/ehsanmok/flare)'s runtime (MIT) — its
`ThreadHandle` shape, its opaque `pthread_mutex_t` blob, the heap-allocated
worker context that must outlive the join, and its hard-won note that the
scheduler's stop flag really wanted to be an atomic. The code is rewritten for
current Mojo rather than copied: flare pins `mojo==1.0.0b1`, and this tin has
to compile on both 1.0.0 and nightly.

Licensed under the Apache License, Version 2.0 — see `LICENSE` and `NOTICE`.
