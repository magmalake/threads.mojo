"""`threads-mojo` — see what `parallel_for` buys you on this machine.

```console
$ threads-mojo bench
$ threads-mojo bench --workers 4
$ threads-mojo info
```

`bench` runs two deliberately different workloads at 1, 2, 4, ... workers up to
`num_cpus()` and prints the speedup against the single-worker run:

- **sum** — add up 100M `Int64`s. Compute bound after the first pass warms the
  pages; this is the one that should scale.
- **memcpy** — copy 800 MB between two buffers. Memory-bandwidth bound, so it
  stops scaling once the cores saturate the controller. It is here precisely
  because it *doesn't* scale linearly: a parallel-for that claims otherwise is
  measuring the wrong thing.
"""

from std.memory import alloc
from std.sys import argv
from std.time import perf_counter_ns

from threads import (
    OpaquePtr,
    current_thread_id,
    i64_ptr,
    num_cpus,
    opaque_ptr,
    parallel_for,
)

comptime USAGE = """usage: threads-mojo <command> [options]

commands:
  bench    scale a parallel sum and a parallel memcpy from 1 worker to num_cpus
  info     print num_cpus and this thread's id

options:
  --workers N   cap the worker count (default: num_cpus)
  --elements N  elements in the sum benchmark (default: 100000000)
"""

# Context layout for both benchmarks, in 64-bit cells.
comptime CTX_CELLS: Int = 6
comptime C_SRC: Int = 0  # source buffer address
comptime C_DST: Int = 1  # destination buffer address
comptime C_N: Int = 2  # element count
comptime C_CHUNKS: Int = 3  # number of tasks the range is split into
comptime C_PARTIALS: Int = 4  # address of the per-task partial-sum array
comptime C_UNUSED: Int = 5


@always_inline
def _chunk_bounds(i: Int, n: Int, chunks: Int) -> Tuple[Int, Int]:
    """Split `[0, n)` into `chunks` near-equal pieces and return piece `i`."""
    var base = n // chunks
    var extra = n % chunks
    var start = i * base + (i if i < extra else extra)
    var length = base + (1 if i < extra else 0)
    return (start, start + length)


def _sum_chunk(i: Int, ctx: OpaquePtr) -> None:
    """Sum one slice of the source array into this task's own partial slot."""
    var cells = i64_ptr(Int(ctx))
    var src = i64_ptr(Int(cells[C_SRC]))
    var partials = i64_ptr(Int(cells[C_PARTIALS]))
    var bounds = _chunk_bounds(i, Int(cells[C_N]), Int(cells[C_CHUNKS]))
    var total = Int64(0)
    for k in range(bounds[0], bounds[1]):
        total += src[k]
    partials[i] = total


def _copy_chunk(i: Int, ctx: OpaquePtr) -> None:
    """Copy one slice of the source array into the destination."""
    var cells = i64_ptr(Int(ctx))
    var src = i64_ptr(Int(cells[C_SRC]))
    var dst = i64_ptr(Int(cells[C_DST]))
    var bounds = _chunk_bounds(i, Int(cells[C_N]), Int(cells[C_CHUNKS]))
    for k in range(bounds[0], bounds[1]):
        dst[k] = src[k]


def _pad(text: String, width: Int) -> String:
    """Right-align `text` in `width` columns."""
    var out = String()
    for _ in range(width - text.byte_length()):
        out += " "
    out += text
    return out^


def _two_dp(hundredths: Int) -> String:
    """Render a fixed-point hundredths value as `d.dd`."""
    var frac = hundredths % 100
    var lead = "0" if frac < 10 else ""
    return String(hundredths // 100, ".", lead, frac)


def _ms(ns: Int) -> String:
    """Format nanoseconds as milliseconds with two decimals."""
    return _two_dp((ns + 5_000) // 10_000)


def _speedup(base_ns: Int, ns: Int) -> String:
    """Format `base_ns / ns` with two decimals."""
    if ns <= 0:
        return "n/a"
    return _two_dp((base_ns * 100 + ns // 2) // ns)


def _ladder(max_workers: Int) -> List[Int]:
    """1, 2, 4, 8, ... up to `max_workers`, with `max_workers` itself last."""
    var steps = List[Int]()
    var w = 1
    while w <= max_workers:
        steps.append(w)
        w *= 2
    if steps[len(steps) - 1] != max_workers:
        steps.append(max_workers)
    return steps^


def run_bench(max_workers: Int, n: Int) raises:
    """Scale both workloads from 1 worker up to `max_workers`."""
    print("threads-mojo bench")
    print("  logical CPUs :", num_cpus())
    print("  workers      : 1 ..", max_workers)
    print("  elements     :", n, "Int64 (", n * 8 // (1 << 20), "MiB )")
    print()

    var src = alloc[Int64](n)
    var dst = alloc[Int64](n)
    for i in range(n):
        src[i] = Int64(i & 0xFFFF)
        dst[i] = 0

    var block = alloc[Int64](CTX_CELLS)
    var ctx = opaque_ptr(Int(block))
    var cells = i64_ptr(Int(block))
    cells[C_SRC] = Int64(Int(src))
    cells[C_DST] = Int64(Int(dst))
    cells[C_N] = Int64(n)
    cells[C_UNUSED] = 0

    # Four tasks per worker: enough to hide any straggler, few enough that the
    # one atomic per task is invisible.
    var max_chunks = max_workers * 4
    var partials = alloc[Int64](max_chunks)
    cells[C_PARTIALS] = Int64(Int(partials))

    # Warm the pages so the first row is not paying for first touch.
    cells[C_CHUNKS] = Int64(max_chunks)
    parallel_for[_sum_chunk](max_chunks, ctx, num_workers=max_workers)

    print("  parallel sum over", n, "Int64")
    print("    workers      ms   speedup   checksum")
    var base_ns = 0
    var expected = Int64(0)
    var steps = _ladder(max_workers)
    for step in range(len(steps)):
        var workers = steps[step]
        var chunks = workers * 4
        cells[C_CHUNKS] = Int64(chunks)
        var t0 = perf_counter_ns()
        parallel_for[_sum_chunk](chunks, ctx, num_workers=workers)
        var t1 = perf_counter_ns()
        var total = Int64(0)
        for i in range(chunks):
            total += partials[i]
        if workers == 1:
            base_ns = t1 - t0
            expected = total
        if total != expected:
            raise Error("parallel sum disagreed with the single-worker result")
        print(
            _pad(String(workers), 11),
            _pad(_ms(t1 - t0), 9),
            _pad(_speedup(base_ns, t1 - t0) + "x", 9),
            _pad(String(total), 14),
        )
    print()

    print("  parallel memcpy of", n * 8 // (1 << 20), "MiB")
    print("    workers      ms   speedup   GB/s")
    base_ns = 0
    for step in range(len(steps)):
        var workers = steps[step]
        var chunks = workers * 4
        cells[C_CHUNKS] = Int64(chunks)
        var t0 = perf_counter_ns()
        parallel_for[_copy_chunk](chunks, ctx, num_workers=workers)
        var t1 = perf_counter_ns()
        if workers == 1:
            base_ns = t1 - t0
        var elapsed = t1 - t0 if t1 > t0 else 1
        # Bytes moved counts both the read and the write; bytes-per-ns is GB/s.
        var gb_per_s = (2 * n * 8 * 100) // elapsed
        print(
            _pad(String(workers), 11),
            _pad(_ms(elapsed), 9),
            _pad(_speedup(base_ns, elapsed) + "x", 9),
            _pad(_two_dp(gb_per_s), 8),
        )
    for i in range(0, n, max(1, n // 16)):
        if dst[i] != src[i]:
            raise Error("parallel memcpy produced a wrong byte at ", i)
    print()
    print("  (memcpy is bandwidth bound — it is here to show where scaling")
    print("   stops, not to show a straight line.)")

    partials.unsafe_free()
    block.unsafe_free()
    dst.unsafe_free()
    src.unsafe_free()


def main() raises:
    var args = argv()
    if len(args) < 2:
        print(USAGE)
        return
    var command = String(args[1])

    var max_workers = num_cpus()
    var elements = 100_000_000
    var i = 2
    while i < len(args):
        var flag = String(args[i])
        if flag == "--workers" and i + 1 < len(args):
            max_workers = Int(String(args[i + 1]))
            i += 2
        elif flag == "--elements" and i + 1 < len(args):
            elements = Int(String(args[i + 1]))
            i += 2
        else:
            print(USAGE)
            raise Error("unknown option: ", flag)

    if command == "info":
        print("logical CPUs :", num_cpus())
        print("thread id    :", current_thread_id())
    elif command == "bench":
        if max_workers < 1:
            raise Error("--workers must be at least 1")
        if elements < 1:
            raise Error("--elements must be at least 1")
        run_bench(max_workers, elements)
    else:
        print(USAGE)
        raise Error("unknown command: ", command)
