"""OS threads: `ThreadHandle`, `num_cpus`, `current_thread_id`, CPU pinning.

A thin, honest wrapper over `pthread_create` / `pthread_join`. There is no
thread pool here and no runtime — one `ThreadHandle` is one kernel thread, and
you are responsible for joining it.

## The send contract

`pthread_create` takes a C function pointer and a single `void *`. That is the
entire channel, and this library does not pretend otherwise:

- the start routine must be a **thin** (non-capturing) function
  `def(OpaquePtr) thin -> OpaquePtr`;
- everything the thread needs travels through the `OpaquePtr` argument;
- the caller guarantees that whatever the pointer addresses **outlives the
  join**.

That last point is not a style preference. Mojo destroys a value at its last
*use*, not at the end of the enclosing scope, so a context that the main thread
stops mentioning after `spawn` can be freed while the worker is still reading
it. Keep the context alive across the join — mention it after `join()`, or heap
allocate it and free it after `join()`. `threads.parallel.parallel_for` does the
latter, and `tests/test_threads.mojo` has a test that fails loudly if the
ordering is ever broken.

## Platforms

pthread symbols are resolved with plain `external_call`, no `dlopen`. On macOS
pthread lives in `libSystem`, which is always linked; on Linux with glibc 2.34
or newer `libpthread` is folded into `libc`. Both are in the default dynamic
link namespace, so the symbols resolve without any link flag. This is verified
on `ubuntu-latest` and `macos-latest` in CI.
"""

from std.ffi import external_call, c_int, c_size_t
from std.memory.alloc import unsafe_alloc
from std.sys.info import CompilationTarget

from .ffi import OpaquePtr, NULL


comptime StartFn = def(OpaquePtr) thin -> OpaquePtr
"""The one callable shape that can cross a thread boundary: a non-capturing
function taking and returning the opaque context pointer."""


# ── ThreadHandle ─────────────────────────────────────────────────────────────


@fieldwise_init
struct ThreadHandle(Movable):
    """An owning handle to one live OS thread.

    `Movable` but deliberately **not** `Copyable`. A `pthread_t` names one
    kernel thread and POSIX forbids joining the same one twice, so
    "exactly one owner, exactly one join" belongs in the type system rather
    than in a comment.

    As defence in depth, `join()` also zeroes the stored id, so a second
    `join()` on the same handle short-circuits instead of calling
    `pthread_join` on a stale id (which is undefined behaviour, not an error
    return).
    """

    var _thread_id: UInt64
    """The `pthread_t`, stored as 64 bits — `unsigned long` on Linux, an opaque
    pointer on macOS, both pointer-sized. Zero means "joined or detached".
    Do not depend on the bit pattern."""

    @staticmethod
    def spawn[start: StartFn](arg: OpaquePtr) raises -> ThreadHandle:
        """Start a thread running `start(arg)`.

        Parameters:
            start: The thread body. Thin (non-capturing) and non-raising —
                pthread has no exception channel, so convert any failure into a
                value the caller can read out of `arg`.

        Args:
            arg: The context pointer handed to `start`. Must stay valid until
                the thread is joined.

        Returns:
            A handle the caller must `join()`.

        Raises:
            Error: If `pthread_create` fails, carrying its POSIX return code
                (`EAGAIN` when the process is out of threads, `EPERM` for
                scheduling-attribute refusals).
        """
        var tid = UInt64(0)
        var tid_ptr = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=tid))
        )
        # A NULL attr means PTHREAD_CREATE_JOINABLE with default stack size.
        var rc = external_call[
            "pthread_create",
            c_int,
            Pointer[UInt64, MutUntrackedOrigin],  # pthread_t *
            Int,  # const pthread_attr_t * (NULL)
            StartFn,  # void *(*)(void *)
            OpaquePtr,  # void *
        ](tid_ptr, NULL, start, arg)
        if rc != c_int(0):
            raise Error("pthread_create failed, rc=", Int(rc))
        return ThreadHandle(_thread_id=tid)

    def join(mut self) raises:
        """Wait for the thread to finish; discard its return value.

        Idempotent: after a successful join the handle's id is zeroed, so
        calling `join()` again returns immediately instead of re-entering
        `pthread_join` with a stale id.

        Raises:
            Error: If `pthread_join` fails. The handle is left untouched so the
                caller can retry or propagate.
        """
        if self._thread_id == 0:
            return
        var rc = external_call["pthread_join", c_int, UInt64, Int](
            self._thread_id, NULL
        )
        if rc != c_int(0):
            raise Error("pthread_join failed, rc=", Int(rc))
        self._thread_id = 0

    def detach(mut self) raises:
        """Release the thread without waiting for it; it cleans itself up when
        it returns.

        After this the handle is inert — `join()` becomes a no-op. Only detach
        a thread whose context outlives the whole process, or one that owns its
        context outright: nothing is left to tell you when it stopped reading.

        Raises:
            Error: If `pthread_detach` fails.
        """
        if self._thread_id == 0:
            return
        var rc = external_call["pthread_detach", c_int, UInt64](self._thread_id)
        if rc != c_int(0):
            raise Error("pthread_detach failed, rc=", Int(rc))
        self._thread_id = 0

    @always_inline
    def is_joinable(self) -> Bool:
        """Whether this handle still names a thread.

        Returns:
            False once `join()` or `detach()` has succeeded.
        """
        return self._thread_id != 0

    def pin_to_cpu(self, cpu: Int) raises:
        """Pin the thread to one logical CPU.

        On Linux this is a real, hard affinity set via
        `pthread_setaffinity_np`. **On macOS it is a documented no-op**: Darwin
        exposes only `thread_policy_set(THREAD_AFFINITY_POLICY)`, which is an
        affinity *hint* used to group threads onto a shared cache, not a pin,
        and pretending otherwise would be worse than doing nothing.

        Args:
            cpu: Zero-based logical CPU index.

        Raises:
            Error: On Linux if `pthread_setaffinity_np` fails — typically
                `EINVAL` for a cpu index outside the process's cpuset, which is
                what happens inside a restricted container. Never raises on
                macOS.
        """
        comptime if CompilationTarget.is_linux():
            # glibc's cpu_set_t is 1024 bits. Zero it, set one bit.
            comptime CPUSET_BYTES: Int = 128
            var mask = unsafe_alloc[UInt8](CPUSET_BYTES)
            for i in range(CPUSET_BYTES):
                mask[unsafe_offset=i] = 0
            var byte_index = cpu // 8
            if byte_index < CPUSET_BYTES:
                mask[unsafe_offset=byte_index] = mask[
                    unsafe_offset=byte_index
                ] | UInt8(1 << (cpu % 8))
            var rc = external_call[
                "pthread_setaffinity_np",
                c_int,
                UInt64,  # pthread_t
                c_size_t,  # cpusetsize
                Pointer[UInt8, MutUntrackedOrigin],  # cpu_set_t *
            ](self._thread_id, c_size_t(CPUSET_BYTES), mask)
            mask.unsafe_free()
            if rc != c_int(0):
                raise Error("pthread_setaffinity_np failed, rc=", Int(rc))
        else:
            # macOS: no hard pin exists. Leave the scheduler alone.
            pass


# ── Process- and thread-level queries ────────────────────────────────────────


def num_cpus() -> Int:
    """The number of logical CPUs available to this process.

    Uses `sysconf(_SC_NPROCESSORS_ONLN)`, whose constant differs between the
    platforms (84 on Linux, 58 on Darwin).

    Returns:
        At least 1 — a failed `sysconf` is reported as 1 rather than 0, so
        callers can divide by it.
    """
    comptime if CompilationTarget.is_linux():
        comptime SC_NPROCESSORS_ONLN: c_int = 84
        var n = external_call["sysconf", Int, c_int](SC_NPROCESSORS_ONLN)
        return n if n > 0 else 1
    else:
        comptime SC_NPROCESSORS_ONLN: c_int = 58
        var n = external_call["sysconf", Int, c_int](SC_NPROCESSORS_ONLN)
        return n if n > 0 else 1


@always_inline
def current_thread_id() -> UInt64:
    """The calling thread's `pthread_self()`.

    Returns:
        An opaque, process-unique-while-live identifier. Useful for asserting
        that work really did run somewhere else; not a stable OS-wide tid.
    """
    return external_call["pthread_self", UInt64]()


@always_inline
def yield_now():
    """Hand the rest of this time slice to another runnable thread
    (`sched_yield`). The polite thing to do inside a spin loop."""
    _ = external_call["sched_yield", c_int]()
