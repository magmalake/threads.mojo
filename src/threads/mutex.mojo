"""A `pthread_mutex_t` (and `pthread_cond_t`) you can share between threads.

Neither type has a portable size, so both are handled as an opaque
**64-byte blob**. That covers every libc this tin targets with room to spare:
`pthread_mutex_t` is 40 bytes on glibc x86-64, 48 on musl x86-64, 8 on glibc
aarch64 and 64 on macOS; `pthread_cond_t` is 48 on glibc and 48 on macOS. The
blob is heap allocated, so its address is stable across moves of the owning
Mojo value — which matters, because the address is what the worker threads see.

## Owning vs. viewing

`Mutex` owns its blob: it initialises on construction, destroys and frees in
`__del__`, and is move-only so there is exactly one owner. But a mutex is
useless unless several threads reach the *same* one, and threads only get a raw
context pointer. So `Mutex.address()` hands out the blob address and
`MutexRef.at(address)` rebuilds a lock/unlock view on the worker side. The view
is copyable and owns nothing.

## Why there is no scope guard

The obvious `with_lock` / RAII `Guard` is deliberately missing. Mojo destroys a
value at its **last use**, not at the end of its scope, so a guard whose only
job is to unlock in `__del__` gets destroyed — and therefore unlocks — at the
point you stop mentioning it, which is usually the first line of the critical
section. That is a silent correctness bug, not a compile error. Call `lock()`
and `unlock()` explicitly, or use `with_lock`, which takes a thin function and
brackets it properly.
"""

from std.ffi import external_call, c_int
from std.memory import alloc

from .ffi import OpaquePtr, NULL


comptime MUTEX_BYTES: Int = 64
"""Size of the opaque `pthread_mutex_t` blob. Generous upper bound over
glibc / musl / macOS on x86-64 and aarch64."""

comptime COND_BYTES: Int = 64
"""Size of the opaque `pthread_cond_t` blob."""

comptime BlobPtr = UnsafePointer[UInt8, MutUntrackedOrigin]


# ── raw pthread_mutex_t operations on a blob address ─────────────────────────


@always_inline
def _mutex_init(blob: BlobPtr) -> Int:
    return Int(
        external_call["pthread_mutex_init", c_int, BlobPtr, Int](blob, NULL)
    )


@always_inline
def _mutex_destroy(blob: BlobPtr) -> Int:
    return Int(external_call["pthread_mutex_destroy", c_int, BlobPtr](blob))


@always_inline
def _mutex_lock(blob: BlobPtr) -> Int:
    return Int(external_call["pthread_mutex_lock", c_int, BlobPtr](blob))


@always_inline
def _mutex_trylock(blob: BlobPtr) -> Int:
    return Int(external_call["pthread_mutex_trylock", c_int, BlobPtr](blob))


@always_inline
def _mutex_unlock(blob: BlobPtr) -> Int:
    return Int(external_call["pthread_mutex_unlock", c_int, BlobPtr](blob))


# ── MutexRef ─────────────────────────────────────────────────────────────────


@fieldwise_init
struct MutexRef(Copyable, ImplicitlyCopyable, Movable):
    """A non-owning lock/unlock view of an initialised mutex blob.

    This is what a worker thread reconstructs from an address it found in its
    context block. It initialises nothing and frees nothing.
    """

    var _blob: BlobPtr
    """The mutex blob, already initialised by the owning `Mutex`."""

    @staticmethod
    @always_inline
    def at(address: Int) -> Self:
        """View the mutex whose blob starts at `address`.

        Args:
            address: From `Mutex.address()`.

        Returns:
            A lock/unlock view.
        """
        return Self(BlobPtr(unsafe_from_address=address))

    @always_inline
    def address(self) -> Int:
        """Return the blob address.

        Returns:
            The address this view was built from.
        """
        return Int(self._blob)

    @always_inline
    def lock(self):
        """Block until this thread holds the mutex."""
        _ = _mutex_lock(self._blob)

    @always_inline
    def unlock(self):
        """Release the mutex. Only the holding thread may call this."""
        _ = _mutex_unlock(self._blob)

    @always_inline
    def try_lock(self) -> Bool:
        """Take the mutex if it is free, without blocking.

        Returns:
            True if the mutex is now held by this thread.
        """
        return _mutex_trylock(self._blob) == 0

    def with_lock[body: def(OpaquePtr) thin -> None](self, ctx: OpaquePtr):
        """Run `body(ctx)` with the mutex held, then release it.

        Parameters:
            body: A thin, non-raising function. Non-raising is load-bearing:
                an escaping exception would skip the unlock.

        Args:
            ctx: Passed straight through to `body`.
        """
        self.lock()
        body(ctx)
        self.unlock()


# ── Mutex ────────────────────────────────────────────────────────────────────


struct Mutex(Movable):
    """An owned `pthread_mutex_t`. Move-only; destroys and frees its blob.

    Default (non-recursive, non-error-checking) attributes: locking one twice
    from the same thread deadlocks, and unlocking one you do not hold is
    undefined. That is plain POSIX behaviour, not something this wrapper adds.
    """

    var _blob: BlobPtr
    """Heap-allocated blob. Stable address across moves of this value."""

    def __init__(out self) raises:
        """Allocate and initialise the mutex.

        Raises:
            Error: If `pthread_mutex_init` fails.
        """
        self._blob = alloc[UInt8](MUTEX_BYTES)
        for i in range(MUTEX_BYTES):
            self._blob[i] = 0
        var rc = _mutex_init(self._blob)
        if rc != 0:
            self._blob.unsafe_free()
            raise Error("pthread_mutex_init failed, rc=", rc)

    def __deinit__(deinit self):
        """Destroy and free the mutex. Destroying a *locked* mutex is undefined
        under POSIX, so join every thread that can touch it first."""
        _ = _mutex_destroy(self._blob)
        self._blob.unsafe_free()

    @always_inline
    def address(self) -> Int:
        """The blob address, to be stashed in a worker context block.

        Returns:
            The address; rebuild a usable view with `MutexRef.at`.
        """
        return Int(self._blob)

    @always_inline
    def ref(self) -> MutexRef:
        """A copyable non-owning view of this mutex.

        Returns:
            The view.
        """
        return MutexRef(self._blob)

    @always_inline
    def lock(self):
        """Block until this thread holds the mutex."""
        _ = _mutex_lock(self._blob)

    @always_inline
    def unlock(self):
        """Release the mutex."""
        _ = _mutex_unlock(self._blob)

    @always_inline
    def try_lock(self) -> Bool:
        """Take the mutex if it is free, without blocking.

        Returns:
            True if the mutex is now held by this thread.
        """
        return _mutex_trylock(self._blob) == 0

    def with_lock[body: def(OpaquePtr) thin -> None](self, ctx: OpaquePtr):
        """Run `body(ctx)` with the mutex held, then release it.

        Parameters:
            body: A thin, non-raising function.

        Args:
            ctx: Passed straight through to `body`.
        """
        self.lock()
        body(ctx)
        self.unlock()


# ── CondVar ──────────────────────────────────────────────────────────────────


@fieldwise_init
struct CondVarRef(Copyable, ImplicitlyCopyable, Movable):
    """A non-owning view of an initialised condition variable, paired with the
    mutex address the waiter holds."""

    var _blob: BlobPtr
    """The condition-variable blob."""

    @staticmethod
    @always_inline
    def at(address: Int) -> Self:
        """View the condition variable at `address`.

        Args:
            address: From `CondVar.address()`.

        Returns:
            A wait/signal view.
        """
        return Self(BlobPtr(unsafe_from_address=address))

    @always_inline
    def wait(self, mutex: MutexRef):
        """Atomically release `mutex` and block until signalled, then re-take
        it. Spurious wakeups are permitted by POSIX — always re-check your
        predicate in a loop.

        Args:
            mutex: The mutex this thread currently holds.
        """
        _ = external_call["pthread_cond_wait", c_int, BlobPtr, BlobPtr](
            self._blob, BlobPtr(unsafe_from_address=mutex.address())
        )

    @always_inline
    def signal(self):
        """Wake at least one waiter."""
        _ = external_call["pthread_cond_signal", c_int, BlobPtr](self._blob)

    @always_inline
    def broadcast(self):
        """Wake every waiter."""
        _ = external_call["pthread_cond_broadcast", c_int, BlobPtr](self._blob)


struct CondVar(Movable):
    """An owned `pthread_cond_t`. Move-only; destroys and frees its blob.

    Always used with a `Mutex`: take the mutex, check the predicate in a
    `while`, `wait` if it does not hold.
    """

    var _blob: BlobPtr
    """Heap-allocated blob, stable across moves."""

    def __init__(out self) raises:
        """Allocate and initialise the condition variable.

        Raises:
            Error: If `pthread_cond_init` fails.
        """
        self._blob = alloc[UInt8](COND_BYTES)
        for i in range(COND_BYTES):
            self._blob[i] = 0
        var rc = external_call["pthread_cond_init", c_int, BlobPtr, Int](
            self._blob, NULL
        )
        if rc != c_int(0):
            self._blob.unsafe_free()
            raise Error("pthread_cond_init failed, rc=", Int(rc))

    def __deinit__(deinit self):
        """Destroy and free the condition variable."""
        _ = external_call["pthread_cond_destroy", c_int, BlobPtr](self._blob)
        self._blob.unsafe_free()

    @always_inline
    def address(self) -> Int:
        """The blob address, to be stashed in a worker context block.

        Returns:
            The address; rebuild a usable view with `CondVarRef.at`.
        """
        return Int(self._blob)

    @always_inline
    def ref(self) -> CondVarRef:
        """A copyable non-owning view.

        Returns:
            The view.
        """
        return CondVarRef(self._blob)

    @always_inline
    def wait(self, mutex: MutexRef):
        """Wait on this condition variable. See `CondVarRef.wait`.

        Args:
            mutex: The mutex this thread currently holds.
        """
        self.ref().wait(mutex)

    @always_inline
    def signal(self):
        """Wake at least one waiter."""
        self.ref().signal()

    @always_inline
    def broadcast(self):
        """Wake every waiter."""
        self.ref().broadcast()
