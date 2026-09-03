"""Minimal pthreads for Mojo — the `threads` package of threads.mojo.

Spawn and join OS threads, share state through atomics and a mutex, and fan a
loop out over cores with `parallel_for`. No dependencies beyond the Mojo
standard library and the pthread symbols every libc already exports.

```mojo
from threads import parallel_for, num_cpus, AtomicCounter
```

Start with `threads.parallel` for the `parallel_for` contract, `threads.pool`
for long-lived workers that run until stopped, `threads.thread` for the thread
and send-contract rules, and `threads.atomic` for why this tin does not simply
re-export `std.atomic`.
"""

from .atomic import (
    AtomicCounter,
    AtomicFlag,
    atomic_fence_acquire,
    atomic_fence_release,
    atomic_fetch_add,
    atomic_load_acquire,
    atomic_load_relaxed,
    atomic_store_relaxed,
    atomic_store_release,
)
from .ffi import I64Ptr, NULL, OpaquePtr, i64_ptr, opaque_ptr
from .mutex import (
    COND_BYTES,
    CondVar,
    CondVarRef,
    MUTEX_BYTES,
    Mutex,
    MutexRef,
)
from .parallel import (
    TaskFn,
    ThreadGroup,
    WorkFn,
    join_all,
    parallel_for,
    spawn_n,
)
from .pool import (
    PoolTaskFn,
    TypedPool,
    WorkerFn,
    WorkerPool,
)
from .thread import (
    StartFn,
    ThreadHandle,
    current_thread_id,
    num_cpus,
    yield_now,
)
