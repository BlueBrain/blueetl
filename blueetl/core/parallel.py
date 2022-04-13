import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from joblib import Parallel, delayed

from blueetl.core.utils import L


@dataclass
class TaskContext:
    task_id: int
    log_level: int
    seed: Optional[int]


def run_parallel(
    tasks: Iterable[Callable[[TaskContext], Any]],
    jobs: Optional[int],
    backend: Optional[str],
    base_seed: Optional[int] = None,
) -> List[Any]:
    """Run tasks in parallel.

    Args:
        tasks: iterable of callable objects that will be called in separate threads or processes.
            The callable must accept a single parameter ctx, that will contain a TaskContext.
        jobs: number of jobs. If not specified, use the BLUEETL_PARALLEL_JOBS env variable,
            or use half of the available cpus. Set to 1 to disable parallelization.
        backend: backend passed to joblib. If not specified, use the joblib default (loky).
            Possible values: loky, multiprocessing, threading.
        base_seed: initial base seed. If specified, a different seed is added to the task context,
            and passed to each callable object.

    Returns:
        list of objects returned by the callable objects, in the same order.
    """
    log_level = L.getEffectiveLevel()
    # If verbose is more than 10, all iterations are printed to stderr.
    # Above 50, the output is sent to stdout.
    verbose = 0 if log_level >= logging.WARNING else 10
    if jobs is None:
        jobs = int(os.getenv("BLUEETL_PARALLEL_JOBS", "0")) or max((os.cpu_count() or 1) // 2, 1)
    parallel = Parallel(n_jobs=jobs, backend=backend, verbose=verbose)
    return parallel(
        delayed(task)(
            ctx=TaskContext(
                task_id=i,
                log_level=log_level,
                seed=None if base_seed is None else base_seed + i,
            )
        )
        for i, task in enumerate(tasks)
    )
