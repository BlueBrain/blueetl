import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
from joblib import Parallel, delayed

from blueetl.core import L
from blueetl.core.constants import BLUEETL_JOBLIB_JOBS, BLUEETL_JOBLIB_VERBOSE
from blueetl.utils import setup_logging


@dataclass
class TaskContext:
    task_id: int
    loglevel: int
    seed: Optional[int]


class Task:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, ctx: TaskContext) -> Any:
        logformat = f"%(asctime)s %(levelname)s %(name)s [task={ctx.task_id}]: %(message)s"
        setup_logging(loglevel=ctx.loglevel, logformat=logformat)
        if ctx.seed is not None:
            np.random.seed(ctx.seed)
        return self.func()


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
    loglevel = L.getEffectiveLevel()
    verbose = os.getenv(BLUEETL_JOBLIB_VERBOSE)
    verbose = int(verbose) if verbose else 0 if loglevel >= logging.WARNING else 10
    jobs = jobs or os.getenv(BLUEETL_JOBLIB_JOBS)
    jobs = int(jobs) if jobs else max((os.cpu_count() or 1) // 2, 1)
    parallel = Parallel(n_jobs=jobs, backend=backend, verbose=verbose)
    return parallel(
        delayed(task)(
            ctx=TaskContext(
                task_id=i,
                loglevel=loglevel,
                seed=None if base_seed is None else base_seed + i,
            )
        )
        for i, task in enumerate(tasks)
    )
