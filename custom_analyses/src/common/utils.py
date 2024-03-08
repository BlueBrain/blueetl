import functools
import json
import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path

import click

L = logging.getLogger("analysis")


def setup_logging(*, log_format: str, log_level: str | int) -> None:
    """Setup logging."""
    logging.basicConfig(format=log_format, level=log_level)


def load_json(path: Path | str) -> dict:
    """Load json from file."""
    with open(path, "r") as f:
        return json.load(f)


def dump_json(content: dict, path: Path | str) -> None:
    """Dump json to file."""
    with open(path, "w") as f:
        json.dump(content, f)


def run_analysis(func: Callable[[dict], dict]) -> Callable[..., dict]:
    """Decorator to be applied to the main function.

    The decorated function should accept the config dict in input, and return the output dict.

    If the script containing the decorated function is called from the CLI, the parameters are read
    from the CLI arguments, and the function is automatically executed.

    Example:

        @run_analysis
        def main(analysis_config: dict) -> dict:
    """

    @functools.wraps(func)
    def wrapper(
        *,
        analysis_config: dict,
        analysis_output: str | Path | None = None,
        log_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
        log_level: str | int = logging.INFO,
    ) -> dict:
        """Call the wrapped function, and write the result to file."""
        setup_logging(log_format=log_format, log_level=log_level)
        result = func(analysis_config)
        if analysis_output:
            dump_json(result, analysis_output)
        return result

    if func.__module__ == "__main__":
        # if the script is called directly, automatically wrap and execute the function
        @click.command()
        @click.option(
            "--log-level",
            default="INFO",
            help="Log level",
            type=click.Choice(
                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                case_sensitive=False,
            ),
            show_default=True,
        )
        @click.argument("analysis_config_file", type=click.Path(exists=True))
        @click.argument("analysis_output_file", type=click.Path(exists=False))
        def cli_wrapper(analysis_config_file, analysis_output_file, log_level):
            wrapper(
                analysis_config=load_json(Path(analysis_config_file)),
                analysis_output=Path(analysis_output_file),
                log_level=log_level,
            )

        cli_wrapper()

    return wrapper


def clean_slurm_env():
    """Remove PMI/SLURM variables that can cause issues when launching other slurm jobs.

    These variable are unset because launching slurm jobs with submitit from a node
    allocated using salloc would fail with the error:
        srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE
        are mutually exclusive.
    """
    for key in os.environ:
        if key.startswith(("PMI_", "SLURM_")) and not key.endswith(("_ACCOUNT", "_PARTITION")):
            L.debug("Deleting env variable %s", key)
            del os.environ[key]


def wait_for_slurm():
    """Wait for some time to allow sacct to return the correct status of the submitted jobs.

    This may be needed when the slurm ids have been reset, and re-used.

    See https://github.com/facebookincubator/submitit/issues/1660.
    """
    initial_sleep = float(os.getenv("SUBMIT_JOBS_INITIAL_SLEEP", "10"))
    L.debug("SUBMIT_JOBS_INITIAL_SLEEP=%s", initial_sleep)
    time.sleep(initial_sleep)


def isolated(func):
    """Isolate a function in a separate process.

    Note: it does not work as a decorator.

    Args:
        func (function): function to isolate.

    Returns:
        the isolated function.
    """

    def func_isolated(*args, **kwargs):
        with multiprocessing.Pool(1, maxtasksperchild=1) as pool:
            return pool.apply(func, args, kwargs)

    return func_isolated
