import functools
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

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
        clean_slurm_env()
        setup_logging(log_format=log_format, log_level=log_level)
        result = func(analysis_config)
        if analysis_output:
            dump_json(result, analysis_output)
        return result

    if func.__module__ == "__main__":
        # if the script is called directly, automatically execute the function
        wrapper(
            analysis_config=load_json(Path(sys.argv[1])),
            analysis_output=Path(sys.argv[2]),
        )

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
