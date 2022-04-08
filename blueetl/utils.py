import os.path
import time
from contextlib import contextmanager
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Union

import pandas as pd
import yaml

from blueetl.constants import DTYPES


@contextmanager
def timed(log, msg, *args):
    """Context manager to log the execution time using the specified logger function."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start_time
        log(f"{msg} [{elapsed:.2f} seconds]", *args)


def load_yaml(filepath):
    """Load from YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(filepath, data):
    """Dump to YAML file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, stream=f)


def ensure_list(x):
    """Return x if x is a list or a tuple, [x] otherwise."""
    return x if isinstance(x, (list, tuple)) else [x]


def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the columns cast to the predefined types."""
    return df.astype({k: DTYPES[k] for k in df.columns if k in DTYPES})


def import_by_string(full_name):
    module_name, _, func_name = full_name.rpartition(".")
    return getattr(import_module(module_name), func_name)


def resolve_path(*paths: Union[str, PathLike], symlinks=False) -> Path:
    """Make the path absolute and return a new path object."""
    if symlinks:
        # resolve any symlinks
        return Path(*paths).resolve()
    # does not resolve symbolic links
    return Path(os.path.abspath(Path(*paths)))
