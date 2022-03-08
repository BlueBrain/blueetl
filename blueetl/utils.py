import time
from contextlib import contextmanager
from importlib import import_module

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


def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the columns cast to the predefined types."""
    return df.astype({k: DTYPES[k] for k in df.columns if k in DTYPES})


def import_by_string(full_name):
    module_name, _, func_name = full_name.rpartition(".")
    return getattr(import_module(module_name), func_name)
