"""Common utilities."""
import hashlib
import itertools
import json
import logging
import os.path
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import yaml
from pydantic import BaseModel

from blueetl.constants import DTYPES
from blueetl.types import StrOrPath


@contextmanager
def timed(log: Callable, msg: str, *args) -> Iterator[None]:
    """Context manager to log the execution time using the specified logger function."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start_time
        log(f"{msg} [{elapsed:.2f} seconds]", *args)


def timed_log(log: Callable) -> Callable:
    """Return a function to log the elapsed time since the initialization or the last call."""

    def _timed_log(msg, *args):
        nonlocal start_time
        now = time.monotonic()
        elapsed = now - start_time
        start_time = now
        log(f"{msg} [{elapsed:.2f} seconds]", *args)

    start_time = time.monotonic()
    return _timed_log


def setup_logging(loglevel: Union[int, str], logformat: Optional[str] = None, **logparams) -> None:
    """Setup logging."""
    logformat = logformat or "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel, **logparams)


def load_yaml(filepath: StrOrPath) -> Any:
    """Load from YAML file."""
    with open(filepath, encoding="utf-8") as f:
        # Any conversion when loading back the values can be done later,
        # so that the loaded object can be validated using jsonschema.
        return yaml.load(f, Loader=yaml.SafeLoader)


def dump_yaml(filepath: StrOrPath, data: Any, **kwargs) -> None:
    """Dump to YAML file."""
    with open(filepath, "w", encoding="utf-8") as f:
        # The custom dumper dumps unsupported types (for example Path) as simple strings.
        yaml.dump(data, stream=f, sort_keys=False, Dumper=_get_internal_yaml_dumper(), **kwargs)


def ensure_list(x: Any) -> Union[list, tuple]:
    """Return x if x is a list or a tuple, [x] otherwise."""
    return x if isinstance(x, (list, tuple)) else [x]


def ensure_dtypes(
    df: pd.DataFrame, desired_dtypes: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Return a DataFrame with the columns and index cast to the desired types.

    Args:
        df: original Pandas DataFrame.
        desired_dtypes: dict of names and desired dtypes. If None, the predefined dtypes are used.
            If the dict contains names not present in the columns or in the index, they are ignored.
            In the index, any (u)int16 or (u)int32 dtype are considered as (u)int64,
            since Pandas doesn't have a corresponding Index type for them.

    Returns:
        A new DataFrame with the desired dtypes, or the same DataFrame if the columns are unchanged.
    """
    if desired_dtypes is None:
        desired_dtypes = DTYPES
    # convert the columns data types
    if dtypes := {
        k: desired_dtypes[k]
        for k in df.columns
        if k in desired_dtypes and desired_dtypes[k] != df.dtypes.at[k]
    }:
        df = df.astype(dtypes)
    # convert the index data types
    if dtypes := {
        k: desired_dtypes[k]
        for k in df.index.names
        if k in desired_dtypes and desired_dtypes[k] != df.index.etl.dtypes.at[k]
    }:
        df.index = df.index.etl.astype(dtypes)
    return df


def import_by_string(full_name: str) -> Callable:
    """Import and return a function by name.

    Args:
        full_name: full name of the function, using dot as a separator if in a submodule.

    Returns:
        The imported function.
    """
    module_name, _, func_name = full_name.rpartition(".")
    return getattr(import_module(module_name), func_name)


def resolve_path(*paths: StrOrPath, symlinks: bool = False) -> Path:
    """Make the path absolute and return a new path object."""
    if symlinks:
        # resolve any symlinks
        return Path(*paths).resolve()
    # does not resolve symbolic links
    return Path(os.path.abspath(Path(*paths)))


def checksum(filepath: StrOrPath, chunk_size: int = 65536) -> str:
    """Calculate and return the checksum of the given file."""
    filehash = hashlib.blake2b()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            filehash.update(chunk)
    return filehash.hexdigest()


def checksum_str(s: str) -> str:
    """Calculate and return the checksum of the given string."""
    return hashlib.blake2b(s.encode("utf-8")).hexdigest()


def checksum_json(obj: Any) -> str:
    """Calculate and return the checksum of the given object converted to json."""
    return checksum_str(json.dumps(obj, sort_keys=True))


@cache
def _get_internal_yaml_dumper() -> type[yaml.SafeDumper]:
    """Return the custom internal yaml dumper class."""
    _representers = {
        Path: str,
        BaseModel: lambda data: data.dict(),
    }

    class Dumper(yaml.SafeDumper):
        """Custom YAML Dumper."""

        def represent_data(self, data):
            for cls in type(data).__mro__:
                if func := _representers.get(cls):
                    return self.represent_data(func(data))
            return super().represent_data(data)

    return Dumper


def dict_product(d: dict) -> Iterator[tuple]:
    """Iterate over the product of the values of the given dict.

    Yield tuples of tuples, where each item is composed by:

    - key
    - value
    - index of the key in the original list

    Example:
        Passing the dictionary:

        .. code-block:: python

            d = {"a": ["a1", "a2"], "b": ["b1", "b2"]}

        Yields tuples from:

        .. code-block:: python

            [
                (("a", "a1", 0), ("b", "b1", 0)),
                (("a", "a1", 0), ("b", "b2", 1)),
                (("a", "a2", 1), ("b", "b1", 0)),
                (("a", "a2", 1), ("b", "b2", 1)),
            ]

    """
    if d:
        yield from itertools.product(
            *[[(key, v, i) for i, v in enumerate(values)] for key, values in d.items()]
        )
