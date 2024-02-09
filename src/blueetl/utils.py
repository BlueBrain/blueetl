"""Common utilities."""

import hashlib
import importlib
import itertools
import json
import logging
import os.path
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import yaml

from blueetl.constants import DTYPES
from blueetl.types import StrOrPath


class CachedPropertyMixIn:
    """MixIn to be used with classes using cached_property to be skipped when pickled."""

    def __getstate__(self) -> dict:
        """Get the object state when the object is pickled."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(getattr(self.__class__, key, None), cached_property)
        }

    def __setstate__(self, state: dict) -> None:
        """Set the object state when the object is unpickled."""
        self.__dict__.update(state)


@contextmanager
def timed(log: Callable, msg, *args) -> Iterator[None]:
    """Context manager to log the execution time using the specified logger function."""
    log(f"{msg}...", *args)
    start_time = time.monotonic()
    status = "failed"
    try:
        yield
        status = "done"
    finally:
        elapsed = time.monotonic() - start_time
        log(f"{msg} [{status} in {elapsed:.2f} seconds]", *args)


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
        # The custom dumper dumps some unsupported types (for example Path) as simple strings.
        yaml.dump(data, stream=f, sort_keys=False, Dumper=_get_internal_yaml_dumper(), **kwargs)


def load_json(filepath: StrOrPath, *, encoding: str = "utf-8", **kwargs) -> Any:
    """Load from JSON file."""
    with open(filepath, encoding=encoding) as f:
        return json.load(f, **kwargs)


def dump_json(
    filepath: StrOrPath, data: Any, *, encoding: str = "utf-8", indent: int = 2, **kwargs
) -> None:
    """Dump to JSON file."""
    with open(filepath, mode="w", encoding=encoding) as fp:
        json.dump(data, fp, indent=indent, **kwargs)


def relpath(path: StrOrPath, start: StrOrPath) -> Path:
    """Return a relative filepath to path from the start directory.

    In Python>=3.12 it would be possible to use ``Path.relative_to`` with walk_up=True.
    """
    return Path(os.path.relpath(path, start=start))


def ensure_list(x: Any) -> list:
    """Always return a list from the given argument."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


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
    return getattr(importlib.import_module(module_name), func_name)


def resolve_path(*paths: StrOrPath, symlinks: bool = False) -> Path:
    """Make the path absolute and return a new path object.

    It may be different from calling Path.resolve(), because Path.resolve() always resolve symlinks.
    It may be different from calling Path.absolute(), because Path.absolute() doesn't remove the
    relative paths. For example, Path('/tmp/..').absolute() == PosixPath('/tmp/..').
    """
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
    # pylint: disable=import-outside-toplevel
    # imported here because optional
    from pydantic import BaseModel

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


def extract_items(obj: dict[str, Any], path: Optional[str] = None) -> Iterator[tuple[str, Any]]:
    """Yield tuples (path, item) from the traversal of the given dict.

    Each yielded path is obtained from the concatenation of the nested keys, separated by ".".

    All the keys of the dicts are expected to be strings, not containing ".".

    For examples, iterating over the result of:

    .. code-block:: python

        extract_items(
            {
                "latency": {"params": {"onset": False}},
                "decay": {"params": {"ratio": [0.25, 0.5, 0.75]}},
                "baseline_PSTH": {"params": {"bin_size": 0.5, "sigma": 0, "offset": -6}},
            }
        )

    would yield the following tuples:

    .. code-block:: python

        ("latency.params.onset", False)
        ("decay.params.ratio",  [0.25, 0.5, 0.75])
        ("baseline_PSTH.params.bin_size", 0.5)
        ("baseline_PSTH.params.sigma", 0)
        ("baseline_PSTH.params.offset", -6)

    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str)
            assert "." not in k
            yield from extract_items(v, path=k if path is None else f"{path}.{k}")
    else:
        yield path, obj


def all_equal(iterable: Iterable) -> bool:
    """Return True if all the items of the given iterable are equal, False otherwise.

    Notes:
    - The items doesn't need to be hashable, because they are compared with equality.
    - The function returns True also when the iterable doesn't contain any item.
    """
    prev = None
    for n, item in enumerate(iterable):
        if n > 0 and item != prev:
            return False
        prev = item
    return True


def import_optional_dependency(name: str) -> Any:
    """Import an optional dependency.

    If a dependency is missing, an ImportError with a custom message is raised. Based on:
    https://github.com/pandas-dev/pandas/blob/0d853e77/pandas/compat/_optional.py#L85

    Args:
        name: The module name.

    Returns:
        ModuleType: The imported module, if found.
    """
    try:
        module = importlib.import_module(name)
    except ImportError as ex:
        msg = f"Missing optional dependency {name!r}. Use pip to install it."
        raise ImportError(msg) from ex

    return module


def copy_config(src: StrOrPath, dst: StrOrPath) -> None:
    """Copy the analysis configuration file to a different location.

    If the simulation_campaign path is relative, then it's resolved
    relatively to the directory containing the original configuration file.

    The output path, instead, is not resolved even when it's relative.

    Note that any comment present in the original file is not preserved.
    """
    src = Path(src)
    config = load_yaml(src)
    config["simulation_campaign"] = resolve_path(src.parent, config["simulation_campaign"])
    dump_yaml(dst, config, default_flow_style=None)
