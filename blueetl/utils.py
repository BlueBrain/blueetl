"""Common utilities."""
import hashlib
import json
import logging
import os.path
import time
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from pathlib import Path, PosixPath
from typing import Any, Callable, Iterator, List, Optional, Tuple, Type, Union

import pandas as pd
import yaml

from blueetl.constants import DTYPES
from blueetl.extract.types import StrOrPath


@contextmanager
def timed(log: Callable, msg: str, *args) -> Iterator[None]:
    """Context manager to log the execution time using the specified logger function."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start_time
        log(f"{msg} [{elapsed:.2f} seconds]", *args)


def setup_logging(loglevel: int, logformat: Optional[str] = None, **logparams) -> None:
    """Setup logging."""
    logformat = logformat or "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel, **logparams)


def load_yaml(filepath: StrOrPath) -> Any:
    """Load from YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_get_internal_yaml_loader())


def dump_yaml(filepath: StrOrPath, data: Any) -> None:
    """Dump to YAML file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, stream=f, sort_keys=False, Dumper=_get_internal_yaml_dumper())


def ensure_list(x: Any) -> Union[List, Tuple]:
    """Return x if x is a list or a tuple, [x] otherwise."""
    return x if isinstance(x, (list, tuple)) else [x]


def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the columns cast to the predefined types."""
    return df.astype({k: DTYPES[k] for k in df.columns if k in DTYPES})


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


def checksum_json(obj: Any) -> str:
    """Calculate and return the checksum of the given object converted to json."""
    return hashlib.blake2b(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _get_internal_yaml_dumper() -> Type[yaml.SafeDumper]:
    """Return the custom internal yaml dumper class."""

    class Dumper(yaml.SafeDumper):
        """Custom YAML Dumper."""

    def _path_representer(dumper, data):
        return dumper.represent_scalar("!path", str(data))

    Dumper.add_representer(PosixPath, _path_representer)
    return Dumper


@lru_cache(maxsize=None)
def _get_internal_yaml_loader() -> Type[yaml.SafeLoader]:
    """Return the custom internal yaml loader class."""

    class Loader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
        """Custom YAML Loader."""

    def _path_constructor(loader, node):
        return Path(loader.construct_scalar(node))

    Loader.add_constructor("!path", _path_constructor)
    return Loader
