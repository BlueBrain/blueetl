"""Parquet data store."""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blueetl.store.base import BaseStore
from blueetl.types import StrOrPath
from blueetl.utils import timed

L = logging.getLogger(__name__)


def _get_unified_schema(path: StrOrPath) -> pa.Schema:
    """Infer and return the unified schema from all the parquet files in the given directory.

    Needed because pq.read_table() would infer the schema from just the first file,
    unless a metadata file is used. See:
    https://github.com/ueshin/apache-arrow/blob/0a2cf3ac/python/pyarrow/parquet.py#L724

    This can cause problems when a column in the first file contains empty lists,
    since they are considered as:

      times: list<element: null>
        child 0, element: null

    while in other files containing lists of floats they are considered as:

      times: list<element: double>
        child 0, element: double

    and with pyarrow 15.0.0 this would raise:

      ArrowNotImplementedError:
        Unsupported cast from double to null using function cast_null

    Processing all the parquet files could require some time (around 4 seconds for 13824 files
    stored in the shared memory /dev/shm), so it doesn't parse all the files if possible.

    If the performance are improved, we could replace the content of the function with:

        schemas = [pq.read_schema(file_path) for file_path in Path(path).iterdir()]
        return pa.unify_schemas(schemas)
    """
    with timed(L.debug, f"Inferring schema for {path}") as messages:
        schemas = []
        null_list_type = pa.list_(pa.null())
        valid_fields: dict[str, bool] = {}
        for file_path in Path(path).iterdir():
            all_valid = True
            file_schema = pq.read_schema(file_path)
            for field in file_schema:
                if valid_fields.get(field.name, False) is False:
                    valid_fields[field.name] = not field.type.equals(null_list_type)
                    all_valid = all_valid and valid_fields[field.name]
            schemas.append(file_schema)
            if all_valid:
                # break because the merged schema doesn't contain any list of nulls
                break
        messages.append(f"with {len(schemas)} loaded schemas")
        return pa.unify_schemas(schemas)


class ParquetStore(BaseStore):
    """Parquet data store."""

    def __init__(self, basedir: StrOrPath) -> None:
        """Initialize the object."""
        super().__init__(basedir=basedir)
        self._dump_options: dict[str, Any] = {
            "engine": "pyarrow",
            # zstd typically provides a higher compression ratio than snappy,
            # and the cpu time is similar
            "compression": "zstd",
        }
        self._load_options: dict[str, Any] = {
            "engine": "pyarrow",
        }

    @property
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""
        return "parquet"

    def dump(self, df: pd.DataFrame, name: str, **kwargs) -> None:
        """Save a dataframe to file, using the given name and the class extension."""
        path = self.path(name)
        # Unless the parameter "index" is explicitly enforced, ensure that RangeIndex
        # is converted to Int64Index in MultiIndexes with Pandas >= 1.5.0.
        # See https://github.com/apache/arrow/issues/33030
        index = True if isinstance(df.index, pd.MultiIndex) else None
        with timed(L.debug, f"Writing {name or 'files'} to {path}"):
            df.to_parquet(path=path, **{"index": index, **self._dump_options, **kwargs})

    def load(self, name: str = "", **kwargs) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension.

        If name is empty, then consider and load all the files in the directory.
        """
        path = self.path(name)
        if not path.exists():
            return None
        if "schema" in kwargs:
            schema = kwargs.pop("schema")
        elif path.is_dir():
            schema = _get_unified_schema(path)
        else:
            schema = None
        with timed(L.debug, f"Reading {name or 'files'} from {path}"):
            return pd.read_parquet(path=path, **{"schema": schema, **self._load_options, **kwargs})
