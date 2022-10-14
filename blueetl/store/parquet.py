"""Parquet data store."""
import logging
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.types import StrOrPath
from blueetl.utils import timed

L = logging.getLogger(__name__)


class ParquetStore(BaseStore):
    """Parquet data store."""

    def __init__(self, basedir: StrOrPath) -> None:
        """Initialize the object."""
        super().__init__(basedir=basedir)
        self._dump_options = {
            "engine": "pyarrow",
            # "engine": "fastparquet",
            # "compression": "snappy",
            # Ensure that RangeIndex is converted to Int64Index in MultiIndexes with Pandas 1.5.0
            # See https://issues.apache.org/jira/browse/ARROW-17806
            # and https://github.com/pandas-dev/pandas/issues/46675
            "index": True,
            # "partition_cols": None,
            # "storage_options": None,
        }
        self._load_options = {
            # pyarrow (8.0.0, 9.0.0) may be affected by a memory leak,
            # and it's slower than fastparquet when reading dataframes with columns
            # containing lists encoded using the Dremel encoding.
            # See https://issues.apache.org/jira/browse/ARROW-17399
            # However, using a different engine for writing and reading may be less safe.
            "engine": "pyarrow",
            # "engine": "fastparquet",
            # "columns": None,
            # "storage_options": None,
            # "use_nullable_dtypes": False,
        }

    @property
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""
        return "parquet"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataframe to file, using the given name and the class extension."""
        path = self.path(name)
        with timed(L.debug, "Writing %s to %s", name, path):
            df.to_parquet(path=path, **self._dump_options)

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension."""
        path = self.path(name)
        if not path.exists():
            return None
        with timed(L.debug, "Reading %s from %s", name, path):
            return pd.read_parquet(path=path, **self._load_options)
