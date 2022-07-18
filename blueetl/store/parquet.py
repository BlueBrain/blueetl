"""Parquet data store."""
import logging
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)


class ParquetStore(BaseStore):
    """Parquet data store."""

    @property
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""
        return "parquet"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataframe to file, using the given name and the class extension."""
        path = self.path(name)
        with timed(L.debug, "Writing %s to %s", name, path):
            df.to_parquet(
                path=path,
                # engine="auto",
                # compression="snappy",
                # index=None,
                # partition_cols=None,
                # storage_options=None,
            )

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension."""
        path = self.path(name)
        if not path.exists():
            return None
        with timed(L.debug, "Reading %s from %s", name, path):
            return pd.read_parquet(
                path=path,
                # engine="auto",
                # columns=None,
                # storage_options=None,
                # use_nullable_dtypes=False,
            )
