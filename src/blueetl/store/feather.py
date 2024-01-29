"""Feather data store."""

import logging
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)
_idx_prefix = "_index"
_idx_sep = ":"


def _index_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert index to columns in the given DataFrame.

    Feather does not support serializing generic Indexes and any MultiIndex for the index
    see https://github.com/pandas-dev/pandas/blob/v1.4.1/pandas/io/feather_format.py#L59-L81
    """
    if (
        not isinstance(df.index, (pd.Index, pd.RangeIndex))
        or df.index.name
        or not df.index.equals(pd.RangeIndex.from_range(range(len(df))))
    ):
        new_columns = {}
        for i, name in enumerate(df.index.names):
            # in case of unnamed MultiIndex, df.reset_index() automatically assigns level_N
            key = name or f"level_{i}"
            value = f"{_idx_prefix}{_idx_sep}{i}{_idx_sep}{name or ''}"
            new_columns[key] = value
        return df.reset_index().rename(columns=new_columns)
    return df


def _columns_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to index in the given DataFrame."""
    mapping = {}
    for col in df.columns:
        if col.startswith(_idx_prefix):
            _, _, name = col.split(_idx_sep, 2)
            mapping[col] = name or None
    if mapping:
        df = df.set_index(list(mapping))
        # it works also for single level indexes
        df.index.set_names([mapping.get(name) for name in df.index.names], inplace=True)
    return df


class FeatherStore(BaseStore):
    """Feather data store."""

    @property
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""
        return "feather"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataframe to file, using the given name and the class extension."""
        path = self.path(name)
        with timed(L.debug, "Writing %s to %s", name, path):
            df = _index_to_columns(df)
            df.to_feather(path)

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension."""
        path = self.path(name)
        if not path.exists():
            return None
        with timed(L.debug, "Reading %s from %s", name, path):
            df = pd.read_feather(path)
            return _columns_to_index(df)
