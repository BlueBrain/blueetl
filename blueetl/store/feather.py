import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)
_idx_prefix = "index:"


def _index_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    # feather does not support serializing generic Indexes and any MultiIndex for the index
    # see https://github.com/pandas-dev/pandas/blob/v1.4.1/pandas/io/feather_format.py#L59-L81
    if (
        not isinstance(df.index, (pd.Index, pd.RangeIndex))
        or df.index.name
        or not df.index.equals(pd.RangeIndex.from_range(range(len(df))))
    ):
        new_columns = {name: f"{_idx_prefix}{name}" for name in df.index.names}
        return df.reset_index().rename(columns=new_columns)
    return df


def _columns_to_index(df: pd.DataFrame) -> pd.DataFrame:
    columns = {col: col[len(_idx_prefix) :] for col in df.columns if col.startswith(_idx_prefix)}
    if columns:
        return df.rename(columns=columns).set_index(list(columns.values()))
    return df


class FeatherStore(BaseStore):
    def _get_path(self, name: str) -> Path:
        return self.basedir / f"{name}.feather"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        path = self._get_path(name)
        with timed(L.info, "Writing %s to %s", name, path):
            df = _index_to_columns(df)
            df.to_feather(path)

    def load(self, name: str) -> Optional[pd.DataFrame]:
        path = self._get_path(name)
        if not path.exists():
            return None
        with timed(L.info, "Reading %s from %s", name, path):
            df = pd.read_feather(path)
            return _columns_to_index(df)
