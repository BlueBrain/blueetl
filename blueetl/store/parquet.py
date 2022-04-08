import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)


class ParquetStore(BaseStore):
    def _get_path(self, name: str) -> Path:
        return self.basedir / f"{name}.parquet"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        path = self._get_path(name)
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
        path = self._get_path(name)
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
