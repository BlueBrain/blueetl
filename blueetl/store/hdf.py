import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)


def _category_to_object(df):
    # Prevent error: Cannot store a category dtype in a HDF5 dataset that uses format="fixed"
    return df.astype({name: "object" for name in df.columns[df.dtypes == "category"]})


class HDFStore(BaseStore):
    def _get_path(self, name: str) -> Path:
        return self.basedir / f"{name}.h5"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        path = self._get_path(name)
        with timed(L.debug, "Writing %s to %s", name, path):
            df = _category_to_object(df)
            df.to_hdf(
                path,
                key=name,
                mode="w",
                # complib="blosc",
                # complevel=9,
                # format="fixed",
                # format="table",
            )

    def load(self, name: str) -> Optional[pd.DataFrame]:
        path = self._get_path(name)
        if not path.exists():
            return None
        with timed(L.debug, "Reading %s from %s", name, path):
            return pd.read_hdf(path, key=name)
