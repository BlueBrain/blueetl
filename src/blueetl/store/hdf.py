"""HDF data store."""

import logging
from typing import Optional

import pandas as pd

from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)


def _category_to_object(df: pd.DataFrame) -> pd.DataFrame:
    # Prevent error: Cannot store a category dtype in a HDF5 dataset that uses format="fixed"
    return df.astype({name: "object" for name in df.columns[df.dtypes == "category"]})


class HDFStore(BaseStore):
    """HDF data store."""

    @property
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""
        return "h5"

    def dump(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataframe to file, using the given name and the class extension."""
        path = self.path(name)
        with timed(L.debug, "Writing %s to %s", name, path):
            df = _category_to_object(df)
            df.to_hdf(
                str(path),
                key=name,
                mode="w",
                # complib="blosc",
                # complevel=9,
                # format="fixed",
                # format="table",
            )

    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension."""
        path = self.path(name)
        if not path.exists():
            return None
        with timed(L.debug, "Reading %s from %s", name, path):
            return pd.read_hdf(path, key=name)
