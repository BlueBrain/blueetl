import logging
from pathlib import Path

import pandas as pd

from blueetl.utils import timed

L = logging.getLogger(__name__)


class HDFStore:
    def __init__(self, basedir):
        self.basedir = Path(basedir).resolve()
        self.basedir.mkdir(exist_ok=True)

    def dump(self, df, name):
        path = self.basedir / f"{name}.h5"
        with timed(L.info, "Writing %s to %s", name, path):
            df.to_hdf(
                path,
                key=name,
                mode="w",
                # complib="blosc",
                # complevel=9,
                # format="fixed",
                # format="table",
            )

    def load(self, name):
        path = self.basedir / f"{name}.h5"
        if not path.exists():
            return None
        with timed(L.info, "Reading %s from %s", name, path):
            return pd.read_hdf(path, key=name)
