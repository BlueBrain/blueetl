import logging
from itertools import chain

import numpy as np
import pandas as pd

from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    COUNT,
    GID,
    NEURON_CLASS,
    SIMULATION,
    SIMULATION_ID,
)
from blueetl.utils import ensure_dtypes, timed

L = logging.getLogger(__name__)


class Neurons:
    def __init__(self, df: pd.DataFrame):
        assert set(df.columns) == {CIRCUIT_ID, NEURON_CLASS, GID}
        self._df = ensure_dtypes(df)
        # FIXME: do we need to ensure that the neurons are sorted?
        self._df = self._df.sort_values([CIRCUIT_ID, NEURON_CLASS, GID], ignore_index=True)

    @property
    def df(self):
        return self._df

    @staticmethod
    def _get_gids(circuit, target, neuron_classes, limit=None, sort=False):
        properties = list(set(chain.from_iterable(neuron_classes.values())))
        properties = [p for p in properties if not p.startswith("$")]
        with timed(L.info, "Cells loaded from circuit"):
            cells = circuit.cells.get(group={"$target": target}, properties=properties)
        result = {}
        for neuron_class, group in neuron_classes.items():
            group = group.copy()
            neuron_limit = group.pop("$limit", limit)
            neuron_sort = group.pop("$sort", sort)
            gids = cells.etl.query_dict(group).index.to_numpy()
            neuron_count = len(gids)
            if neuron_limit and neuron_count > neuron_limit:
                gids = np.random.choice(gids, size=neuron_limit, replace=False)
            if neuron_sort:
                gids.sort()
            result[neuron_class] = gids
            L.info(
                "Selected gids for %s: %s/%s (limit=%s, sort=%s)",
                neuron_class,
                len(gids),
                neuron_count,
                neuron_limit,
                neuron_sort,
            )
        return result

    @classmethod
    def from_simulations(cls, simulations, target, neuron_classes, limit=None):
        grouped = simulations.df.groupby([CIRCUIT_ID], sort=False)[CIRCUIT].first()
        records = []
        for circuit_id, circuit in grouped.items():
            gids_by_class = cls._get_gids(circuit, target, neuron_classes, limit=limit)
            records.extend(
                (circuit_id, neuron_class, gid)
                for neuron_class, gids in gids_by_class.items()
                for gid in gids
            )
        df = pd.DataFrame.from_records(records, columns=[CIRCUIT_ID, NEURON_CLASS, GID])
        return cls(df)

    @classmethod
    def from_pandas(cls, df):
        return cls(df)

    def to_pandas(self):
        """Dump neurons to a dataframe that can be serialized and stored."""
        # Prevent error: Cannot store a category dtype in a HDF5 dataset that uses format="fixed"
        return self.df.astype({NEURON_CLASS: object})

    def as_series(self):
        columns = [CIRCUIT_ID, NEURON_CLASS]
        return self.df.set_index(columns)[GID]

    def count_by_class(self):
        """Return the number of gids for each circuit and neuron class."""
        return self.df.groupby([CIRCUIT_ID, NEURON_CLASS])[GID].count()
