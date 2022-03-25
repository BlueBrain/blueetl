import logging
from itertools import chain

import numpy as np
import pandas as pd

from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS
from blueetl.extract.base import BaseExtractor
from blueetl.utils import timed

L = logging.getLogger(__name__)


class Neurons(BaseExtractor):
    COLUMNS = [CIRCUIT_ID, NEURON_CLASS, GID]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # ensure that the neurons are sorted
        self._df = self._df.sort_values(self.COLUMNS, ignore_index=True)

    @staticmethod
    def _get_gids(circuit, target, neuron_classes, limit=None):
        properties = list(set(chain.from_iterable(neuron_classes.values())))
        properties = [p for p in properties if not p.startswith("$")]
        with timed(L.info, "Cells loaded from circuit"):
            cells = circuit.cells.get(group={"$target": target}, properties=properties)
        result = {}
        for neuron_class, group in neuron_classes.items():
            group = group.copy()
            neuron_limit = group.pop("$limit", limit)
            gids = cells.etl.q(group).index.to_numpy()
            neuron_count = len(gids)
            if neuron_limit and neuron_count > neuron_limit:
                gids = np.random.choice(gids, size=neuron_limit, replace=False)
            result[neuron_class] = gids
            L.info(
                "Selected gids for %s: %s/%s (limit=%s)",
                neuron_class,
                len(gids),
                neuron_count,
                neuron_limit,
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

    def count_by_neuron_class(self):
        """Return the number of gids for each circuit and neuron class."""
        return self.df.groupby([CIRCUIT_ID, NEURON_CLASS])[GID].count()
