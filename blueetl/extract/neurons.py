import logging
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, NEURON_CLASS_INDEX
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.utils import timed

L = logging.getLogger(__name__)


class Neurons(BaseExtractor):
    COLUMNS = [CIRCUIT_ID, NEURON_CLASS, GID, NEURON_CLASS_INDEX]

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        # ensure that the neurons are sorted
        self._df: pd.DataFrame = self._df.sort_values(self.COLUMNS, ignore_index=True)

    @staticmethod
    def _get_gids(
        circuit, target: Optional[str], neuron_classes: Dict[str, Dict], limit: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        properties = list(set(chain.from_iterable(neuron_classes.values())))
        properties = [p for p in properties if not p.startswith("$")]
        with timed(L.info, "Cells loaded from circuit"):
            cells_group = {"$target": target} if target else None
            cells = circuit.cells.get(group=cells_group, properties=properties)
        result = {}
        for neuron_class, group in neuron_classes.items():
            group = group.copy()
            neuron_limit = group.pop("$limit", limit)
            # selection by gid is different because the gid index has no name
            selected_gids = group.pop(GID, None)
            gids = cells.etl.q(group).index.to_numpy()
            if selected_gids:
                gids = np.intersect1d(gids, selected_gids)
            neuron_count = len(gids)
            if neuron_limit and neuron_count > neuron_limit:
                gids = np.random.choice(gids, size=neuron_limit, replace=False)
            gids.sort()
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
    def from_simulations(
        cls,
        simulations: Simulations,
        target: Optional[str],
        neuron_classes: Dict[str, Dict],
        limit: Optional[int] = None,
    ) -> "Neurons":
        grouped = simulations.df.groupby([CIRCUIT_ID])[CIRCUIT].first()
        records: List[Tuple[int, str, int, int]] = []
        for circuit_id, circuit in grouped.items():
            gids_by_class = cls._get_gids(circuit, target, neuron_classes, limit=limit)
            records.extend(
                (circuit_id, neuron_class, gid, neuron_class_index)
                for neuron_class, gids in gids_by_class.items()
                for neuron_class_index, gid in enumerate(gids)
            )
        df = pd.DataFrame.from_records(records, columns=cls.COLUMNS)
        return cls(df)

    def count_by_neuron_class(self, observed=True) -> pd.Series:
        """Return the number of gids for each circuit and neuron class."""
        return self.df.groupby([CIRCUIT_ID, NEURON_CLASS], observed=observed)[GID].count()
