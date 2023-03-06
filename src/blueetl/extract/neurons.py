"""Neurons extractor."""
import logging
from itertools import chain
from typing import Optional

import numpy as np
import pandas as pd
from bluepy import Circuit

from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, NEURON_CLASS_INDEX
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.utils import timed

L = logging.getLogger(__name__)


class Neurons(BaseExtractor):
    """Neurons extractor class."""

    COLUMNS = [CIRCUIT_ID, NEURON_CLASS, GID, NEURON_CLASS_INDEX]

    def __init__(self, df: pd.DataFrame, cached: bool, filtered: bool) -> None:
        """Initialize the extractor."""
        super().__init__(df, cached=cached, filtered=filtered)
        # ensure that the neurons are sorted
        self._df: pd.DataFrame = self._df.sort_values(self.COLUMNS, ignore_index=True)

    @staticmethod
    def _get_gids(
        circuit: Circuit,
        target: Optional[str],
        neuron_classes: dict[str, dict],
        limit: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        cells_cache = {}
        properties = set(chain.from_iterable(neuron_classes.values()))
        properties = [p for p in properties if not p.startswith("$")]

        def _load_cells(_target):
            if _target not in cells_cache:
                with timed(L.info, "Loading cells from circuit for target %s", _target):
                    _cells_group = {"$target": _target} if _target else None
                    _cells = circuit.cells.get(group=_cells_group, properties=properties)
                    cells_cache[_target] = _cells
            return cells_cache[_target]

        result = {}
        for neuron_class, group in neuron_classes.items():
            group = group.copy()
            neuron_limit = group.pop("$limit", limit)
            cells = _load_cells(group.pop("$target", target))
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
        neuron_classes: dict[str, dict],
        limit: Optional[int] = None,
    ) -> "Neurons":
        """Return a new Neurons instance from the given simulations and configuration.

        Args:
            simulations: Simulations extractor.
            target: target string, or None to not filter by target.
            neuron_classes: configuration dict of neuron classes to be extracted.
            limit: if specified, limit the number of extracted neurons.

        Returns:
            Neurons: new instance.
        """
        grouped = simulations.df.groupby([CIRCUIT_ID])[CIRCUIT].first()
        records: list[tuple[int, str, int, int]] = []
        for circuit_id, circuit in grouped.items():
            gids_by_class = cls._get_gids(circuit, target, neuron_classes, limit=limit)
            records.extend(
                (circuit_id, neuron_class, gid, neuron_class_index)
                for neuron_class, gids in gids_by_class.items()
                for neuron_class_index, gid in enumerate(gids)
            )
        df = pd.DataFrame.from_records(records, columns=cls.COLUMNS)
        return cls(df, cached=False, filtered=False)

    def count_by_neuron_class(self, observed: bool = True) -> pd.Series:
        """Return the number of gids for each circuit and neuron class.

        Args:
            observed: If True: only show observed values for categorical groupers.
                If False: show all values for categorical groupers.
        """
        return self.df.groupby([CIRCUIT_ID, NEURON_CLASS], observed=observed)[GID].count()
