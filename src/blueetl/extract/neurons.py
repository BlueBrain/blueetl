"""Neurons extractor."""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from bluepy import Circuit

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, NEURON_CLASS_INDEX
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.utils import ensure_list, timed

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
        neuron_classes: dict[str, NeuronClassConfig],
        limit: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        def _get_property_names() -> list[str]:
            """Return the list of properties to be retrieved from the cells DataFrame."""
            properties_set = set()
            for conf in neuron_classes.values():
                for query_dict in ensure_list(conf.query):
                    properties_set.update(query_dict)
            return sorted(properties_set)

        def _load_cells(_target: Optional[str]) -> pd.DataFrame:
            """Load and return the cells for a given target from the circuit or from the cache.

            If target is None or empty string, all the cells are loaded.
            """
            _target = _target or ""
            if _target not in cells_cache:
                msg = "Loading cells from circuit "
                msg += f"using target {_target}" if _target else "without using target"
                with timed(L.info, msg):
                    _cells_group = {"$target": _target} if _target else None
                    _cells = circuit.cells.get(group=_cells_group, properties=property_names)
                    cells_cache[_target] = _cells
            return cells_cache[_target]

        def _filter_gids_by_neuron_class(name: str, config: NeuronClassConfig) -> np.ndarray:
            neuron_limit = limit if config.limit is None else config.limit
            neuron_target = target if config.target is None else config.target
            cells = _load_cells(neuron_target)
            gids = cells.etl.q(config.query).index.to_numpy()
            if config.gids:
                gids = np.intersect1d(gids, config.gids)
            neuron_count = len(gids)
            if neuron_limit and neuron_count > neuron_limit:
                gids = np.random.choice(gids, size=neuron_limit, replace=False)
            gids.sort()
            L.info(
                "Selected gids for %s: %s/%s (limit=%s, target=%s)",
                name,
                len(gids),
                neuron_count,
                neuron_limit,
                neuron_target,
            )
            L.debug("Configured query: %s", config.query)
            L.debug("Configured gids: %s", config.gids)
            return gids

        cells_cache: dict[str, pd.DataFrame] = {}
        property_names = _get_property_names()
        return {
            name: _filter_gids_by_neuron_class(name, config)
            for name, config in neuron_classes.items()
        }

    @classmethod
    def from_simulations(
        cls,
        simulations: Simulations,
        target: Optional[str],
        neuron_classes: dict[str, NeuronClassConfig],
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
