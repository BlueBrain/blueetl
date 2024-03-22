"""Neurons extractor."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from blueetl.adapters.circuit import CircuitAdapter as Circuit
from blueetl.adapters.node_sets import NodeSetsAdapter as NodeSets
from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, NEURON_CLASS_INDEX
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.utils import ensure_list, timed

L = logging.getLogger(__name__)

# cached node_ids for each (population, node_set, node_sets_file)
CellsCache = dict[tuple[Optional[str], Optional[str], Optional[Path]], pd.DataFrame]


def _get_property_names(neuron_classes: dict[str, NeuronClassConfig]) -> list[str]:
    """Return the list of properties to be retrieved from the cells DataFrame."""
    properties_set = set()
    for conf in neuron_classes.values():
        for query_dict in ensure_list(conf.query):
            properties_set.update(query_dict)
    return sorted(properties_set)


def _load_cells(
    circuit: Circuit,
    property_names: list[str],
    cells_cache: CellsCache,
    population: Optional[str],
    node_set: Optional[str],
    node_sets_file: Optional[Path],
) -> pd.DataFrame:
    """Load and return the cells for the given population and node_set.

    Data are retrieved from the circuit or from the cache.

    If node_set is None or empty string, all the cells of the population are loaded.
    """
    node_set = node_set or None
    key = (population, node_set, node_sets_file)
    if key not in cells_cache:
        msg = f"Loading nodes using {population=}, {node_set=}, {node_sets_file=}"
        with timed(L.info, msg):
            if node_set and node_sets_file:
                node_sets = NodeSets.from_file(circuit.node_sets_file)
                node_sets |= NodeSets.from_file(node_sets_file)
                node_set = node_sets.instance[node_set]
            _cells = circuit.nodes[population].get(group=node_set, properties=property_names)
            cells_cache[key] = _cells
    return cells_cache[key]


def _filter_gids_by_neuron_class(
    circuit: Circuit,
    property_names: list[str],
    cells_cache: CellsCache,
    name: str,
    config: NeuronClassConfig,
) -> np.ndarray:
    """Return the array of node_ids filtered by neuron class."""
    cells = _load_cells(
        circuit=circuit,
        property_names=property_names,
        cells_cache=cells_cache,
        population=config.population,
        node_set=config.node_set,
        node_sets_file=config.node_sets_file,
    )
    gids = cells.etl.q(config.query).index.to_numpy()
    if config.node_id is not None:
        gids = np.intersect1d(gids, config.node_id)
    neuron_count = len(gids)
    if config.limit and neuron_count > config.limit:
        gids = np.random.choice(gids, size=config.limit, replace=False)
    gids.sort()
    L.info(
        "Selected gids for %s: %s/%s (limit=%s, population=%s, node_set=%s)",
        name,
        len(gids),
        neuron_count,
        config.limit,
        config.population,
        config.node_set,
    )
    L.debug("Configured query: %s", config.query)
    L.debug("Configured node_id: %s", config.node_id)
    return gids


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
        circuit: Circuit, neuron_classes: dict[str, NeuronClassConfig]
    ) -> dict[str, np.ndarray]:
        """Return a dict containing name: node_ids for each neuron class."""
        cells_cache: CellsCache = {}
        property_names = _get_property_names(neuron_classes=neuron_classes)
        return {
            name: _filter_gids_by_neuron_class(circuit, property_names, cells_cache, name, config)
            for name, config in neuron_classes.items()
        }

    @classmethod
    def from_simulations(
        cls, simulations: Simulations, neuron_classes: dict[str, NeuronClassConfig]
    ) -> "Neurons":
        """Return a new Neurons instance from the given simulations and configuration.

        Args:
            simulations: Simulations extractor.
            neuron_classes: configuration dict of neuron classes to be extracted.

        Returns:
            Neurons: new instance.
        """
        grouped = simulations.df.groupby([CIRCUIT_ID])[CIRCUIT].first()
        records: list[tuple[int, str, int, int]] = []
        for circuit_id, circuit in grouped.items():
            gids_by_class = cls._get_gids(circuit, neuron_classes=neuron_classes)
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
