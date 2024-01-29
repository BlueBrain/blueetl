"""TrialSteps extractor."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from blueetl.adapters.circuit import CircuitAdapter as Circuit
from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.config.analysis_model import TrialStepsConfig
from blueetl.constants import CIRCUIT_ID, SIMULATION_ID, TRIAL_STEPS_LABEL, TRIAL_STEPS_VALUE
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.utils import import_by_string, timed

L = logging.getLogger(__name__)


class TrialSteps(BaseExtractor):
    """TrialSteps extractor class."""

    COLUMNS = [SIMULATION_ID, CIRCUIT_ID, TRIAL_STEPS_LABEL, TRIAL_STEPS_VALUE]
    # allow additional columns that can be used to store more details
    _allow_extra_columns = True
    # trial_steps are optional
    _allow_empty_data = True

    @classmethod
    def _load_spikes(
        cls,
        simulation: Simulation,
        circuit: Circuit,
        population: Optional[str],
        node_set: Optional[str],
        limit: Optional[int],
        initial_offset: float,
        t_start: float,
        t_stop: float,
    ) -> np.ndarray:
        # pylint: disable=too-many-arguments
        # circuit is passed explicitly instead of loading it from simulation.circuit
        # to take advantage of any circuit already loaded in memory
        with timed(L.info, "Loading cells from circuit"):
            cells_group = node_set or None
            gids = circuit.nodes[population].ids(group=cells_group)
        neuron_count = len(gids)
        if limit and neuron_count > limit:
            gids = np.random.choice(gids, size=limit, replace=False)
        L.info("Selected %s/%s gids", len(gids), neuron_count)
        spikes = (
            simulation.spikes[population]
            .get(gids, t_start=initial_offset + t_start, t_stop=initial_offset + t_stop)
            .index.to_numpy()
        )
        spikes -= initial_offset
        L.info("Selected %s spikes", len(spikes))
        return spikes

    @classmethod
    def from_simulations(
        cls, simulations: Simulations, trial_steps_config: dict[str, TrialStepsConfig]
    ) -> "TrialSteps":
        """Return a new TrialSteps instance from the given simulations and configuration.

        Args:
            simulations: Simulations extractor.
            trial_steps_config: configuration dict.

        Returns:
            TrialSteps: new instance.
        """
        # pylint: disable=too-many-locals
        results = []
        for trial_steps_label, config in trial_steps_config.items():
            func = import_by_string(config.function)
            t_start, t_stop = config.bounds
            initial_offset = config.initial_offset
            for _, rec in simulations.df.etl.iter():
                L.info(
                    "Processing trial_steps_label=%s, simulation_id=%s, circuit_id=%s",
                    trial_steps_label,
                    rec.simulation_id,
                    rec.circuit_id,
                )
                spikes = cls._load_spikes(
                    simulation=rec.simulation,
                    circuit=rec.circuit,
                    population=config.population,
                    node_set=config.node_set,
                    limit=config.limit,
                    initial_offset=initial_offset,
                    t_start=t_start,
                    t_stop=t_stop,
                )
                trial_steps_result = func(spikes, config.dict())
                try:
                    trial_steps_value = trial_steps_result.pop(TRIAL_STEPS_VALUE)
                except KeyError:
                    L.error(
                        "The dictionary returned by %r must contain the %r key",
                        config.function,
                        TRIAL_STEPS_VALUE,
                    )
                    raise
                L.info("trial_steps_value=%s", trial_steps_value)
                results.append(
                    {
                        SIMULATION_ID: rec.simulation_id,
                        CIRCUIT_ID: rec.circuit_id,
                        TRIAL_STEPS_LABEL: trial_steps_label,
                        TRIAL_STEPS_VALUE: trial_steps_value,
                        **trial_steps_result,
                    }
                )
        df = pd.DataFrame(results) if results else pd.DataFrame([], columns=cls.COLUMNS)
        return cls(df, cached=False, filtered=False)
