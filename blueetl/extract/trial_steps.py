import logging

import numpy as np
import pandas as pd

from blueetl.constants import CIRCUIT_ID, SIMULATION_ID, TRIAL_STEPS_LABEL, TRIAL_STEPS_VALUE
from blueetl.extract.base import BaseExtractor
from blueetl.utils import import_by_string, timed

L = logging.getLogger(__name__)


class TrialSteps(BaseExtractor):
    COLUMNS = [SIMULATION_ID, CIRCUIT_ID, TRIAL_STEPS_LABEL, TRIAL_STEPS_VALUE]

    @classmethod
    def _validate(cls, df):
        # allow additional arbitrary columns useful for inspection
        cls._validate_columns(df, allow_extra=True)

    @classmethod
    def _load_spikes(
        cls, simulation, circuit, target, limit, initial_offset, t_start, t_end
    ) -> np.ndarray:
        """

        Args:

        Returns:

        """
        # circuit is passed explicitly instead of loading it from simulation.circuit
        # to take advantage of any circuit already loaded in memory
        with timed(L.info, "Cells loaded from circuit"):
            gids = circuit.cells.get(group={"$target": target}, properties=[]).index.to_numpy()
        neuron_count = len(gids)
        if limit and neuron_count > limit:
            gids = np.random.choice(gids, size=limit, replace=False)
        L.info("Selected %s/%s gids", len(gids), neuron_count)
        spikes = simulation.spikes.get(
            gids, t_start=initial_offset + t_start, t_end=initial_offset + t_end
        ).index.to_numpy()
        spikes -= initial_offset
        L.info("Selected %s spikes", len(spikes))
        return spikes

    @classmethod
    def from_simulations(cls, simulations, config):
        target = config["target"]
        limit = config["limit"]
        results = []
        for trial_steps_label, trial_steps_params in config.get("trial_steps", {}).items():
            func = import_by_string(trial_steps_params["function"])
            t_start, t_end = trial_steps_params["bounds"]
            initial_offset = trial_steps_params["initial_offset"]
            for index, rec in simulations.df.etl.q(complete=True).etl.iter():
                L.info(
                    "Processing trial_steps_label=%s, simulation_id=%s, circuit_id=%s",
                    trial_steps_label,
                    rec.simulation_id,
                    rec.circuit_id,
                )
                spikes = cls._load_spikes(
                    simulation=rec.simulation,
                    circuit=rec.circuit,
                    target=target,
                    limit=limit,
                    initial_offset=initial_offset,
                    t_start=t_start,
                    t_end=t_end,
                )
                trial_steps_result = func(spikes, trial_steps_params)
                try:
                    trial_steps_value = trial_steps_result.pop(TRIAL_STEPS_VALUE)
                except KeyError:
                    L.error(
                        "The dictionary returned by %r must contain the %r key",
                        trial_steps_params["function"],
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
        return cls(df)
