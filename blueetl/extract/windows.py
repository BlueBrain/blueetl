import logging

import numpy as np
import pandas as pd

from blueetl.constants import (
    CIRCUIT_ID,
    DURATION,
    SIMULATION_ID,
    T_START,
    T_STOP,
    TRIAL,
    TRIAL_STEPS_VALUE,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Windows(BaseExtractor):
    _columns = [SIMULATION_ID, CIRCUIT_ID, WINDOW, TRIAL, T_START, T_STOP, DURATION, WINDOW_TYPE]

    @classmethod
    def _validate(cls, df):
        cls._validate_columns(df)
        # check that all the trials in the same window have the same duration
        if not np.all(df.groupby(WINDOW)[DURATION].nunique() == 1):
            raise ValueError("Inconsistent window duration in some trial(s)")
        # check that the trials numbers are unique and starting from 0
        if not np.all(
            df.groupby([SIMULATION_ID, WINDOW])[TRIAL].agg(
                lambda x: sorted(x) == list(range(len(x)))
            )
        ):
            raise ValueError("Inconsistent trial index in some window(s)")

    @classmethod
    def from_simulations(cls, simulations, trial_steps, config):
        """Load and expand windows for each simulation."""
        results = []
        for index, rec in simulations.df.etl.iter():
            for name, win in config["windows"].items():
                L.info(
                    "Processing simulation_id=%s, circuit_id=%s, window=%s",
                    rec.simulation_id,
                    rec.circuit_id,
                    name,
                )
                t_start, t_stop = win["bounds"]
                duration = t_stop - t_start
                window_type = win.get("window_type", "")
                number_of_trials = win.get("n_trials", 1)
                trial_steps_value = win.get("trial_steps_value", 0)
                trial_steps_label = win.get("trial_steps_label", "")
                if trial_steps_label:
                    trial_steps_value = trial_steps.df.etl.one(
                        simulation_id=rec.simulation_id,
                        circuit_id=rec.circuit_id,
                        trial_steps_label=trial_steps_label,
                    )[TRIAL_STEPS_VALUE]
                    L.info("Using the calculated trial_steps_value=%s", trial_steps_value)
                elif trial_steps_value:
                    L.info("Using the configured trial_steps_value=%s", trial_steps_value)
                if number_of_trials > 1 and not trial_steps_value:
                    raise ValueError("trial_steps_value cannot be 0 when n_trials > 1")
                for index in range(number_of_trials):
                    results.append(
                        {
                            SIMULATION_ID: rec.simulation_id,
                            CIRCUIT_ID: rec.circuit_id,
                            WINDOW: name,
                            TRIAL: index,
                            T_START: t_start + trial_steps_value * index,
                            T_STOP: t_stop + trial_steps_value * index,
                            DURATION: duration,
                            WINDOW_TYPE: window_type,
                        }
                    )
        df = pd.DataFrame(results)
        return cls(df)

    def get_bounds(self, simulation_id, window, trial=0):
        """Return (t_start, t_stop) for the specified window and trial."""
        rec = self.df.etl.first(simulation_id=simulation_id, window=window, trial=trial)
        return rec[[T_START, T_STOP]]

    def get_duration(self, window):
        """Return the duration of the specified window."""
        return self.df.etl.first(window=window)[DURATION]

    def get_number_of_trials(self, window):
        """Return the number of trials for the specified window."""
        return np.max(self.df.etl.q(window=window)[TRIAL]) + 1
