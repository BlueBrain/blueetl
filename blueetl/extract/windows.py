import logging

import numpy as np
import pandas as pd

from blueetl.constants import DURATION, T_START, T_STOP, TRIAL, WINDOW
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Windows(BaseExtractor):
    @staticmethod
    def _validate(df):
        assert set(df.columns) == {WINDOW, TRIAL, T_START, T_STOP, DURATION}
        grouped = df.groupby(WINDOW)
        # check that all the trials in the same window have the same duration
        assert np.all(
            grouped[DURATION].nunique() == 1
        ), "Inconsistent window duration in some trial"
        # check that the trials numbers are unique and starting from 0
        assert np.all(
            grouped[TRIAL].agg(lambda x: sorted(x) == list(range(len(x))))
        ), "Inconsistent trial index in some window"

    @classmethod
    def from_config(cls, config):
        """Load and expand windows from the configuration."""
        windows = []
        for name, w in config["windows"].items():
            t_start, t_stop = w["bounds"]
            trial_step = w.get("trial_step", 0)
            duration = t_stop - t_start
            for i in range(w.get("n_trials", 1)):
                windows.append(
                    {
                        WINDOW: name,
                        TRIAL: i,
                        T_START: t_start + trial_step * i,
                        T_STOP: t_stop + trial_step * i,
                        DURATION: duration,
                    }
                )
        df = pd.DataFrame(windows)
        return cls(df)

    def get_bounds(self, window, trial=0):
        """Return (t_start, t_stop) for the specified window and trial."""
        df = self.df
        df = df[(df[WINDOW] == window) & (df[TRIAL] == trial)]
        return tuple(df[[T_START, T_STOP]].iloc[0])

    def get_duration(self, window):
        """Return the duration of the specified window."""
        df = self.df
        return df[df[WINDOW] == window][DURATION].iat[0]

    def get_number_of_trials(self, window):
        """Return the number of trials for the specified window."""
        return np.sum(self.df[WINDOW] == window)
