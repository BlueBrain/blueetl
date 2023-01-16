"""Soma Potentials extractor."""
import logging

import pandas as pd
from bluepy import Simulation

from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION_ID, TIME, VALUE, WINDOW
from blueetl.extract.report import ReportExtractor

L = logging.getLogger(__name__)


class SomaReport(ReportExtractor):
    """SomaReport extractor class."""

    COLUMNS = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, TIME, GID, VALUE]

    @classmethod
    def _load_values(
        cls, simulation: Simulation, gids, windows_df: pd.DataFrame, name: str
    ) -> pd.DataFrame:
        """Return a DataFrame for the given simulation, gids, and windows."""
        df_list = []
        report = simulation.report(name)
        for rec in windows_df.itertuples():
            win = cls.calculate_window_slice(rec)
            df = report.get(t_start=win.t_start, t_end=win.t_stop, t_step=win.t_step, gids=gids)
            df = df.unstack().reset_index().rename(columns={0: VALUE})
            df[WINDOW] = win.name
            df_list.append(df)
        return pd.concat(df_list)
