"""Soma Potentials extractor."""

import logging
from typing import Optional

import pandas as pd
from blueetl_core.utils import smart_concat

from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.constants import (
    CIRCUIT_ID,
    GID,
    NEURON_CLASS,
    SIMULATION_ID,
    TIME,
    TRIAL,
    VALUE,
    WINDOW,
)
from blueetl.extract.report import ReportExtractor

L = logging.getLogger(__name__)


class SomaReport(ReportExtractor):
    """SomaReport extractor class."""

    COLUMNS = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, TRIAL, TIME, GID, VALUE]

    @classmethod
    def _load_values(
        cls,
        *,
        simulation: Simulation,
        population: Optional[str],
        gids,
        windows_df: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """Return a DataFrame for the given simulation, population, gids, and windows."""
        df_list = []
        report = simulation.reports[name][population]
        for rec in windows_df.itertuples():
            win = cls.calculate_window_slice(rec)
            df = report.get(group=gids, t_start=win.t_start, t_stop=win.t_stop, t_step=win.t_step)
            df.index.rename(TIME, inplace=True)
            df.columns.rename(GID, inplace=True)
            df = df.unstack().reset_index()
            df.rename(columns={0: VALUE}, inplace=True)
            df[WINDOW] = win.name
            df[TRIAL] = win.trial
            # make the times relative to the offset
            df[TIME] -= win.offset
            df_list.append(df)
        return smart_concat(df_list)
