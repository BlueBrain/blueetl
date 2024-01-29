"""Spikes extractor."""

import logging
from typing import Optional

import pandas as pd
from blueetl_core.utils import smart_concat

from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION_ID, TIME, TRIAL, WINDOW
from blueetl.extract.report import ReportExtractor

L = logging.getLogger(__name__)


class Spikes(ReportExtractor):
    """Spikes extractor class."""

    COLUMNS = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, TRIAL, TIME, GID]

    @classmethod
    def _assign_window(cls, df: pd.DataFrame, rec) -> pd.DataFrame:
        win = cls.calculate_window_slice(rec)
        df = df[(df[TIME] >= win.t_start) & (df[TIME] < win.t_stop)].copy()
        df[WINDOW] = win.name
        df[TRIAL] = win.trial
        # make the spike times relative to the offset
        df[TIME] -= win.offset
        return df

    @classmethod
    def _load_values(
        cls,
        simulation: Simulation,
        population: Optional[str],
        gids,
        windows_df: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """Filter and aggregate the spikes in bins according to the given windows.

        Args:
            simulation: simulation containing the SpikeReport of times and gids.
            population: node population name.
            gids: array of gids to be selected.
            windows_df: windows dataframe with columns [window, trial, t_start, t_stop]
            name: name of the report in the simulation configuration, ignored.

        Returns:
            pd.DataFrame: dataframe with columns [window, time, gid]
        """
        df = simulation.spikes[population].get(gids).reset_index()
        # in snap the columns are named `times` and `ids`
        df.columns.array[0:2] = [TIME, GID]
        df = smart_concat(cls._assign_window(df, rec) for rec in windows_df.itertuples())
        df = df.reset_index(drop=True)
        return df
