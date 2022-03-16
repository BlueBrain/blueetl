import logging

import pandas as pd

from blueetl.constants import (
    CIRCUIT_ID,
    GID,
    NEURON_CLASS,
    SIMULATION,
    SIMULATION_ID,
    TIME,
    WINDOW,
    TRIAL,
)
from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Spikes(BaseExtractor):
    _columns = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, TRIAL, TIME, GID]

    @classmethod
    def _assign_window(cls, df: pd.DataFrame, name: str, trial: int, t_start: float, t_stop: float):
        df = df[(df[TIME] >= t_start) & (df[TIME] < t_stop)].copy()
        df.loc[:, [WINDOW, TRIAL]] = [name, trial]
        # make the spike times relative to the start of the window
        df.loc[:, TIME] -= t_start
        return df

    @classmethod
    def _load_spikes(cls, spikes, gids, windows_df: pd.DataFrame):
        """Filter and aggregate the spikes in bins according to the given windows.

        Args:
            spikes: SpikeReport of times and gids.
            gids: array of gids to be selected.
            windows_df: windows dataframe with columns [window, trial, t_start, t_stop]

        Returns:
            pd.DataFrame: dataframe with columns [window, time, gid]

        """
        df = spikes.get(gids).reset_index()
        df = df.rename(columns={"t": TIME})
        df = pd.concat(
            cls._assign_window(df, rec.window, rec.trial, rec.t_start, rec.t_stop)
            for rec in windows_df.itertuples()
        )
        df = df.reset_index(drop=True)
        L.info("Selected spikes: %s", len(df))
        return df

    @classmethod
    def from_simulations(cls, simulations, neurons, windows):
        merged = pd.merge(simulations.df, neurons.df, sort=False)
        # group the gids together
        columns = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS]
        grouped = merged.groupby(columns, sort=False, observed=True).agg(
            {GID: tuple, SIMULATION: "first"}
        )
        df_list = []
        for index, rec in grouped.etl.iter():
            # TODO: load spikes in parallel?
            L.info(
                "Processing simulation_id=%s, circuit_id=%s, neuron_class=%s: %s gids",
                *index,
                len(rec.gid),
            )
            windows_df = windows.df.etl.q(simulation_id=index.simulation_id)
            tmp_df = cls._load_spikes(rec.simulation.spikes, rec.gid, windows_df)
            tmp_df[columns] = list(index)
            # TODO: verify if converting NEURON_CLASS to category here using CategoricalDtype
            #  can reduce the memory temporarily needed during the process.
            df_list.append(tmp_df)
        df = pd.concat(df_list, ignore_index=True, copy=False)
        return cls(df)

    def grouped_by_neuron_class(self):
        """Group the dataframe by some columns and yield each record as a tuple (key, df).

        Group by columns: SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW
        Returned columns in each yielded df (sorted): TRIAL, GID, TIME

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        yield from self.df.etl.grouped_by(
            groupby_columns=[SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW],
            selected_columns=[TRIAL, GID, TIME],
        )

    def grouped_by_gid(self):
        """Group the dataframe by some columns and yield each record as a tuple (key, df).

        Group by columns: SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID
        Returned columns in each yielded df (sorted): TRIAL, TIME

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        yield from self.df.etl.grouped_by(
            groupby_columns=[SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID],
            selected_columns=[TRIAL, TIME],
        )
