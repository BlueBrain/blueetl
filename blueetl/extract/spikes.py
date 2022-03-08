import collections
import logging
from functools import partial
from typing import Dict, List, Tuple

import pandas as pd

from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION, SIMULATION_ID, TIME, WINDOW
from blueetl.utils import ensure_dtypes, timed

L = logging.getLogger(__name__)


class Spikes:
    def __init__(self, df: pd.DataFrame):
        assert set(df.columns) == {TIME, GID, WINDOW, SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS}
        self._df: pd.DataFrame = ensure_dtypes(df)
        # FIXME: do we need to ensure that the neurons or times are sorted?
        # self._df = self._df.sort_values(
        #     [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID, TIME], ignore_index=True
        # )

    @property
    def df(self):
        return self._df

    @classmethod
    def _assign_window(cls, df, bins, labels: List[str]):
        df = df.copy()
        df[WINDOW] = pd.cut(df[TIME], bins).cat.rename_categories(labels)
        # remove the rows that are outside the time intervals
        return df.dropna()

    @classmethod
    def _load_spikes(cls, spikes, gids, windows: Dict[str, Tuple[float, float]]):
        """Filter and aggregate the spikes in bins according to the given windows.

        Args:
            spikes: SpikeReport of times and gids.
            gids: array of gids to be selected.
            windows: dict of windows represented as label -> [start_time, end_time].

        Returns:
            pd.DataFrame: dataframe with columns [window, time, gid]
        """

        with timed(L.info, "Completed partial spikes extraction"):
            df = spikes.get(gids).reset_index()
            df = df.rename(columns={"t": TIME})
            # assign the correct window to each row.
            interval_index_from_tuples = partial(
                pd.IntervalIndex.from_tuples, closed="left", name=WINDOW
            )
            bins = interval_index_from_tuples(list(windows.values()))
            labels = list(windows.keys())
            if not bins.is_overlapping:
                df = cls._assign_window(df, bins, labels)
            else:
                # Overlapping IntervalIndex is not accepted by cut
                L.info("Overlapping windows, iterating over each interval...")
                df = pd.concat(
                    cls._assign_window(df, bins=pd.IntervalIndex([interval]), labels=[label])
                    for interval, label in zip(bins, labels)
                )
            df = df.reset_index(drop=True)
            L.info("Selected spikes: %s", len(df))
            return df

    @classmethod
    def from_simulations(cls, simulations, neurons, windows):
        # ensure that all the intervals are tuples and not lists
        windows = {label: tuple(interval) for label, interval in windows.items()}
        L.info("Extracting spikes using windows: %s", windows)
        merged = pd.merge(simulations.df, neurons.df, sort=False)
        # group the gids together
        columns = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS]
        grouped = merged.groupby(columns, sort=False, observed=True).agg(
            {GID: tuple, SIMULATION: "first"}
        )
        df_list = []
        for index, value in grouped.etl.iter_named_items():
            # TODO: load spikes in parallel?
            L.info(
                "Processing simulation_id=%s, circuit_id=%s, neuron_class=%s: %s gids",
                *index,
                len(value.gid),
            )
            tmp_df = cls._load_spikes(value.simulation.spikes, value.gid, windows=windows)
            tmp_df[columns] = list(index)
            # TODO: verify if converting NEURON_CLASS to category here using CategoricalDtype
            #  can reduce the memory temporarily needed during the process.
            df_list.append(tmp_df)
        df = pd.concat(df_list, ignore_index=True, copy=False)
        return cls(df)

    @classmethod
    def from_pandas(cls, df):
        return cls(df)

    def to_pandas(self):
        """Dump spikes to a dataframe that can be serialized and stored."""
        return self.df

    def as_series(self):
        columns = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID]
        return self.df.set_index(columns)[TIME]

    def grouped_by_neuron_class(self):
        """Group the dataframe by some columns and yield each record as a tuple (key, df).

        Group by columns: SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        yield from self.df.etl.grouped_by(
            groupby_columns=[SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW],
            selected_columns=[TIME, GID],
        )

    def grouped_by_gid(self):
        """Group the dataframe by some columns and yield each record as a tuple (key, df).

        Group by columns: SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        yield from self.df.etl.grouped_by(
            groupby_columns=[SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS, WINDOW, GID],
            selected_columns=[TIME],
        )
