"""Generic Report extractor."""
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar

import pandas as pd
from bluepy import Simulation

from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION, SIMULATION_ID
from blueetl.extract.base import BaseExtractor
from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.windows import Windows

L = logging.getLogger(__name__)
ReportExtractorT = TypeVar("ReportExtractorT", bound="ReportExtractor")


@dataclass
class WindowSlice:
    """Window slice attributes."""

    t_start: float
    t_stop: float
    t_step: Optional[float]
    offset: float
    name: str
    trial: int


class ReportExtractor(BaseExtractor, metaclass=ABCMeta):
    """Report extractor class."""

    @staticmethod
    def calculate_window_slice(rec) -> WindowSlice:
        """Calculate and return the window slice attributes."""
        # increment t_start and t_stop because they are relative to offset
        t_start = rec.offset + rec.t_start
        t_stop = rec.offset + rec.t_stop
        t_step = rec.t_step or None
        return WindowSlice(
            t_start=t_start,
            t_stop=t_stop,
            t_step=t_step,
            offset=rec.offset,
            name=rec.window,
            trial=rec.trial,
        )

    @classmethod
    @abstractmethod
    def _load_values(
        cls, simulation: Simulation, gids, windows_df: pd.DataFrame, name: str
    ) -> pd.DataFrame:
        """Return a DataFrame for the given simulation, gids, and windows.

        Args:
            simulation: simulation containing the report.
            gids: array of gids to be selected.
            windows_df: windows dataframe.
            name: name of the report in the simulation configuration.

        Returns:
            pd.DataFrame: dataframe with the needed columns.
        """

    @classmethod
    def from_simulations(
        cls: type[ReportExtractorT],
        simulations: Simulations,
        neurons: Neurons,
        windows: Windows,
        name: str,
    ) -> ReportExtractorT:
        """Return a new instance from the given simulations, neurons, and windows.

        Args:
            simulations: Simulations extractor.
            neurons: Neurons extractor.
            windows: Windows extractor.
            name: name of the report in the simulation configuration.

        Returns:
            New instance.
        """
        merged = pd.merge(simulations.df, neurons.df, sort=False)
        # group the gids together
        columns = [SIMULATION_ID, CIRCUIT_ID, NEURON_CLASS]
        grouped = merged.groupby(columns, sort=False, observed=True).agg(
            {GID: tuple, SIMULATION: "first"}
        )
        df_list = []
        for index, rec in grouped.etl.iter():
            L.info(
                "Extracting %s for simulation_id=%s, circuit_id=%s, neuron_class=%s: %s gids",
                cls.__name__,
                *index,
                len(rec.gid),
            )
            windows_df = windows.df.etl.q(simulation_id=index.simulation_id)
            tmp_df = cls._load_values(
                simulation=rec.simulation, gids=rec.gid, windows_df=windows_df, name=name
            )
            tmp_df[columns] = list(index)
            df_list.append(tmp_df)
        df = pd.concat(df_list, ignore_index=True, copy=False)
        return cls(df)
