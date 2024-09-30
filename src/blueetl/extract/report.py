"""Generic Report extractor."""

import logging
import tempfile
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple, Optional, TypeVar

import pandas as pd
from blueetl_core.utils import smart_concat

from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, POPULATION, SIMULATION, SIMULATION_ID
from blueetl.extract.base import BaseExtractor
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.windows import Windows
from blueetl.parallel import merge_filter
from blueetl.store.parquet import ParquetStore
from blueetl.utils import ensure_dtypes, get_shmdir, timed

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
        cls,
        *,
        simulation: Simulation,
        population: Optional[str],
        gids,
        windows_df: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """Return a DataFrame for the given simulation, population, gids, and windows.

        Args:
            simulation: simulation containing the report.
            population: node population name.
            gids: array of gids to be selected.
            windows_df: windows dataframe.
            name: name of the report in the simulation configuration.

        Returns:
            pd.DataFrame: dataframe with the needed columns.
        """

    @classmethod
    def from_simulations(
        cls: type[ReportExtractorT],
        *,
        simulations: Simulations,
        neurons: Neurons,
        windows: Windows,
        neuron_classes: NeuronClasses,
        name: str,
    ) -> ReportExtractorT:
        """Return a new instance from the given simulations, neurons, and windows.

        Args:
            simulations: Simulations extractor.
            neurons: Neurons extractor.
            windows: Windows extractor.
            neuron_classes: NeuronClasses extractor.
            name: name of the report in the simulation configuration.

        Returns:
            New instance.
        """
        with tempfile.TemporaryDirectory(prefix="blueetl_", dir=get_shmdir()) as _temp_folder:
            with timed(L.info, "Executing merge_filter "):
                func = partial(
                    _merge_filter_func,
                    temp_folder=Path(_temp_folder),
                    name=name,
                    neuron_classes_df=neuron_classes.df,
                    dataframe_builder=cls._load_values,
                )
                merge_filter(
                    df_list=[simulations.df, neurons.df, windows.df],
                    groupby=[SIMULATION_ID, CIRCUIT_ID],
                    func=func,
                )
            with timed(L.info, "Executing concatenation"):
                df = ParquetStore(Path(_temp_folder)).load()
                df = ensure_dtypes(df)
            return cls(df, cached=False, filtered=False)


def _merge_filter_func(
    *,
    task_index: int,
    key: NamedTuple,
    df_list: list[pd.DataFrame],
    temp_folder: Path,
    name: str,
    neuron_classes_df: pd.DataFrame,
    dataframe_builder: Callable[..., pd.DataFrame],
) -> None:
    """Executed in a subprocess, write a partial DataFrame to temp_folder."""
    # pylint: disable=too-many-locals
    simulations_df, neurons_df, windows_df = df_list
    simulation_id, simulation = simulations_df.etl.one()[[SIMULATION_ID, SIMULATION]]
    assert simulation_id == key.simulation_id  # type: ignore[attr-defined]
    df_list = []
    for inner_key, df in neurons_df.etl.groupby_iter([CIRCUIT_ID, NEURON_CLASS]):
        population = neuron_classes_df.etl.one(
            circuit_id=inner_key.circuit_id, neuron_class=inner_key.neuron_class
        )[POPULATION]
        result_df = dataframe_builder(
            simulation=simulation,
            population=population,
            gids=df[GID],
            windows_df=windows_df,
            name=name,
        )
        result_df[[SIMULATION_ID, *inner_key._fields]] = [simulation_id, *inner_key]
        df_list.append(result_df)
    result_df = smart_concat(df_list, ignore_index=True)
    # the conversion to the desired dtype here is important to reduce memory usage and cpu time
    result_df = ensure_dtypes(result_df)
    ParquetStore(temp_folder).dump(result_df, name=f"{task_index:08d}")
