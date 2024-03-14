"""Windows extractor."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from blueetl.adapters.circuit import CircuitAdapter as Circuit
from blueetl.adapters.node_sets import NodeSetsAdapter as NodeSets
from blueetl.adapters.simulation import SimulationAdapter as Simulation
from blueetl.config.analysis_model import TrialStepsConfig, WindowConfig
from blueetl.constants import (
    CHECKSUM_SEP,
    CIRCUIT_ID,
    DURATION,
    LEVEL_SEP,
    OFFSET,
    SIMULATION_ID,
    T_START,
    T_STEP,
    T_STOP,
    TRIAL,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.resolver import Resolver
from blueetl.utils import import_by_string, timed

L = logging.getLogger(__name__)


def _load_dynamic_gids(
    circuit: Circuit,
    population: Optional[str],
    node_set: Optional[str],
    node_sets_file: Optional[Path],
    limit: Optional[int],
) -> np.ndarray:
    """Return the node ids to consider."""
    node_set = node_set or None
    with timed(L.info, "Loading nodes from circuit for dynamic offset"):
        if node_set and node_sets_file:
            node_sets = NodeSets.from_file(circuit.node_sets_file)
            node_sets |= NodeSets.from_file(node_sets_file)
            node_set = node_sets.instance[node_set]
        gids = circuit.nodes[population].ids(group=node_set)
    neuron_count = len(gids)
    if limit and neuron_count > limit:
        gids = np.random.choice(gids, size=limit, replace=False)
    L.info("Selected %s/%s gids", len(gids), neuron_count)
    return gids


def _load_dynamic_spikes(
    simulation: Simulation,
    population: Optional[str],
    gids: np.ndarray,
    offset: float,
    t_start: float,
    t_stop: float,
) -> np.ndarray:
    """Return the spikes extracted for the given parameters."""
    spikes = (
        simulation.spikes[population]
        .get(gids, t_start=offset + t_start, t_stop=offset + t_stop)
        .index.to_numpy()
    )
    spikes -= offset
    L.info("Selected %s spikes", len(spikes))
    return spikes


def _calculate_dynamic_offset(
    simulation: Simulation,
    circuit: Circuit,
    initial_offset: float,
    step_offsets: list[float],
    trial_steps_config: TrialStepsConfig,
) -> float:
    """Calculate the dynamic offset according to NSETM-2281."""
    # circuit is passed explicitly instead of loading it from simulation.circuit
    # to take advantage of any data already loaded and cached
    gids = _load_dynamic_gids(
        circuit=circuit,
        population=trial_steps_config.population,
        node_set=trial_steps_config.node_set,
        node_sets_file=trial_steps_config.node_sets_file,
        limit=trial_steps_config.limit,
    )
    spikes_list = []
    t_start, t_stop = trial_steps_config.bounds
    for step_offset in step_offsets:
        spikes = _load_dynamic_spikes(
            simulation=simulation,
            population=trial_steps_config.population,
            gids=gids,
            offset=initial_offset + step_offset,
            t_start=t_start,
            t_stop=t_stop,
        )
        spikes_list.append(spikes)
    func = import_by_string(trial_steps_config.function)
    result = func(spikes_list, trial_steps_config.dict())
    if not np.issubdtype(type(result), np.number):
        raise ValueError(f"The function {trial_steps_config.function} must return a number")
    return result


class Windows(BaseExtractor):
    """Windows extractor class."""

    COLUMNS = [
        SIMULATION_ID,
        CIRCUIT_ID,
        WINDOW,
        TRIAL,
        OFFSET,
        T_START,
        T_STOP,
        T_STEP,
        DURATION,
        WINDOW_TYPE,
    ]

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)
        # check that all the trials in the same window have the same t_start, t_stop, duration
        if not np.all(
            df.groupby(WINDOW, observed=True)[[T_START, T_STOP, DURATION]].nunique() == 1
        ):
            raise ValueError("Inconsistent window t_start, t_stop, or duration in some trial(s)")
        # check that the trials numbers are unique and starting from 0
        if not np.all(
            df.groupby([SIMULATION_ID, WINDOW], observed=True)[TRIAL].agg(
                lambda x: sorted(x) == list(range(len(x)))
            )
        ):
            raise ValueError("Inconsistent trial index in some window(s)")

    @classmethod
    def _load_records_from_resolver(
        cls,
        name: str,
        win: str,
        simulation_id: int,
        circuit_id: int,
        resolver: Resolver,
    ) -> list[dict[str, Any]]:
        # example of valid win: spikes.extraction.windows.w1#checksum
        win, _, _checksum = win.rpartition(CHECKSUM_SEP)
        ref, _, window = win.rpartition(LEVEL_SEP)
        windows = resolver.get(ref)
        df = windows.df.etl.q(simulation_id=simulation_id, circuit_id=circuit_id, window=window)
        if len(df) == 0:
            raise RuntimeError(f"The window referenced by {win} is empty or undefined")
        return df.assign(**{WINDOW: name}).to_dict(orient="records")

    @classmethod
    def _load_records_from_config(
        cls,
        name: str,
        rec: Any,  # row from simulations DataFrame
        win: WindowConfig,
        trial_steps_config: Optional[TrialStepsConfig],
    ) -> list[dict[str, Any]]:
        """Load the records from the window configuration."""
        t_start, t_stop = win.bounds
        t_step = win.t_step
        duration = t_stop - t_start
        if win.trial_steps_list:
            step_offsets = win.trial_steps_list
        else:
            step_offsets = [win.trial_steps_value * i for i in range(win.n_trials or 1)]
        if trial_steps_config:
            dynamic_offset = _calculate_dynamic_offset(
                simulation=rec.simulation,
                circuit=rec.circuit,
                initial_offset=win.initial_offset,
                step_offsets=step_offsets,
                trial_steps_config=trial_steps_config,
            )
        else:
            dynamic_offset = 0.0
        L.info(
            "Using window=%s, initial_offset=%s, dynamic_offset=%s, step_offsets=%s, "
            "t_start=%s, t_stop=%s, t_step=%s, duration=%s",
            name,
            win.initial_offset,
            dynamic_offset,
            step_offsets,
            t_start,
            t_stop,
            t_step,
            duration,
        )
        return [
            {
                SIMULATION_ID: rec.simulation_id,
                CIRCUIT_ID: rec.circuit_id,
                WINDOW: name,
                TRIAL: index,
                OFFSET: win.initial_offset + dynamic_offset + step_offset,
                T_START: t_start,
                T_STOP: t_stop,
                T_STEP: t_step,
                DURATION: duration,
                WINDOW_TYPE: win.window_type,
            }
            for index, step_offset in enumerate(step_offsets)
        ]

    @classmethod
    def from_simulations(
        cls,
        simulations: Simulations,
        windows_config: dict[str, Union[str, WindowConfig]],
        trial_steps_config: dict[str, TrialStepsConfig],
        resolver: Resolver,
    ) -> "Windows":
        """Return a new Windows instance from the given simulations and configuration.

        Args:
            simulations: Simulations extractor.
            windows_config: configuration dict.
            trial_steps_config: configuration dict.
            resolver: resolver instance.

        Returns:
            Windows: new instance.
        """
        results = []
        for _, rec in simulations.df.etl.iter():
            for name, win in windows_config.items():
                L.info(
                    "Processing simulation_id=%s, circuit_id=%s, window=%s",
                    rec.simulation_id,
                    rec.circuit_id,
                    name,
                )
                if isinstance(win, str):
                    partial_results = cls._load_records_from_resolver(
                        name=name,
                        win=win,
                        simulation_id=rec.simulation_id,
                        circuit_id=rec.circuit_id,
                        resolver=resolver,
                    )
                else:
                    partial_results = cls._load_records_from_config(
                        name=name,
                        win=win,
                        rec=rec,
                        trial_steps_config=(
                            trial_steps_config[win.trial_steps_label]
                            if win.trial_steps_label
                            else None
                        ),
                    )
                results.extend(partial_results)

        df = pd.DataFrame(results)
        return cls(df, cached=False, filtered=False)

    def get_bounds(self, window: str) -> tuple[float, float]:
        """Return the interval (t_start, t_stop) for the specified window.

        The returned values don't depend on the simulation or the trial,
        because they are relative the offset, that is the only changing value.
        """
        rec = self.df.etl.first(window=window)
        t_start, t_stop = rec[[T_START, T_STOP]]
        return t_start, t_stop

    def get_duration(self, window: str) -> float:
        """Return the duration of the specified window."""
        return self.df.etl.first(window=window)[DURATION]

    def get_number_of_trials(self, window: str) -> int:
        """Return the number of trials for the specified window."""
        return np.max(self.df.etl.q(window=window)[TRIAL]) + 1
