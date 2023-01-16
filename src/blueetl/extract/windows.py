"""Windows extractor."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    TRIAL_STEPS_VALUE,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract.base import BaseExtractor
from blueetl.extract.simulations import Simulations
from blueetl.extract.trial_steps import TrialSteps
from blueetl.resolver import Resolver

L = logging.getLogger(__name__)


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
        if not np.all(df.groupby(WINDOW)[[T_START, T_STOP, DURATION]].nunique() == 1):
            raise ValueError("Inconsistent window t_start, t_stop, or duration in some trial(s)")
        # check that the trials numbers are unique and starting from 0
        if not np.all(
            df.groupby([SIMULATION_ID, WINDOW])[TRIAL].agg(
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
        resolver: Optional[Resolver],
    ) -> List[Dict[str, Any]]:
        assert resolver is not None
        # example of valid win: spikes.repo.windows.w1#checksum
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
        win: Dict[str, Any],
        simulation_id: int,
        circuit_id: int,
        trial_steps: TrialSteps,
    ) -> List[Dict[str, Any]]:
        initial_offset = win.get("initial_offset", 0)
        t_start, t_stop = win["bounds"]
        t_step = win.get("t_step", 0)
        duration = t_stop - t_start
        window_type = win.get("window_type", "")
        number_of_trials = win.get("n_trials", 1)
        trial_steps_value = win.get("trial_steps_value", 0)
        trial_steps_label = win.get("trial_steps_label", "")
        if trial_steps_label:
            trial_steps_value = trial_steps.df.etl.one(
                simulation_id=simulation_id,
                circuit_id=circuit_id,
                trial_steps_label=trial_steps_label,
            )[TRIAL_STEPS_VALUE]
            L.info("Using the calculated trial_steps_value=%s", trial_steps_value)
        elif trial_steps_value:
            L.info("Using the configured trial_steps_value=%s", trial_steps_value)
        if number_of_trials > 1 and not trial_steps_value:
            raise ValueError("trial_steps_value cannot be 0 when n_trials > 1")
        return [
            {
                SIMULATION_ID: simulation_id,
                CIRCUIT_ID: circuit_id,
                WINDOW: name,
                TRIAL: index,
                OFFSET: initial_offset + trial_steps_value * index,
                T_START: t_start,
                T_STOP: t_stop,
                T_STEP: t_step,
                DURATION: duration,
                WINDOW_TYPE: window_type,
            }
            for index in range(number_of_trials)
        ]

    @classmethod
    def from_simulations(
        cls,
        simulations: Simulations,
        trial_steps: TrialSteps,
        config: Dict[str, Any],
        resolver: Optional[Resolver],
    ) -> "Windows":
        """Return a new Windows instance from the given simulations and configuration.

        Args:
            simulations: Simulations extractor.
            trial_steps: TrialSteps extractor.
            config: configuration dict containing the "windows" key.
            resolver: resolver instance.

        Returns:
            Windows: new instance.
        """
        # pylint: disable=too-many-locals
        results = []
        for _, rec in simulations.df.etl.iter():
            for name, win in config["windows"].items():
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
                        simulation_id=rec.simulation_id,
                        circuit_id=rec.circuit_id,
                        trial_steps=trial_steps,
                    )
                results.extend(partial_results)

        df = pd.DataFrame(results)
        return cls(df)

    def get_bounds(self, window: str) -> Tuple[float, float]:
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
