"""Repository."""
import json
import logging
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT_ID, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import ExtractorT
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.spikes import Spikes
from blueetl.extract.trial_steps import TrialSteps
from blueetl.extract.windows import Windows
from blueetl.utils import timed

L = logging.getLogger(__name__)


class Repository:
    """Repository class."""

    def __init__(
        self,
        simulations_config: SimulationsConfig,
        extraction_config: Dict[str, Any],
        cache_manager: CacheManager,
        simulations_filter: Optional[Dict] = None,
    ) -> None:
        """Initialize the repository."""
        self._extraction_config = extraction_config
        self._simulations_config = simulations_config
        self._cache_manager = cache_manager
        self._simulations_filter = simulations_filter
        self._names = [
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
        ]

    def __getstate__(self) -> Dict:
        """Get the object state when the object is pickled."""
        if not self.is_extracted():
            # ensure that the dataframes are extracted and stored to disk,
            # because we want to be able to use the cached data in the subprocesses.
            L.info("Extracting dataframes before serialization")
            self.extract()
        # Copy the object's state from self.__dict__, excluding the unpicklable entries.
        return {k: v for k, v in self.__dict__.items() if k not in self.names}

    def __setstate__(self, state: Dict) -> None:
        """Set the object state when the object is unpickled."""
        # Restore instance attributes
        self.__dict__.update(state)

    @property
    def names(self) -> List[str]:
        """Return the list of names of the extracted objects."""
        return self._names

    @cached_property
    def simulations(self) -> Simulations:
        """Return the Simulations extraction."""
        return self._extract_simulations()

    @cached_property
    def neurons(self) -> Neurons:
        """Return the Neurons extraction."""
        return self._extract_neurons()

    @cached_property
    def neuron_classes(self) -> NeuronClasses:
        """Return the NeuronClasses extraction."""
        return self._extract_neuron_classes()

    @cached_property
    def trial_steps(self) -> TrialSteps:
        """Return the TrialSteps extraction."""
        return self._extract_trial_steps()

    @cached_property
    def windows(self) -> Windows:
        """Return the Windows extraction."""
        return self._extract_windows()

    @cached_property
    def spikes(self) -> Spikes:
        """Return the Spikes extraction."""
        return self._extract_spikes()

    @cached_property
    def simulation_ids(self) -> List[int]:
        """Return the list of simulation ids, possibly filtered."""
        return self.simulations.df[SIMULATION_ID].to_list()

    def _extraction_wrapper(
        self,
        name: str,
        extract_cached: Callable[[pd.DataFrame], ExtractorT],
        extract_new: Callable[[], ExtractorT],
    ) -> ExtractorT:
        """Return an object extracted from the cache or as new.

        Args:
            name: name of the dataframe.
            extract_cached: function to extract the object from a cached dataframe.
            extract_new: function to extract the object as new.
        """
        with timed(L.info, "Executed extraction of %s", name):
            is_new = is_modified = False
            df = self._cache_manager.load_repo(name)
            if df is not None:
                initial_len = len(df)
                instance = extract_cached(df)
                is_modified = initial_len != len(instance.df)
            else:
                instance = extract_new()
                is_new = True
            assert instance is not None, "The extraction didn't return a valid instance."
            if is_new or is_modified:
                self._cache_manager.dump_repo(df=instance.to_pandas(), name=name)
            return instance

    def _extract_simulations(self) -> Simulations:
        def _extract_cached(df: pd.DataFrame) -> Simulations:
            return Simulations.from_pandas(df, query=self._simulations_filter)

        def _extract_new() -> Simulations:
            return Simulations.from_config(
                config=self._simulations_config,
                query=self._simulations_filter,
            )

        return self._extraction_wrapper(
            name="simulations",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def _extract_neurons(self) -> Neurons:
        def _extract_cached(df: pd.DataFrame) -> Neurons:
            query = {}
            if self.simulation_ids:
                selected_sims = self.simulations.df.etl.q(simulation_id=self.simulation_ids)
                query = {CIRCUIT_ID: sorted(set(selected_sims[CIRCUIT_ID]))}
            return Neurons.from_pandas(df, query=query)

        def _extract_new() -> Neurons:
            return Neurons.from_simulations(
                simulations=self.simulations,
                target=self._extraction_config["target"],
                neuron_classes=self._extraction_config["neuron_classes"],
                limit=self._extraction_config["limit"],
            )

        return self._extraction_wrapper(
            name="neurons",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def _extract_neuron_classes(self) -> NeuronClasses:
        def _extract_cached(df: pd.DataFrame) -> NeuronClasses:
            query = {}
            if self.simulation_ids:
                selected_sims = self.simulations.df.etl.q(simulation_id=self.simulation_ids)
                query = {CIRCUIT_ID: sorted(set(selected_sims[CIRCUIT_ID]))}
            return NeuronClasses.from_pandas(df, query=query)

        def _extract_new() -> NeuronClasses:
            return NeuronClasses.from_neurons(
                neurons=self.neurons,
                target=self._extraction_config["target"],
                neuron_classes=self._extraction_config["neuron_classes"],
                limit=self._extraction_config["limit"],
            )

        return self._extraction_wrapper(
            name="neuron_classes",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def _extract_trial_steps(self) -> TrialSteps:
        def _extract_cached(df: pd.DataFrame) -> TrialSteps:
            query = {SIMULATION_ID: self.simulation_ids} if self.simulation_ids else {}
            return TrialSteps.from_pandas(df, query=query)

        def _extract_new() -> TrialSteps:
            return TrialSteps.from_simulations(
                simulations=self.simulations,
                config=self._extraction_config,
            )

        return self._extraction_wrapper(
            name="trial_steps",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def _extract_windows(self) -> Windows:
        def _extract_cached(df: pd.DataFrame) -> Windows:
            query = {SIMULATION_ID: self.simulation_ids} if self.simulation_ids else {}
            return Windows.from_pandas(df, query=query)

        def _extract_new() -> Windows:
            return Windows.from_simulations(
                simulations=self.simulations,
                trial_steps=self.trial_steps,
                config=self._extraction_config,
            )

        return self._extraction_wrapper(
            name="windows",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def _extract_spikes(self) -> Spikes:
        def _extract_cached(df: pd.DataFrame) -> Spikes:
            query = {SIMULATION_ID: self.simulation_ids} if self.simulation_ids else {}
            return Spikes.from_pandas(df, query=query)

        def _extract_new() -> Spikes:
            return Spikes.from_simulations(
                simulations=self.simulations,
                neurons=self.neurons,
                windows=self.windows,
            )

        return self._extraction_wrapper(
            name="spikes",
            extract_cached=_extract_cached,
            extract_new=_extract_new,
        )

    def extract(self) -> None:
        """Extract all the dataframes."""
        for name in self.names:
            getattr(self, name)
        self.check_extractions()

    def is_extracted(self) -> bool:
        """Return True if all the dataframes have been extracted."""
        # note: the cached_property is stored as an attribute after it's accessed
        return all(name in self.__dict__ for name in self.names)

    def check_extractions(self) -> None:
        """Check that all the dataframes have been extracted."""
        if not self.is_extracted():
            raise RuntimeError("Not all the dataframes have been extracted")

    def missing_simulations(self) -> pd.DataFrame:
        """Return a DataFrame with the simulations ignored because of missing spikes.

        Returns:
            pd.DataFrame with simulation_path as columns, simulation conditions as index,
                and one record for each ignored and missing simulation.
        """
        all_simulations = self._simulations_config.to_pandas().rename(SIMULATION_PATH)
        extracted_simulations = self.simulations.df[SIMULATION_PATH]
        return (
            pd.merge(
                all_simulations,
                extracted_simulations,
                left_on=[*all_simulations.index.names, SIMULATION_PATH],
                right_on=[*extracted_simulations.index.names, SIMULATION_PATH],
                how="left",
                indicator=True,
            )
            .etl.q(_merge="left_only")
            .drop(columns="_merge")
        )

    def print(self) -> None:
        """Print some information about the instance.

        Only for debug and internal use, it may be removed in a future release.
        """
        print("### extraction_config")
        print(json.dumps(self._extraction_config, indent=2))
        print("### simulations_config")
        print(json.dumps(self._simulations_config.to_dict(), indent=2))
        names = ["simulations", "neurons", "neuron_classes", "trial_steps", "windows", "spikes"]
        for name in names:
            print(f"### {name}.df")
            df = getattr(getattr(self, name), "df")
            print(df)
            print(df.dtypes)
