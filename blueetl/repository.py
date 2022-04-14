import json
import logging
from typing import Any, Dict, Type

import pandas as pd
from lazy_object_proxy import Proxy

from blueetl import DefaultStore
from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import SIMULATION_PATH
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.spikes import Spikes
from blueetl.extract.trial_steps import TrialSteps
from blueetl.extract.windows import Windows
from blueetl.store.base import BaseStore
from blueetl.utils import timed

L = logging.getLogger(__name__)


class Repository:
    def __init__(
        self,
        simulations_config: SimulationsConfig,
        extraction_config: Dict[str, Any],
        store_dir,
        store_class: Type[BaseStore] = DefaultStore,
        use_cache: bool = False,
    ) -> None:
        self._extraction_config = extraction_config
        self._simulations_config = simulations_config
        self._store = store_class(store_dir)
        self._use_cache = use_cache
        self._names = {
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
        }
        self._init_proxies()

    def _init_proxies(self):
        self.simulations: Simulations = Proxy(self._extract_simulations)
        self.neurons: Neurons = Proxy(self._extract_neurons)
        self.neuron_classes: NeuronClasses = Proxy(self._extract_neuron_classes)
        self.trial_steps: TrialSteps = Proxy(self._extract_trial_steps)
        self.windows: Windows = Proxy(self._extract_windows)
        self.spikes: Spikes = Proxy(self._extract_spikes)

    @property
    def names(self):
        return sorted(self._names)

    def __getstate__(self):
        """Get the object state when the object is pickled."""
        if not self.is_extracted():
            # ensure that the dataframes are extracted and stored to disk
            L.info("Extracting dataframes before serialization")
            self.extract()
        # Copy the object's state from self.__dict__
        state = self.__dict__.copy()
        for name in self.names:
            # Remove the unpicklable entries.
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        """Set the object state when the object is unpickled."""
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore the previous state.
        self._init_proxies()

    def _extract_simulations(self) -> Simulations:
        name = "simulations"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = Simulations.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = Simulations.from_config(self._simulations_config)
            self._store.dump(instance.to_pandas(), name)
        return instance

    def _extract_neurons(self) -> Neurons:
        name = "neurons"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = Neurons.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = Neurons.from_simulations(
                    simulations=self.simulations,
                    target=self._extraction_config["target"],
                    neuron_classes=self._extraction_config["neuron_classes"],
                    limit=self._extraction_config["limit"],
                )
            self._store.dump(instance.to_pandas(), name)
        return instance

    def _extract_neuron_classes(self) -> NeuronClasses:
        name = "neuron_classes"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = NeuronClasses.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = NeuronClasses.from_neurons(
                    neurons=self.neurons,
                    target=self._extraction_config["target"],
                    neuron_classes=self._extraction_config["neuron_classes"],
                    limit=self._extraction_config["limit"],
                )
            self._store.dump(instance.to_pandas(), name)
        return instance

    def _extract_trial_steps(self) -> TrialSteps:
        name = "trial_steps"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = TrialSteps.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = TrialSteps.from_simulations(
                    simulations=self.simulations,
                    config=self._extraction_config,
                )
            self._store.dump(instance.to_pandas(), name)
        return instance

    def _extract_windows(self) -> Windows:
        name = "windows"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = Windows.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = Windows.from_simulations(
                    simulations=self.simulations,
                    trial_steps=self.trial_steps,
                    config=self._extraction_config,
                )
            self._store.dump(instance.to_pandas(), name)
        return instance

    def _extract_spikes(self) -> Spikes:
        name = "spikes"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.debug("Loading cached %s", name)
            instance = Spikes.from_pandas(df)
        else:
            L.debug("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                instance = Spikes.from_simulations(
                    simulations=self.simulations,
                    neurons=self.neurons,
                    windows=self.windows,
                )
            self._store.dump(instance.to_pandas(), name)
        return instance

    def extract(self):
        """Extract all the dataframes."""
        for name in self.names:
            # resolve all the proxies
            proxy = getattr(self, name)
            getattr(proxy, "__wrapped__")
        self.check_extractions()

    def is_extracted(self):
        """Return True if all the dataframes have been extracted."""
        return all(getattr(self, name).__resolved__ for name in self.names)

    def check_extractions(self):
        """Check that all the dataframes have been extracted."""
        if not self.is_extracted():
            raise RuntimeError("Not all the dataframes have been extracted")

    def missing_simulations(self):
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

    def print(self):
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
