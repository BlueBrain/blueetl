import json
import logging
from typing import Any, Dict, Type

import pandas as pd

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
        self.simulations: Simulations
        self.neurons: Neurons
        self.neuron_classes: NeuronClasses
        self.trial_steps: TrialSteps
        self.windows: Windows
        self.spikes: Spikes
        self._names = {
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
        }

    @property
    def names(self):
        return sorted(self._names)

    def extract_simulations(self):
        name = "simulations"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.simulations = Simulations.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.simulations = Simulations.from_config(self._simulations_config)
            self._store.dump(self.simulations.to_pandas(), name)

    def extract_neurons(self):
        name = "neurons"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.neurons = Neurons.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.neurons = Neurons.from_simulations(
                    simulations=self.simulations,
                    target=self._extraction_config["target"],
                    neuron_classes=self._extraction_config["neuron_classes"],
                    limit=self._extraction_config["limit"],
                )
            self._store.dump(self.neurons.to_pandas(), name)

    def extract_neuron_classes(self):
        name = "neuron_classes"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.neuron_classes = NeuronClasses.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.neuron_classes = NeuronClasses.from_neurons(
                    neurons=self.neurons,
                    target=self._extraction_config["target"],
                    neuron_classes=self._extraction_config["neuron_classes"],
                    limit=self._extraction_config["limit"],
                )
            self._store.dump(self.neuron_classes.to_pandas(), name)

    def extract_trial_steps(self):
        name = "trial_steps"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.trial_steps = TrialSteps.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.trial_steps = TrialSteps.from_simulations(
                    simulations=self.simulations,
                    config=self._extraction_config,
                )
            self._store.dump(self.trial_steps.to_pandas(), name)

    def extract_windows(self):
        name = "windows"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.windows = Windows.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.windows = Windows.from_simulations(
                    simulations=self.simulations,
                    trial_steps=self.trial_steps,
                    config=self._extraction_config,
                )
            self._store.dump(self.windows.to_pandas(), name)

    def extract_spikes(self):
        name = "spikes"
        df = self._store.load(name) if self._use_cache else None
        if df is not None:
            L.info("Loading cached %s", name)
            self.spikes = Spikes.from_pandas(df)
        else:
            L.info("Extracting %s", name)
            with timed(L.info, "Completed extraction of %s", name):
                self.spikes = Spikes.from_simulations(
                    simulations=self.simulations,
                    neurons=self.neurons,
                    windows=self.windows,
                )
            self._store.dump(self.spikes.to_pandas(), name)

    def extract(self):
        self.extract_simulations()
        self.extract_neurons()
        self.extract_neuron_classes()
        self.extract_trial_steps()
        self.extract_windows()
        self.extract_spikes()
        self.check_extractions()

    def check_extractions(self):
        """Check that all the dataframes have been extracted."""
        if any(getattr(self, name, None) is None for name in self.names):
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
