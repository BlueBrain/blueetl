import json
import logging
from typing import Any, Dict, Type

from blueetl import DefaultStore
from blueetl.config.simulations import SimulationsConfig
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
        self.extraction_config = extraction_config
        self.simulations_config = simulations_config
        self.store = store_class(store_dir)
        self.use_cache = use_cache
        self.simulations: Simulations
        self.neurons: Neurons
        self.neuron_classes: NeuronClasses
        self.trial_steps: TrialSteps
        self.windows: Windows
        self.spikes: Spikes
        self._all_names = [
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
        ]

    def extract_simulations(self):
        df = self.store.load("simulations") if self.use_cache else None
        if df is not None:
            L.info("simulations dataframe loaded from existing data")
            self.simulations = Simulations.from_pandas(df)
        else:
            L.info("Extracting simulations...")
            with timed(L.info, "Completed simulations extraction"):
                self.simulations = Simulations.from_config(self.simulations_config)
            L.info("Dumping simulations...")
            self.store.dump(self.simulations.to_pandas(), "simulations")

    def extract_neurons(self):
        df = self.store.load("neurons") if self.use_cache else None
        if df is not None:
            L.info("neurons dataframe loaded from existing data")
            self.neurons = Neurons.from_pandas(df)
        else:
            L.info("Extracting neurons...")
            with timed(L.info, "Completed neurons extraction"):
                self.neurons = Neurons.from_simulations(
                    simulations=self.simulations,
                    target=self.extraction_config["target"],
                    neuron_classes=self.extraction_config["neuron_classes"],
                    limit=self.extraction_config["limit"],
                )
            L.info("Dumping neurons...")
            self.store.dump(self.neurons.to_pandas(), "neurons")

    def extract_neuron_classes(self):
        df = self.store.load("neuron_classes") if self.use_cache else None
        if df is not None:
            L.info("neuron_classes dataframe loaded from existing data")
            self.neuron_classes = NeuronClasses.from_pandas(df)
        else:
            L.info("Extracting neuron_classes...")
            with timed(L.info, "Completed neuron_classes extraction"):
                self.neuron_classes = NeuronClasses.from_neurons(
                    neurons=self.neurons,
                    target=self.extraction_config["target"],
                    neuron_classes=self.extraction_config["neuron_classes"],
                    limit=self.extraction_config["limit"],
                )
            L.info("Dumping neuron_classes...")
            self.store.dump(self.neuron_classes.to_pandas(), "neuron_classes")

    def extract_trial_steps(self):
        df = self.store.load("trial_steps") if self.use_cache else None
        if df is not None:
            L.info("trial_steps dataframe loaded from existing data")
            self.trial_steps = TrialSteps.from_pandas(df)
        else:
            L.info("Extracting trial_steps...")
            with timed(L.info, "Completed trial_steps extraction"):
                self.trial_steps = TrialSteps.from_simulations(
                    simulations=self.simulations,
                    config=self.extraction_config,
                )
            L.info("Dumping trial_steps...")
            self.store.dump(self.trial_steps.to_pandas(), "trial_steps")

    def extract_windows(self):
        df = self.store.load("windows") if self.use_cache else None
        if df is not None:
            L.info("windows dataframe loaded from existing data")
            self.windows = Windows.from_pandas(df)
        else:
            L.info("Extracting windows...")
            with timed(L.info, "Completed windows extraction"):
                self.windows = Windows.from_simulations(
                    simulations=self.simulations,
                    trial_steps=self.trial_steps,
                    config=self.extraction_config,
                )
            L.info("Dumping windows...")
            self.store.dump(self.windows.to_pandas(), "windows")

    def extract_spikes(self):
        df = self.store.load("spikes") if self.use_cache else None
        if df is not None:
            L.info("spikes dataframe loaded from existing data")
            self.spikes = Spikes.from_pandas(df)
        else:
            L.info("Extracting spikes...")
            with timed(L.info, "Completed spikes extraction"):
                self.spikes = Spikes.from_simulations(
                    simulations=self.simulations,
                    neurons=self.neurons,
                    windows=self.windows,
                )
            L.info("Dumping spikes...")
            self.store.dump(self.spikes.to_pandas(), "spikes")

    def extract(self):
        self.extract_simulations()
        self.extract_neurons()
        self.extract_neuron_classes()
        self.extract_trial_steps()
        self.extract_windows()
        self.extract_spikes()
        self.check_extractions()

    def check_extractions(self):
        if any(getattr(self, name, None) is None for name in self._all_names):
            raise RuntimeError("Not all the dataframes have been extracted")

    def print(self):
        print("### extraction_config")
        print(json.dumps(self.extraction_config, indent=2))
        print("### simulations_config")
        print(json.dumps(self.simulations_config.to_dict(), indent=2))
        names = ["simulations", "neurons", "neuron_classes", "trial_steps", "windows", "spikes"]
        for name in names:
            print(f"### {name}.df")
            df = getattr(getattr(self, name), "df")
            print(df)
            print(df.dtypes)
