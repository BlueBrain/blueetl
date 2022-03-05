import json
import logging

from blueetl.extract.neurons import Neurons
from blueetl.extract.simulations import Simulations
from blueetl.extract.spikes import Spikes
from blueetl.store.hdf import HDFStore

L = logging.getLogger(__name__)


class Repository:
    def __init__(self, simulations_config, extraction_config, cache_dir, use_cache=False):
        self.extraction_config = extraction_config
        self.simulations_config = simulations_config
        self.store = HDFStore(cache_dir)
        self.use_cache = use_cache
        self.simulations = None
        self.neurons = None
        self.spikes = None

    def extract_simulations(self):
        df = self.store.load("simulations") if self.use_cache else None
        if df is not None:
            L.info("Simulations loaded from existing dataframe")
            self.simulations = Simulations.from_pandas(df)
        else:
            L.info("Extracting simulations...")
            self.simulations = Simulations.from_config(self.simulations_config)
            L.info("Dumping simulations...")
            self.store.dump(self.simulations.to_pandas(), "simulations")

    def extract_neurons(self):
        df = self.store.load("neurons") if self.use_cache else None
        if df is not None:
            L.info("Neurons loaded from existing dataframe")
            self.neurons = Neurons.from_pandas(df)
        else:
            L.info("Extracting neurons...")
            self.neurons = Neurons.from_simulations(
                simulations=self.simulations,
                target=self.extraction_config["target"],
                neuron_classes=self.extraction_config["neuron_classes"],
                limit=self.extraction_config["limit"],
            )
            L.info("Dumping neurons...")
            self.store.dump(self.neurons.to_pandas(), "neurons")

    def extract_spikes(self):
        df = self.store.load("spikes") if self.use_cache else None
        if df is not None:
            L.info("Spikes loaded from existing dataframe")
            self.spikes = Spikes.from_pandas(df)
        else:
            L.info("Extracting spikes...")
            self.spikes = Spikes.from_simulations(
                simulations=self.simulations,
                neurons=self.neurons,
                windows=self.extraction_config["windows"],
            )
            L.info("Dumping spikes...")
            self.store.dump(self.spikes.to_pandas(), "spikes")

    def extract(self):
        self.extract_simulations()
        self.extract_neurons()
        self.extract_spikes()

    def print(self):
        print("### extraction_config")
        print(json.dumps(self.extraction_config, indent=2))
        print("### simulations_config")
        print(json.dumps(self.simulations_config.to_dict(), indent=2))
        print("### simulations.df")
        print(self.simulations.df)
        print(self.simulations.df.dtypes)
        print("### neurons.df")
        print(self.neurons.df)
        print(self.neurons.df.dtypes)
        print("### spikes.df")
        print(self.spikes.df)
        print(self.spikes.df.dtypes)
