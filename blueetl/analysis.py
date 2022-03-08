import logging

import numpy as np
import pandas as pd

from blueetl import etl
from blueetl.config.simulations import SimulationsConfig
from blueetl.constants import CIRCUIT_ID, GID, NEURON_CLASS, WINDOW
from blueetl.repository import Repository
from blueetl.utils import load_yaml, import_by_string

L = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, analysis_config, use_cache=False):
        self.analysis_config = analysis_config
        self.repo = Repository(
            simulations_config=SimulationsConfig.load(analysis_config["simulation_campaign"]),
            extraction_config=analysis_config["extraction"],
            cache_dir=self.analysis_config["output"],
            use_cache=use_cache,
        )

    def initialize(self):
        self.repo.extract()
        self.repo.print()

    def _get_window_limits(self, window):
        return self.analysis_config["extraction"]["windows"][window]

    def calculate_features_by_gid(self):
        feature_collections = self.analysis_config["analysis"]["features"].get("gid", [])
        records = []
        for key, df in self.repo.spikes.grouped_by_gid():
            record = key._asdict()
            for feature_collection in feature_collections:
                func = import_by_string(feature_collection["function"])
                params = feature_collection.get("params", {})
                record.update(
                    func(
                        analysis=self,
                        circuit_id=record[CIRCUIT_ID],
                        neuron_class=record[NEURON_CLASS],
                        window=record[WINDOW],
                        gid=record[GID],
                        df=df,
                        params=params,
                    )
                )
            records.append(record)
        # in the returned df, the type of `neuron_class` and `window` is `object`
        return pd.DataFrame(records)

    def calculate_features_by_neuron_class(self):
        feature_collections = self.analysis_config["analysis"]["features"].get("neuron_class", [])
        records = []
        for key, df in self.repo.spikes.grouped_by_neuron_class():
            record = key._asdict()
            for feature_collection in feature_collections:
                func = import_by_string(feature_collection["function"])
                params = feature_collection.get("params", {})
                record.update(
                    func(
                        analysis=self,
                        circuit_id=record[CIRCUIT_ID],
                        neuron_class=record[NEURON_CLASS],
                        window=record[WINDOW],
                        df=df,
                        params=params,
                    )
                )
            records.append(record)
        # in the returned df, the type of `neuron_class` and `window` is `object`
        return pd.DataFrame(records)


def main():
    loglevel = logging.INFO
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel)
    np.random.seed(0)
    etl.register_accessors()
    analysis_config = load_yaml("./tests/data/tmp/analysis_config_01.yaml")
    # analysis_config = load_yaml("./tests/data/tmp/analysis_config_02.yaml")
    a = Analyzer(analysis_config, use_cache=True)
    a.initialize()
    features_by_gid = a.calculate_features_by_gid()
    print("### features by gid")
    print(features_by_gid)
    features_by_neuron_class = a.calculate_features_by_neuron_class()
    print("### features by neuron_class")
    print(features_by_neuron_class)


if __name__ == "__main__":
    main()
