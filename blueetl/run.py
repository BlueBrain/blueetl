import logging

import numpy as np
import pandas as pd

from blueetl.analysis import Analyzer
from blueetl.constants import *
from blueetl.utils import load_yaml

# global variables only for interactive inspection
a = features_by_gid = features_by_neuron_class = generic_features = None

analysis_config_file = "./tests/data/tmp/analysis_config_01.yaml"
# analysis_config_file = "./tests/data/tmp/analysis_config_02.yaml"


def main():
    global a, features_by_gid, features_by_neuron_class, generic_features
    loglevel = logging.INFO
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel)
    np.random.seed(0)
    analysis_config = load_yaml(analysis_config_file)
    a = Analyzer(analysis_config, use_cache=True)
    a.initialize()
    # features_by_gid = a.calculate_features_by_gid()
    # print("### features_by_gid")
    # print(features_by_gid)
    # features_by_neuron_class = a.calculate_features_by_neuron_class()
    # print("### features_by_neuron_class")
    # print(features_by_neuron_class)
    generic_features = a.calculate_generic_features()
    print("### generic_features")
    for k, v in generic_features.items():
        print("#", k)
        print(v)


if __name__ == "__main__":
    main()
