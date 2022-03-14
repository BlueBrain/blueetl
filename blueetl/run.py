import logging

import numpy as np

from blueetl.analysis import Analyzer
from blueetl.utils import load_yaml

loglevel = logging.INFO
logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=logformat, level=loglevel)
np.random.seed(0)
analysis_config = load_yaml("./tests/data/tmp/analysis_config_01.yaml")
# analysis_config = load_yaml("./tests/data/tmp/analysis_config_02.yaml")
a = Analyzer(analysis_config, use_cache=True)
a.initialize()
features_by_gid = a.calculate_features_by_gid()
print("### features_by_gid")
print(features_by_gid)
features_by_neuron_class = a.calculate_features_by_neuron_class()
print("### features_by_neuron_class")
print(features_by_neuron_class)
