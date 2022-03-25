import logging

import numpy as np
import pandas as pd

from blueetl.analysis import Analyzer
from blueetl.constants import *
from blueetl.utils import load_yaml

analysis_config_file = "./tests/data/tmp/analysis_config_01.yaml"
# analysis_config_file = "./tests/data/tmp/analysis_config_02.yaml"


def main():
    loglevel = logging.INFO
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel)
    np.random.seed(0)
    analysis_config = load_yaml(analysis_config_file)
    a = Analyzer(analysis_config, use_cache=True)
    a.run()
    return a


if __name__ == "__main__":
    a = main()
