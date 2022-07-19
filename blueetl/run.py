"""Run script example."""
# pylint: disable=wildcard-import,unused-import,unused-wildcard-import
import logging

import numpy as np
import pandas as pd

from blueetl.analysis import Analyzer
from blueetl.constants import *
from blueetl.utils import load_yaml

analysis_config_file = "./tests/data/tmp/analysis_config_01.yaml"
# analysis_config_file = "./tests/data/tmp/analysis_config_02.yaml"
# analysis_config_file = "./tests/data/tmp/analysis_config_04_NSETM-1891.yaml"


def main():
    """Run script example, only for demonstration."""
    loglevel = logging.INFO
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=logformat, level=loglevel)
    np.random.seed(0)
    analysis_config = load_yaml(analysis_config_file)
    return Analyzer(analysis_config)


if __name__ == "__main__":
    a = main()
    a.extract_repo()
    a.calculate_features()
