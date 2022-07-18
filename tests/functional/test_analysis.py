import numpy as np
import pandas as pd
import pytest

from utils_functional import GPFS_DATA_PATH, TEST_DATA_PATH, change_directory
from pandas.testing import assert_frame_equal

from blueetl.analysis import Analyzer
from blueetl.constants import CIRCUIT, SIMULATION
from blueetl.utils import load_yaml


def _load_df(path):
    return pd.read_parquet(path)


@pytest.mark.parametrize(
    "analysis_config_file, expected",
    [
        ("analysis_config_01.yaml", "expected/analysis_01"),
        ("analysis_config_02.yaml", "expected/analysis_02"),
    ],
)
def test_analyzer(analysis_config_file, expected, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = GPFS_DATA_PATH / expected

    with change_directory(tmp_path):
        a = Analyzer(analysis_config)

        a.extract_repo()

        expected_df = _load_df(expected_path / "repo" / "simulations.parquet")
        assert_frame_equal(a.repo.simulations.df.drop(columns=[SIMULATION, CIRCUIT]), expected_df)

        expected_df = _load_df(expected_path / "repo" / "neurons.parquet")
        assert_frame_equal(a.repo.neurons.df, expected_df)

        expected_df = _load_df(expected_path / "repo" / "neuron_classes.parquet")
        assert_frame_equal(a.repo.neuron_classes.df, expected_df)

        expected_df = _load_df(expected_path / "repo" / "trial_steps.parquet")
        assert_frame_equal(a.repo.trial_steps.df, expected_df)

        expected_df = _load_df(expected_path / "repo" / "windows.parquet")
        assert_frame_equal(a.repo.windows.df, expected_df)

        expected_df = _load_df(expected_path / "repo" / "spikes.parquet")
        assert_frame_equal(a.repo.spikes.df, expected_df)

        a.calculate_features()

        expected_df = _load_df(expected_path / "features" / "by_gid.parquet")
        assert_frame_equal(a.features.by_gid.df, expected_df)

        expected_df = _load_df(expected_path / "features" / "by_gid_and_trial.parquet")
        assert_frame_equal(a.features.by_gid_and_trial.df, expected_df)

        expected_df = _load_df(expected_path / "features" / "by_neuron_class.parquet")
        assert_frame_equal(a.features.by_neuron_class.df, expected_df)

        expected_df = _load_df(expected_path / "features" / "by_neuron_class_and_trial.parquet")
        assert_frame_equal(a.features.by_neuron_class_and_trial.df, expected_df)

        expected_df = _load_df(expected_path / "features" / "histograms.parquet")
        assert_frame_equal(a.features.histograms.df, expected_df)
