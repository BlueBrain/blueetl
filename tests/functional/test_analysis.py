from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from utils_functional import GPFS_DATA_PATH, TEST_DATA_PATH, change_directory

from blueetl.analysis import Analyzer
from blueetl.constants import CIRCUIT, SIMULATION
from blueetl.utils import load_yaml


def _get_subattr(obj, attrs):
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _load_df(path):
    return pd.read_parquet(path)


def _dump_df(df, path):
    df.to_parquet(path)


def _dump_all(a, path):
    conf = {
        "repo": [
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
        ],
        "features": [
            "by_gid",
            "by_gid_and_trial",
            "by_neuron_class",
            "by_neuron_class_and_trial",
            "histograms",
        ],
    }
    path = Path(path).resolve()
    for container, names in conf.items():
        for name in names:
            df = _get_subattr(a, [container, name, "df"])
            df = df[[col for col in df.columns if col not in [SIMULATION, CIRCUIT]]]
            filepath = path / container / f"{name}.parquet"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            _dump_df(df, filepath)


def _test_repo(a, path):
    a.extract_repo()

    expected_df = _load_df(path / "repo" / "simulations.parquet")
    assert_frame_equal(a.repo.simulations.df.drop(columns=[SIMULATION, CIRCUIT]), expected_df)

    expected_df = _load_df(path / "repo" / "neurons.parquet")
    assert_frame_equal(a.repo.neurons.df, expected_df)

    expected_df = _load_df(path / "repo" / "neuron_classes.parquet")
    assert_frame_equal(a.repo.neuron_classes.df, expected_df)

    expected_df = _load_df(path / "repo" / "trial_steps.parquet")
    assert_frame_equal(a.repo.trial_steps.df, expected_df)

    expected_df = _load_df(path / "repo" / "windows.parquet")
    assert_frame_equal(a.repo.windows.df, expected_df)

    expected_df = _load_df(path / "repo" / "spikes.parquet")
    assert_frame_equal(a.repo.spikes.df, expected_df)


def _test_features(a, path):
    a.calculate_features()

    expected_df = _load_df(path / "features" / "by_gid.parquet")
    assert_frame_equal(a.features.by_gid.df, expected_df)

    expected_df = _load_df(path / "features" / "by_gid_and_trial.parquet")
    assert_frame_equal(a.features.by_gid_and_trial.df, expected_df)

    expected_df = _load_df(path / "features" / "by_neuron_class.parquet")
    assert_frame_equal(a.features.by_neuron_class.df, expected_df)

    expected_df = _load_df(path / "features" / "by_neuron_class_and_trial.parquet")
    assert_frame_equal(a.features.by_neuron_class_and_trial.df, expected_df)

    expected_df = _load_df(path / "features" / "histograms.parquet")
    assert_frame_equal(a.features.histograms.df, expected_df)


def _test_filter_in_memory(a, path):
    a2 = a.apply_filter()
    if not a.simulations_filter_in_memory:
        assert a2 is a
    else:
        assert a2 is not a
        # test that the new DataFrames have been filtered
        _test_repo(a2, path / "_filtered")
        _test_features(a2, path / "_filtered")
        # test that the original DataFrames are unchanged
        _test_repo(a, path)
        _test_features(a, path)


def _update_expected_files(a, path):
    # to be used only when test cases are added or modified
    a2 = a.apply_filter()
    _dump_all(a, path)
    if a.simulations_filter_in_memory:
        _dump_all(a2, path / "_filtered")


@pytest.mark.parametrize(
    "analysis_config_file, expected",
    [
        ("analysis_config_01.yaml", "expected/analysis_01"),
        ("analysis_config_02.yaml", "expected/analysis_02"),
        ("analysis_config_03.yaml", "expected/analysis_03"),
        ("analysis_config_04.yaml", "expected/analysis_04"),
    ],
)
@pytest.mark.skip(reason="to be executed only to create or overwrite the expected files")
def test_update_expected_files(analysis_config_file, expected, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = GPFS_DATA_PATH / expected

    with change_directory(tmp_path), Analyzer(analysis_config) as analyzer:
        _update_expected_files(analyzer, expected_path)


@pytest.mark.parametrize(
    "analysis_config_file, expected",
    [
        ("analysis_config_01.yaml", "expected/analysis_01"),
        ("analysis_config_02.yaml", "expected/analysis_02"),
        ("analysis_config_03.yaml", "expected/analysis_03"),
        ("analysis_config_04.yaml", "expected/analysis_04"),
    ],
)
def test_analyzer(analysis_config_file, expected, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = GPFS_DATA_PATH / expected

    # test without cache
    with change_directory(tmp_path), Analyzer(analysis_config) as analyzer:
        _test_repo(analyzer, expected_path)
        _test_features(analyzer, expected_path)
        _test_filter_in_memory(analyzer, expected_path)

    # test with cache
    with change_directory(tmp_path), Analyzer(analysis_config) as analyzer:
        _test_repo(analyzer, expected_path)
        _test_features(analyzer, expected_path)
        _test_filter_in_memory(analyzer, expected_path)
