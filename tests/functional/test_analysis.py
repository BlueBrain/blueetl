from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from blueetl.analysis import Analyzer
from blueetl.constants import CIRCUIT, SIMULATION
from blueetl.utils import load_yaml
from tests.functional.utils import (
    EXPECTED_PATH,
    TEST_DATA_PATH,
    assertion_error_message,
    change_directory,
)

# analysis_config_file, expected_dir
analysis_configs = [
    ("analysis_config_01.yaml", "analysis_01"),
    ("analysis_config_02.yaml", "analysis_02"),
    ("analysis_config_03.yaml", "analysis_03"),
    ("analysis_config_04.yaml", "analysis_04"),
    ("analysis_config_07.yaml", "analysis_07"),
    ("analysis_config_09.yaml", "analysis_09"),
]


def _get_subattr(obj, attrs):
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _load_df(path):
    return pd.read_parquet(path)


def _dump_df(df, path):
    df.to_parquet(path)


def _dump_all(a, path):
    # used only when test cases are added or modified
    path = Path(path).resolve()
    for container_name in "repo", "features":
        container = getattr(a, container_name)
        for name in getattr(container, "names"):
            df = _get_subattr(container, [name, "df"])
            df = df[[col for col in df.columns if col not in [SIMULATION, CIRCUIT]]]
            filepath = path / container_name / f"{name}.parquet"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            _dump_df(df, filepath)


def _test_repo(a, path):
    a.extract_repo()
    path = path / "repo"
    assert len(a.repo.names) > 0
    assert sorted(a.repo.names) == sorted(p.stem for p in path.glob("*.parquet"))
    for name in a.repo.names:
        expected_df = _load_df(path / f"{name}.parquet")
        df = _get_subattr(a, ["repo", name, "df"])
        df = df[[col for col in df.columns if col not in [SIMULATION, CIRCUIT]]]
        with assertion_error_message(f"In repo {name}:"):
            assert_frame_equal(df, expected_df)


def _test_features(a, path):
    a.calculate_features()
    path = path / "features"
    assert len(a.features.names) > 0
    assert sorted(a.features.names) == sorted(p.stem for p in path.glob("*.parquet"))
    for name in a.features.names:
        expected_df = _load_df(path / f"{name}.parquet")
        actual_df = _get_subattr(a, ["features", name, "df"])
        with assertion_error_message(f"In features {name}:"):
            assert_frame_equal(actual_df, expected_df)


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
    # used only when test cases are added or modified
    a2 = a.apply_filter()
    _dump_all(a, path)
    if a.simulations_filter_in_memory:
        _dump_all(a2, path / "_filtered")


@pytest.mark.skip(reason="to be executed only to create or overwrite the expected files")
@pytest.mark.parametrize("analysis_config_file, expected_dir", analysis_configs)
def test_update_expected_files(analysis_config_file, expected_dir, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = EXPECTED_PATH / expected_dir

    with change_directory(tmp_path), Analyzer(analysis_config) as analyzer:
        _update_expected_files(analyzer, expected_path)


@pytest.mark.parametrize("analysis_config_file, expected_dir", analysis_configs)
def test_analyzer(analysis_config_file, expected_dir, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = EXPECTED_PATH / expected_dir

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
