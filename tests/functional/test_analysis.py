from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from blueetl.analysis import MultiAnalyzer
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
    ("analysis_config_05.yaml", "analysis_05"),
    ("analysis_config_06.yaml", "analysis_06"),
    ("analysis_config_07.yaml", "analysis_07"),
    ("analysis_config_08.yaml", "analysis_08"),
    ("analysis_config_09.yaml", "analysis_09"),
    ("analysis_config_10.yaml", "analysis_10"),
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
    a.extract_repo()
    a.calculate_features()
    for container_name in "repo", "features":
        container = getattr(a, container_name)
        for name in getattr(container, "names"):
            df = _get_subattr(container, [name, "df"])
            df = df[[col for col in df.columns if col not in [SIMULATION, CIRCUIT]]]
            filepath = path / container_name / f"{name}.parquet"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            _dump_df(df, filepath)


def _dump_all_multi(ma, path):
    # used only when test cases are added or modified
    path = Path(path).resolve()
    ma.extract_repo()
    ma.calculate_features()
    for name in ma.names:
        partial_path = path / name
        a = getattr(ma, name)
        _dump_all(a, partial_path)


def _test_repo(a, path):
    a.extract_repo()
    path = path / "repo"
    assert len(a.repo.names) > 0
    assert sorted(a.repo.names) == sorted(p.stem for p in path.glob("*.parquet"))
    for name in a.repo.names:
        expected_df = _load_df(path / f"{name}.parquet")
        df = _get_subattr(a, ["repo", name, "df"])
        df = df[[col for col in df.columns if col not in [SIMULATION, CIRCUIT]]]
        with assertion_error_message(f"Difference in repo {name!r}"):
            assert_frame_equal(df, expected_df)


def _test_features(a, path):
    a.calculate_features()
    path = path / "features"
    assert sorted(a.features.names) == sorted(p.stem for p in path.glob("*.parquet"))
    for name in a.features.names:
        expected_df = _load_df(path / f"{name}.parquet")
        actual_df = _get_subattr(a, ["features", name, "df"])
        with assertion_error_message(f"Difference in features {name!r}"):
            assert_frame_equal(actual_df, expected_df)


def _test_repo_multi(ma, path):
    ma.extract_repo()
    for name in ma.names:
        partial_path = path / name
        a = getattr(ma, name)
        _test_repo(a, partial_path)


def _test_features_multi(ma, path):
    ma.calculate_features()
    for name in ma.names:
        partial_path = path / name
        a = getattr(ma, name)
        _test_features(a, partial_path)


def _test_filter_in_memory(ma, path):
    ma2 = ma.apply_filter()
    if not ma.global_config.simulations_filter_in_memory:
        assert ma2 is ma
    else:
        assert ma2 is not ma
        # test that the new DataFrames have been filtered
        _test_repo_multi(ma2, path / "_filtered")
        _test_features_multi(ma2, path / "_filtered")
        # test that the original DataFrames are unchanged
        _test_repo_multi(ma, path)
        _test_features_multi(ma, path)


def _update_expected_files(ma, path):
    # used only when test cases are added or modified
    _dump_all_multi(ma, path)
    if ma.global_config.simulations_filter_in_memory:
        ma2 = ma.apply_filter()
        _dump_all_multi(ma2, path / "_filtered")


@pytest.mark.skip(reason="to be executed only to create or overwrite the expected files")
@pytest.mark.parametrize("analysis_config_file, expected_dir", analysis_configs)
def test_update_expected_files(analysis_config_file, expected_dir, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = EXPECTED_PATH / expected_dir

    with change_directory(tmp_path), MultiAnalyzer(analysis_config) as multi_analyzer:
        _update_expected_files(multi_analyzer, expected_path)

    assert 0, (
        "This test should be executed only to create or overwrite the expected files. "
        "You can ignore this error if this is the case."
    )


@pytest.mark.parametrize("analysis_config_file, expected_dir", analysis_configs)
def test_analyzer(analysis_config_file, expected_dir, tmp_path):
    np.random.seed(0)
    analysis_config = load_yaml(TEST_DATA_PATH / analysis_config_file)
    expected_path = EXPECTED_PATH / expected_dir

    # test without cache
    with change_directory(tmp_path), MultiAnalyzer(analysis_config) as multi_analyzer:
        _test_repo_multi(multi_analyzer, expected_path)
        _test_features_multi(multi_analyzer, expected_path)
        _test_filter_in_memory(multi_analyzer, expected_path)

    # test with cache
    with change_directory(tmp_path), MultiAnalyzer(analysis_config) as multi_analyzer:
        _test_repo_multi(multi_analyzer, expected_path)
        _test_features_multi(multi_analyzer, expected_path)
        _test_filter_in_memory(multi_analyzer, expected_path)
