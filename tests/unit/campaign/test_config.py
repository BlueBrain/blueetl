import json
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pytest_lazyfixture import lazy_fixture

from blueetl.campaign import config as test_module
from blueetl.utils import dump_yaml, load_yaml
from tests.unit.utils import TEST_DATA_PATH, assert_frame_equal


@pytest.mark.parametrize(
    "input_dict, expected_obj",
    [
        (
            lazy_fixture("xarray_config_dict_simple"),
            lazy_fixture("blueetl_config_obj_simple"),
        ),
        (
            lazy_fixture("xarray_config_dict_coupled"),
            lazy_fixture("blueetl_config_obj_coupled"),
        ),
        (
            lazy_fixture("blueetl_config_dict_simple"),
            lazy_fixture("blueetl_config_obj_simple"),
        ),
        (
            lazy_fixture("blueetl_config_dict_coupled"),
            lazy_fixture("blueetl_config_obj_coupled"),
        ),
    ],
)
def test_simulations_config_load(tmp_path, input_dict, expected_obj):
    config_path = tmp_path / "config.yaml"
    dump_yaml(config_path, input_dict)

    result = test_module.SimulationCampaign.load(config_path)

    assert isinstance(result, test_module.SimulationCampaign)
    assert result.name == expected_obj.name
    assert result.attrs == expected_obj.attrs
    assert_frame_equal(result._data, expected_obj._data)
    assert result == expected_obj


@pytest.mark.parametrize(
    "input_obj, expected_dict",
    [
        (
            lazy_fixture("blueetl_config_obj_simple"),
            lazy_fixture("blueetl_config_dict_simple"),
        ),
        (
            lazy_fixture("blueetl_config_obj_coupled"),
            lazy_fixture("blueetl_config_dict_coupled"),
        ),
    ],
)
def test_simulations_config_dump(tmp_path, input_obj, expected_dict):
    config_path = tmp_path / "config.yaml"

    input_obj.dump(config_path)

    content = load_yaml(config_path)
    assert content == expected_dict


@pytest.mark.parametrize(
    "input_dict",
    [
        lazy_fixture("blueetl_config_dict_simple"),
        lazy_fixture("blueetl_config_dict_coupled"),
    ],
)
def test_simulations_config_load_dump_roundtrip(tmp_path, input_dict):
    config_path_1 = tmp_path / "config_1.yaml"
    config_path_2 = tmp_path / "config_2.yaml"
    dump_yaml(config_path_1, input_dict)
    simulations_config_1 = test_module.SimulationCampaign.load(config_path_1)
    simulations_config_1.dump(config_path_2)
    simulations_config_2 = test_module.SimulationCampaign.load(config_path_2)

    assert simulations_config_1.name == simulations_config_2.name
    assert simulations_config_1.attrs == simulations_config_2.attrs
    assert_frame_equal(simulations_config_1._data, simulations_config_2._data)
    assert simulations_config_1 == simulations_config_2


@pytest.mark.parametrize(
    "input_dict, expected_obj",
    [
        (
            lazy_fixture("blueetl_config_dict_simple"),
            lazy_fixture("blueetl_config_obj_simple"),
        ),
        (
            lazy_fixture("blueetl_config_dict_coupled"),
            lazy_fixture("blueetl_config_obj_coupled"),
        ),
    ],
)
def test_simulations_config_from_dict(input_dict, expected_obj):
    result = test_module.SimulationCampaign.from_dict(input_dict)

    assert isinstance(result, test_module.SimulationCampaign)
    assert result == expected_obj


@pytest.mark.parametrize(
    "input_dict, expected_obj",
    [
        (
            lazy_fixture("xarray_config_dict_simple"),
            lazy_fixture("blueetl_config_obj_simple"),
        ),
        (
            lazy_fixture("xarray_config_dict_coupled"),
            lazy_fixture("blueetl_config_obj_coupled"),
        ),
    ],
)
def test_simulations_config_from_xarray_dict(input_dict, expected_obj):
    result = test_module.SimulationCampaign.from_xarray_dict(input_dict)

    assert isinstance(result, test_module.SimulationCampaign)
    assert result == expected_obj


@pytest.mark.parametrize(
    "input_obj, expected_dict",
    [
        (
            lazy_fixture("blueetl_config_obj_simple"),
            lazy_fixture("blueetl_config_dict_simple"),
        ),
        (
            lazy_fixture("blueetl_config_obj_coupled"),
            lazy_fixture("blueetl_config_dict_coupled"),
        ),
    ],
)
def test_simulations_config_to_dict(input_obj, expected_dict):
    result = input_obj.to_dict()

    assert isinstance(result, dict)
    assert result == expected_dict


@pytest.mark.parametrize(
    "input_obj, expected_dict",
    [
        (
            lazy_fixture("blueetl_config_obj_simple"),
            lazy_fixture("xarray_config_dict_simple"),
        ),
        (
            lazy_fixture("blueetl_config_obj_coupled"),
            lazy_fixture("xarray_config_dict_coupled"),
        ),
    ],
)
def test_simulations_config_to_xarray_dict(input_obj, expected_dict):
    result = input_obj.to_xarray_dict()

    assert isinstance(result, dict)
    result = json.loads(json.dumps(result))  # convert tuples to lists
    assert result == expected_dict


@pytest.mark.parametrize(
    "input_obj, expected_df",
    [
        (
            lazy_fixture("blueetl_config_obj_simple"),
            lazy_fixture("blueetl_config_dataframe_simple"),
        ),
        (
            lazy_fixture("blueetl_config_obj_coupled"),
            lazy_fixture("blueetl_config_dataframe_coupled"),
        ),
    ],
)
def test_simulations_config_get_all(input_obj, expected_df):
    result = input_obj.get()

    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected_df)
    assert result is not input_obj._data


@pytest.mark.parametrize(
    "input_obj",
    [
        lazy_fixture("blueetl_config_obj_simple"),
        lazy_fixture("blueetl_config_obj_coupled"),
    ],
)
@pytest.mark.parametrize(
    "filename, expected",
    [
        ("circuit_sonata.json", True),
        ("CircuitConfig", False),
    ],
)
def test_simulations_config_is_sonata(input_obj, filename, expected):
    input_obj.attrs["circuit_config"] = f"/path/to/{filename}"
    result = input_obj.is_sonata()
    assert result == expected


@pytest.mark.parametrize(
    "input_obj",
    [
        lazy_fixture("blueetl_config_obj_simple"),
        lazy_fixture("blueetl_config_obj_coupled"),
    ],
)
def test_simulations_config_is_sonata_raises(input_obj):
    del input_obj.attrs["circuit_config"]
    with pytest.raises(RuntimeError, match="circuit_config is missing in the simulation campaign"):
        input_obj.is_sonata()


@pytest.mark.parametrize(
    "input_obj, expected_len",
    [
        (lazy_fixture("blueetl_config_obj_simple"), 4),
        (lazy_fixture("blueetl_config_obj_coupled"), 2),
    ],
)
def test_simulations_config_len(input_obj, expected_len):
    result = len(input_obj)
    assert result == expected_len


@pytest.mark.parametrize(
    "input_obj, expected_len",
    [
        (lazy_fixture("blueetl_config_obj_simple"), 4),
        (lazy_fixture("blueetl_config_obj_coupled"), 2),
    ],
)
def test_simulations_config_iter(input_obj, expected_len):
    count = 0
    for sim in input_obj:
        assert isinstance(sim, test_module.SimulationRow)
        count += 1
    assert count == expected_len


@pytest.mark.parametrize(
    "input_obj, expected_len",
    [
        (lazy_fixture("blueetl_config_obj_simple"), 4),
        (lazy_fixture("blueetl_config_obj_coupled"), 2),
    ],
)
def test_simulations_config_getitem(input_obj, expected_len):
    for i in range(expected_len):
        sim = input_obj[i]
        assert isinstance(sim, test_module.SimulationRow)


@pytest.mark.parametrize(
    "input_obj, expected_data, expected_index",
    [
        (lazy_fixture("blueetl_config_obj_simple"), [[2.0, 4.0]], [3]),
        (lazy_fixture("blueetl_config_obj_coupled"), [[2.0, 4.0]], [1]),
    ],
)
def test_simulations_config_get(input_obj, expected_data, expected_index):
    result = input_obj.get(ca=2.0, depolarization=4.0)

    columns = ["ca", "depolarization"]
    expected_df = pd.DataFrame(expected_data, columns=columns, index=expected_index)
    assert_frame_equal(result[columns], expected_df)


@pytest.mark.parametrize(
    "input_obj, expected_ids",
    [
        (lazy_fixture("blueetl_config_obj_simple"), [3]),
        (lazy_fixture("blueetl_config_obj_coupled"), [1]),
    ],
)
def test_simulations_config_ids(input_obj, expected_ids):
    result = input_obj.ids(ca=2.0, depolarization=4.0)

    assert_array_equal(result, expected_ids)


@pytest.mark.parametrize(
    "config_file",
    [
        "config_01_relative_bbp_workflow.json",
        "config_02_relative_blueetl.yaml",
    ],
)
def test_simulations_config_with_relative_paths(config_file):
    config_file = TEST_DATA_PATH / "simulation_campaign" / config_file

    result = test_module.SimulationCampaign.load(config_file)

    assert Path(result.attrs["path_prefix"]).is_absolute()
    assert Path(result.attrs["circuit_config"]).is_absolute()
