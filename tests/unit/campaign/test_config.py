import pandas as pd
import pytest
import xarray as xr

from blueetl.campaign.config import SimulationCampaignConfig
from blueetl.utils import load_yaml
from tests.unit.utils import TEST_DATA_PATH, assert_frame_equal, assert_xr_equal


@pytest.mark.parametrize(
    "config_path, expected_name, expected_attrs, expected_conditions, expected_data",
    [
        pytest.param(
            "analysis_config/simulations_config_01_blueetl.yaml",
            "dummy_name",
            {"k1": "v1", "k2": "v2"},
            ["ca", "seed"],
            [
                {"simulation_path": "/path/to/1", "ca": 1.1, "seed": 1},
                {"simulation_path": "/path/to/2", "ca": 1.2, "seed": 1},
            ],
            id="01_blueetl",
        ),
        pytest.param(
            "analysis_config/simulations_config_02_bbpwf_simple.json",
            "c26abad3-0861-4f7d-863c-a42daffd029f",
            {"path_prefix": "/path/to/simulation/campaign"},
            ["ca", "shotn_mean_pct", "shotn_sd_pct", "seed"],
            [
                {
                    "ca": 1.05,
                    "shotn_mean_pct": 50,
                    "shotn_sd_pct": 40,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/c26abad3-0861-4f7d-863c-a42daffd029f/000",
                },
                {
                    "ca": 1.05,
                    "shotn_mean_pct": 60,
                    "shotn_sd_pct": 40,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/c26abad3-0861-4f7d-863c-a42daffd029f/001",
                },
                {
                    "ca": 1.15,
                    "shotn_mean_pct": 50,
                    "shotn_sd_pct": 40,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/c26abad3-0861-4f7d-863c-a42daffd029f/002",
                },
                {
                    "ca": 1.15,
                    "shotn_mean_pct": 60,
                    "shotn_sd_pct": 40,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/c26abad3-0861-4f7d-863c-a42daffd029f/003",
                },
            ],
            id="02_bbpwf_simple",
        ),
        pytest.param(
            "analysis_config/simulations_config_03_bbpwf_coupled.json",
            "881857e0-c7ae-49c1-a6f6-8a34f43c2e26",
            {"path_prefix": "/path/to/simulation/campaign"},
            ["coupled", "ca", "stdev_mean_ratio", "fr_scale", "seed"],
            [
                {
                    "coupled": 0,
                    "ca": 1.0,
                    "stdev_mean_ratio": 0.9,
                    "fr_scale": 0.125,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/881857e0-c7ae-49c1-a6f6-8a34f43c2e26/000",
                },
                {
                    "coupled": 1,
                    "ca": 1.15,
                    "stdev_mean_ratio": 1.1,
                    "fr_scale": 1.0,
                    "seed": 628462,
                    "simulation_path": "/path/to/simulation/campaign/881857e0-c7ae-49c1-a6f6-8a34f43c2e26/001",
                },
            ],
            id="03_bbpwf_coupled",
        ),
    ],
)
def test_simulations_config_load(
    config_path, expected_name, expected_attrs, expected_conditions, expected_data
):
    result = SimulationCampaignConfig.load(TEST_DATA_PATH / config_path)

    assert isinstance(result, SimulationCampaignConfig)
    assert result.name == expected_name
    assert result.attrs == expected_attrs
    assert result.conditions == expected_conditions
    expected = pd.DataFrame(expected_data)
    expected.attrs = expected_attrs
    assert_frame_equal(result.data, expected)


def test_simulations_config_dump(tmp_path, simulations_config, simulations_config_dict):
    expected = simulations_config_dict
    path = tmp_path / "tmp_config.yaml"
    simulations_config.dump(path)

    content = load_yaml(path)
    assert content == expected


@pytest.mark.parametrize(
    "config_path",
    [
        "analysis_config/simulations_config_01_blueetl.yaml",
        "analysis_config/simulations_config_02_bbpwf_simple.json",
        "analysis_config/simulations_config_03_bbpwf_coupled.json",
    ],
)
def test_simulations_config_load_dump_roundtrip(tmp_path, config_path):
    simulations_config_1 = SimulationCampaignConfig.load(TEST_DATA_PATH / config_path)
    new_path = tmp_path / "tmp_config.yaml"
    simulations_config_1.dump(new_path)
    simulations_config_2 = SimulationCampaignConfig.load(new_path)

    assert simulations_config_1.name == simulations_config_2.name
    assert simulations_config_1.attrs == simulations_config_2.attrs
    assert simulations_config_1.conditions == simulations_config_2.conditions
    assert_frame_equal(simulations_config_1.data, simulations_config_2.data)


def test_simulations_config_from_dict(simulations_config_dict, simulations_config_dataframe):
    d = simulations_config_dict
    result = SimulationCampaignConfig.from_dict(d)

    assert isinstance(result, SimulationCampaignConfig)
    assert result.name == d["name"]
    assert result.attrs == d["attrs"]
    assert result.conditions == d["conditions"]
    assert_frame_equal(result.data, simulations_config_dataframe)


def test_simulations_config_from_xarray(simulations_config_xarray, simulations_config_dataframe):
    da = simulations_config_xarray
    result = SimulationCampaignConfig.from_xarray(da)

    assert isinstance(result, SimulationCampaignConfig)
    assert result.name == da.name
    assert result.attrs == da.attrs
    assert_frame_equal(result.data, simulations_config_dataframe)


def test_simulations_config_to_dict(simulations_config, simulations_config_dict):
    result = simulations_config.to_dict()

    assert isinstance(result, dict)
    assert result == simulations_config_dict


def test_simulations_config_to_xarray(simulations_config, simulations_config_xarray):
    expected = simulations_config_xarray
    result = simulations_config.to_xarray()

    assert isinstance(result, xr.DataArray)
    assert_xr_equal(result, expected)


def test_simulations_config_to_pandas(simulations_config, simulations_config_dataframe):
    expected = simulations_config_dataframe
    result = simulations_config.to_pandas()

    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected)
