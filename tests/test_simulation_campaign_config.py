from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from blueetl.simulation_campaign.config import SimulationCampaignResult

TEST_DATA_PATH = Path(__file__).parent / "data"


def _get_simulation_campaign_result():
    """Return a SimulationCampaignResult instance."""
    return SimulationCampaignResult(
        name="dummy_name",
        attrs={"k1": "v1", "k2": "v2"},
        data=[
            {"ca": 1.1, "seed": 1, "path": "path1"},
            {"ca": 1.2, "seed": 1, "path": "path2"},
        ],
    )


def _get_dict():
    """Return a dict consistent with _get_simulation_campaign_result."""
    return {
        "name": "dummy_name",
        "attrs": {"k1": "v1", "k2": "v2"},
        "data": [
            {"ca": 1.1, "path": "path1", "seed": 1},
            {"ca": 1.2, "path": "path2", "seed": 1},
        ],
    }


def _get_pandas():
    """Return a Series consistent with _get_simulation_campaign_result."""
    s = pd.Series(
        data=["path1", "path2"],
        index=pd.MultiIndex.from_tuples([(1.1, 1), (1.2, 1)], names=["ca", "seed"]),
        name="dummy_name",
    )
    s.attrs = {"k1": "v1", "k2": "v2"}
    return s


def _get_xarray():
    """Return a DataArray consistent with _build_simulation_campaign_result."""
    return xr.DataArray(
        data=[["path1"], ["path2"]],
        coords={"ca": [1.1, 1.2], "seed": [1]},
        attrs={"k1": "v1", "k2": "v2"},
        name="dummy_name",
    )


def test_simulation_campaign_result_load():
    result = SimulationCampaignResult.load(TEST_DATA_PATH / "simulation_campaign_result.yaml")
    assert isinstance(result, SimulationCampaignResult)
    assert result.name == "dummy_name"
    assert result.attrs == {"k1": "v1", "k2": "v2"}
    assert result.data == [
        {"ca": 1.1, "path": "path1", "seed": 1},
        {"ca": 1.2, "path": "path2", "seed": 1},
    ]


def test_simulation_campaign_result_from_dict():
    d = _get_dict()
    result = SimulationCampaignResult.from_dict(d)
    assert isinstance(result, SimulationCampaignResult)
    assert result.name == d["name"]
    assert result.attrs == d["attrs"]
    assert result.data == d["data"]


def test_simulation_campaign_result_from_pandas():
    s = _get_pandas()
    result = SimulationCampaignResult.from_pandas(s)
    assert isinstance(result, SimulationCampaignResult)
    assert result.name == s.name
    assert result.attrs == s.attrs
    assert result.data == [
        {"ca": 1.1, "path": "path1", "seed": 1},
        {"ca": 1.2, "path": "path2", "seed": 1},
    ]


def test_simulation_campaign_result_from_xarray():
    da = _get_xarray()
    result = SimulationCampaignResult.from_xarray(da)
    assert isinstance(result, SimulationCampaignResult)
    assert result.name == da.name
    assert result.attrs == da.attrs
    assert result.data == [
        {"ca": 1.1, "path": "path1", "seed": 1},
        {"ca": 1.2, "path": "path2", "seed": 1},
    ]


def test_simulation_campaign_result_dump(tmp_path):
    expected = _get_dict()
    path = tmp_path / "tmp_config.yaml"
    _get_simulation_campaign_result().dump(path)
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    assert content == expected


def test_simulation_campaign_result_to_dict():
    expected = _get_dict()
    result = _get_simulation_campaign_result().to_dict()
    assert isinstance(result, dict)
    assert result == expected


def test_simulation_campaign_result_to_pandas():
    expected = _get_pandas()
    result = _get_simulation_campaign_result().to_pandas()
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, expected)
    # not tested with assert_series_equal
    assert result.attrs == expected.attrs


def test_simulation_campaign_result_to_xarray():
    expected = _get_xarray()
    result = _get_simulation_campaign_result().to_xarray()
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_equal(result, expected)
    # not tested with assert_equal
    assert result.name == expected.name
    assert result.attrs == expected.attrs
