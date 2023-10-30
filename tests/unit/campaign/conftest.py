import pandas as pd
import pytest
import xarray as xr

from blueetl.campaign.config import SimulationCampaign


@pytest.fixture
def xarray_config_dict_simple():
    return {
        "dims": ["ca", "depolarization"],
        "attrs": {
            "path_prefix": "/tmp/simple",
            "blue_config_template": "simulation_sonata.tmpl",
            "circuit_config": "/tmp/tests/data/circuit_sonata.json",
        },
        "data": [["", "uuid/1"], ["uuid/2", "uuid/3"]],
        "coords": {
            "ca": {"dims": ["ca"], "attrs": {}, "data": [1.0, 2.0]},
            "depolarization": {"dims": ["depolarization"], "attrs": {}, "data": [3.0, 4.0]},
        },
        "name": "uuid",
    }


@pytest.fixture
def xarray_config_dict_coupled():
    return {
        "dims": ["coupled"],
        "attrs": {
            "path_prefix": "/tmp/coupled",
            "blue_config_template": "simulation_sonata.tmpl",
            "circuit_config": "/tmp/tests/data/circuit_sonata.json",
        },
        "data": ["", "uuid/1"],
        "coords": {
            "ca": {"dims": ["coupled"], "attrs": {}, "data": [1.0, 2.0]},
            "depolarization": {"dims": ["coupled"], "attrs": {}, "data": [3.0, 4.0]},
        },
        "name": "uuid",
    }


@pytest.fixture
def blueetl_config_dict_simple():
    return {
        "format": "blueetl",
        "version": 1,
        "name": "uuid",
        "attrs": {
            "path_prefix": "/tmp/simple",
            "blue_config_template": "simulation_sonata.tmpl",
            "circuit_config": "/tmp/tests/data/circuit_sonata.json",
        },
        "data": [
            {
                "ca": 1.0,
                "depolarization": 3.0,
                "simulation_path": "",
            },
            {
                "ca": 1.0,
                "depolarization": 4.0,
                "simulation_path": "/tmp/simple/uuid/1/simulation_config.json",
            },
            {
                "ca": 2.0,
                "depolarization": 3.0,
                "simulation_path": "/tmp/simple/uuid/2/simulation_config.json",
            },
            {
                "ca": 2.0,
                "depolarization": 4.0,
                "simulation_path": "/tmp/simple/uuid/3/simulation_config.json",
            },
        ],
    }


@pytest.fixture
def blueetl_config_dict_coupled():
    return {
        "format": "blueetl",
        "version": 1,
        "name": "uuid",
        "attrs": {
            "path_prefix": "/tmp/coupled",
            "blue_config_template": "simulation_sonata.tmpl",
            "circuit_config": "/tmp/tests/data/circuit_sonata.json",
            "__coupled__": "coupled",
        },
        "data": [
            {
                "ca": 1.0,
                "depolarization": 3.0,
                "simulation_path": "",
            },
            {
                "ca": 2.0,
                "depolarization": 4.0,
                "simulation_path": "/tmp/coupled/uuid/1/simulation_config.json",
            },
        ],
    }


@pytest.fixture
def xarray_config_obj_simple(xarray_config_dict_simple):
    return xr.DataArray.from_dict(xarray_config_dict_simple)


@pytest.fixture
def xarray_config_obj_coupled(xarray_config_dict_coupled):
    return xr.DataArray.from_dict(xarray_config_dict_coupled)


@pytest.fixture
def blueetl_config_dataframe_simple(blueetl_config_dict_simple):
    d = blueetl_config_dict_simple
    return pd.DataFrame.from_records(d["data"])


@pytest.fixture
def blueetl_config_dataframe_coupled(blueetl_config_dict_coupled):
    d = blueetl_config_dict_coupled
    return pd.DataFrame.from_records(d["data"])


@pytest.fixture
def blueetl_config_obj_simple(blueetl_config_dict_simple):
    d = blueetl_config_dict_simple
    return SimulationCampaign(
        data=pd.DataFrame.from_records(d["data"]),
        name=d["name"],
        attrs=d["attrs"],
    )


@pytest.fixture
def blueetl_config_obj_coupled(blueetl_config_dict_coupled):
    d = blueetl_config_dict_coupled
    return SimulationCampaign(
        data=pd.DataFrame.from_records(d["data"]),
        name=d["name"],
        attrs=d["attrs"],
    )
