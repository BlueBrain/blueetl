import pandas as pd
import pytest
import xarray as xr

from blueetl.config.simulations import SimulationsConfig


@pytest.fixture
def simulations_config():
    """Return a SimulationsConfig instance."""
    return SimulationsConfig(
        data=pd.DataFrame(
            [
                {"ca": 1.1, "seed": 1, "simulation_path": "/path/to/1/BlueConfig"},
                {"ca": 1.2, "seed": 1, "simulation_path": "/path/to/2/BlueConfig"},
            ]
        ),
        name="dummy_name",
        attrs={"k1": "v1", "k2": "v2"},
    )


@pytest.fixture
def simulations_config_dict():
    """Return a dict consistent with simulations_config."""
    return {
        "format": "blueetl",
        "version": 1,
        "name": "dummy_name",
        "attrs": {"k1": "v1", "k2": "v2"},
        "conditions": ["ca", "seed"],
        "data": [
            {"ca": 1.1, "seed": 1, "simulation_path": "/path/to/1/BlueConfig"},
            {"ca": 1.2, "seed": 1, "simulation_path": "/path/to/2/BlueConfig"},
        ],
    }


@pytest.fixture
def simulations_config_dataframe():
    """Return a DataFrame consistent with simulations_config."""
    df = pd.DataFrame(
        [
            {"ca": 1.1, "seed": 1, "simulation_path": "/path/to/1/BlueConfig"},
            {"ca": 1.2, "seed": 1, "simulation_path": "/path/to/2/BlueConfig"},
        ]
    )
    df.attrs = {"k1": "v1", "k2": "v2"}
    return df


@pytest.fixture
def simulations_config_xarray():
    """Return a DataArray consistent with simulations_config."""
    return xr.DataArray(
        data=[
            ["/path/to/1/BlueConfig"],
            ["/path/to/2/BlueConfig"],
        ],
        coords={"ca": [1.1, 1.2], "seed": [1]},
        attrs={"k1": "v1", "k2": "v2"},
        name="dummy_name",
    )
