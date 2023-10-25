"""Simulation Campaign configuration."""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from blueetl.constants import SIMULATION_PATH
from blueetl.types import StrOrPath
from blueetl.utils import dump_yaml, load_yaml, resolve_path

L = logging.getLogger(__name__)


class SimulationCampaignConfig:
    """Result of a Simulation Campaign."""

    def __init__(
        self,
        data: pd.DataFrame,
        name: Optional[str] = None,
        attrs: Optional[dict] = None,
        conditions: Optional[list[str]] = None,
    ) -> None:
        """Init the configuration.

        Args:
            data: DataFrame of simulation paths, having as columns: simulation_path,
                and the parameters used for each simulation (for example: seed, ca...).
            name: optional name of the Simulation Campaign.
                If not specified, use 'data.name'.
            attrs: optional dict of custom attributes.
                If not specified, use 'data.attrs'.
            conditions: optional list of parameters used the simulations.
                If not specified, all the columns of data except simulation_path are used.
        """
        data = data.copy()
        if name is not None:
            data.name = name
        if attrs is not None:
            data.attrs = attrs
        self._conditions = conditions or [c for c in data.columns if c != SIMULATION_PATH]
        self._data = data
        self._resolve_paths()
        self._validate()

    def _resolve_paths(self) -> None:
        def _to_absolute(path: str) -> str:
            """Convert to absolute path, but don't check if it exists."""
            path = resolve_path(path_prefix, path)
            if path.is_dir():
                # use circuit_config to determine the expected config path of the simulations,
                # because the paths defined in the simulation campaign don't include the filename
                if not self.attrs or "circuit_config" not in self.attrs:
                    raise RuntimeError("circuit_config is missing in the simulation campaign")
                if self.attrs["circuit_config"].endswith(".json"):
                    path = path / "simulation_config.json"
                else:
                    path = path / "BlueConfig"
            return str(path)

        path_prefix = self.attrs.get("path_prefix", "")
        self.data[SIMULATION_PATH] = self.data[SIMULATION_PATH].apply(_to_absolute)

    def _validate(self) -> None:
        if SIMULATION_PATH not in self.data.columns:
            raise ValueError(f"Missing required column: {SIMULATION_PATH}")
        if diff := set(self.conditions) - set(self.data.columns):
            raise ValueError(f"Invalid extra conditions: {sorted(diff)}")

    def __eq__(self, other: object) -> bool:
        """Return True if the objects are considered equal, False otherwise."""
        if not isinstance(other, SimulationCampaignConfig):
            # https://docs.python.org/3/library/constants.html#NotImplemented
            return NotImplemented
        return (
            self.name == other.name
            and self.attrs == other.attrs
            and self.conditions == other.conditions
            and self.data.equals(other.data)
        )

    @property
    def name(self) -> str:
        """Return the name of the simulations campaign."""
        return self.data.name or ""

    @property
    def attrs(self) -> dict:
        """Return the attributes dict associated with the simulations campaign."""
        return self.data.attrs

    @property
    def conditions(self) -> list[str]:
        """Return the list of conditions associated with the simulations campaign.

        The conditions are the names of columns considered as parameters of the simulations.
        """
        return self._conditions

    @property
    def data(self) -> pd.DataFrame:
        """Return the wrapped dataframe."""
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the wrapped dataframe.

        It can be used to avoid any unintentional modification of the internal dataframe.
        """
        return self.data.copy()

    @classmethod
    def load(cls, path: StrOrPath) -> "SimulationCampaignConfig":
        """Load the configuration from file.

        Different file formats are supported:

        - xarray (json or yaml): configuration produced by bbp-workflow.
        - blueetl (json or yaml): configuration produced by this same class.

        Args:
            path: path to the configuration file.

        Returns:
            SimulationCampaignConfig: new instance.
        """
        config = load_yaml(path)
        if config.get("format") == "blueetl":
            L.info("Detected blueetl configuration format")
            return cls.from_dict(config)
        if set(config) == {"name", "attrs", "data", "dims", "coords"}:
            L.info("Detected xarray configuration format")
            da = xr.DataArray.from_dict(config)
            return cls.from_xarray(da)
        raise ValueError("Unable to detect the configuration format")

    def dump(self, path: StrOrPath) -> None:
        """Save the configuration to file."""
        dump_yaml(path, data=self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationCampaignConfig":
        """Load the configuration from dict."""
        data = pd.DataFrame.from_dict(d["data"])
        return cls(data=data, name=d["name"], attrs=d["attrs"], conditions=d["conditions"])

    def to_dict(self) -> dict:
        """Convert the configuration to dict."""
        return {
            "format": "blueetl",
            "version": 1,
            "name": self.name,
            "attrs": self.attrs,
            "conditions": self.conditions,
            "data": self.data.to_dict(orient="records"),
        }

    @classmethod
    def from_xarray(cls, da: xr.DataArray) -> "SimulationCampaignConfig":
        """Load the configuration from xarray.DataArray."""
        data = da.rename(SIMULATION_PATH).to_dataframe().reset_index()
        return cls(data=data, name=str(da.name), attrs=da.attrs)

    def to_xarray(self) -> xr.DataArray:
        """Convert the configuration to xarray.DataArray."""
        s = self.data.set_index(self.conditions)[SIMULATION_PATH]
        s.name = self.name
        # may be added by bbp-workflow when using coupled coordinates
        coupled = "coupled"
        if coupled not in self.conditions:
            # generated with the task GenerateSimulationCampaign
            da = xr.DataArray.from_series(s)
            da.attrs = self.attrs
        else:
            # generated with the task GenerateCoupledCoordsSimulationCampaign
            index, indexer = s.index.sortlevel(coupled)
            if not np.array_equal(indexer, np.arange(len(indexer))):
                raise ValueError("Incorrect coupled indexer")
            index = s.index.droplevel(coupled)
            da = xr.DataArray(
                list(s),
                name=s.name,
                dims=coupled,
                coords={coupled: index},
                attrs=self.attrs,
            )
            da = da.reset_index(coupled)
        return da
