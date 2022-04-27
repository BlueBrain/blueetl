"""Simulation Campaign configuration."""
import logging
from os import PathLike
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from blueetl.constants import SIMULATION_PATH
from blueetl.utils import dump_yaml, load_yaml, resolve_path

L = logging.getLogger(__name__)


class SimulationsConfig:
    """Result of a Simulation Campaign."""

    def __init__(
        self,
        data: pd.DataFrame,
        name: Optional[str] = None,
        attrs: Optional[Dict] = None,
        conditions: Optional[List[str]] = None,
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

    def _resolve_paths(self):
        def _to_absolute(path):
            full_path = resolve_path(path_prefix, path)
            if full_path.is_dir():
                L.warning("%s is a directory, using BlueConfig", path)
                full_path = full_path / "BlueConfig"
            if not full_path.exists():
                L.warning("%s doesn't exist, proceeding anyway", path)
            return str(full_path)

        path_prefix = self.attrs.get("path_prefix", "")
        self.data[SIMULATION_PATH] = self.data[SIMULATION_PATH].apply(_to_absolute)

    def _validate(self):
        if SIMULATION_PATH not in self.data.columns:
            raise ValueError(f"Missing required column: {SIMULATION_PATH}")
        diff = set(self.conditions) - set(self.data.columns)
        if diff:
            raise Exception(f"Invalid extra conditions: {diff}")

    @property
    def name(self) -> str:
        return self.data.name or ""

    @property
    def attrs(self) -> Dict:
        return self.data.attrs

    @property
    def conditions(self) -> List[str]:
        return self._conditions

    @property
    def data(self):
        return self._data

    def to_pandas(self):
        return self.data.copy()

    @classmethod
    def load(cls, path: Union[str, PathLike]) -> "SimulationsConfig":
        """Load the configuration from file.

        Args:
            path: path to the configuration file.
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

    def dump(self, path):
        """Save the configuration to file."""
        dump_yaml(path, data=self.to_dict())

    @classmethod
    def from_dict(cls, d):
        """Load the configuration from dict."""
        data = pd.DataFrame.from_dict(d["data"])
        return cls(data=data, name=d["name"], attrs=d["attrs"], conditions=d["conditions"])

    def to_dict(self):
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
    def from_xarray(cls, da):
        """Load the configuration from xarray.DataArray."""
        data = da.rename(SIMULATION_PATH).to_dataframe().reset_index()
        return cls(data=data, name=da.name, attrs=da.attrs)

    def to_xarray(self):
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
