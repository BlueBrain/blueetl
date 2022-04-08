"""Simulation Campaign configuration."""
import logging
from typing import Any, Dict, List

import pandas as pd
import xarray as xr

from blueetl.constants import SIMULATION_PATH
from blueetl.utils import dump_yaml, load_yaml, resolve_path

L = logging.getLogger(__name__)


class SimulationsConfig:
    """Result of a Simulation Campaign."""

    def __init__(self, name: str, attrs: Dict, data: List[Dict[str, Any]]):
        """Init the configuration.

        Args:
            name: name of the Simulation Campaign
            attrs: custom attributes
            data: list of dicts, one for each generated simulation. Each dict should contain:
                - path: path to the simulation configuration.
                - parameters identifying the simulation (for example: seed, ca...).
                  The value of each coordinate needs to be the single value used in this simulation.
        """
        self.name = name or ""
        self.attrs = attrs or {}
        self.data = data or []
        self._resolve_paths()

    def _resolve_paths(self):
        path_prefix = self.attrs.get("path_prefix", "")
        for d in self.data:
            path = resolve_path(path_prefix, d[SIMULATION_PATH])
            if path.is_dir():
                L.warning(
                    "%r is a directory, assuming BlueConfig as the config file", d[SIMULATION_PATH]
                )
                path = path / "BlueConfig"
            d[SIMULATION_PATH] = str(path)

    @classmethod
    def load(cls, path):
        """Load the configuration from file."""
        config = load_yaml(path)
        keys = set(config)
        if keys == {"name", "attrs", "data", "dims", "coords"}:
            L.info("Detected xarray config")
            da = xr.DataArray.from_dict(config)
            return cls.from_xarray(da)
        if keys == {"name", "attrs", "data"}:
            L.info("Detected internal config")
            return cls.from_dict(config)
        raise ValueError("Unable to detect the configuration type")

    @classmethod
    def from_dict(cls, d):
        """Load the configuration from dict."""
        return cls(**d)

    @classmethod
    def from_pandas(cls, s):
        """Load the configuration from pandas.Series."""
        return cls(
            name=s.name,
            attrs=s.attrs,
            data=s.rename(SIMULATION_PATH).reset_index().to_dict(orient="records"),
        )

    @classmethod
    def from_xarray(cls, da):
        """Load the configuration from xarray.DataArray."""
        return cls(
            name=da.name,
            attrs=da.attrs,
            data=da.to_series().rename(SIMULATION_PATH).reset_index().to_dict(orient="records"),
        )

    def dump(self, path):
        """Save the configuration to file."""
        dump_yaml(path, data=self.to_dict())

    def to_dict(self):
        """Convert the configuration to dict."""
        return {"name": self.name, "attrs": self.attrs, "data": self.data}

    def to_pandas(self):
        """Convert the configuration to pandas.Series."""
        df = pd.DataFrame.from_dict(self.data)
        s = df.set_index([col for col in df.columns if col != SIMULATION_PATH])[SIMULATION_PATH]
        s.attrs = self.attrs
        s.name = self.name
        return s

    def to_xarray(self):
        """Convert the configuration to xarray.DataArray."""
        s = self.to_pandas()
        da = xr.DataArray.from_series(s)
        da.attrs = s.attrs
        return da
