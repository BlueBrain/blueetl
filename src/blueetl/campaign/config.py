"""Simulation Campaign configuration."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from blueetl.constants import SIMULATION_PATH
from blueetl.types import StrOrPath
from blueetl.utils import dump_yaml, load_yaml, resolve_path

L = logging.getLogger(__name__)

SONATA_CONFIG = "simulation_config.json"
BLUEPY_CONFIG = "BlueConfig"


def _resolve_simulation_path(path_prefix: str, path: str, filename: str) -> str:
    """Convert to full absolute path, without checking if it exists."""
    if not path or path.startswith("https://"):
        # do not convert excluded simulations or nexus urls
        return path
    path_obj = resolve_path(path_prefix, path)
    if path_obj.name != filename:
        # the paths in the xarray simulation campaign config don't include the filename
        path_obj = path_obj / filename
    return str(path_obj)


def _reduce_simulation_path(path_prefix: str, path: str, filename: str) -> str:
    """Reduce to shorter path, without path_prefix and filename."""
    if not path or path.startswith("https://"):
        # do not convert excluded simulations or nexus urls
        return path
    path_obj = Path(path)
    if path_obj.is_relative_to(path_prefix):
        path_obj = path_obj.relative_to(path_prefix)
    if path_obj.name == filename:
        # the paths in the xarray simulation campaign config don't include the filename
        path_obj = path_obj.parent
    return str(path_obj)


@dataclass
class SimulationRow:
    """Simulation row in the simulation campaign."""

    index: int
    path: str
    conditions: dict[str, Any]

    @property
    def empty(self) -> bool:
        """Return True if the simulation has been excluded when running the campaign."""
        return not self.path


class SimulationCampaign:
    """Simulation campaign configuration."""

    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        attrs: dict,
        config_dir: Optional[Path] = None,
    ) -> None:
        """Init the configuration.

        Args:
            data: DataFrame of simulation paths, having as columns: simulation_path,
                and the parameters used for each simulation (for example: seed, ca...).
            name: name of the Simulation Campaign.
            attrs: dict of custom attributes.
            config_dir: if specified, it's used to resolve relative paths in attrs.
        """
        self._name = name
        self._attrs = attrs.copy()
        if config_dir:
            if "path_prefix" in self._attrs:
                self._attrs["path_prefix"] = str(
                    resolve_path(config_dir, self._attrs["path_prefix"])
                )
            if "circuit_config" in self._attrs:
                self._attrs["circuit_config"] = str(
                    resolve_path(config_dir, self._attrs["circuit_config"])
                )
        self._data = data.copy()
        self._data[SIMULATION_PATH] = self._resolve_paths(self._data[SIMULATION_PATH])

    def _get_simulation_filename(self) -> str:
        """Return the filename of each simulation in the campaign."""
        return SONATA_CONFIG if self.is_sonata() else BLUEPY_CONFIG

    def _resolve_paths(self, simulation_paths: pd.Series) -> pd.Series:
        """Resolve the simulation paths."""
        return simulation_paths.apply(
            lambda path: _resolve_simulation_path(
                path_prefix=self.attrs.get("path_prefix", ""),
                path=path,
                filename=self._get_simulation_filename(),
            )
        )

    def _reduce_paths(self, simulation_paths: pd.Series) -> pd.Series:
        """Reduce the simulation paths."""
        return simulation_paths.apply(
            lambda path: _reduce_simulation_path(
                path_prefix=self.attrs.get("path_prefix", ""),
                path=path,
                filename=self._get_simulation_filename(),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Return True if the objects are considered equal, False otherwise."""
        if not isinstance(other, SimulationCampaign):
            # https://docs.python.org/3/library/constants.html#NotImplemented
            return NotImplemented
        return all(
            (
                self.name == other.name,
                self.attrs == other.attrs,
                self._data.equals(other._data),
            )
        )

    @property
    def name(self) -> str:
        """Return the name of the simulations campaign."""
        return self._name

    @property
    def attrs(self) -> dict:
        """Return the attributes dict associated with the simulations campaign."""
        return self._attrs

    @property
    def condition_names(self) -> list[str]:
        """Return the names of the parameters used to run the simulations."""
        return [c for c in self._data.columns if c != SIMULATION_PATH]

    @property
    def conditions(self) -> pd.DataFrame:
        """Return the DataFrame of the parameters used to run the simulations."""
        return self._data[self.condition_names]

    def is_coupled(self):
        """Return True if the coords are coupled, False otherwise."""
        return bool(self.attrs.get("__coupled__"))

    def is_sonata(self):
        """Return True if the simulations are in SONATA format, False otherwise."""
        circuit_config = self.attrs.get("circuit_config", "")
        if not circuit_config:
            raise RuntimeError("circuit_config is missing in the simulation campaign")
        return circuit_config.endswith(".json")

    @classmethod
    def load(cls, path: StrOrPath) -> "SimulationCampaign":
        """Load the configuration from file.

        Different file formats are supported:

        - xarray (json or yaml): configuration produced by bbp-workflow.
        - blueetl (json or yaml): configuration produced by this same class.

        Args:
            path: path to the configuration file.

        Returns:
            SimulationCampaign: new instance.
        """
        config = load_yaml(path)
        if config.get("format") == "blueetl":
            L.debug("Detected blueetl configuration format")
            return cls.from_dict(config, config_dir=Path(path).parent)
        if set(config) == {"name", "attrs", "data", "dims", "coords"}:
            L.debug("Detected xarray configuration format")
            return cls.from_xarray_dict(config, config_dir=Path(path).parent)
        raise ValueError("Unable to detect the configuration format")

    def dump(self, path: StrOrPath) -> None:
        """Save the configuration to file."""
        dump_yaml(path, data=self.to_dict())

    @classmethod
    def from_dict(cls, d: dict, config_dir: Optional[Path] = None) -> "SimulationCampaign":
        """Load the configuration from dict."""
        data = pd.DataFrame.from_records(d["data"])
        return cls(data=data, name=d["name"], attrs=d["attrs"], config_dir=config_dir)

    def to_dict(self) -> dict:
        """Convert the configuration to dict."""
        return {
            "format": "blueetl",
            "version": 1,
            "name": self.name,
            "attrs": self.attrs,
            "data": self._data.to_dict(orient="records"),
        }

    @classmethod
    def from_xarray(
        cls, da: xr.DataArray, config_dir: Optional[Path] = None
    ) -> "SimulationCampaign":
        """Load the configuration from xarray.DataArray."""
        df = da.to_dataframe(SIMULATION_PATH)
        # If the campaign is not coupled, the result after calling `to_dataframe()` is like:
        #
        #                    simulation_path
        # ca  depolarization
        # 1.0 3.0                     uuid/0
        #     4.0
        # 2.0 3.0                     uuid/2
        #     4.0                     uuid/3
        #
        # Otherwise, the result is like:
        #
        #           ca  depolarization simulation_path
        # coupled
        # 0        1.0             3.0          uuid/0
        # 1        2.0             4.0
        attrs = da.attrs.copy()
        if len(df.columns) == 1:
            coupled = None
        else:
            coupled = df.index.name
            assert coupled
            assert isinstance(df.index, pd.RangeIndex)
            # save the __coupled__ attribute to be able to convert back to xarray
            attrs["__coupled__"] = coupled
        data = df.reset_index(drop=bool(coupled))
        return cls(data=data, name=str(da.name), attrs=attrs, config_dir=config_dir)

    def to_xarray(self) -> xr.DataArray:
        """Convert the configuration to xarray.DataArray."""
        s = self._data.set_index(self.condition_names)[SIMULATION_PATH]
        s = self._reduce_paths(s)
        s.name = self.name
        attrs = self.attrs.copy()
        coupled = attrs.pop("__coupled__", None)
        if not coupled:
            # generated by GenerateSimulationCampaign
            da = xr.DataArray.from_series(s)
            da.attrs = attrs
        else:
            # generated by GenerateCoupledCoordsSimulationCampaign
            da = xr.DataArray(
                list(s),
                name=s.name,
                dims=coupled,
                coords={coupled: s.index},
                attrs=attrs,
            ).reset_index(coupled)
        return da

    @classmethod
    def from_xarray_dict(cls, d: dict, config_dir: Optional[Path] = None) -> "SimulationCampaign":
        """Load the configuration from a dict representing xarray.DataArray."""
        da = xr.DataArray.from_dict(d)
        return cls.from_xarray(da, config_dir=config_dir)

    def to_xarray_dict(self) -> dict:
        """Return the configuration as a dict representing xarray.DataArray."""
        return self.to_xarray().to_dict()

    def __len__(self) -> int:
        """Return the number of simulations."""
        return len(self._data)

    def __iter__(self) -> Iterator[SimulationRow]:
        """Iterate over the simulation rows."""
        for i, (_, sim_dict) in enumerate(self._data.etl.iterdict()):
            path = sim_dict.pop(SIMULATION_PATH)
            yield SimulationRow(index=i, path=path, conditions=sim_dict)

    def __getitem__(self, index: int) -> SimulationRow:
        """Return a specific simulation row."""
        sim_dict = self._data.loc[index].to_dict()
        path = sim_dict.pop(SIMULATION_PATH)
        return SimulationRow(index=index, path=path, conditions=sim_dict)

    def get(self, *args, **kwargs) -> pd.DataFrame:
        """Return a DataFrame with the selected simulations.

        See ``etl.q`` for the filter syntax.
        """
        return self._data.copy().etl.q(*args, **kwargs)

    def ids(self, *args, **kwargs) -> np.ndarray:
        """Return a numpy array with the ids of the selected simulations.

        See ``etl.q`` for the filter syntax.
        """
        return self.get(*args, **kwargs).index.to_numpy()
