"""Simulation Campaign configuration."""
import pandas as pd
import xarray as xr
import yaml

# used for conversion to pandas.Series or xarray.DataArray
_DATA_KEY = "path"


class SimulationCampaignResult:
    """Result of a Simulation Campaign."""

    def __init__(self, name, attrs, data):
        """Init the configuration."""
        assert all(_DATA_KEY in d for d in data)  # TODO: proper validation
        self.name = name
        self.attrs = attrs
        self.data = data

    @classmethod
    def load(cls, path):
        """Load the configuration from file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

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
            data=s.rename(_DATA_KEY).reset_index().to_dict(orient="records"),
        )

    @classmethod
    def from_xarray(cls, da):
        """Load the configuration from xarray.DataArray."""
        return cls(
            name=da.name,
            attrs=da.attrs,
            data=da.to_series().rename(_DATA_KEY).reset_index().to_dict(orient="records"),
        )

    def dump(self, path):
        """Save the configuration to file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), stream=f)

    def to_dict(self):
        """Convert the configuration to dict."""
        return {"name": self.name, "attrs": self.attrs, "data": self.data}

    def to_pandas(self):
        """Convert the configuration to pandas.Series."""
        df = pd.DataFrame.from_dict(self.data)
        s = df.set_index([col for col in df.columns if col != _DATA_KEY])[_DATA_KEY]
        s.attrs = self.attrs
        s.name = self.name
        return s

    def to_xarray(self):
        """Convert the configuration to xarray.DataArray."""
        s = self.to_pandas()
        da = xr.DataArray.from_series(s)
        da.attrs = s.attrs
        return da
