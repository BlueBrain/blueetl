from pathlib import Path

import pandas as pd
import xarray as xr

TEST_DATA_PATH = Path(__file__).parent.parent / "data"


def assert_frame_equal(actual, expected):
    pd.testing.assert_frame_equal(actual, expected)
    # check explicitly attrs because not considered by pd.testing.assert_frame_equal
    assert actual.attrs == expected.attrs


def assert_xr_equal(actual, expected):
    xr.testing.assert_equal(actual, expected)
    # check explicitly name and attrs because not considered by xr.testing.assert_equal
    assert actual.name == expected.name
    assert actual.attrs == expected.attrs
