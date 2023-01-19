from pathlib import Path

import pandas as pd
import pydantic
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


def iterallvalues(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iterallvalues(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iterallvalues(v)
    elif isinstance(obj, pydantic.BaseModel):
        for k, v in obj:
            yield from iterallvalues(v)
    yield obj


def assert_not_duplicates(obj):
    # verify that obj doesn't contain duplicate instances
    ids = set()
    for v in iterallvalues(obj):
        if isinstance(v, (dict, list, tuple)):
            assert id(v) not in ids, f"Duplicate {type(v).__name__}: {v}"
            ids.add(id(v))
