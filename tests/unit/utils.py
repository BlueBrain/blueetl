import importlib
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pydantic
import xarray as xr

try:
    import bluepy

    BLUEPY_AVAILABLE = True
except ImportError:
    BLUEPY_AVAILABLE = False

TEST_DATA_PATH = Path(__file__).parent / "data"
TEST_NODE_SETS_FILE = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets.json"
TEST_NODE_SETS_FILE_EXTRA = TEST_DATA_PATH / "circuit" / "sonata" / "node_sets_extra.json"
TEST_CIRCUIT_CONFIG = TEST_DATA_PATH / "circuit" / "sonata" / "circuit_config.json"
TEST_SIMULATION_CONFIG = TEST_DATA_PATH / "simulation" / "sonata" / "simulation_config.json"


class PicklableMock(MagicMock):
    """Mock that can be pickled without errors.

    Neither the internal status nor any attribute set after the initialization is saved or restored!
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    __reduce__ = lambda self: (partial(MagicMock, **self._kwargs), self._args)


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
    """Verify that obj doesn't contain duplicate instances."""
    ids = set()
    for v in iterallvalues(obj):
        if isinstance(v, (dict, list, tuple)):
            assert id(v) not in ids, f"Duplicate {type(v).__name__}: {v}"
            ids.add(id(v))


def assert_isinstance(instance, class_name):
    """Verify that instance is an instance of class_name (given as string)."""
    module_name, _, class_name = class_name.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    assert isinstance(instance, cls)
