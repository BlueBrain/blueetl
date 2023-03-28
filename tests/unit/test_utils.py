import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from pandas.api.types import is_integer_dtype

from blueetl import utils as test_module
from blueetl.constants import DTYPES


@pytest.mark.parametrize(
    "x, expected",
    [
        ([], []),
        ((), ()),
        (None, [None]),
        ("", [""]),
        ("a", ["a"]),
        (["a"], ["a"]),
        (("a",), ("a",)),
    ],
)
def test_ensure_list(x, expected):
    result = test_module.ensure_list(x)
    assert result == expected


@pytest.mark.parametrize("dtypes", [None, DTYPES])
def test_ensure_dtypes(dtypes):
    df = pd.DataFrame([{k: i for i, k in enumerate(DTYPES)}])
    df["other"] = 99
    df.index = pd.MultiIndex.from_frame(df)

    result = test_module.ensure_dtypes(df, desired_dtypes=dtypes)

    expected = [*list(DTYPES.values()), np.int64]
    assert_equal(result.dtypes.to_numpy(), expected)
    # convert any int to int64 in the expected list
    expected = [np.int64 if is_integer_dtype(item) else item for item in expected]
    assert_equal(result.index.dtypes.to_numpy(), expected)


def test_import_by_string():
    name = "json.load"
    result = test_module.import_by_string(name)
    assert result is json.load


def test_resolve_path(tmp_path):
    subdir = Path("subdir")
    my_file = Path("my_file")
    my_symlink = Path("my_symlink")
    non_existent = Path("non_existent")

    tmp_path = tmp_path.resolve()
    subdir_absolute = tmp_path / subdir
    my_file_absolute = tmp_path / subdir / my_file
    my_symlink_absolute = tmp_path / subdir / my_symlink
    non_existent_absolute = tmp_path / subdir / non_existent

    subdir_absolute.mkdir()
    my_file_absolute.touch()
    my_symlink_absolute.symlink_to(my_file)

    result = test_module.resolve_path(tmp_path, subdir, my_file, symlinks=False)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_file, symlinks=True)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_symlink, symlinks=False)
    assert result == my_symlink_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_symlink, symlinks=True)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, non_existent, symlinks=False)
    assert result == non_existent_absolute

    result = test_module.resolve_path(tmp_path, subdir, non_existent, symlinks=True)
    assert result == non_existent_absolute


def test_dump_yaml(tmp_path):
    data = {
        "dict": {"str": "mystr", "int": 123},
        "list_of_int": [1, 2, 3],
        "list_of_str": ["1", "2", "3"],
        "path": Path("/custom/path"),
    }
    expected = """
dict:
  str: mystr
  int: 123
list_of_int:
- 1
- 2
- 3
list_of_str:
- '1'
- '2'
- '3'
path: /custom/path
    """
    filepath = tmp_path / "test.yaml"

    test_module.dump_yaml(filepath, data)
    dumped_content = filepath.read_text(encoding="utf-8")
    assert dumped_content.strip() == expected.strip()


def test_load_yaml(tmp_path):
    data = """
dict:
  str: mystr
  int: 123
list_of_int:
- 1
- 2
- 3
list_of_str:
- '1'
- '2'
- '3'
path: /custom/path
    """
    expected = {
        "dict": {"str": "mystr", "int": 123},
        "list_of_int": [1, 2, 3],
        "list_of_str": ["1", "2", "3"],
        "path": "/custom/path",
    }
    filepath = tmp_path / "test.yaml"
    filepath.write_text(data, encoding="utf-8")

    loaded_data = test_module.load_yaml(filepath)
    assert loaded_data == expected


@pytest.mark.parametrize(
    "d, expected",
    [
        (
            {},
            [],
        ),
        (
            {"key1": []},
            [],
        ),
        (
            {"key1": [100, 200]},
            [
                (("key1", 100, 0),),
                (("key1", 200, 1),),
            ],
        ),
        (
            {"key1": [100, 200], "key2": []},
            [],
        ),
        (
            {"key1": [100, 200], "key2": [300]},
            [
                (("key1", 100, 0), ("key2", 300, 0)),
                (("key1", 200, 1), ("key2", 300, 0)),
            ],
        ),
        (
            {"key1": [100, 200], "key2": [300, 400]},
            [
                (("key1", 100, 0), ("key2", 300, 0)),
                (("key1", 100, 0), ("key2", 400, 1)),
                (("key1", 200, 1), ("key2", 300, 0)),
                (("key1", 200, 1), ("key2", 400, 1)),
            ],
        ),
    ],
)
def test_dict_product(d, expected):
    result = test_module.dict_product(d)
    assert list(result) == expected


def test_extract_items():
    obj = {
        "latency": {"params": {"onset": False}},
        "decay": {"params": {"ratio": [0.25, 0.5, 0.75]}},
        "baseline_PSTH": {"params": {"bin_size": 0.5, "sigma": 0, "offset": -6}},
    }
    expected = [
        ("latency.params.onset", False),
        ("decay.params.ratio", [0.25, 0.5, 0.75]),
        ("baseline_PSTH.params.bin_size", 0.5),
        ("baseline_PSTH.params.sigma", 0),
        ("baseline_PSTH.params.offset", -6),
    ]
    result = list(test_module.extract_items(obj))
    assert result == expected


@pytest.mark.parametrize(
    "iterable, expected",
    [
        ([], True),
        ([1], True),
        ([1, 1], True),
        ([1, 1, 2], False),
        ("", True),
        ("aaaaaa", True),
        ("aaaaaba", False),
        ([[123, 456]], True),
        ([[123, 456], [123, 456]], True),
        ([[123, 456], [123, 456], [123, 999]], False),
    ],
)
def test_all_equal(iterable, expected):
    # with a list
    result = test_module.all_equal(iterable)
    assert result == expected
    # with an iterator
    result = test_module.all_equal(iter(iterable))
    assert result == expected
