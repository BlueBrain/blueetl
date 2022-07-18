import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl.core import utils as test_module


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ({}, {}, True),
        ({}, {"key": 1}, False),
        ({"key": 1}, {}, True),
        ({"key": 1}, {"key": 1}, True),
        ({"key": 1}, {"key": [1, 2]}, True),
        ({"key": 1}, {"key": {"isin": [1, 2]}}, True),
        ({"key": 1}, {"key": 2}, False),
        ({"key": 1}, {"key": [2, 3]}, False),
        ({"key": 1}, {"key": {"isin": [2, 3]}}, False),
        ({"key1": 1, "key2": 2}, {"key1": 1}, True),
        ({"key1": 1}, {"key1": 1, "key2": 2}, False),
        ({"key": {"isin": [1, 2]}}, {"key": 1}, False),
        ({"key": {"ne": 3}}, {"key": {"ne": 3}}, True),
        ({"key": {"ne": 3}}, {"key": {"ne": 4}}, False),
        ({"key": {"gt": 3}}, {"key": {"gt": 2}}, True),
        ({"key": {"gt": 3}}, {"key": {"gt": 3}}, True),
        ({"key": {"gt": 3}}, {"key": {"gt": 4}}, False),
        ({"key": {"ge": 3}}, {"key": {"ge": 2}}, True),
        ({"key": {"ge": 3}}, {"key": {"ge": 3}}, True),
        ({"key": {"ge": 3}}, {"key": {"ge": 4}}, False),
        ({"key": {"lt": 3}}, {"key": {"lt": 2}}, False),
        ({"key": {"lt": 3}}, {"key": {"lt": 3}}, True),
        ({"key": {"lt": 3}}, {"key": {"lt": 4}}, True),
        ({"key": {"le": 3}}, {"key": {"le": 2}}, False),
        ({"key": {"le": 3}}, {"key": {"le": 3}}, True),
        ({"key": {"le": 3}}, {"key": {"le": 4}}, True),
        ({"key": {"le": 3, "ge": 1}}, {"key": {"le": 4}}, True),
        ({"key": {"le": 3, "ge": 1}}, {"key": {"le": 4, "ge": 0}}, True),
        ({"key": 1}, {"key": {"eq": 1}}, True),
        ({"key": 1}, {"key": {"eq": 1, "isin": [1, 2]}}, True),
        ({"key": 1}, {"key": {"eq": 1, "isin": [2, 3]}}, False),
    ],
)
def test_is_subfilter(left, right, expected):
    result = test_module.is_subfilter(left, right)
    assert result == expected


def test_safe_concat_series(series1):
    obj1 = series1.copy() + 1
    obj2 = series1.copy() + 2
    iterable = [obj1, obj2]
    expected = pd.Series(
        [1, 2, 3, 4, 2, 3, 4, 5],
        index=pd.MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")] * 2, names=["i0", "i1"]
        ),
        name="values",
    )

    result = test_module.safe_concat(iterable)
    assert_series_equal(result, expected)


def test_safe_concat_series_having_indexes_with_different_level_order(series1):
    # test that the result is consistent when the levels of the indexes are ordered differently.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = series1.copy() + 1
    obj2 = series1.copy().reorder_levels(["i1", "i0"]) + 2
    iterable = [obj1, obj2]
    expected = pd.Series(
        [1, 2, 3, 4, 2, 3, 4, 5],
        index=pd.MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")] * 2, names=["i0", "i1"]
        ),
        name="values",
    )
    result = test_module.safe_concat(iterable)
    assert_series_equal(result, expected)


def test_safe_concat_series_having_indexes_with_different_level_name(series1):
    # test that an exception is raised when the levels of the indexes are different.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = series1.copy() + 1
    obj2 = series1.copy() + 2
    obj2.index.set_names({"i0": "x0"}, inplace=True)
    iterable = [obj1, obj2]
    with pytest.raises(RuntimeError, match="Levels not found: i0"):
        test_module.safe_concat(iterable)


def test_safe_concat_series_having_indexes_with_different_number_of_levels(series1):
    # test that an exception is raised when the levels of the indexes are different.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = series1.copy() + 1
    obj2 = series1.copy() + 2
    obj2 = obj2.to_frame()
    obj2["i2"] = 999
    obj2 = obj2.set_index("i2", append=True)["values"]
    iterable = [obj1, obj2]
    match = re.escape("Length of order must be same as number of levels (3), got 2")
    with pytest.raises(RuntimeError, match=match):
        test_module.safe_concat(iterable)


def test_safe_concat_dataframes(dataframe1):
    obj1 = dataframe1.copy() + 1
    obj2 = dataframe1.copy() + 2
    iterable = [obj1, obj2]
    expected = pd.DataFrame(
        {"v0": [1, 2, 3, 4, 2, 3, 4, 5], "v1": [5, 6, 7, 8, 6, 7, 8, 9]},
        index=pd.MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")] * 2, names=["i0", "i1"]
        ),
    )
    result = test_module.safe_concat(iterable)
    assert_frame_equal(result, expected)


def test_safe_concat_dataframes_having_indexes_with_different_level_order(dataframe1):
    # test that the result is consistent when the levels of the indexes are ordered differently.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = dataframe1.copy() + 1
    obj2 = dataframe1.copy().reorder_levels(["i1", "i0"]) + 2
    iterable = [obj1, obj2]
    expected = pd.DataFrame(
        {"v0": [1, 2, 3, 4, 2, 3, 4, 5], "v1": [5, 6, 7, 8, 6, 7, 8, 9]},
        index=pd.MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")] * 2, names=["i0", "i1"]
        ),
    )
    result = test_module.safe_concat(iterable)
    assert_frame_equal(result, expected)


def test_safe_concat_dataframes_having_indexes_with_different_level_name(dataframe1):
    # test that an exception is raised when the levels of the indexes are different.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = dataframe1.copy() + 1
    obj2 = dataframe1.copy() + 2
    obj2.index.set_names({"i0": "x0"}, inplace=True)
    iterable = [obj1, obj2]
    with pytest.raises(RuntimeError, match="Levels not found: i0"):
        test_module.safe_concat(iterable)


def test_safe_concat_dataframes_having_indexes_with_different_number_of_levels(dataframe1):
    # test that an exception is raised when the levels of the indexes are different.
    # plain pd.concat would blindly concatenate the indexes, ignoring the names of the levels.
    obj1 = dataframe1.copy() + 1
    obj2 = dataframe1.copy() + 2
    obj2["i2"] = 999
    obj2 = obj2.set_index("i2", append=True)
    iterable = [obj1, obj2]
    match = re.escape("Length of order must be same as number of levels (3), got 2")
    with pytest.raises(RuntimeError, match=match):
        test_module.safe_concat(iterable)
