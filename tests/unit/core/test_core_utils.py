import re

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl.core import utils as test_module


def test_andor_mask():
    query_list = [
        {"col1": 10},  # select row 1
        {"col1": 11, "col2": 111},  # select row 2
        {"col1": 99, "col2": 100},  # select none
    ]
    df = pd.DataFrame(
        {
            "col1": [0, 10, 11, 20, 21],
            "col2": [100, 110, 111, 111, 111],
        }
    )

    def _filter_func(query):
        return [df[key] == val for key, val in query.items()]

    mask = test_module._and_or_mask(query_list, _filter_func)

    assert_array_equal(mask, [0, 1, 1, 0, 0])


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


@pytest.mark.parametrize(
    "iterables, expected",
    [
        ([[], []], 0),
        ([[], [1, 2, 3]], 0),
        ([[1, 2], [1, 2, 3]], 2),
        ([[1, 2, 3], [1, 2]], 2),
        ([[1, 2, 3], [1, 2, 3]], 3),
        ([[2, 3, 1], [1, 2, 3]], 0),
        ([[1, 4, 3], [1, 2, 3]], 1),
        ([(i for i in "abc"), ["a", "b", "c"]], 3),
        ([(i for i in "acc"), ["a", "b", "c"]], 1),
    ],
)
def test_longest_match_count(iterables, expected):
    result = test_module.longest_match_count(*iterables)
    assert result == expected


def test_cached_dataframe():
    df = pd.DataFrame(
        {
            "simulation_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "circuit_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "window": ["w0", "w0", "w1", "w1", "w0", "w0", "w1", "w1"],
            "trial": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    query = {"simulation_id": 1, "circuit_id": 0, "window": "w0"}
    expected_df = pd.DataFrame(
        {
            "simulation_id": [1, 1],
            "circuit_id": [0, 0],
            "window": ["w0", "w0"],
            "trial": [0, 1],
        },
        index=[4, 5],
    )
    expected_stack = [
        test_module.CachedItem(
            df=pd.DataFrame(
                {
                    "simulation_id": [1, 1, 1, 1],
                    "circuit_id": [0, 0, 0, 0],
                    "window": ["w0", "w0", "w1", "w1"],
                    "trial": [0, 1, 0, 1],
                },
                index=[4, 5, 6, 7],
            ),
            key="simulation_id",
            value=1,
        ),
        test_module.CachedItem(
            df=pd.DataFrame(
                {
                    "simulation_id": [1, 1, 1, 1],
                    "circuit_id": [0, 0, 0, 0],
                    "window": ["w0", "w0", "w1", "w1"],
                    "trial": [0, 1, 0, 1],
                },
                index=[4, 5, 6, 7],
            ),
            key="circuit_id",
            value=0,
        ),
        test_module.CachedItem(
            df=pd.DataFrame(
                {
                    "simulation_id": [1, 1],
                    "circuit_id": [0, 0],
                    "window": ["w0", "w0"],
                    "trial": [0, 1],
                },
                index=[4, 5],
            ),
            key="window",
            value="w0",
        ),
    ]
    cache = test_module.CachedDataFrame(df=df)

    result = cache.query(query=query)
    assert_frame_equal(result, expected_df)
    assert cache._stack == expected_stack
    assert cache._matched == 0

    query["trial"] = 1

    result = cache.query(query=query)
    assert_frame_equal(result, expected_df.iloc[[1]])
    assert cache._stack == expected_stack + [
        test_module.CachedItem(
            df=pd.DataFrame(
                {
                    "simulation_id": [1],
                    "circuit_id": [0],
                    "window": ["w0"],
                    "trial": [1],
                },
                index=[5],
            ),
            key="trial",
            value=1,
        )
    ]
    assert cache._matched == len(expected_stack)

    query["trial"] = 0

    result = cache.query(query=query)
    assert_frame_equal(result, expected_df.iloc[[0]])
    assert cache._stack == expected_stack + [
        test_module.CachedItem(
            df=pd.DataFrame(
                {
                    "simulation_id": [1],
                    "circuit_id": [0],
                    "window": ["w0"],
                    "trial": [0],
                },
                index=[4],
            ),
            key="trial",
            value=0,
        )
    ]
    assert cache._matched == len(expected_stack)

    del query["trial"]

    result = cache.query(query=query)
    assert_frame_equal(result, expected_df)
    assert cache._stack == expected_stack
    assert cache._matched == len(expected_stack)
