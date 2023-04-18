from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from pandas.testing import assert_index_equal, assert_series_equal

from blueetl.core.etl import ETLSeriesAccessor


def test_etl_instance(series1):
    obj = series1
    result = obj.etl
    assert isinstance(result, ETLSeriesAccessor)


def test_conditions(series1):
    obj = series1
    result = obj.etl.conditions()
    assert_array_equal(result, ["i0", "i1"])


@pytest.mark.parametrize(
    "conditions, expected",
    [
        pytest.param("i0", ["i1"], id="single"),
        pytest.param(["i0"], ["i1"], id="list"),
        pytest.param([], ["i0", "i1"], id="empty"),
        pytest.param(["i0", "i1"], [], id="all"),
        pytest.param(["iX"], ["i0", "i1"], id="nonexistent"),
    ],
)
def test_complementary_conditions(series1, conditions, expected):
    obj = series1
    result = obj.etl.complementary_conditions(conditions)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "conditions, expected",
    [
        pytest.param(None, [pd.Index(["a", "b"], name="i0"), pd.Index(["c", "d"], name="i1")]),
        pytest.param(["i0"], [pd.Index(["a", "b"], name="i0")], id="one"),
    ],
)
def test_labels(series1, conditions, expected):
    obj = series1
    result = obj.etl.labels(conditions)
    assert len(result) == len(expected)
    for actual_index, expected_index in zip(result, expected):
        assert_index_equal(actual_index, expected_index)


def test_labels_of(series1):
    obj = series1
    result = obj.etl.labels_of("i0")
    assert_array_equal(result, ["a", "b"])


def test_labels_of_nonexistent_condition(series1):
    obj = series1
    with pytest.raises(KeyError, match="Level iX not found"):
        obj.etl.labels_of("iX")


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param("i0", id="str"),
        pytest.param(["i0"], id="list"),
    ],
)
def test_remove_conditions(series1, conditions):
    obj = series1
    result = obj.etl.remove_conditions(conditions)
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param("i1", id="str"),
        pytest.param(["i1"], id="list"),
    ],
)
def test_keep_conditions(series1, conditions):
    obj = series1
    result = obj.etl.keep_conditions(conditions)
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


@pytest.mark.parametrize(
    "conditions, values",
    [
        pytest.param("i2", "e", id="str"),
        pytest.param(["i2"], ["e"], id="list"),
    ],
)
def test_add_conditions(series1, conditions, values):
    obj = series1
    result = obj.etl.add_conditions(conditions, values)
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.MultiIndex.from_tuples(
                [("e", "a", "c"), ("e", "a", "d"), ("e", "b", "c"), ("e", "b", "d")],
                names=["i2", "i0", "i1"],
            ),
            name="values",
        ),
    )


@pytest.mark.parametrize(
    "conditions, values, dtypes",
    [
        pytest.param("i2", 123, float, id="str"),
        pytest.param(["i2"], [123], [float], id="list"),
    ],
)
def test_add_conditions_with_dtypes(series1, conditions, values, dtypes):
    obj = series1
    result = obj.etl.add_conditions(conditions, values, dtypes=dtypes)
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.MultiIndex.from_tuples(
                [(123.0, "a", "c"), (123.0, "a", "d"), (123.0, "b", "c"), (123.0, "b", "d")],
                names=["i2", "i0", "i1"],
            ),
            name="values",
        ),
    )
    assert_equal(result.index.dtypes.to_numpy(), [float, object, object])


@pytest.mark.parametrize(
    "inner, drop, expected_conditions",
    [
        (False, False, ["i2", "i0", "i1"]),
        (True, False, ["i0", "i1", "i2"]),
        (False, True, ["i2"]),
        (True, True, ["i2"]),
    ],
)
def test_add_conditions_inner_drop(series1, inner, drop, expected_conditions):
    obj = series1
    conditions = ["i2"]
    values = ["e"]
    result = obj.etl.add_conditions(conditions, values, inner=inner, drop=drop)
    assert result.index.names == expected_conditions


def test_select_with_drop_level(series1):
    obj = series1
    result = obj.etl.select(i1="c", drop_level=True)
    assert_series_equal(
        result,
        pd.Series(
            [0, 2],
            index=pd.Index(["a", "b"], name="i0"),
            name="values",
        ),
    )


def test_select_without_drop_level(series1):
    obj = series1
    result = obj.etl.select(i1="c", drop_level=False)
    assert_series_equal(
        result,
        pd.Series(
            [0, 2],
            index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
            name="values",
        ),
    )


def test_select_without_args(series1):
    obj = series1
    result = obj.etl.select()
    assert_series_equal(result, obj)


def test_unpool_from_series_of_scalars(series1):
    def func(x):
        return pd.Series([x, x + 20], index=pd.Index(["x0", "x2"], name="func"))

    obj = series1
    result = obj.etl.unpool(func)
    assert_series_equal(
        result,
        pd.Series(
            [0, 20, 1, 21, 2, 22, 3, 23],
            index=pd.MultiIndex.from_tuples(
                [
                    ("a", "c", "x0"),
                    ("a", "c", "x2"),
                    ("a", "d", "x0"),
                    ("a", "d", "x2"),
                    ("b", "c", "x0"),
                    ("b", "c", "x2"),
                    ("b", "d", "x0"),
                    ("b", "d", "x2"),
                ],
                names=["i0", "i1", "func"],
            ),
            name=None,
        ),
    )


def test_unpool_from_series_of_lists(series1):
    def func(x):
        return pd.Series([x[0], x[2]], index=pd.Index(["x0", "x2"], name="func"))

    obj = series1
    # transform each element to a list of integers
    obj = obj.transform(lambda x: [x, x + 10, x + 20])
    result = obj.etl.unpool(func)
    assert_series_equal(
        result,
        pd.Series(
            [0, 20, 1, 21, 2, 22, 3, 23],
            index=pd.MultiIndex.from_tuples(
                [
                    ("a", "c", "x0"),
                    ("a", "c", "x2"),
                    ("a", "d", "x0"),
                    ("a", "d", "x2"),
                    ("b", "c", "x0"),
                    ("b", "c", "x2"),
                    ("b", "d", "x0"),
                    ("b", "d", "x2"),
                ],
                names=["i0", "i1", "func"],
            ),
            name=None,
        ),
    )


def test_pool_with_func_returning_scalar(series1):
    def func(x):
        return sum(x)

    obj = series1
    result = obj.etl.pool(["i1"], func)
    assert_series_equal(
        result,
        pd.Series(
            [1, 5],
            index=pd.Index(["a", "b"], name="i0"),
            name="values",
        ),
    )


def test_pool_with_func_returning_series(series1):
    def func(x):
        return pd.Series(
            [sum(x[:, "d"]), sum(x[:, "c"])], index=pd.Index(["x0", "x1"], name="func")
        )

    obj = series1
    result = obj.etl.pool(["i1"], func)
    assert_series_equal(
        result,
        pd.Series(
            [1, 0, 3, 2],
            index=pd.MultiIndex.from_tuples(
                [
                    ("a", "x0"),
                    ("a", "x1"),
                    ("b", "x0"),
                    ("b", "x1"),
                ],
                names=["i0", "func"],
            ),
            name="values",
        ),
    )


@pytest.mark.skip(reason="experimental")
def test_remodel(series1):
    def func(x):
        for index, value in x.etl.iter():
            yield value + 1000, {**index._asdict(), "i2": "e"}
            yield value + 2000, {**index._asdict(), "i2": "f"}

    obj = series1
    result = obj.etl.remodel(func)
    assert_series_equal(
        result,
        pd.Series(
            [1000, 2000, 1001, 2001, 1002, 2002, 1003, 2003],
            index=pd.MultiIndex.from_tuples(
                [
                    ("a", "c", "e"),
                    ("a", "c", "f"),
                    ("a", "d", "e"),
                    ("a", "d", "f"),
                    ("b", "c", "e"),
                    ("b", "c", "f"),
                    ("b", "d", "e"),
                    ("b", "d", "f"),
                ],
                names=["i0", "i1", "i2"],
            ),
        ),
    )


@pytest.mark.parametrize("as_type", ["dict", "list", "kwargs"])
@pytest.mark.parametrize(
    "query, expected",
    [
        (
            {"i0": "a"},
            pd.Series(
                [0, 1],
                index=pd.MultiIndex.from_tuples([("a", "c"), ("a", "d")], names=["i0", "i1"]),
                name="values",
            ),
        ),
        (
            {"i0": "a", "i1": "d"},
            pd.Series(
                [1],
                index=pd.MultiIndex.from_tuples([("a", "d")], names=["i0", "i1"]),
                name="values",
            ),
        ),
        (
            {"i0": "a", "i1": "x"},
            pd.Series(
                [],
                index=pd.MultiIndex.from_tuples([], names=["i0", "i1"]),
                name="values",
                dtype=np.int64,
            ),
        ),
        (
            {"i0": ["a", "b"], "i1": "c"},
            pd.Series(
                [0, 2],
                index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
                name="values",
            ),
        ),
        (
            {"i0": {"ge": "b", "lt": "z"}, "i1": {"isin": ["d", "x"]}},
            pd.Series(
                [3],
                index=pd.MultiIndex.from_tuples([("b", "d")], names=["i0", "i1"]),
                name="values",
            ),
        ),
    ],
)
def test_query(series1, query, expected, as_type):
    obj = series1
    assert isinstance(query, dict)

    if as_type == "dict":
        result = obj.etl.q(query)
    elif as_type == "list":
        result = obj.etl.q([query])
    elif as_type == "kwargs":
        result = obj.etl.q(**query)
    else:
        raise ValueError(f"Invalid as_type: {as_type}")

    # check_index_type=False needed to avoid Attribute "inferred_type" are different
    assert_series_equal(result, expected, check_index_type=False)


def test_query_with_multiple_dicts(series1):
    obj = series1
    query = [
        {"i0": "a"},
        {"i0": "b", "i1": "d"},
    ]
    expected = pd.Series(
        [0, 1, 3],
        index=pd.MultiIndex.from_tuples([("a", "c"), ("a", "d"), ("b", "d")], names=["i0", "i1"]),
        name="values",
    )

    result = obj.etl.q(query)

    assert_series_equal(result, expected)


def test_query_with_invalid_params(series1):
    obj = series1
    with pytest.raises(ValueError, match="Query and params cannot be specified together"):
        obj.etl.q({"i0": "a"}, i1="b")


def test_one(series1):
    obj = series1
    result = obj.etl.one(i0="b", i1="d")
    assert result == 3


def test_one_raises_when_multiple_results(series1):
    obj = series1
    with pytest.raises(RuntimeError, match="The query returned 2 records instead of 1."):
        obj.etl.one(i0="b")


def test_one_raises_when_no_results(series1):
    obj = series1
    with pytest.raises(RuntimeError, match="The query returned 0 records instead of 1."):
        obj.etl.one(i0="x")


def test_first(series1):
    obj = series1
    result = obj.etl.first(i0="b")
    assert result == 2


def test_first_raises_when_no_results(series1):
    obj = series1
    with pytest.raises(RuntimeError, match="The query returned 0 records."):
        obj.etl.first(i0="x")


def test_iter(series1):
    obj = series1
    it = obj.etl.iter()
    assert isinstance(it, Iterator)
    index, value = next(it)
    assert isinstance(index, tuple)
    assert index == ("a", "c")
    assert value == 0
    assert index.i0 == "a"
    assert index.i1 == "c"


def test_iterdict(series1):
    obj = series1
    it = obj.etl.iterdict()
    assert isinstance(it, Iterator)
    index, value = next(it)
    assert isinstance(index, dict)
    assert index == {"i0": "a", "i1": "c"}
    assert value == 0
