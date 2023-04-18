from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from blueetl.core.etl import ETLDataFrameAccessor


def test_etl_instance(dataframe1):
    obj = dataframe1
    result = obj.etl
    assert isinstance(result, ETLDataFrameAccessor)


def test_conditions(dataframe1):
    obj = dataframe1
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
def test_complementary_conditions(dataframe1, conditions, expected):
    obj = dataframe1
    result = obj.etl.complementary_conditions(conditions)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "conditions, expected",
    [
        pytest.param(None, [pd.Index(["a", "b"], name="i0"), pd.Index(["c", "d"], name="i1")]),
        pytest.param(["i0"], [pd.Index(["a", "b"], name="i0")], id="one"),
    ],
)
def test_labels(dataframe1, conditions, expected):
    obj = dataframe1
    result = obj.etl.labels(conditions)
    assert len(result) == len(expected)
    for actual_index, expected_index in zip(result, expected):
        assert_index_equal(actual_index, expected_index)


def test_labels_of(dataframe1):
    obj = dataframe1
    result = obj.etl.labels_of("i0")
    assert_array_equal(result, ["a", "b"])


def test_labels_of_nonexistent_condition(dataframe1):
    obj = dataframe1
    with pytest.raises(KeyError, match="Level iX not found"):
        obj.etl.labels_of("iX")


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param("i0", id="str"),
        pytest.param(["i0"], id="list"),
    ],
)
def test_remove_conditions(dataframe1, conditions):
    obj = dataframe1
    result = obj.etl.remove_conditions(conditions)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param("i1", id="str"),
        pytest.param(["i1"], id="list"),
    ],
)
def test_keep_conditions(dataframe1, conditions):
    obj = dataframe1
    result = obj.etl.keep_conditions(conditions)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


@pytest.mark.parametrize(
    "conditions, values",
    [
        pytest.param("i2", "e", id="str"),
        pytest.param(["i2"], ["e"], id="list"),
    ],
)
def test_add_conditions(dataframe1, conditions, values):
    obj = dataframe1
    result = obj.etl.add_conditions(conditions, values)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.MultiIndex.from_tuples(
                [("e", "a", "c"), ("e", "a", "d"), ("e", "b", "c"), ("e", "b", "d")],
                names=["i2", "i0", "i1"],
            ),
        ),
    )


@pytest.mark.parametrize(
    "conditions, values, dtypes",
    [
        pytest.param("i2", 123, float, id="str"),
        pytest.param(["i2"], [123], [float], id="list"),
    ],
)
def test_add_conditions_with_dtypes(dataframe1, conditions, values, dtypes):
    obj = dataframe1
    result = obj.etl.add_conditions(conditions, values, dtypes=dtypes)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.MultiIndex.from_tuples(
                [(123.0, "a", "c"), (123.0, "a", "d"), (123.0, "b", "c"), (123.0, "b", "d")],
                names=["i2", "i0", "i1"],
            ),
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
def test_add_conditions_inner_drop(dataframe1, inner, drop, expected_conditions):
    obj = dataframe1
    conditions = ["i2"]
    values = ["e"]
    result = obj.etl.add_conditions(conditions, values, inner=inner, drop=drop)
    assert result.index.names == expected_conditions


def test_select_with_drop_level(dataframe1):
    obj = dataframe1
    result = obj.etl.select(i1="c", drop_level=True)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 2], "v1": [4, 6]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_select_without_drop_level(dataframe1):
    obj = dataframe1
    result = obj.etl.select(i1="c", drop_level=False)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 2], "v1": [4, 6]},
            index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
        ),
    )


def test_select_without_args(dataframe1):
    obj = dataframe1
    result = obj.etl.select()
    assert_frame_equal(result, obj)


@pytest.mark.skip(reason="need to verify the requirements and the implementation")
def test_unpool_from_series_of_scalars(dataframe1):
    def func(x):
        return pd.Series([x, x + 20], index=pd.Index(["x0", "x2"], name="func"))

    obj = dataframe1
    result = obj.etl.unpool(func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 20, 1, 21, 2, 22, 3, 23], "v1": [4, 24, 5, 25, 6, 26, 7, 27]},
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
        ),
    )


@pytest.mark.skip(reason="need to verify the requirements and the implementation")
def test_unpool_from_series_of_lists(dataframe1):
    def func(x):
        return pd.Series([x[0], x[2]], index=pd.Index(["x0", "x2"], name="func"))

    obj = dataframe1
    # transform each element to a list of integers
    obj = obj.transform(lambda x: [x, x + 10, x + 20])
    result = obj.etl.unpool(func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            # [0, 20, 1, 21, 2, 22, 3, 23],
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
        ),
    )


def test_pool_with_func_returning_scalar(dataframe1):
    def func(x):
        # x is a dataframe with the same index as the original obj,
        # but the only varying level is the one specified in the pool call.
        # Example:
        #        v0  v1
        # i0 i1
        # a  c    0   4
        #    d    1   5
        return x.to_numpy().sum()

    obj = dataframe1
    result = obj.etl.pool(["i1"], func)
    assert_series_equal(
        result,
        pd.Series(
            [10, 18],
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_pool_with_func_returning_series(dataframe1):
    def func(x):
        # x is a dataframe with the same index as the original obj,
        # but the only varying level is the one specified in the pool call.
        # Example:
        #        v0  v1
        # i0 i1
        # a  c    0   4
        #    d    1   5
        return x.sum()

    obj = dataframe1
    result = obj.etl.pool(["i1"], func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [1, 5], "v1": [9, 13]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_pool_with_func_returning_series_with_different_shape(dataframe1):
    def func(x):
        # x is a dataframe with the same index as the original obj,
        # but the only varying level is the one specified in the pool call.
        # Example:
        #        v0  v1
        # i0 i1
        # a  c    0   4
        #    d    1   5
        return pd.Series(
            [x["v0"].sum(), x["v1"].sum(), x.to_numpy().sum()],
            index=pd.Index(["v0_sum", "v1_sum", "sum"]),
        )

    obj = dataframe1
    result = obj.etl.pool(["i1"], func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0_sum": [1, 5], "v1_sum": [9, 13], "sum": [10, 18]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


@pytest.mark.parametrize("as_type", ["dict", "list", "kwargs"])
@pytest.mark.parametrize(
    "query, expected",
    [
        (
            {"v0": 0},
            pd.DataFrame(
                [[0, 4]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("a", "c")], names=["i0", "i1"]),
            ),
        ),
        (
            {"v0": 1, "v1": 5},
            pd.DataFrame(
                [[1, 5]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("a", "d")], names=["i0", "i1"]),
            ),
        ),
        (
            {"v0": 1, "v1": 8},
            pd.DataFrame(
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([], names=["i0", "i1"]),
                dtype=np.int64,
            ),
        ),
        (
            {"v0": [1, 2]},
            pd.DataFrame(
                [[1, 5], [2, 6]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("a", "d"), ("b", "c")], names=["i0", "i1"]),
            ),
        ),
        (
            {"v0": [1, 2], "v1": [6, 7]},
            pd.DataFrame(
                [[2, 6]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("b", "c")], names=["i0", "i1"]),
            ),
        ),
        (
            {"v0": {"ge": 1, "le": 2}, "v1": {"isin": [6, 7]}},
            pd.DataFrame(
                [[2, 6]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("b", "c")], names=["i0", "i1"]),
            ),
        ),
    ],
)
def test_query(dataframe1, query, expected, as_type):
    obj = dataframe1
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
    assert_frame_equal(result, expected, check_index_type=False)


def test_query_with_multiple_dicts(dataframe1):
    obj = dataframe1
    query = [
        {"v0": 0},
        {"v0": 1, "v1": 5},
    ]
    expected = pd.DataFrame(
        [[0, 4], [1, 5]],
        columns=["v0", "v1"],
        index=pd.MultiIndex.from_tuples([("a", "c"), ("a", "d")], names=["i0", "i1"]),
    )

    result = obj.etl.q(query)

    assert_frame_equal(result, expected)


def test_query_with_invalid_params(dataframe1):
    obj = dataframe1
    with pytest.raises(ValueError, match="Query and params cannot be specified together"):
        obj.etl.q({"v0": 0}, v1=1)


def test_one(dataframe1):
    obj = dataframe1
    result = obj.etl.one(v0=0)
    assert_series_equal(result, pd.Series({"v0": 0, "v1": 4}, name=("a", "c")))


def test_one_raises_when_multiple_results(dataframe1):
    obj = dataframe1
    with pytest.raises(RuntimeError, match="The query returned 2 records instead of 1."):
        obj.etl.one(i0="b")


def test_one_raises_when_no_results(dataframe1):
    obj = dataframe1
    with pytest.raises(RuntimeError, match="The query returned 0 records instead of 1."):
        obj.etl.one(i0="x")


def test_first(dataframe1):
    obj = dataframe1
    result = obj.etl.first(i0="b")
    assert_series_equal(result, pd.Series({"v0": 2, "v1": 6}, name=("b", "c")))


def test_first_raises_when_no_results(dataframe1):
    obj = dataframe1
    with pytest.raises(RuntimeError, match="The query returned 0 records."):
        obj.etl.first(i0="x")


def test_iter(dataframe1):
    obj = dataframe1
    it = obj.etl.iter()
    assert isinstance(it, Iterator)
    index, value = next(it)
    assert isinstance(index, tuple)
    assert isinstance(value, tuple)
    assert index == ("a", "c")
    assert value == (0, 4)
    assert index.i0 == "a"
    assert index.i1 == "c"
    assert value.v0 == 0
    assert value.v1 == 4


def test_iterdict(dataframe1):
    obj = dataframe1
    it = obj.etl.iterdict()
    assert isinstance(it, Iterator)
    index, value = next(it)
    assert isinstance(index, dict)
    assert isinstance(value, dict)
    assert index == {"i0": "a", "i1": "c"}
    assert value == {"v0": 0, "v1": 4}


@pytest.mark.parametrize(
    "params, expected_key, expected_df",
    [
        pytest.param(
            {"groupby_columns": ["i1"]},
            ("c",),
            pd.DataFrame(
                [[0, 4], [2, 6]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
            ),
        ),
        pytest.param(
            {"groupby_columns": ["i1"], "selected_columns": ["v1"]},
            ("c",),
            pd.DataFrame(
                [[4], [6]],
                columns=["v1"],
                index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
            ),
        ),
        pytest.param(
            {"groupby_columns": ["v0"]},
            (0,),
            pd.DataFrame(
                [[0, 4]],
                columns=["v0", "v1"],
                index=pd.MultiIndex.from_tuples([("a", "c")], names=["i0", "i1"]),
            ),
        ),
        pytest.param(
            {"groupby_columns": ["v0"], "selected_columns": ["v1"]},
            (0,),
            pd.DataFrame(
                [[4]],
                columns=["v1"],
                index=pd.MultiIndex.from_tuples([("a", "c")], names=["i0", "i1"]),
            ),
        ),
    ],
)
def test_groupby_iter(dataframe1, params, expected_key, expected_df):
    obj = dataframe1
    it = obj.etl.groupby_iter(**params)
    assert isinstance(it, Iterator)
    key, group_df = next(it)
    assert key == expected_key
    assert_frame_equal(group_df, expected_df)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"groupby_columns": ["i0"]},
            [
                (("a",), [{"v0": 0, "v1": 4}, {"v0": 1, "v1": 5}]),
                (("b",), [{"v0": 2, "v1": 6}, {"v0": 3, "v1": 7}]),
            ],
        ),
        (
            {"groupby_columns": ["v0"]},
            [
                ((0,), [{"v0": 0, "v1": 4}]),
                ((1,), [{"v0": 1, "v1": 5}]),
                ((2,), [{"v0": 2, "v1": 6}]),
                ((3,), [{"v0": 3, "v1": 7}]),
            ],
        ),
        (
            {"groupby_columns": ["i0", "v0"]},
            [
                (("a", 0), [{"v0": 0, "v1": 4}]),
                (("a", 1), [{"v0": 1, "v1": 5}]),
                (("b", 2), [{"v0": 2, "v1": 6}]),
                (("b", 3), [{"v0": 3, "v1": 7}]),
            ],
        ),
        (
            {"groupby_columns": ["i0", "v0"], "selected_columns": ["v1"]},
            [
                (("a", 0), [{"v1": 4}]),
                (("a", 1), [{"v1": 5}]),
                (("b", 2), [{"v1": 6}]),
                (("b", 3), [{"v1": 7}]),
            ],
        ),
    ],
)
def test_groupby_run_parallel(dataframe1, params, expected):
    def func(key, df):
        return tuple(key), df.to_dict(orient="records")

    obj = dataframe1
    result = obj.etl.groupby_run_parallel(**params, func=func)
    assert result == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"groupby_columns": ["i0"]},
            pd.Series(
                [4, 6, 8, 10],
                index=pd.MultiIndex.from_tuples(
                    [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")], names=["i0", "i1"]
                ),
            ),
        ),
    ],
)
def test_groupby_apply_parallel(dataframe1, params, expected):
    def func(key, df):
        return df.sum(axis=1)

    obj = dataframe1
    result = obj.etl.groupby_apply_parallel(**params, func=func)
    assert_series_equal(result, expected)
