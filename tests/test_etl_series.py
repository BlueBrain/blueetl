import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from blueetl.core.etl import ETLSeriesAccessor


def _build_series():
    """Return a simple Series.

    i0  i1
    a   c     0
        d     1
    b   c     2
        d     3
    Name: values, dtype: int64
    """
    return pd.Series(
        [0, 1, 2, 3],
        index=pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["i0", "i1"]),
        name="values",
    )


def test_etl_instance():
    s = _build_series()
    result = s.etl
    assert isinstance(result, ETLSeriesAccessor)


def test_conditions():
    s = _build_series()
    result = s.etl.conditions()
    assert_array_equal(result, ["i0", "i1"])


def test_complementary_conditions_with_str():
    s = _build_series()
    result = s.etl.complementary_conditions("i0")
    assert_array_equal(result, ["i1"])


def test_complementary_conditions_with_list():
    s = _build_series()
    result = s.etl.complementary_conditions(["i0"])
    assert_array_equal(result, ["i1"])


def test_complementary_conditions_with_empty_list():
    s = _build_series()
    result = s.etl.complementary_conditions([])
    assert_array_equal(result, ["i0", "i1"])


def test_complementary_conditions_with_all_conditions():
    s = _build_series()
    result = s.etl.complementary_conditions(["i0", "i1"])
    assert_array_equal(result, [])


def test_complementary_conditions_with_nonexistent_condition():
    s = _build_series()
    result = s.etl.complementary_conditions(["iX"])
    assert_array_equal(result, ["i0", "i1"])


def test_labels():
    s = _build_series()
    result = s.etl.labels()
    assert_array_equal(result, [["a", "b"], ["c", "d"]])


def test_labels_of():
    s = _build_series()
    result = s.etl.labels_of("i0")
    assert_array_equal(result, ["a", "b"])


def test_labels_of_nonexistent_condition():
    s = _build_series()
    with pytest.raises(KeyError, match="Level iX not found"):
        s.etl.labels_of("iX")


def test_filter_drop_level():
    s = _build_series()
    result = s.etl.filter(i1="c", drop_level=True)
    assert_series_equal(
        result,
        pd.Series(
            [0, 2],
            index=pd.Index(["a", "b"], name="i0"),
            name="values",
        ),
    )


def test_filter_no_drop_level():
    s = _build_series()
    result = s.etl.filter(i1="c", drop_level=False)
    assert_series_equal(
        result,
        pd.Series(
            [0, 2],
            index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
            name="values",
        ),
    )


def test_remove_condition_with_str():
    s = _build_series()
    result = s.etl.remove_condition("i0")
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


def test_remove_condition_with_list():
    s = _build_series()
    result = s.etl.remove_condition(["i0"])
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


def test_keep_condition_with_str():
    s = _build_series()
    result = s.etl.keep_condition("i1")
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


def test_keep_condition_with_list():
    s = _build_series()
    result = s.etl.keep_condition(["i1"])
    assert_series_equal(
        result,
        pd.Series(
            [0, 1, 2, 3],
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
            name="values",
        ),
    )


def test_add_condition():
    s = _build_series()
    result = s.etl.add_condition("i2", "e")
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


def test_unpool_from_series_of_scalars():
    def func(x):
        return pd.Series([x, x + 20], index=pd.Index(["x0", "x2"], name="func"))

    s = _build_series()
    result = s.etl.unpool(func)
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


def test_unpool_from_series_of_lists():
    def func(x):
        return pd.Series([x[0], x[2]], index=pd.Index(["x0", "x2"], name="func"))

    s = _build_series()
    # transform each element to a list of integers
    s = s.transform(lambda x: [x, x + 10, x + 20])
    result = s.etl.unpool(func)
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


def test_pool_with_func_returning_scalar():
    def func(x):
        return sum(x)

    s = _build_series()
    result = s.etl.pool(["i1"], func)
    assert_series_equal(
        result,
        pd.Series(
            [1, 5],
            index=pd.Index(["a", "b"], name="i0"),
            name="values",
        ),
    )


def test_pool_with_func_returning_series():
    def func(x):
        return pd.Series(
            [sum(x[:, "d"]), sum(x[:, "c"])], index=pd.Index(["x0", "x1"], name="func")
        )

    s = _build_series()
    result = s.etl.pool(["i1"], func)
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


def test_remodel():
    def func(x):
        for index, value in x.etl.iter():
            yield value + 1000, {**index._asdict(), "i2": "e"}
            yield value + 2000, {**index._asdict(), "i2": "f"}

    s = _build_series()
    result = s.etl.remodel(func)
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
