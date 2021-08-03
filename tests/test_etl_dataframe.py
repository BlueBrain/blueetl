import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl import etl


def _build_dataframe():
    """Return a simple DataFrame.

           v0  v1
    i0 i1
    a  c    0   4
       d    1   5
    b  c    2   6
       d    3   7
    """
    return pd.DataFrame(
        {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
        index=pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["i0", "i1"]),
    )


def test_etl_instance():
    df = _build_dataframe()
    result = df.etl
    assert isinstance(result, etl.ETLDataFrameAccessor)


def test_conditions():
    df = _build_dataframe()
    result = df.etl.conditions()
    assert_array_equal(result, ["i0", "i1"])


def test_complementary_conditions_with_str():
    df = _build_dataframe()
    result = df.etl.complementary_conditions("i0")
    assert_array_equal(result, ["i1"])


def test_complementary_conditions_with_list():
    df = _build_dataframe()
    result = df.etl.complementary_conditions(["i0"])
    assert_array_equal(result, ["i1"])


def test_complementary_conditions_with_empty_list():
    df = _build_dataframe()
    result = df.etl.complementary_conditions([])
    assert_array_equal(result, ["i0", "i1"])


def test_complementary_conditions_with_all_conditions():
    df = _build_dataframe()
    result = df.etl.complementary_conditions(["i0", "i1"])
    assert_array_equal(result, [])


def test_complementary_conditions_with_nonexistent_condition():
    df = _build_dataframe()
    result = df.etl.complementary_conditions(["iX"])
    assert_array_equal(result, ["i0", "i1"])


def test_labels():
    df = _build_dataframe()
    result = df.etl.labels()
    assert_array_equal(result, [["a", "b"], ["c", "d"]])


def test_labels_of():
    df = _build_dataframe()
    result = df.etl.labels_of("i0")
    assert_array_equal(result, ["a", "b"])


def test_labels_of_nonexistent_condition():
    df = _build_dataframe()
    with pytest.raises(KeyError, match="Level iX not found"):
        df.etl.labels_of("iX")


def test_filter_drop_level():
    df = _build_dataframe()
    result = df.etl.filter(i1="c", drop_level=True)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 2], "v1": [4, 6]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_filter_no_drop_level():
    df = _build_dataframe()
    result = df.etl.filter(i1="c", drop_level=False)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 2], "v1": [4, 6]},
            index=pd.MultiIndex.from_tuples([("a", "c"), ("b", "c")], names=["i0", "i1"]),
        ),
    )


def test_remove_condition_with_str():
    df = _build_dataframe()
    result = df.etl.remove_condition("i0")
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


def test_remove_condition_with_list():
    df = _build_dataframe()
    result = df.etl.remove_condition(["i0"])
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


def test_keep_condition_with_str():
    df = _build_dataframe()
    result = df.etl.keep_condition("i1")
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


def test_keep_condition_with_list():
    df = _build_dataframe()
    result = df.etl.keep_condition(["i1"])
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [0, 1, 2, 3], "v1": [4, 5, 6, 7]},
            index=pd.Index(["c", "d", "c", "d"], name="i1"),
        ),
    )


def test_add_condition():
    df = _build_dataframe()
    result = df.etl.add_condition("i2", "e")
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


@pytest.mark.skip(reason="need to verify the requirements and the implementation")
def test_unpool_from_series_of_scalars():
    def func(x):
        return pd.Series([x, x + 20], index=pd.Index(["x0", "x2"], name="func"))

    df = _build_dataframe()
    result = df.etl.unpool(func)
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
def test_unpool_from_series_of_lists():
    def func(x):
        return pd.Series([x[0], x[2]], index=pd.Index(["x0", "x2"], name="func"))

    df = _build_dataframe()
    # transform each element to a list of integers
    df = df.transform(lambda x: [x, x + 10, x + 20])
    result = df.etl.unpool(func)
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


def test_pool_with_func_returning_scalar():
    def func(x):
        # x is a dataframe with the same index as the original df,
        # but the only varying level is the one specified in the pool call.
        # Example:
        #        v0  v1
        # i0 i1
        # a  c    0   4
        #    d    1   5
        return x.to_numpy().sum()

    df = _build_dataframe()
    result = df.etl.pool(["i1"], func)
    assert_series_equal(
        result,
        pd.Series(
            [10, 18],
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_pool_with_func_returning_series():
    def func(x):
        # x is a dataframe with the same index as the original df,
        # but the only varying level is the one specified in the pool call.
        # Example:
        #        v0  v1
        # i0 i1
        # a  c    0   4
        #    d    1   5
        return x.sum()

    df = _build_dataframe()
    result = df.etl.pool(["i1"], func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0": [1, 5], "v1": [9, 13]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )


def test_pool_with_func_returning_series_with_different_shape():
    def func(x):
        # x is a dataframe with the same index as the original df,
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

    df = _build_dataframe()
    result = df.etl.pool(["i1"], func)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"v0_sum": [1, 5], "v1_sum": [9, 13], "sum": [10, 18]},
            index=pd.Index(["a", "b"], name="i0"),
        ),
    )
