import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal


@pytest.fixture
def storable_df_with_unnamed_index():
    # Format: column -> (rows, dtype)
    data = {
        "a": ([0, 1, 2], int),
        "b": ([0, 1, 2.0], float),
        "c": ([np.nan, np.nan, np.nan], float),
        "d": ([0, None, np.nan], float),
        "e": ([None, None, None], object),
        "f": (["s0", "s1", "s2"], object),
        "g": (["s0", None, np.nan], object),
        "h": ([[1.1, 1.2], [5], None], object),
        "i": ([[1.1, 1.2], [], None], object),
        "j": ([[1.1, 1.2], [], [3, 4]], object),
        "k": ([[1.1, 1.2], [2], [3, 4]], object),
    }
    df = pd.DataFrame({k: np.array(v[0], dtype=v[1]) for k, v in data.items()})
    expected_dtypes = pd.Series({k: v[1] for k, v in data.items()})
    assert_series_equal(df.dtypes, expected_dtypes)
    return df


@pytest.fixture
def storable_df_with_named_index(storable_df_with_unnamed_index):
    df = storable_df_with_unnamed_index
    df.index.name = "i0"
    return df


@pytest.fixture
def storable_df_with_unnamed_multiindex(storable_df_with_unnamed_index):
    df = storable_df_with_unnamed_index
    df.index = pd.MultiIndex.from_tuples([(0, 0), (1, 0), (1, 1)])
    return df


@pytest.fixture
def storable_df_with_named_multiindex(storable_df_with_unnamed_index):
    df = storable_df_with_unnamed_index
    df.index = pd.MultiIndex.from_tuples([(0, 0), (1, 0), (1, 1)], names=["i0", "i1"])
    return df
