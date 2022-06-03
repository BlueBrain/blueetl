import pandas as pd
import pytest


@pytest.fixture
def dataframe1():
    """Return a simple DataFrame with MultiIndex.

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


@pytest.fixture
def series1():
    """Return a simple Series with MultiIndex.

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


@pytest.fixture
def index1():
    """Return a MultiIndex.

    MultiIndex([('a', 'c'),
                ('a', 'd'),
                ('b', 'c'),
                ('b', 'd')],
               names=['i0', 'i1'])
    """
    return pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["i0", "i1"])


@pytest.fixture
def index2():
    """Return a MultiIndex without names.

    MultiIndex([('a', 'c'),
                ('a', 'd'),
                ('b', 'c'),
                ('b', 'd')],
               )
    """
    return pd.MultiIndex.from_product([["a", "b"], ["c", "d"]])
