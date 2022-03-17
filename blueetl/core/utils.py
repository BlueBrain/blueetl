import logging
from itertools import chain
from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

L = logging.getLogger(__name__)


def query_frame(df: pd.DataFrame, query: Dict) -> pd.DataFrame:
    """Given a query dictionary, return the DataFrame filtered by columns and index."""
    if not query:
        return df
    # map each name to columns or index
    # if the same key is present in both columns and index, use columns
    mapping = {
        **{k: "index" for k in df.index.names if k is not None},
        **{k: "columns" for k in df.columns if k is not None},
    }
    # dictionary with query keys split into columns and index
    q = {"columns": {}, "index": {}}
    for key, value in query.items():
        q[mapping[key]][key] = value
    # filter by columns and index
    masks = chain(
        (eqin(df[key], value) for key, value in q["columns"].items()),
        (eqin(df.index.get_level_values(key), value) for key, value in q["index"].items()),
    )
    return df.loc[np.all(list(masks), axis=0)]


def query_series(series: pd.Series, query: Dict) -> pd.Series:
    """Given a query dictionary, return the Series filtered by index."""
    if not query:
        return series
    # filter by columns and index
    masks = (eqin(series.index.get_level_values(key), value) for key, value in query.items())
    return series.loc[np.all(list(masks), axis=0)]


def eqin(obj, value):
    """As 'isin', but value can be a list or a scalar value for equality check.

    Args:
        obj: a DataFrame, Series, or Index.
        value: a scalar or list-like value.
    """
    if is_list_like(value):
        return obj.isin(value)
    # more efficient than using isin with a list of one element
    return obj == value


def safe_concat(iterable, *args, **kwargs):
    """Build and return a Series or a Dataframe from an iterable of objects with the same index.

    Args:
        iterable: iterable of Series or DataFrames.
            All the objects must be of the same type and they must have the same index,
            or an exception is raised.

    Returns:
        (pd.Series, pd.DataFrame) result of the concatenation, same type of the input elements.
    """

    def _ordered(obj):
        nonlocal order
        if order is None:
            order = obj.index.names
        return obj if order == obj.index.names else obj.reorder_levels(order)

    order = None
    return pd.concat((_ordered(obj) for obj in iterable), *args, **kwargs)


def concat_tuples(iterable, *args, **kwargs):
    """Build and return a Series from an iterable of tuples (value, conditions).

    Args:
        iterable: iterable of tuples (value, conditions), where
            - value is a single value that will be added to the Series
            - conditions is a dict containing the conditions to be used for the MultiIndex.
              The keys of the conditions must be the same for each tuple of the iterable,
              or an exception is raised.

    Returns:
        (pd.Series) result of the concatenation.
    """

    def _index(conditions):
        arrays = [[v] for v in conditions.values()]
        names = list(conditions)
        return pd.MultiIndex.from_arrays(arrays, names=names)

    iterable = (pd.Series([data], index=_index(conditions)) for data, conditions in iterable)
    return safe_concat(iterable, *args, **kwargs)
