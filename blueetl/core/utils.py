"""Core utilities."""
import operator
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from blueetl.core import L


def query_frame(df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
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
    q: Dict[str, Any] = {"columns": {}, "index": {}}
    for key, value in query.items():
        q[mapping[key]][key] = value
    # filter by columns and index
    masks = chain(
        (compare(df[key], value) for key, value in q["columns"].items()),
        (compare(df.index.get_level_values(key), value) for key, value in q["index"].items()),
    )
    return df.loc[np.all(list(masks), axis=0)]


def query_series(series: pd.Series, query: Dict) -> pd.Series:
    """Given a query dictionary, return the Series filtered by index."""
    if not query:
        return series
    # filter by columns and index
    masks = (compare(series.index.get_level_values(key), value) for key, value in query.items())
    return series.loc[np.all(list(masks), axis=0)]


def compare(obj: Union[pd.DataFrame, pd.Series, pd.Index], value: Any) -> np.ndarray:
    """Return the result of the comparison between obj and value.

    Args:
        obj: a DataFrame, Series, or Index.
        value: value used for comparison.
            - if scalar, use equality
            - if list-like, use isin
            - if dict, any supported operator can be specified

    Examples:
        >>> df = pd.DataFrame({"gid": [0, 2, 3, 7, 8]})
        >>> compare(df["gid"], 3)
            array([False, False,  True, False, False])
        >>> compare(df["gid"], [3, 5, 8])
            array([False, False,  True, False,  True])
        >>> compare(df["gid"], {"ge": 3, "lt": 8})
            array([False, False,  True,  True, False])

    """
    if isinstance(value, dict):
        operators = {
            "eq": "__eq__",
            "ne": "__ne__",
            "le": "__le__",
            "lt": "__lt__",
            "ge": "__ge__",
            "gt": "__gt__",
            "isin": "isin",
        }
        masks = []
        for op, v in value.items():
            if op in operators:
                masks.append(getattr(obj, operators[op])(v))
            else:
                raise ValueError(f"Unsupported operator: {op}")
        result = np.all(masks, axis=0) if masks else np.full(len(obj), True)
    elif is_list_like(value):
        result = np.asarray(obj.isin(value))
    else:
        # more efficient than using isin with a list of one element
        result = np.asarray(obj == value)
    return result


def is_subfilter(left: Dict, right: Dict) -> bool:
    """Return True if ``left`` is a subfilter of ``right``, False otherwise.

    ``left`` is a subfilter of ``right`` if it's equal or more specific.

    Examples:
        >>> print(is_subfilter({}, {}))
        True
        >>> print(is_subfilter({}, {"key": 1}))
        False
        >>> print(is_subfilter({"key": 1}, {}))
        True
        >>> print(is_subfilter({"key": 1}, {"key": 1}))
        True
        >>> print(is_subfilter({"key": 1}, {"key": [1, 2]}))
        True
        >>> print(is_subfilter({"key": 1}, {"key": {"isin": [1, 2]}}))
        True
        >>> print(is_subfilter({"key": 1}, {"key": 2}))
        False
        >>> print(is_subfilter({"key": 1}, {"key": [2, 3]}))
        False
        >>> print(is_subfilter({"key": 1}, {"key": {"isin": [2, 3]}}))
        False
        >>> print(is_subfilter({"key1": 1, "key2": 2}, {"key1": 1}))
        True
        >>> print(is_subfilter({"key1": 1}, {"key1": 1, "key2": 2}))
        False
    """

    def _to_dict(obj) -> Dict:
        """Return a normalized filter, i.e. a dict where "eq" is replaced by "isin"."""
        obj = deepcopy(obj)
        if isinstance(obj, dict):
            if "eq" in obj:
                # convert "eq" to "isin", and set "isin" to the new value,
                # or to an empty list if "eq" and "isin" are incompatible
                value = obj.pop("eq")
                obj["isin"] = [value] if "isin" not in obj or value in obj["isin"] else []
            return obj
        if isinstance(obj, list):
            return {"isin": obj}
        # any other type of object is considered for equality with "isin"
        return {"isin": [obj]}

    def _is_subdict(d1: Dict, d2: Dict) -> bool:
        """Return True if d1 is a subdict of d2."""
        # mapping operator -> operation
        operators = {
            "ne": operator.eq,
            "le": operator.le,
            "lt": operator.le,
            "ge": operator.ge,
            "gt": operator.ge,
            "isin": lambda a, b: set(a).issubset(b),
        }
        assert set(operators).issuperset(d1), "Invalid keys in d1"
        assert set(operators).issuperset(d2), "Invalid keys in d2"
        unmatched_keys = set()
        # for each operator in the operators mapping,
        # if the operator is present in d2 but not in d1,
        # or if the given operation is not satisfied,
        # then d1 cannot be a subdict of d2
        for op, operation in operators.items():
            if op in d2 and (op not in d1 or not operation(d1[op], d2[op])):
                unmatched_keys.add(op)
        L.debug("unmatched keys: %s", sorted(unmatched_keys))
        return len(unmatched_keys) == 0

    for key in right:
        if key not in left:
            return False
        dict_left = _to_dict(left[key])
        dict_right = _to_dict(right[key])
        if not _is_subdict(dict_left, dict_right):
            return False
    return True


def safe_concat(iterable, *args, **kwargs):
    """Build and return a Series or a Dataframe from an iterable of objects with the same index.

    Args:
        iterable: iterable of Series or DataFrames.
            All the objects must be of the same type, and they must have the same index,
            or an exception is raised.

    Returns:
        (pd.Series, pd.DataFrame) result of the concatenation, same type of the input elements.
    """

    def _reorder_levels(obj, order):
        # wrap reorder_levels to ensure that some c
        if len(order) != obj.index.nlevels:
            # reorder_levels would raise an AssertionError
            raise RuntimeError(
                f"Length of order must be same as number of "
                f"levels ({obj.index.nlevels}), got {len(order)}"
            )
        if diff := set(order).difference(obj.index.names):
            # reorder_levels would raise a KeyError
            raise RuntimeError(f"Levels not found: {''.join(diff)}")
        return obj.reorder_levels(order)

    def _ordered(obj):
        nonlocal order
        if order is None:
            order = obj.index.names
        return obj if order == obj.index.names else _reorder_levels(obj, order)

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
