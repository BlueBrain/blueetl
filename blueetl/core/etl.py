"""Pandas accessors."""
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import pandas as pd

from blueetl.core import L
from blueetl.core.parallel import Task, run_parallel
from blueetl.core.utils import query_frame, query_series
from blueetl.utils import ensure_list

# Naming conventions
#  level: number, or name of the level in the MultiIndex
#  condition: name of the level (e.g. seed), similar to level, but it cannot be numeric
#  labels: possible values for a specific level of the index


class ETLBaseAccessor(ABC):
    """Accessor with methods common to Series and DataFrame."""

    def __init__(self, pandas_obj):
        """Initialize the accessor."""
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj) -> None:
        """Validate the wrapped object."""
        # assert all(obj.index.names), "All the index levels must have a name"
        # assert len(set(obj.index.names)) == obj.index.nlevels, "The level names must be unique"

    def conditions(self) -> List[str]:
        """Names for each of the index levels."""
        return self._obj.index.names

    def complementary_conditions(self, conditions: Union[str, List[str]]) -> List[str]:
        """Return the difference between the object conditions and the specified conditions.

        Args:
            conditions: single condition or list of conditions used to calculate the difference.
        """
        conditions = ensure_list(conditions)
        # TODO: raise a KeyError if any condition is not found in self.conditions?
        return self._obj.index.names.difference(conditions)

    def labels(self, conditions: Optional[List[str]] = None) -> List[pd.Index]:
        """Unique labels for each level, or for the specified levels.

        Args:
            conditions: list of condition names, or None to consider all the levels.

        Returns:
            list of indexes with unique labels, one for each level.
        """
        conditions = conditions or self.conditions()
        return [self.labels_of(condition) for condition in conditions]

    def labels_of(self, condition: str) -> pd.Index:
        """Unique labels for a specific level in the index.

        Args:
            condition: condition name.

        Returns:
            indexes with unique labels.
        """
        return self._obj.index.unique(condition)

    def remove_conditions(
        self, conditions: Union[str, List[str]]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Remove one or more conditions.

        Args:
            conditions: single condition or list of conditions to remove.

        Returns:
            resulting Series or DataFrame.
        """
        return self._obj.droplevel(conditions, axis=0)

    def keep_conditions(self, conditions: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """Remove the conditions not specified.

        Args:
            conditions: single condition or list of conditions to keep.

        Returns:
            resulting Series or DataFrame.
        """
        return self._obj.droplevel(self.complementary_conditions(conditions), axis=0)

    def add_conditions(
        self, conditions: Union[str, List[str]], values: Any, inner=False, drop=False
    ) -> Union[pd.Series, pd.DataFrame]:
        """Add one or multiple conditions into the outermost or innermost position.

        Args:
            conditions: single conditions or list of conditions to be added.
            values: single value or list of values, one for each condition.
            inner (bool): if True, add the conditions in the innermost position.
            drop (bool): if True, drop the existing conditions.

        Returns:
            resulting Series or DataFrame.
        """
        conditions = ensure_list(conditions)
        values = ensure_list(values)
        if len(conditions) != len(values):
            raise ValueError("Conditions and values must have the same length")
        result = pd.concat([self._obj], axis="index", keys=[tuple(values)], names=conditions)
        if drop:
            # levels to be dropped, for example: [-3, -2, -1]
            levels = list(range(-self._obj.columns.nlevels, 0))
            result = result.droplevel(levels)
        elif inner:
            # rotate the levels: (0 1) 2 3 4 5 -> 2 3 4 5 (0 1)
            order = list(range(result.index.nlevels))
            order = order[len(conditions) :] + order[: len(conditions)]
            result = result.reorder_levels(order)
        return result

    def select(self, drop_level=True, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Filter the series or dataframe based on some conditions on the index.

        Note: if the level doesn't need to be dropped, it's possible to use `etl.q` instead.
        TODO: consider if it can be deprecated in favour of etl.q, and removed.

        Args:
            drop_level (bool): True to drop the conditions from the returned object.
            kwargs: conditions used to filter, specified as name=value.
        """
        if not kwargs:
            return self._obj
        labels, values = zip(*kwargs.items())
        return self._obj.xs(level=labels, key=values, drop_level=drop_level, axis=0)

    def select_one(self, drop_level=True, **kwargs):
        return self.select(drop_level=drop_level, **kwargs).iat[0]

    def groupby_except(
        self, conditions: Union[str, List[str]], *args, **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """Group by all the conditions except for the ones specified.

        Args:
            conditions: single condition or list of conditions to be excluded from the groupby
        """
        complementary_conditions = self.complementary_conditions(conditions)
        return self._obj.groupby(complementary_conditions, *args, **kwargs)

    def pool(self, conditions: Union[str, List[str]], func) -> Union[pd.Series, pd.DataFrame]:
        """Remove one or more conditions grouping by the remaining conditions.

        Args:
            conditions: single condition or list of conditions to be removed from the index.
            func: function that should accept a single element.
                If the returned value is a Series, it will be used as an additional level
                in the MultiIndex of the returned object.
        """
        return self.groupby_except(conditions).apply(func)

    @abstractmethod
    def _query_dict(self, query: Dict) -> Union[pd.Series, pd.DataFrame]:
        """Given a query dictionary, return the filtered Series or DataFrame."""

    def q(self, _query: Optional[Dict] = None, /, **params) -> Union[pd.Series, pd.DataFrame]:
        """Given a query dict or some query parameters, return the filtered Series or DataFrame.

        Filter by columns (for DataFrames) and index names (for both Series and DataFrames).
        If a name is present in both columns and index names, only the column is considered.

        Each value can be a scalar or a list. If it's a list, any matching record will be selected.

        Query and params cannot be specified together. If they are both empty or missing,
        return the original Series or DataFrame.

        This method is similar to the `query` method for DataFrames, but it accepts a dict instead
        of a string, and has some limitations on the possible filters to be applied.

        Args:
            _query: query dictionary, with:
                    keys: columns or index levels
                    values: value or list of values
                All the keys are combined with AND, while each list of values is combined with OR.
                Examples
                    {"mtype": "SO_BP", "etype": "cNAC"} -> mtype == SO_BP AND etype == cNAC
                    {"mtype": ["SO_BP", "SP_AA"]} -> mtype == SO_BP OR mtype == SP_AA
        """
        if _query and params:
            raise ValueError("Query and params cannot be specified together")
        return self._query_dict(_query or params)

    def one(self, _query: Optional[Dict] = None, /, **params) -> Any:
        """Execute the query and return the unique resulting record."""
        result = self.q(_query, **params)
        if len(result) != 1:
            raise RuntimeError(f"The query returned {len(result)} records instead of 1.")
        return result.iloc[0]

    def first(self, _query: Optional[Dict] = None, /, **params) -> Any:
        """Execute the query and return the first resulting record."""
        result = self.q(_query, **params)
        if len(result) == 0:
            raise RuntimeError("The query returned 0 records.")
        return result.iloc[0]


class ETLSeriesAccessor(ETLBaseAccessor):
    """Series accessor."""

    def unpool(self, func):
        """Apply the given function to the object elements and add a condition to the index.

        Args:
            func: function that should accept a single element and return a Series object.
                The name of that Series will be used as the name of the new level
                in the MultiIndex of the returned object.
        """
        return self._obj.apply(func).stack()

    # def merge(self, other):
    #     # FIXME: to be removed if redundant
    #     return pd.concat([self._obj, other.reindex_like(self._obj)])
    #
    # def map(self, func):
    #     # FIXME: to be removed if redundant
    #     return self._obj.map(func)

    # def remodel(self, func):
    #     """Apply func and return a new Series.
    #
    #     Args:
    #         func: generator function accepting the Series as argument, and yielding tuples
    #             (value, conditions) that will be concatenated to build a new Series.
    #
    #     Returns:
    #         (pd.Series) result of the concatenation.
    #     """
    #     return concat_tuples(func(self._obj))

    def iter(self):
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned named_index is a namedtuple representing the value of the index.
        The returned value is the actual value of each element of the series.
        """
        return zip(self._obj.index.etl.iter(), iter(self._obj))

    def _query_dict(self, query: Dict) -> pd.Series:
        """Given a query dictionary, return the Series filtered by index."""
        return query_series(self._obj, query)


class ETLDataFrameAccessor(ETLBaseAccessor):
    """DataFrame accessor."""

    def iter(self):
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned ``named_index`` is a namedtuple representing the value of the index.
        The returned ``value`` is a namedtuple as returned by pandas.DataFrame.itertuples.
        """
        return zip(self._obj.index.etl.iter(), self._obj.itertuples(index=False, name="Values"))

    def _query_dict(self, query: Dict) -> pd.DataFrame:
        """Given a query dictionary, return the DataFrame filtered by columns and index."""
        return query_frame(self._obj, query)

    def groupby_iter(
        self,
        groupby_columns: List[str],
        selected_columns: Optional[List[str]] = None,
        sort: bool = True,
        observed: bool = True,
    ) -> Iterator[Tuple[NamedTuple, pd.DataFrame]]:
        """Group the dataframe by columns and yield each record as a tuple (key, df).

        It can be used as a replacement for the iteration over df.groupby, but:
            - the yielded keys are namedtuples, instead of tuples
            - the yielded dataframes contain only the varying columns, instead of all the columns

        Args:
            groupby_columns: list of column names to group by.
            selected_columns: list of column names to be included in the yielded dataframes.
                If None, only the varying columns are included.
            sort: Sort group keys.
            observed: This only applies if any of the groupers are Categoricals.
                If True: only show observed values for categorical groupers.
                If False: show all values for categorical groupers.

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        if selected_columns is None:
            groupby_set = set(groupby_columns)
            selected_columns = [col for col in self._obj.columns if col not in groupby_set]
        grouped = self._obj.groupby(groupby_columns, sort=sort, observed=observed)
        grouped = grouped[selected_columns]
        RecordKey = namedtuple("RecordKey", groupby_columns)  # type: ignore
        for key, df in grouped:
            yield RecordKey(*ensure_list(key)), df

    def groupby_run_parallel(
        self,
        groupby_columns: List[str],
        selected_columns: Optional[List[str]] = None,
        *,
        sort: bool = True,
        observed: bool = True,
        func: Optional[Callable] = None,
        jobs: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> List[Any]:
        """Call groupby_iter and apply the given function in parallel, returning the results.

        Args:
            groupby_columns: see groupby_iter.
            selected_columns: see groupby_iter.
            sort: see groupby_iter.
            observed: see groupby_iter.
            func: callable accepting the parameters: key (NamedTuple), df (pd.DataFrame)
            jobs: number of jobs (see run_parallel)
            backend: parallel backend (see run_parallel)

        Returns:
            list of results.
        """
        assert func is not None, "A callable must be specified."
        grouped = self.groupby_iter(
            groupby_columns=groupby_columns,
            selected_columns=selected_columns,
            sort=sort,
            observed=observed,
        )
        tasks_generator = (Task(partial(func, key=key, df=group_df)) for key, group_df in grouped)
        return run_parallel(tasks_generator, jobs=jobs, backend=backend)

    def groupby_apply_parallel(
        self,
        groupby_columns: List[str],
        selected_columns: Optional[List[str]] = None,
        *,
        sort: bool = True,
        observed: bool = True,
        func: Optional[Callable] = None,
        jobs: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> pd.DataFrame:
        """Call groupby_iter and apply the given function in parallel, returning a DataFrame.

        To work as expected, func should return a DataFrame and all the DataFrames should have
        the same index and columns.

        Still experimental.
        """
        results = self.groupby_run_parallel(
            groupby_columns=groupby_columns,
            selected_columns=selected_columns,
            sort=sort,
            observed=observed,
            func=func,
            jobs=jobs,
            backend=backend,
        )
        return pd.concat(results)


class ETLIndexAccessor:
    def __init__(self, pandas_obj):
        """Initialize the accessor."""
        self._obj = pandas_obj

    def iter(self):
        """Iterate over the index, yielding a namedtuple for each element.

        It can be used as an alternative to the pandas iteration over the index
        to yield named tuples instead of standard tuples.

        It works with both Indexes and MultiIndexes.
        """
        names = self._obj.names
        Index = namedtuple("Index", names, rename=True)
        for i in self._obj:
            yield Index(*ensure_list(i))


def register_accessors():
    """Register the accessors.

    It must be called once, before accessing the etl namespace.
    """
    L.info("Registering etl pandas accessors")
    # alternatively, the accessors could be registered with a decorator,
    # but this call is more explicit and it doesn't need any unused import
    pd.api.extensions.register_series_accessor("etl")(ETLSeriesAccessor)
    pd.api.extensions.register_dataframe_accessor("etl")(ETLDataFrameAccessor)
    pd.api.extensions.register_index_accessor("etl")(ETLIndexAccessor)


# TODO: deprecated, to be removed
ETLBaseAccessor.filter = ETLBaseAccessor.select  # type: ignore
ETLBaseAccessor.remove_condition = ETLBaseAccessor.remove_conditions  # type: ignore
ETLBaseAccessor.keep_condition = ETLBaseAccessor.keep_conditions  # type: ignore
ETLBaseAccessor.add_condition = ETLBaseAccessor.add_conditions  # type: ignore
ETLBaseAccessor.groupby_excluding = ETLBaseAccessor.groupby_except  # type: ignore
ETLDataFrameAccessor.grouped_by = ETLDataFrameAccessor.groupby_iter  # type: ignore
ETLDataFrameAccessor.group_run_parallel = ETLDataFrameAccessor.groupby_run_parallel  # type: ignore
ETLDataFrameAccessor.group_apply_parallel = ETLDataFrameAccessor.groupby_apply_parallel  # type: ignore
