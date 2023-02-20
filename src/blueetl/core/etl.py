"""Pandas accessors."""
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Iterator
from functools import partial
from typing import Any, Callable, Generic, NamedTuple, Optional, TypeVar, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from blueetl.core import L
from blueetl.core.parallel import Task, run_parallel
from blueetl.core.utils import query_frame, query_series
from blueetl.utils import ensure_list

# Naming conventions
#  level: number, or name of the level in the MultiIndex
#  condition: name of the level (e.g. seed), similar to level, but it cannot be numeric
#  labels: possible values for a specific level of the index

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)
PandasGroupByT = TypeVar("PandasGroupByT", SeriesGroupBy, DataFrameGroupBy)


class ETLBaseAccessor(ABC, Generic[PandasT, PandasGroupByT]):
    """Accessor with methods common to Series and DataFrame."""

    def __init__(self, pandas_obj: PandasT) -> None:
        """Initialize the accessor."""
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: PandasT) -> None:
        """Validate the wrapped object."""
        # assert all(obj.index.names), "All the index levels must have a name"
        # assert len(set(obj.index.names)) == obj.index.nlevels, "The level names must be unique"

    def conditions(self) -> list[str]:
        """Names for each of the index levels."""
        return self._obj.index.names

    def complementary_conditions(self, conditions: Union[str, list[str]]) -> list[str]:
        """Return the difference between the object conditions and the specified conditions.

        Args:
            conditions: single condition or list of conditions used to calculate the difference.
        """
        conditions = ensure_list(conditions)
        # TODO: raise a KeyError if any condition is not found in self.conditions?
        return self._obj.index.names.difference(conditions)

    def labels(self, conditions: Optional[list[str]] = None) -> list[pd.Index]:
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

    def remove_conditions(self, conditions: Union[str, list[str]]) -> PandasT:
        """Remove one or more conditions.

        Args:
            conditions: single condition or list of conditions to remove.

        Returns:
            resulting Series or DataFrame.
        """
        return self._obj.droplevel(conditions, axis=0)

    def keep_conditions(self, conditions: Union[str, list[str]]) -> PandasT:
        """Remove the conditions not specified.

        Args:
            conditions: single condition or list of conditions to keep.

        Returns:
            resulting Series or DataFrame.
        """
        return self._obj.droplevel(self.complementary_conditions(conditions), axis=0)

    def add_conditions(
        self,
        conditions: Union[str, list[str]],
        values: Any,
        inner: bool = False,
        drop: bool = False,
        dtypes: Any = None,
    ) -> PandasT:
        """Add one or multiple conditions into the outermost or innermost position.

        Args:
            conditions: single conditions or list of conditions to be added.
            values: single value or list of values, one for each condition.
            inner (bool): if True, add the conditions in the innermost position.
            drop (bool): if True, drop the existing conditions.
            dtypes: if not None, it's used to enforce the dtype of the new levels.
                It can be a single dtype, or a list of dtypes, one for each condition.
                Examples: int, float, "category"...

        Returns:
            resulting Series or DataFrame.
        """
        conditions = ensure_list(conditions)
        values = ensure_list(values)
        if len(conditions) != len(values):
            raise ValueError("Conditions and values must have the same length")
        result = pd.concat([self._obj], axis="index", keys=[tuple(values)], names=conditions)
        if dtypes:
            dtypes = ensure_list(dtypes)
            if len(conditions) != len(dtypes):
                raise ValueError("Conditions and dtypes must have the same length")
            result.index = result.index.etl.astype(dict(zip(conditions, dtypes)))
        if drop:
            # levels to be dropped, for example: [-3, -2, -1]
            levels = list(range(-self._obj.index.nlevels, 0))
            result = result.droplevel(levels)
        elif inner:
            # rotate the levels: (0 1) 2 3 4 5 -> 2 3 4 5 (0 1)
            order = list(range(result.index.nlevels))
            order = order[len(conditions) :] + order[: len(conditions)]
            result = result.reorder_levels(order)
        return result

    def select(self, drop_level: bool = True, **kwargs) -> PandasT:
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

    def groupby_except(self, conditions: Union[str, list[str]], *args, **kwargs) -> PandasGroupByT:
        """Group by all the conditions except for the ones specified.

        Args:
            conditions: single condition or list of conditions to be excluded from the groupby
        """
        complementary_conditions = self.complementary_conditions(conditions)
        return self._obj.groupby(complementary_conditions, *args, **kwargs)

    def pool(self, conditions: Union[str, list[str]], func: Callable) -> PandasT:
        """Remove one or more conditions grouping by the remaining conditions.

        Args:
            conditions: single condition or list of conditions to be removed from the index.
            func: function that should accept a single element.
                If the returned value is a Series, it will be used as an additional level
                in the MultiIndex of the returned object.
        """
        return self.groupby_except(conditions).apply(func)

    @abstractmethod
    def _query_dict(self, query: dict) -> PandasT:
        """Given a query dictionary, return the filtered Series or DataFrame."""

    def q(self, _query: Optional[dict] = None, /, **params) -> PandasT:
        """Given a query dict or some query parameters, return the filtered Series or DataFrame.

        Filter by columns (for DataFrames) and index names (for both Series and DataFrames).
        If a name is present in both columns and index names, only the column is considered.

        All the keys are combined in AND, while the values can be scalar, list, or dict.

        * If value is a scalar, the exact value will be matched.
        * If value is a list, the values in the list are considered in OR.
        * If value is a dict, a more advanced filter can be specified using the
            supported operators: ``eq, ne, le, lt, ge, gt, isin``.

        Query and named params cannot be specified together.
        If they are both empty or missing, the original Series or DataFrame is returned.

        This method is similar to the standard `query` method for pandas DataFrames, but it accepts
        a dict instead of a string, and has some limitations on the possible filters to be applied.

        Args:
            _query: query dictionary, where the keys are columns or index levels, and the values
                can be scalar, list, or dict values.
            **params: named params can be specified as an alternative to the _query dictionary.

        Examples:
            * {"mtype": "SO_BP", "etype": "cNAC"} -> mtype == SO_BP AND etype == cNAC
            * {"mtype": ["SO_BP", "SP_AA"]} -> mtype == SO_BP OR mtype == SP_AA
            * {"gid": {"ge": 3, "lt": 8} -> gid >= 3 AND gid < 8
        """
        if _query and params:
            raise ValueError("Query and params cannot be specified together")
        return self._query_dict(_query or params)

    def one(self, _query: Optional[dict] = None, /, **params) -> Any:
        """Execute the query and return the unique resulting record."""
        result = self.q(_query, **params)
        if len(result) != 1:
            raise RuntimeError(f"The query returned {len(result)} records instead of 1.")
        return result.iloc[0]

    def first(self, _query: Optional[dict] = None, /, **params) -> Any:
        """Execute the query and return the first resulting record."""
        result = self.q(_query, **params)
        if len(result) == 0:
            raise RuntimeError("The query returned 0 records.")
        return result.iloc[0]


class ETLSeriesAccessor(ETLBaseAccessor[pd.Series, SeriesGroupBy]):
    """Series accessor."""

    def unpool(self, func: Callable) -> pd.Series:
        """Apply the given function to the object elements and add a condition to the index.

        Args:
            func: function that should accept a single element and return a Series object.
                The name of that Series will be used as the name of the new level
                in the MultiIndex of the returned object.
        """
        return self._obj.apply(func).stack()

    def iter(self) -> Iterator[tuple[NamedTuple, Any]]:
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned named_index is a namedtuple representing the value of the index.
        The returned value is the actual value of each element of the series.
        """
        return zip(self._obj.index.etl.iter(), iter(self._obj))

    def iterdict(self) -> Iterator[tuple[dict, Any]]:
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned named_index is a dict representing the value of the index.
        The returned value is the actual value of each element of the series.

        This method can be used as an alternative to ``iter`` when:

        - The index names contain invalid identifiers, or
        - it's more convenient to work with dictionaries.

        Valid identifiers consist of letters, digits, and underscores but do not start
        with a digit or underscore and cannot be a Python keyword.
        """
        return zip(self._obj.index.etl.iterdict(), iter(self._obj))

    def _query_dict(self, query: dict) -> pd.Series:
        """Given a query dictionary, return the Series filtered by index."""
        return query_series(self._obj, query)


class ETLDataFrameAccessor(ETLBaseAccessor[pd.DataFrame, DataFrameGroupBy]):
    """DataFrame accessor."""

    def iter(self) -> Iterator[tuple[NamedTuple, NamedTuple]]:
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned ``named_index`` is a namedtuple representing the value of the index.
        The returned ``value`` is a namedtuple as returned by pandas.DataFrame.itertuples.
        """
        return zip(self._obj.index.etl.iter(), self._obj.itertuples(index=False, name="Values"))

    def iterdict(self) -> Iterator[tuple[dict, dict]]:
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned ``named_index`` is a dict representing the value of the index.
        The returned ``value`` is a dict containing a key for each column.

        This method can be used as an alternative to ``iter`` when:

        - The column or index names contain invalid identifiers, or
        - it's more convenient to work with dictionaries.

        Valid identifiers consist of letters, digits, and underscores but do not start
        with a digit or underscore and cannot be a Python keyword.
        """
        columns = self._obj.columns
        for named_index, value in zip(
            self._obj.index.etl.iterdict(), self._obj.itertuples(index=False)
        ):
            yield named_index, dict(zip(columns, value))

    def _query_dict(self, query: dict) -> pd.DataFrame:
        """Given a query dictionary, return the DataFrame filtered by columns and index."""
        return query_frame(self._obj, query)

    def groupby_iter(
        self,
        groupby_columns: list[str],
        selected_columns: Optional[list[str]] = None,
        sort: bool = True,
        observed: bool = True,
    ) -> Iterator[tuple[NamedTuple, pd.DataFrame]]:
        """Group the dataframe by columns and yield each record as a tuple (key, df).

        It can be used as a replacement for the iteration over df.groupby, but:
            - the yielded keys are namedtuples, instead of tuples
            - the yielded dataframes contain only the selected columns, if specified

        Args:
            groupby_columns: list of column names to group by.
            selected_columns: list of column names to be included in the yielded dataframes.
                If None, all the columns are included.
            sort: Sort group keys.
            observed: This only applies if any of the groupers are Categoricals.
                If True: only show observed values for categorical groupers.
                If False: show all values for categorical groupers.

        Yields:
            a tuple (key, df), where key is a namedtuple with the grouped columns
        """
        # Workaround to avoid: FutureWarning: In a future version of pandas, a length 1 tuple will
        # be returned when iterating over a groupby with a grouper equal to a list of length 1.
        # Don't supply a list with a single grouper to avoid this warning.
        by = groupby_columns[0] if len(groupby_columns) == 1 else groupby_columns
        grouped = self._obj.groupby(by, sort=sort, observed=observed)
        if selected_columns:
            grouped = grouped[selected_columns]
        RecordKey = namedtuple("RecordKey", groupby_columns)  # type: ignore
        for key, df in grouped:
            yield RecordKey(*ensure_list(key)), df

    def groupby_run_parallel(
        self,
        groupby_columns: list[str],
        selected_columns: Optional[list[str]] = None,
        *,
        sort: bool = True,
        observed: bool = True,
        func: Optional[Callable] = None,
        jobs: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> list[Any]:
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
        groupby_columns: list[str],
        selected_columns: Optional[list[str]] = None,
        *,
        sort: bool = True,
        observed: bool = True,
        func: Optional[Callable] = None,
        jobs: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> pd.DataFrame:
        """Call groupby_iter and apply the given function in parallel, returning a DataFrame.

        To work as expected, func should return a DataFrame or a Series, and all the returned
        objects should have the same index and columns.

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
    """Index accessor."""

    def __init__(self, pandas_obj: pd.Index) -> None:
        """Initialize the accessor."""
        self._obj = pandas_obj

    def _mangle_names(self) -> list[str]:
        """Return the index names, replacing missing names with ilevel_N, where N is the level.

        The name ilevel_N is the same name used by convention in DataFrame.query to identify
        the index levels without a name in a MultiIndex.
        """
        return [
            name if name is not None else f"ilevel_{i}" for i, name in enumerate(self._obj.names)
        ]

    def iter(self) -> Iterator[NamedTuple]:
        """Iterate over the index, yielding a namedtuple for each element.

        It can be used as an alternative to the pandas iteration over the index
        to yield named tuples instead of standard tuples.

        It works with both Indexes and MultiIndexes.
        """
        names = self._mangle_names()
        Index = namedtuple("Index", names, rename=True)  # type: ignore
        for i in self._obj:
            yield Index(*ensure_list(i))

    def iterdict(self) -> Iterator[dict]:
        """Iterate over the index, yielding a dict for each element.

        This method can be used as an alternative to ``iter`` when:

        - The index names contain invalid identifiers, or
        - it's more convenient to work with dictionaries.

        It works with both Indexes and MultiIndexes.
        """
        names = self._mangle_names()
        for i in self._obj:
            yield dict(zip(names, ensure_list(i)))

    @property
    def dtypes(self) -> pd.Series:
        """Return the dtypes of the index."""
        if isinstance(self._obj, pd.MultiIndex):
            return self._obj.dtypes
        return pd.Series([self._obj.dtype], index=pd.Index([self._obj.name]))

    def astype(self, dtype) -> pd.Index:
        """Create a new Index with the given dtypes.

        Args:
            dtype: numpy dtype or pandas type, or dict of dtypes when applied to a MultiIndex.
                Any (u)int16 or (u)int32 dtype is considered as (u)int64,
                since Pandas doesn't have a corresponding Index type for them.

        Returns:
            a copy of index using the specified dtypes.
        """
        if not isinstance(self._obj, pd.MultiIndex):
            return self._obj.astype(dtype)
        if not isinstance(dtype, dict):
            raise TypeError("A dict of dtypes must be specified when working with a MultiIndex")
        if diff := set(dtype).difference(self._obj.names):
            raise NameError(f"Some names don't exist in the index levels: {sorted(diff)}")
        dtype_by_num = {i: dtype[name] for i, name in enumerate(self._obj.names) if name in dtype}
        return self._obj.set_levels(
            [self._obj.levels[i].astype(dt) for i, dt in dtype_by_num.items()],
            level=list(dtype_by_num),
        )


def register_accessors() -> None:
    """Register the accessors.

    It must be called once, before accessing the etl namespace.
    """
    L.info("Registering etl pandas accessors")
    # alternatively, the accessors could be registered with a decorator,
    # but this call is more explicit and it doesn't need any unused import
    pd.api.extensions.register_series_accessor("etl")(ETLSeriesAccessor)
    pd.api.extensions.register_dataframe_accessor("etl")(ETLDataFrameAccessor)
    pd.api.extensions.register_index_accessor("etl")(ETLIndexAccessor)
