"""Parallelization utilities."""

import logging
from collections import namedtuple
from collections.abc import Callable, Iterator
from functools import partial
from types import SimpleNamespace
from typing import Any, NamedTuple

import pandas as pd
from blueetl_core.parallel import Task, run_parallel
from blueetl_core.utils import CachedDataFrame

from blueetl.constants import CIRCUIT_ID, SIMULATION_ID

L = logging.getLogger(__name__)


def _unique_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the unique rows of the given DataFrame.

    - Only the specified columns are used to determine the uniqueness of the rows.
    - If a column is not present in the DataFrame, it's ignored.
    """
    if any(df.index.names):
        # reset the index if any columns are contained in the index levels
        if levels := [level for level in columns if level in df.index.names]:
            df = df.reset_index(level=levels)
    # find the unique rows
    df = df[df.columns.intersection(columns)].drop_duplicates(ignore_index=True)
    L.info("Considering %s rows for columns %s", len(df), list(df.columns))
    return df


def _groups(df_list: list[pd.DataFrame], groupby: list[str]) -> pd.DataFrame:
    """Get all the possible groups and return a sorted merged dataframe.

    The result is similar to what you get joining all the DataFrames and using groupby
    on the result to get all the groups, but it should use less memory and may be faster
    with big DataFrames.

    Dataframes not containing any column or index level in groupby are ignored.
    """
    result = None
    for df in df_list:
        df = _unique_rows(df, groupby)
        if len(df) > 0:
            if result is None:
                result = df
            else:
                result = result.merge(df, how="left")
            if set(result.columns) == set(groupby):
                # all the groupby columns have already been considered
                break
    if result is None:
        raise RuntimeError("At least one filtering DataFrame must be specified")
    if diff := set(groupby) - set(result.columns):
        raise ValueError(f"Unknown columns in groupby: {sorted(diff)}")
    # ensure that the groups are sorted by the specified columns
    return result[groupby].sort_values(groupby, ignore_index=True)


def _func_generator(
    df_list: list[pd.DataFrame], groupby: list[str], func: Callable
) -> Iterator[Callable[[], Any]]:
    """Yield functions to be executed in a subprocess."""
    groups = _groups(df_list, groupby=groupby)
    caches = [CachedDataFrame(df) for df in df_list]
    L.info("Tasks to be executed: %s", len(groups))
    # for each group, yield a function that can be called in a subprocess
    for _, key in groups.etl.iter():
        filtered = [df.query(key._asdict(), ignore_unknown_keys=True) for df in caches]
        yield partial(func, key=key, df_list=filtered)


def merge_filter(
    df_list: list[pd.DataFrame],
    groupby: list[str],
    func: Callable[[NamedTuple, list[pd.DataFrame]], Any],
    parallel: bool = True,
) -> Iterator[Any]:
    """Merge the specified columns of the list of DataFrames, and call func for each combination.

    The merge operation is similar to a SQL left outer join.

    Args:
        df_list: list of DataFrames.
        groupby: list of columns to consider across the DataFrames.
        func: callback function accepting ``key: NamedTuple, df_list: list[pd.DataFrames]``,
            executed for each calculated combination of columns.
        parallel: True to call the callback function in subprocesses, False otherwise.

    Yields:
        values returned by the callback function.

    """
    func_generator = _func_generator(df_list=df_list, groupby=groupby, func=func)
    if parallel:
        yield from run_parallel(Task(f) for f in func_generator)
    else:
        yield from (f() for f in func_generator)


def merge_groupby(
    df_list: list[pd.DataFrame], groupby: list[str], parallel=True
) -> Iterator[tuple[NamedTuple, pd.DataFrame]]:
    """Merge a list of DataFrames, group by the given keys, and yield keys and groups.

    The merge operation is similar to a SQL left outer join.

    If parallel is True, the dataframes are filtered in the main process and merged in subprocesses.
    If parallel is False, the dataframes are merged in the same process.
    """

    def _func(key: NamedTuple, df_list: list[pd.DataFrame]) -> tuple[NamedTuple, pd.DataFrame]:
        # executed in a subprocess
        merged = df_list[0]
        for df in df_list[1:]:
            merged = merged.merge(df, how="left", copy=False)
        return key, merged

    yield from merge_filter(df_list=df_list, groupby=groupby, func=_func, parallel=parallel)


def call_by_simulation(
    simulations: pd.DataFrame,
    dataframes_to_filter: dict[str, pd.DataFrame],
    func: Callable,
    how: str = "dataframe",
) -> list[Any]:
    """Execute the given function in parallel, one task for each simulation.

    Args:
        simulations: DataFrame of simulations.
        dataframes_to_filter: dict of DataFrames to filter by simulation_id and/or circuit_id,
            and passed to each subprocess.
        func: callable called for each simulation, accepting:
            - simulation_row: NamedTuple (or the type specified with the `how` parameter)
            - filtered_dataframes: dict of DataFrames filtered by simulation_id and/or circuit_id
            If the function has other parameters, they can be applied using `functools.partials`,
            and they will be serialized and passed unchanged to the subprocesses.
        jobs: number of jobs (see run_parallel)
        backend: parallel backend (see run_parallel)
        how: format the `simulation_row` parameter passed to the func callback.
            It can be one of "namespace", "namedtuple", "dict", "series", "dataframe".

    Returns:
        list of results
    """

    def _func(key: NamedTuple, df_list: list[pd.DataFrame]) -> tuple[NamedTuple, pd.DataFrame]:
        # pylint: disable=unused-argument
        # executed in a subprocess
        simulation_row = convert_row(df_list[0].reset_index())
        filtered_dataframes = dict(zip(dataframes_to_filter, df_list[1:]))
        return func(simulation_row=simulation_row, filtered_dataframes=filtered_dataframes)

    convert_row = {
        "namedtuple": lambda df: namedtuple("Values", df.columns)(**df.iloc[0]),
        "namespace": lambda df: SimpleNamespace(**df.iloc[0]),
        "dict": lambda df: df.iloc[0].to_dict(),
        "series": lambda df: df.iloc[0],
        "dataframe": lambda df: df,
    }[how]
    return list(
        merge_filter(
            df_list=[simulations, *dataframes_to_filter.values()],
            groupby=[SIMULATION_ID, CIRCUIT_ID],
            func=_func,
            parallel=True,
        )
    )
