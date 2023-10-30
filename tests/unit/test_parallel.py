import itertools
import os
from collections.abc import Iterator
from functools import partial
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from blueetl_core.constants import BLUEETL_JOBLIB_JOBS
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from blueetl import parallel as test_module


def merge_groupby_classic(
    df_list: list[pd.DataFrame], groupby: list[str]
) -> Iterator[tuple[NamedTuple, pd.DataFrame]]:
    """Merge a list of DataFrames, group by the given keys, and yield keys and groups."""
    merged = df_list[0]
    for df in df_list[1:]:
        merged = merged.merge(df, how="left")
    yield from merged.groupby(groupby, dropna=False)


def test_merge_groupby():
    all_dataframes = {
        "simulations": pd.DataFrame(
            [
                {
                    "ca": 1.1,
                    "seed": 1,
                    "simulation_path": "/path/to/0",
                    "simulation_id": 0,
                    "circuit_id": 0,
                    "simulation": Mock(),
                    "circuit": Mock(),
                },
                {
                    "ca": 1.2,
                    "seed": 1,
                    "simulation_path": "/path/to/1",
                    "simulation_id": 1,
                    "circuit_id": 0,
                    "simulation": Mock(),
                    "circuit": Mock(),
                },
                {
                    "ca": 1.1,
                    "seed": 1,
                    "simulation_path": "/path/to/2",
                    "simulation_id": 2,
                    "circuit_id": 1,
                    "simulation": Mock(),
                    "circuit": Mock(),
                },
                {
                    "ca": 1.2,
                    "seed": 1,
                    "simulation_path": "/path/to/3",
                    "simulation_id": 3,
                    "circuit_id": 1,
                    "simulation": Mock(),
                    "circuit": Mock(),
                },
            ]
        ),
        "neurons": pd.DataFrame(
            [
                {"circuit_id": 0, "neuron_class": "L2_INH", "gid": 0, "neuron_class_index": 0},
                {"circuit_id": 0, "neuron_class": "L2_INH", "gid": 1, "neuron_class_index": 1},
                {"circuit_id": 0, "neuron_class": "L2_EXC", "gid": 2, "neuron_class_index": 0},
                {"circuit_id": 0, "neuron_class": "L2_EXC", "gid": 3, "neuron_class_index": 1},
            ]
        ),
        "by_neuron_class": pd.DataFrame(
            [
                {"simulation_id": 0, "circuit_id": 0, "neuron_class": "L2_INH", "f1": 111},
                {"simulation_id": 0, "circuit_id": 0, "neuron_class": "L2_EXC", "f1": 222},
                {"simulation_id": 1, "circuit_id": 0, "neuron_class": "L2_INH", "f1": 333},
                {"simulation_id": 1, "circuit_id": 0, "neuron_class": "L2_EXC", "f1": 444},
                {"simulation_id": 2, "circuit_id": 1, "neuron_class": "L2_INH", "f1": 555},
                {"simulation_id": 2, "circuit_id": 1, "neuron_class": "L2_EXC", "f1": 666},
                {"simulation_id": 3, "circuit_id": 1, "neuron_class": "L2_INH", "f1": 777},
                {"simulation_id": 3, "circuit_id": 1, "neuron_class": "L2_EXC", "f1": 888},
            ]
        ),
    }
    df_list = list(all_dataframes.values())
    groupby = ["simulation_id", "circuit_id", "neuron_class"]

    result = test_module.merge_groupby(df_list, groupby=groupby, parallel=False)
    expected = merge_groupby_classic(df_list, groupby=groupby)

    for result_item, expected_item in itertools.zip_longest(result, expected):
        assert isinstance(result_item, tuple)
        assert isinstance(expected_item, tuple)
        result_key, result_group = result_item
        expected_key, expected_group = expected_item
        assert_array_equal(result_key, expected_key)
        assert isinstance(result_group, pd.DataFrame)
        assert isinstance(expected_group, pd.DataFrame)
        # Reset the index because the index is ignored in merge_groupby
        expected_group = expected_group.reset_index(drop=True)
        # dtype may be different because int64 can be converted to float64 when some values are nan
        assert_frame_equal(result_group, expected_group, check_dtype=False)


@pytest.mark.parametrize("how", ["namespace", "namedtuple", "dict", "series", "dataframe"])
@patch.dict(os.environ, {BLUEETL_JOBLIB_JOBS: "1"})
def test_call_by_simulation(how):
    simulations = pd.DataFrame(
        [
            {
                "ca": 1.1,
                "seed": 1,
                "simulation_path": "/path/to/0",
                "simulation_id": 0,
                "circuit_id": 0,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.2,
                "seed": 1,
                "simulation_path": "/path/to/1",
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.1,
                "seed": 1,
                "simulation_path": "/path/to/2",
                "simulation_id": 2,
                "circuit_id": 1,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.2,
                "seed": 1,
                "simulation_path": "/path/to/3",
                "simulation_id": 3,
                "circuit_id": 1,
                "simulation": Mock(),
                "circuit": Mock(),
            },
        ]
    )
    all_dataframes = {
        "neurons": pd.DataFrame(
            [
                {"circuit_id": 0, "neuron_class": "L2_INH", "gid": 0, "neuron_class_index": 0},
                {"circuit_id": 0, "neuron_class": "L2_INH", "gid": 1, "neuron_class_index": 1},
                {"circuit_id": 0, "neuron_class": "L2_EXC", "gid": 2, "neuron_class_index": 0},
                {"circuit_id": 0, "neuron_class": "L2_EXC", "gid": 3, "neuron_class_index": 1},
            ]
        ),
        "by_neuron_class": pd.DataFrame(
            [
                {"simulation_id": 0, "circuit_id": 0, "neuron_class": "L2_INH", "f1": 111},
                {"simulation_id": 0, "circuit_id": 0, "neuron_class": "L2_EXC", "f1": 222},
                {"simulation_id": 1, "circuit_id": 0, "neuron_class": "L2_INH", "f1": 333},
                {"simulation_id": 1, "circuit_id": 0, "neuron_class": "L2_EXC", "f1": 444},
                {"simulation_id": 2, "circuit_id": 1, "neuron_class": "L2_INH", "f1": 555},
                {"simulation_id": 2, "circuit_id": 1, "neuron_class": "L2_EXC", "f1": 666},
                {"simulation_id": 3, "circuit_id": 1, "neuron_class": "L2_INH", "f1": 777},
                {"simulation_id": 3, "circuit_id": 1, "neuron_class": "L2_EXC", "f1": 888},
            ]
        ).set_index(["simulation_id", "circuit_id", "neuron_class"]),
    }

    simulation_row_type = {
        "namespace": SimpleNamespace,
        "namedtuple": tuple,
        "dict": dict,
        "series": pd.Series,
        "dataframe": pd.DataFrame,
    }[how]
    simulation_row_get = {
        "namespace": lambda row, attr: getattr(row, attr),
        "namedtuple": lambda row, attr: getattr(row, attr),
        "dict": lambda row, attr: row[attr],
        "series": lambda row, attr: row.at[attr],
        "dataframe": lambda row, attr: row.at[0, attr],
    }[how]

    def func(simulation_row, filtered_dataframes, param1, param2):
        assert isinstance(simulation_row, simulation_row_type)
        assert isinstance(filtered_dataframes, dict)
        _df = filtered_dataframes["by_neuron_class"]
        assert isinstance(_df, pd.DataFrame)
        simulation_id = simulation_row_get(simulation_row, "simulation_id")
        assert_array_equal(_df.index.unique("simulation_id"), [simulation_id])
        return [simulation_id, sum(_df["f1"]), param1, param2]

    result = test_module.call_by_simulation(
        simulations=simulations,
        dataframes_to_filter=all_dataframes,
        func=partial(func, param1=1, param2=2),
        how=how,
    )

    assert result == [[0, 333, 1, 2], [1, 777, 1, 2], [2, 1221, 1, 2], [3, 1665, 1, 2]]
