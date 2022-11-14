from functools import partial
from unittest.mock import Mock

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from blueetl import features as test_module


@pytest.mark.parametrize("how", ["namedtuple", "dict", "series", "dataframe"])
def test_call_by_simulation(how):
    simulations = pd.DataFrame(
        [
            {
                "ca": 1.1,
                "seed": 1,
                "simulation_path": "/path/to/0/BlueConfig",
                "simulation_id": 0,
                "circuit_id": 0,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.2,
                "seed": 1,
                "simulation_path": "/path/to/1/BlueConfig",
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.1,
                "seed": 1,
                "simulation_path": "/path/to/2/BlueConfig",
                "simulation_id": 2,
                "circuit_id": 1,
                "simulation": Mock(),
                "circuit": Mock(),
            },
            {
                "ca": 1.2,
                "seed": 1,
                "simulation_path": "/path/to/3/BlueConfig",
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
        "namedtuple": tuple,
        "dict": dict,
        "series": pd.Series,
        "dataframe": pd.DataFrame,
    }[how]
    simulation_row_get = {
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
        jobs=1,
        backend=None,
        how=how,
    )

    assert result == [[0, 333, 1, 2], [1, 777, 1, 2], [2, 1221, 1, 2], [3, 1665, 1, 2]]
