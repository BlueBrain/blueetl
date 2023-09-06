import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pandas as pd
from blueetl_core.constants import BLUEETL_JOBLIB_JOBS
from pandas.testing import assert_frame_equal

from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    COUNT,
    DURATION,
    GID,
    GIDS,
    LIMIT,
    NEURON_CLASS,
    NEURON_CLASS_INDEX,
    NODE_SET,
    OFFSET,
    POPULATION,
    QUERY,
    SIMULATION,
    SIMULATION_ID,
    T_START,
    T_STEP,
    T_STOP,
    TRIAL,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract import soma_report as test_module
from blueetl.utils import ensure_dtypes


def _get_soma_report(group=None, t_start=None, t_stop=None, t_step=None):
    """Return a DataFrame as returned by simulation.reports["soma"]["pop"].get()."""
    df = pd.DataFrame(
        {
            100: [-72.1, -72.3, -72.5],
            200: [-70.7, -70.8, -71.0],
            300: [-69.5, -69.8, -70.0],
        },
        index=pd.Index([19.5, 20, 20.5], name="time"),
    )
    return df.etl.q(time={"ge": t_start, "lt": t_stop})[list(group)]


@patch.dict(os.environ, {BLUEETL_JOBLIB_JOBS: "1"})
def test_soma_report_from_simulations():
    mock_circuit = MagicMock()
    mock_sim = MagicMock()
    _report_by_type = mock_sim.reports.__getitem__.return_value
    _report_by_pop = _report_by_type.__getitem__.return_value
    _report_by_pop.get.side_effect = _get_soma_report
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, SIMULATION: mock_sim, CIRCUIT: mock_circuit},
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df

    mock_neurons_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {CIRCUIT_ID: 0, NEURON_CLASS: "L23_EXC", GID: 100, NEURON_CLASS_INDEX: 0},
                {CIRCUIT_ID: 0, NEURON_CLASS: "L23_EXC", GID: 200, NEURON_CLASS_INDEX: 1},
                {CIRCUIT_ID: 0, NEURON_CLASS: "L4_EXC", GID: 300, NEURON_CLASS_INDEX: 0},
            ]
        )
    )
    mock_neurons = Mock()
    type(mock_neurons).df = mock_neurons_df

    mock_windows_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {
                    SIMULATION_ID: 0,
                    CIRCUIT_ID: 0,
                    WINDOW: "w1",
                    TRIAL: 0,
                    OFFSET: 20,
                    T_START: 0,
                    T_STOP: 100,
                    T_STEP: 0,
                    DURATION: 100,
                    WINDOW_TYPE: "spontaneous",
                },
            ]
        )
    )
    mock_windows = Mock()
    type(mock_windows).df = mock_windows_df

    mock_neuron_classes_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {
                    CIRCUIT_ID: 0,
                    NEURON_CLASS: "L23_EXC",
                    COUNT: 2,
                    LIMIT: None,
                    POPULATION: "thalamus_neurons",
                    NODE_SET: None,
                    GIDS: None,
                    QUERY: "{}",
                },
                {
                    CIRCUIT_ID: 0,
                    NEURON_CLASS: "L4_EXC",
                    COUNT: 1,
                    LIMIT: None,
                    POPULATION: "thalamus_neurons",
                    NODE_SET: None,
                    GIDS: None,
                    QUERY: "{}",
                },
            ]
        )
    )
    mock_neuron_classes = Mock()
    type(mock_neuron_classes).df = mock_neuron_classes_df

    result = test_module.SomaReport.from_simulations(
        simulations=mock_simulations,
        neurons=mock_neurons,
        windows=mock_windows,
        neuron_classes=mock_neuron_classes,
        name="soma",
    )

    expected_df = pd.DataFrame(
        [
            {
                "gid": 100,
                "time": 20.0,
                "value": -72.3,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L23_EXC",
            },
            {
                "gid": 100,
                "time": 20.5,
                "value": -72.5,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L23_EXC",
            },
            {
                "gid": 200,
                "time": 20.0,
                "value": -70.8,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L23_EXC",
            },
            {
                "gid": 200,
                "time": 20.5,
                "value": -71.0,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L23_EXC",
            },
            {
                "gid": 300,
                "time": 20.0,
                "value": -69.8,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L4_EXC",
            },
            {
                "gid": 300,
                "time": 20.5,
                "value": -70.0,
                "window": "w1",
                "simulation_id": 0,
                "circuit_id": 0,
                "neuron_class": "L4_EXC",
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.SomaReport)
    assert_frame_equal(result.df, expected_df)
    assert mock_simulations_df.call_count == 1
    assert mock_neurons_df.call_count == 1
    assert mock_windows_df.call_count == 1
