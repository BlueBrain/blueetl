from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pandas as pd
from bluepy.exceptions import BluePyError
from pandas.testing import assert_frame_equal

from blueetl.constants import CIRCUIT_ID, SIMULATION_ID
from blueetl.extract import simulations as test_module


def _get_mock_simulation():
    mock_circuit = Mock()
    mock_circuit.config = {
        "cells": "/path/to/cells",
        "morphologies": "/path/to/morphologies",
        "emodels": "/path/to/emodels",
        "connectome": "/path/to/connectome",
        "projections": {},
        "atlas": "/path/to/atlas",
    }
    mock_simulation = Mock()
    mock_simulation.circuit = mock_circuit
    return mock_simulation


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.side_effect = [mock_simulation0, mock_simulation1]
    config = Mock()
    config.to_pandas.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )
    result = test_module.Simulations.from_config(config)

    expected_df = pd.DataFrame(
        [
            {
                "simulation_path": "path1",
                "seed": 10,
                "Grating Orientation (degrees)": 0,
                "simulation_id": 0,
                "circuit_id": 0,
                "simulation": mock_simulation0,
                "circuit": mock_circuit0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": mock_simulation1,
                "circuit": mock_circuit0,
            },
        ]
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, SIMULATION_ID: np.int16})
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert config.to_pandas.call_count == 1
    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config_filtered_by_simulation_id(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.side_effect = [mock_simulation0, mock_simulation1]
    config = Mock()
    config.to_pandas.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )
    result = test_module.Simulations.from_config(config, simulation_ids={1})

    expected_df = pd.DataFrame(
        [
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": mock_simulation1,
                "circuit": mock_circuit0,
            },
        ],
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, SIMULATION_ID: np.int16})
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert config.to_pandas.call_count == 1
    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config_without_spikes(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.side_effect = [mock_simulation0, mock_simulation1]
    config = Mock()
    config.to_pandas.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )
    # mock the spikes attribute to simulate an incomplete simulation
    type(mock_simulation0).spikes = PropertyMock(side_effect=BluePyError)

    result = test_module.Simulations.from_config(config)

    expected_df = pd.DataFrame(
        [
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": mock_simulation1,
                "circuit": mock_circuit0,
            },
        ],
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, SIMULATION_ID: np.int16})
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert config.to_pandas.call_count == 1
    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.side_effect = [mock_simulation0, mock_simulation1]
    df = pd.DataFrame(
        [
            {
                "simulation_path": "path1",
                "seed": 10,
                "Grating Orientation (degrees)": 0,
                "simulation_id": 0,
                "circuit_id": 0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
            },
        ]
    )
    result = test_module.Simulations.from_pandas(df)
    expected_df = pd.DataFrame(
        [
            {
                "simulation_path": "path1",
                "seed": 10,
                "Grating Orientation (degrees)": 0,
                "simulation_id": 0,
                "circuit_id": 0,
                "simulation": mock_simulation0,
                "circuit": mock_circuit0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": mock_simulation1,
                "circuit": mock_circuit0,
            },
        ]
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, SIMULATION_ID: np.int16})
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


def test_simulations_to_pandas():
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    df = pd.DataFrame(
        [
            {
                "simulation_path": "path1",
                "seed": 10,
                "Grating Orientation (degrees)": 0,
                "simulation_id": 0,
                "circuit_id": 0,
                "simulation": mock_simulation0,
                "circuit": mock_circuit0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
                "simulation": mock_simulation1,
                "circuit": mock_circuit1,
            },
        ]
    )
    result_df = test_module.Simulations(df).to_pandas()
    expected_df = pd.DataFrame(
        [
            {
                "simulation_path": "path1",
                "seed": 10,
                "Grating Orientation (degrees)": 0,
                "simulation_id": 0,
                "circuit_id": 0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 0,
            },
        ]
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, SIMULATION_ID: np.int16})
    assert_frame_equal(result_df, expected_df)
    assert mock_circuit0 != mock_circuit1
