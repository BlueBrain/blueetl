from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import pytest
from bluepy.exceptions import BluePyError
from pandas.testing import assert_frame_equal

from blueetl.extract import simulations as test_module
from blueetl.utils import ensure_dtypes


def _get_mock_simulation(n=0):
    mock_circuit = Mock()
    mock_circuit.config = {
        "cells": f"/path/to/cells/{n}",
        "morphologies": f"/path/to/morphologies/{n}",
        "emodels": f"/path/to/emodels/{n}",
        "connectome": f"/path/to/connectome/{n}",
        "projections": {},
        "atlas": f"/path/to/atlas/{n}",
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
    expected_df = ensure_dtypes(expected_df)
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
    expected_df = ensure_dtypes(expected_df)
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
    # mock the spikes property to simulate an incomplete simulation
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
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert config.to_pandas.call_count == 1
    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_complete_campaign(mock_simulation_class):
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
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_incomplete_campaign(mock_simulation_class):
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
                "simulation_id": 10,
                "circuit_id": 5,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 11,
                "circuit_id": 5,
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
                "simulation_id": 10,
                "circuit_id": 5,
                "simulation": mock_simulation0,
                "circuit": mock_circuit0,
            },
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 11,
                "circuit_id": 5,
                "simulation": mock_simulation1,
                "circuit": mock_circuit0,
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_inconsistent_campaign(mock_simulation_class):
    # the mocked simulations have different circuit_hash,
    # the dataframe simulations have the same circuit_id
    mock_simulation0 = _get_mock_simulation(0)
    mock_simulation1 = _get_mock_simulation(1)
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
    with pytest.raises(RuntimeError, match="Inconsistent simulations"):
        test_module.Simulations.from_pandas(df)

    assert mock_simulation_class.call_count == 2
    assert mock_circuit0 != mock_circuit1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_filtered_by_simulation_id(mock_simulation_class):
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
    result = test_module.Simulations.from_pandas(df, simulation_ids={1})
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
        ]
    )
    expected_df = ensure_dtypes(expected_df)
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
    expected_df = ensure_dtypes(expected_df)
    assert_frame_equal(result_df, expected_df)
    assert mock_circuit0 != mock_circuit1
