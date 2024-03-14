from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from blueetl.campaign.config import SimulationCampaign
from blueetl.extract import simulations as test_module
from blueetl.utils import ensure_dtypes


def _get_mock_simulation(n=0, exists=True, is_complete=True):
    mock_circuit = Mock()
    mock_circuit.checksum.return_value = n
    mock_simulation = Mock()
    mock_simulation.circuit = mock_circuit
    mock_simulation.exists.return_value = exists
    mock_simulation.is_complete.return_value = is_complete
    return mock_simulation


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
    config = MagicMock(SimulationCampaign)
    config.get.return_value = pd.DataFrame(
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

    assert config.get.call_count == 1
    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config_filtered_by_simulation_id(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
    config = MagicMock(SimulationCampaign)
    config.get.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )
    result = test_module.Simulations.from_config(config, query={"simulation_id": [1]})

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

    assert config.get.call_count == 1
    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config_without_spikes(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation(is_complete=False)
    mock_simulation1 = _get_mock_simulation(is_complete=True)
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
    config = MagicMock(SimulationCampaign)
    config.get.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )

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

    assert config.get.call_count == 1
    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_config_first_nonexistent(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation(exists=False)
    mock_simulation1 = _get_mock_simulation(exists=True)
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
    config = MagicMock(SimulationCampaign)
    config.get.return_value = pd.DataFrame(
        [
            {"simulation_path": "path1", "seed": 10, "Grating Orientation (degrees)": 0},
            {"simulation_path": "path2", "seed": 11, "Grating Orientation (degrees)": 45},
        ]
    )

    result = test_module.Simulations.from_config(config)

    expected_df = pd.DataFrame(
        [
            # circuit_id is 1 and not 0, and circuit is mock_circuit1 and not 0,
            # because the first simulation and circuit cannot be loaded and hashed
            {
                "simulation_path": "path2",
                "seed": 11,
                "Grating Orientation (degrees)": 45,
                "simulation_id": 1,
                "circuit_id": 1,
                "simulation": mock_simulation1,
                "circuit": mock_circuit1,
            },
        ],
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Simulations)
    assert_frame_equal(result.df, expected_df)

    assert config.get.call_count == 1
    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 0
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_complete_campaign(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
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

    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_incomplete_campaign(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
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

    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_load_inconsistent_campaign(mock_simulation_class):
    # the circuits of the mocked simulations have different circuit_hash,
    # while the circuit_ids in the dataframe are the same
    mock_simulation0 = _get_mock_simulation(0)
    mock_simulation1 = _get_mock_simulation(1)
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
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
    with pytest.raises(test_module.InconsistentSimulations, match="Inconsistent hash and id"):
        test_module.Simulations.from_pandas(df)

    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 0


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_first_nonexistent(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation(exists=False)
    mock_simulation1 = _get_mock_simulation(exists=True)
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]
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
    with pytest.raises(test_module.InconsistentSimulations, match="Inconsistent cache"):
        test_module.Simulations.from_pandas(df)

    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 0
    assert mock_simulation1.is_complete.call_count == 1


@patch(f"{test_module.__name__}.Simulation", autospec=True)
def test_simulations_from_pandas_filtered_by_simulation_id(mock_simulation_class):
    mock_simulation0 = _get_mock_simulation()
    mock_simulation1 = _get_mock_simulation()
    mock_circuit0 = mock_simulation0.circuit
    mock_circuit1 = mock_simulation1.circuit
    mock_simulation_class.from_file.side_effect = [mock_simulation0, mock_simulation1]

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
    result = test_module.Simulations.from_pandas(df, query={"simulation_id": [1]})
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

    assert mock_simulation_class.from_file.call_count == 2
    assert mock_circuit0 != mock_circuit1
    assert mock_simulation0.exists.call_count == 1
    assert mock_simulation1.exists.call_count == 1
    assert mock_simulation0.is_complete.call_count == 1
    assert mock_simulation1.is_complete.call_count == 1


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
    result_df = test_module.Simulations(df, cached=False, filtered=False).to_pandas()
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
