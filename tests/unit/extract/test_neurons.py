from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION, SIMULATION_ID
from blueetl.extract.neurons import Neurons
from blueetl.utils import ensure_dtypes


def _get_cells():
    """Return a DataFrame as returned by circuit.nodes[population].get()."""
    return pd.DataFrame(
        [
            {
                "layer": 1,
                "mtype": "L1_DAC",
                "etype": "cNAC",
                "region": "S1FL",
                "synapse_class": "INH",
                "x": 4497.1,
                "y": -1404.7,
                "z": -1710.8,
            },
            {
                "layer": 2,
                "mtype": "L2_TPC:A",
                "etype": "cADpyr",
                "region": "S1FL",
                "synapse_class": "EXC",
                "x": 4592.3,
                "y": -1351.1,
                "z": -1987.2,
            },
            {
                "layer": 4,
                "mtype": "L4_BP",
                "etype": "cNAC",
                "region": "S1FL",
                "synapse_class": "INH",
                "x": 3953.9,
                "y": -1279.3,
                "z": -2143.9,
            },
        ],
        index=pd.Index([100, 200, 300], name="node_ids"),
    )


def test_neurons_from_simulations():
    # set the seed because the neurons may be chosen randomly
    np.random.seed(0)
    mock_circuit = Mock()
    mock_circuit = MagicMock()
    mock_circuit.nodes.__getitem__.return_value.get.return_value = _get_cells()
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, SIMULATION: Mock(), CIRCUIT: mock_circuit},
                {SIMULATION_ID: 1, CIRCUIT_ID: 0, SIMULATION: Mock(), CIRCUIT: mock_circuit},
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df
    neuron_classes = {
        "L1_INH": NeuronClassConfig(
            **{"population": "thalamus_neurons", "query": {"layer": [1], "synapse_class": ["INH"]}}
        ),
        "MY_GIDS": NeuronClassConfig(**{"population": "thalamus_neurons", "node_id": [200, 300]}),
        "EMPTY": NeuronClassConfig(**{"population": "thalamus_neurons", "query": {"layer": [999]}}),
        "LIMITED": NeuronClassConfig(
            **{"population": "thalamus_neurons", "query": {"synapse_class": ["INH"]}, "limit": 1}
        ),
    }
    result = Neurons.from_simulations(
        simulations=mock_simulations,
        neuron_classes=neuron_classes,
    )
    expected_df = pd.DataFrame(
        [
            {
                "circuit_id": 0,
                "neuron_class": "L1_INH",
                "gid": 100,
                "neuron_class_index": 0,
            },
            {
                "circuit_id": 0,
                "neuron_class": "MY_GIDS",
                "gid": 200,
                "neuron_class_index": 0,
            },
            {
                "circuit_id": 0,
                "neuron_class": "MY_GIDS",
                "gid": 300,
                "neuron_class_index": 1,
            },
            {
                "circuit_id": 0,
                "neuron_class": "LIMITED",
                "gid": 300,
                "neuron_class_index": 0,
            },
        ],
    )
    expected_df = expected_df.sort_values([CIRCUIT_ID, NEURON_CLASS, GID], ignore_index=True)
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, Neurons)
    assert_frame_equal(result.df, expected_df)
    assert mock_circuit.nodes.__getitem__.return_value.get.call_count == 1
    assert mock_simulations_df.call_count == 1


def test_neurons_from_simulations_without_neurons():
    mock_circuit = MagicMock()
    mock_circuit.nodes.__getitem__.return_value.get.return_value = _get_cells()
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, SIMULATION: Mock(), CIRCUIT: mock_circuit},
                {SIMULATION_ID: 1, CIRCUIT_ID: 0, SIMULATION: Mock(), CIRCUIT: mock_circuit},
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df
    neuron_classes = {
        "EMPTY": NeuronClassConfig(**{"population": "thalamus_neurons", "query": {"layer": [999]}}),
    }

    with pytest.raises(RuntimeError, match="No data extracted to Neurons"):
        Neurons.from_simulations(simulations=mock_simulations, neuron_classes=neuron_classes)
    assert mock_circuit.nodes.__getitem__.return_value.get.call_count == 1
    assert mock_simulations_df.call_count == 1


def test_neurons_count_by_neuron_class():
    df = pd.DataFrame(
        [
            {
                "circuit_id": 0,
                "neuron_class": "NC1",
                "gid": 100,
                "neuron_class_index": 0,
            },
            {
                "circuit_id": 0,
                "neuron_class": "NC1",
                "gid": 200,
                "neuron_class_index": 1,
            },
            {
                "circuit_id": 0,
                "neuron_class": "NC2",
                "gid": 100,
                "neuron_class_index": 0,
            },
            {
                "circuit_id": 1,
                "neuron_class": "NC1",
                "gid": 100,
                "neuron_class_index": 0,
            },
        ],
    )
    neurons = Neurons.from_pandas(df)
    result = neurons.count_by_neuron_class()
    expected_series = ensure_dtypes(
        pd.DataFrame(
            [
                [0, "NC1", 2],
                [0, "NC2", 1],
                [1, "NC1", 1],
            ],
            columns=[CIRCUIT_ID, NEURON_CLASS, GID],
        )
    ).set_index([CIRCUIT_ID, NEURON_CLASS])[GID]

    assert_series_equal(result, expected_series)
