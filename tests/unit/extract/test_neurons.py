from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, GID, NEURON_CLASS, SIMULATION, SIMULATION_ID
from blueetl.extract.neurons import Neurons
from blueetl.utils import ensure_dtypes
from tests.unit.utils import TEST_NODE_SETS_FILE_EXTRA


def test_neurons_from_simulations(mock_circuit):
    # set the seed because the neurons may be chosen randomly
    np.random.seed(0)
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
        "L1_INH": NeuronClassConfig.model_validate(
            {
                "population": "thalamus_neurons",
                "query": {"layer": ["1"], "synapse_class": ["INH"]},
            }
        ),
        "MY_GIDS": NeuronClassConfig(**{"population": "thalamus_neurons", "node_id": [200, 300]}),
        "EMPTY": NeuronClassConfig.model_validate(
            {"population": "thalamus_neurons", "query": {"layer": ["999"]}}
        ),
        "LIMITED": NeuronClassConfig.model_validate(
            {
                "population": "thalamus_neurons",
                "query": {"synapse_class": ["INH"]},
                "limit": 1,
                "node_set": "ExtraLayer2",
                "node_sets_file": TEST_NODE_SETS_FILE_EXTRA,
            }
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
    assert mock_circuit.nodes.__getitem__.return_value.get.call_count == 2
    assert mock_simulations_df.call_count == 1


def test_neurons_from_simulations_without_neurons(mock_circuit):
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
        "EMPTY": NeuronClassConfig(
            **{"population": "thalamus_neurons", "query": {"layer": ["999"]}}
        ),
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
