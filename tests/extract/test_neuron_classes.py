from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from blueetl.constants import CIRCUIT_ID, COUNT, GID, NEURON_CLASS
from blueetl.extract.neuron_classes import NeuronClasses


def _count_by_neuron_class(data):
    df = pd.DataFrame(data, columns=[CIRCUIT_ID, NEURON_CLASS, GID])
    return df.set_index([CIRCUIT_ID, NEURON_CLASS])[GID]


def test_from_neurons():
    mock_neurons = MagicMock()
    mock_neurons.count_by_neuron_class.return_value = _count_by_neuron_class(
        [
            [0, "L23_EXC", 123],
            [0, "L4_INH", 456],
        ]
    )
    result = NeuronClasses.from_neurons(
        mock_neurons,
        target="hex0",
        neuron_classes={
            "L23_EXC": {"layer": [2, 3], "synapse_class": ["EXC"]},
            "L4_INH": {"layer": [4], "synapse_class": ["INH"]},
        },
    )
    assert mock_neurons.count_by_neuron_class.call_count == 1
    assert isinstance(result, NeuronClasses)
    expected_df = pd.DataFrame(
        [
            {
                CIRCUIT_ID: 0,
                NEURON_CLASS: "L23_EXC",
                COUNT: 123,
                "limit": None,
                "target": "hex0",
                "layer": [2, 3],
                "synapse_class": ["EXC"],
            },
            {
                CIRCUIT_ID: 0,
                NEURON_CLASS: "L4_INH",
                COUNT: 456,
                "limit": None,
                "target": "hex0",
                "layer": [4],
                "synapse_class": ["INH"],
            },
        ]
    )
    expected_df = expected_df.astype({CIRCUIT_ID: np.int16, NEURON_CLASS: "category"})
    assert_frame_equal(result.df, expected_df)


def test_from_neurons_without_neurons():
    mock_neurons = MagicMock()
    mock_neurons.count_by_neuron_class.return_value = _count_by_neuron_class([])
    with pytest.raises(RuntimeError, match="All neuron classes are empty"):
        NeuronClasses.from_neurons(
            mock_neurons,
            target="atarget",
            neuron_classes={"aclass": {"region": "any"}},
        )
    assert mock_neurons.count_by_neuron_class.call_count == 1
