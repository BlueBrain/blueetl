import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from blueetl.config.analysis_model import NeuronClassConfig
from blueetl.constants import CIRCUIT_ID, COUNT, GID, NEURON_CLASS
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.utils import ensure_dtypes


def _count_by_neuron_class(data):
    """Return a Series as returned by neurons.count_by_neuron_class()."""
    df = pd.DataFrame(data, columns=[CIRCUIT_ID, NEURON_CLASS, GID])
    return df.set_index([CIRCUIT_ID, NEURON_CLASS])[GID]


def test_neuron_classes_from_neurons():
    mock_neurons = MagicMock()
    mock_neurons.count_by_neuron_class.return_value = _count_by_neuron_class(
        [
            [0, "L23_EXC", 123],
            [0, "L4_INH", 456],
        ]
    )
    result = NeuronClasses.from_neurons(
        mock_neurons,
        neuron_classes={
            "L23_EXC": NeuronClassConfig(
                **{
                    "population": "thalamus_neurons",
                    "query": {"layer": ["2", "3"], "synapse_class": ["EXC"]},
                }
            ),
            "L4_INH": NeuronClassConfig(
                **{
                    "population": "thalamus_neurons",
                    "query": {"layer": ["4"], "synapse_class": ["INH"]},
                }
            ),
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
                "population": "thalamus_neurons",
                "node_set": None,
                "gids": None,
                "query": json.dumps({"layer": ["2", "3"], "synapse_class": ["EXC"]}),
            },
            {
                CIRCUIT_ID: 0,
                NEURON_CLASS: "L4_INH",
                COUNT: 456,
                "limit": None,
                "population": "thalamus_neurons",
                "node_set": None,
                "gids": None,
                "query": json.dumps({"layer": ["4"], "synapse_class": ["INH"]}),
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert_frame_equal(result.df, expected_df)


def test_neuron_classes_from_neurons_without_neurons():
    mock_neurons = MagicMock()
    mock_neurons.count_by_neuron_class.return_value = _count_by_neuron_class([])
    with pytest.raises(RuntimeError, match="No data extracted to NeuronClasses"):
        NeuronClasses.from_neurons(
            mock_neurons,
            neuron_classes={
                "aclass": NeuronClassConfig(
                    **{"population": "thalamus_neurons", "query": {"region": "any"}}
                )
            },
        )
    assert mock_neurons.count_by_neuron_class.call_count == 1
