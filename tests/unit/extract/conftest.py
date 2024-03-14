from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests.unit.utils import TEST_NODE_SETS_FILE


def _get_cells():
    """Return a DataFrame as returned by circuit.nodes[population].get()."""
    return pd.DataFrame(
        [
            {
                "layer": "1",
                "mtype": "L1_DAC",
                "etype": "cNAC",
                "region": "S1FL",
                "synapse_class": "INH",
                "x": 4497.1,
                "y": -1404.7,
                "z": -1710.8,
            },
            {
                "layer": "2",
                "mtype": "L2_TPC:A",
                "etype": "cADpyr",
                "region": "S1FL",
                "synapse_class": "EXC",
                "x": 4592.3,
                "y": -1351.1,
                "z": -1987.2,
            },
            {
                "layer": "4",
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


@pytest.fixture
def mock_circuit():
    """Simplified mock of circuit, providing only get() and ids() for a node population."""
    mock = MagicMock()
    mock.node_sets_file = str(TEST_NODE_SETS_FILE)
    df = _get_cells()
    # circuit.nodes[population]
    mock_population = mock.nodes.__getitem__.return_value
    # circuit.nodes[population].get()
    mock_population.get.return_value = df
    # circuit.nodes[population].ids()
    mock_population.ids.return_value = df.index.to_numpy()
    return mock


@pytest.fixture
def mock_simulation():
    """Simplified mock of simulation, providing only get() for a spikes population."""
    mock = MagicMock()
    # simulation.spikes[population]
    mock_population = mock.spikes.__getitem__.return_value
    # simulation.spikes[population].get() returns a pandas.Series
    # with the spiking node_ids indexed by sorted spike time
    mock_population.get.return_value = pd.Series(
        [200, 100, 200, 200],
        index=[100.1, 120.2, 120.2, 150.0],
    )
    return mock
