from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from blueetl.config.analysis_model import TrialStepsConfig
from blueetl.constants import CIRCUIT, CIRCUIT_ID, SIMULATION, SIMULATION_ID
from blueetl.extract import trial_steps as test_module
from blueetl.utils import ensure_dtypes


def _get_ids():
    """Return a numpy array as returned by circuit.nodes[population].ids()."""
    return np.array([100, 200, 300])


def _get_spikes0(gids, t_start, t_stop):
    """Return a Series as returned by simulation.spikes.get()."""
    spikes = pd.Series(
        [300, 100, 300, 200, 100, 100],
        index=pd.Index([56.05, 82.25, 441.85, 520.025, 609.425, 1167.525], name="times"),
        name="ids",
    )
    return spikes[spikes.isin(gids)][t_start:t_stop]


def _get_spikes1(gids, t_start, t_stop):
    """Return a Series as returned by simulation.spikes.get()."""
    spikes = pd.Series(
        [100, 300, 200, 200],
        index=pd.Index([183.05, 492.87, 899.00, 941.12], name="times"),
        name="ids",
    )
    return spikes[spikes.isin(gids)][t_start:t_stop]


def custom_func(spikes, trial_steps_params):
    return {"trial_steps_value": 12.0}


def test_trial_steps_from_simulations():
    np.random.seed(0)
    mock_circuit = MagicMock()
    mock_circuit.nodes.__getitem__.return_value.get.return_value = _get_ids()
    mock_sim0 = MagicMock()
    mock_sim0.spikes.__getitem__.return_value.get.side_effect = _get_spikes0
    mock_sim1 = MagicMock()
    mock_sim1.spikes.__getitem__.return_value.get.side_effect = _get_spikes1
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, SIMULATION: mock_sim0, CIRCUIT: mock_circuit},
                {SIMULATION_ID: 1, CIRCUIT_ID: 0, SIMULATION: mock_sim1, CIRCUIT: mock_circuit},
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df
    trial_steps_config = {
        "ts1": TrialStepsConfig(
            **{
                "population": "thalamus_neurons",
                "function": f"{__name__}.custom_func",
                "initial_offset": 7000,
                "bounds": [-50, 25],
            }
        ),
    }

    result = test_module.TrialSteps.from_simulations(
        simulations=mock_simulations, trial_steps_config=trial_steps_config
    )

    expected_df = pd.DataFrame(
        [
            {
                "simulation_id": 0,
                "circuit_id": 0,
                "trial_steps_label": "ts1",
                "trial_steps_value": 12.0,
            },
            {
                "simulation_id": 1,
                "circuit_id": 0,
                "trial_steps_label": "ts1",
                "trial_steps_value": 12.0,
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.TrialSteps)
    assert_frame_equal(result.df, expected_df)
    assert mock_circuit.nodes.__getitem__.return_value.ids.call_count == 2
    assert mock_simulations_df.call_count == 1
    assert mock_sim0.spikes.__getitem__.return_value.get.call_count == 1
    assert mock_sim1.spikes.__getitem__.return_value.get.call_count == 1
