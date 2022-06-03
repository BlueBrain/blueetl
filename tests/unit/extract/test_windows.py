from unittest.mock import Mock, PropertyMock

import pandas as pd
from pandas.testing import assert_frame_equal

from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    DURATION,
    OFFSET,
    SIMULATION,
    SIMULATION_ID,
    T_START,
    T_STOP,
    TRIAL,
    TRIAL_STEPS_LABEL,
    TRIAL_STEPS_VALUE,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract import windows as test_module
from blueetl.utils import ensure_dtypes


def test_windows_from_simulations():
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, SIMULATION: Mock(), CIRCUIT: Mock()},
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df

    mock_trial_steps_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {SIMULATION_ID: 0, CIRCUIT_ID: 0, TRIAL_STEPS_LABEL: "ts1", TRIAL_STEPS_VALUE: 150},
            ]
        )
    )
    trial_steps = Mock()
    type(trial_steps).df = mock_trial_steps_df

    config = {
        "windows": {
            "w1": {"bounds": [5000, 6000]},
            "w2": {
                "bounds": [0, 100],
                "initial_offset": 1000,
                "window_type": "custom_type_1",
                "n_trials": 2,
                "trial_steps_value": 100,
            },
            "w3": {
                "bounds": [0, 200],
                "initial_offset": 2000,
                "window_type": "custom_type_2",
                "n_trials": 2,
                "trial_steps_label": "ts1",
            },
        }
    }
    result = test_module.Windows.from_simulations(mock_simulations, trial_steps, config)

    expected_df = pd.DataFrame(
        [
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w1",
                TRIAL: 0,
                OFFSET: 0,
                T_START: 5000,
                T_STOP: 6000,
                DURATION: 1000,
                WINDOW_TYPE: "",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w2",
                TRIAL: 0,
                OFFSET: 1000,
                T_START: 0,
                T_STOP: 100,
                DURATION: 100,
                WINDOW_TYPE: "custom_type_1",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w2",
                TRIAL: 1,
                OFFSET: 1100,
                T_START: 0,
                T_STOP: 100,
                DURATION: 100,
                WINDOW_TYPE: "custom_type_1",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w3",
                TRIAL: 0,
                OFFSET: 2000,
                T_START: 0,
                T_STOP: 200,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_2",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w3",
                TRIAL: 1,
                OFFSET: 2150,
                T_START: 0,
                T_STOP: 200,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_2",
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Windows)
    assert_frame_equal(result.df, expected_df)
    assert mock_simulations_df.call_count == 1
