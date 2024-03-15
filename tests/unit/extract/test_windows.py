from unittest.mock import Mock, PropertyMock

import pandas as pd
from pandas.testing import assert_frame_equal

from blueetl.config.analysis_model import TrialStepsConfig, WindowConfig
from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    DURATION,
    OFFSET,
    SIMULATION,
    SIMULATION_ID,
    T_START,
    T_STEP,
    T_STOP,
    TRIAL,
    WINDOW,
    WINDOW_TYPE,
)
from blueetl.extract import windows as test_module
from blueetl.resolver import AttrResolver
from blueetl.utils import ensure_dtypes
from tests.unit.utils import TEST_NODE_SETS_FILE_EXTRA


def _myfunc1(spikes, params):
    """Calculate and return the cortical onset from spikes"""
    return 250


def _myfunc2(spikes, params):
    """Calculate and return the cortical onset from spikes"""
    return 750


def test_windows_from_simulations(mock_simulation, mock_circuit):
    mock_simulations_df = PropertyMock(
        return_value=pd.DataFrame(
            [
                {
                    SIMULATION_ID: 0,
                    CIRCUIT_ID: 0,
                    SIMULATION: mock_simulation,
                    CIRCUIT: mock_circuit,
                },
            ]
        )
    )
    mock_simulations = Mock()
    type(mock_simulations).df = mock_simulations_df

    root = Mock()
    root.soma.repo.windows.df = pd.DataFrame(
        {
            SIMULATION_ID: 0,
            CIRCUIT_ID: 0,
            WINDOW: "w9",
            TRIAL: 0,
            OFFSET: 1000,
            T_START: 0,
            T_STOP: 900,
            T_STEP: 0,
            DURATION: 900,
            WINDOW_TYPE: "custom_type_9",
        },
        index=[0],
    )
    root.soma.extraction = root.soma.repo
    resolver = AttrResolver(root)

    windows_config = {
        "w1": WindowConfig(bounds=[5000, 6000]),
        "w2": WindowConfig(
            **{
                "bounds": [0, 100],
                "initial_offset": 1000,
                "window_type": "custom_type_1",
                "n_trials": 2,
                "trial_steps_value": 100,
            }
        ),
        "w3": WindowConfig(
            **{
                "bounds": [0, 200],
                "initial_offset": 2000,
                "window_type": "custom_type_2",
                "trial_steps_list": [0, 150],
            }
        ),
        "w4": "soma.extraction.windows.w9#checksum",
        "w5": WindowConfig(
            **{
                "bounds": [0, 200],
                "initial_offset": 2000,
                "window_type": "custom_type_3",
                "trial_steps_label": "ts1",
            }
        ),
        "w6": WindowConfig(
            **{
                "bounds": [0, 200],
                "initial_offset": 2000,
                "window_type": "custom_type_4",
                "trial_steps_list": [50, 150],
                "trial_steps_label": "ts2",
            }
        ),
    }
    trial_steps_config = {
        "ts1": TrialStepsConfig(
            function=f"{__name__}._myfunc1",
            bounds=[-50, 150],
        ),
        "ts2": TrialStepsConfig(
            function=f"{__name__}._myfunc2",
            bounds=[-20, 200],
            node_set="ExtraLayer2",
            node_sets_file=TEST_NODE_SETS_FILE_EXTRA,
        ),
    }

    result = test_module.Windows.from_simulations(
        simulations=mock_simulations,
        windows_config=windows_config,
        trial_steps_config=trial_steps_config,
        resolver=resolver,
    )

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
                T_STEP: 0,
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
                T_STEP: 0,
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
                T_STEP: 0,
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
                T_STEP: 0,
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
                T_STEP: 0,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_2",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w4",
                TRIAL: 0,
                OFFSET: 1000,
                T_START: 0,
                T_STOP: 900,
                T_STEP: 0,
                DURATION: 900,
                WINDOW_TYPE: "custom_type_9",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w5",
                TRIAL: 0,
                OFFSET: 2250,
                T_START: 0,
                T_STOP: 200,
                T_STEP: 0,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_3",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w6",
                TRIAL: 0,
                OFFSET: 2800,
                T_START: 0,
                T_STOP: 200,
                T_STEP: 0,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_4",
            },
            {
                SIMULATION_ID: 0,
                CIRCUIT_ID: 0,
                WINDOW: "w6",
                TRIAL: 1,
                OFFSET: 2900,
                T_START: 0,
                T_STOP: 200,
                T_STEP: 0,
                DURATION: 200,
                WINDOW_TYPE: "custom_type_4",
            },
        ]
    )
    expected_df = ensure_dtypes(expected_df)
    assert isinstance(result, test_module.Windows)
    assert_frame_equal(result.df, expected_df)
    assert mock_simulations_df.call_count == 1
