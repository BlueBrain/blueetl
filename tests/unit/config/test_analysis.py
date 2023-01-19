import pytest

import blueetl.config.analysis
from blueetl.config import analysis as test_module
from blueetl.config.analysis_model import FeaturesConfig
from tests.unit.utils import assert_not_duplicates


@pytest.mark.parametrize(
    "config_list, expected_list",
    [
        (
            [],
            [],
        ),
        (
            [
                {
                    "type": "multi",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "function": "module.func",
                    "params": {"PSD": {}},
                    "params_product": {
                        "ratio": [0.25, 0.5],
                        "CPDF": [
                            {"params": {"bin_size": 1}},
                            {"params": {"bin_size": 2}},
                        ],
                    },
                    "params_zip": {
                        "param1": [10, 20],
                        "param2": [11, 21],
                    },
                }
            ],
            [
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 1}},
                        "PSD": {},
                        "param1": 10,
                        "param2": 11,
                        "ratio": 0.25,
                    },
                    "suffix": "_0_0__0",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 1}},
                        "PSD": {},
                        "param1": 20,
                        "param2": 21,
                        "ratio": 0.25,
                    },
                    "suffix": "_0_0__1",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 2}},
                        "PSD": {},
                        "param1": 10,
                        "param2": 11,
                        "ratio": 0.25,
                    },
                    "suffix": "_0_1__0",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 2}},
                        "PSD": {},
                        "param1": 20,
                        "param2": 21,
                        "ratio": 0.25,
                    },
                    "suffix": "_0_1__1",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 1}},
                        "PSD": {},
                        "param1": 10,
                        "param2": 11,
                        "ratio": 0.5,
                    },
                    "suffix": "_1_0__0",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 1}},
                        "PSD": {},
                        "param1": 20,
                        "param2": 21,
                        "ratio": 0.5,
                    },
                    "suffix": "_1_0__1",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 2}},
                        "PSD": {},
                        "param1": 10,
                        "param2": 11,
                        "ratio": 0.5,
                    },
                    "suffix": "_1_1__0",
                    "type": "multi",
                },
                {
                    "function": "module.func",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "params": {
                        "CPDF": {"params": {"bin_size": 2}},
                        "PSD": {},
                        "param1": 20,
                        "param2": 21,
                        "ratio": 0.5,
                    },
                    "suffix": "_1_1__1",
                    "type": "multi",
                },
            ],
        ),
    ],
)
def test_config__resolve_features(config_list, expected_list):
    config_list = [FeaturesConfig(**d) for d in config_list]
    expected_list = [FeaturesConfig(**d) for d in expected_list]

    result = blueetl.config.analysis._resolve_features(config_list)

    assert result == expected_list
    assert_not_duplicates(result)


def test_config__resolve_features_error():
    config_list = [
        {
            "type": "multi",
            "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
            "function": "module.func",
            "params_zip": {
                "param1": [10, 20],
                "param2": [11, 21, 31],
            },
        }
    ]
    config_list = [FeaturesConfig(**d) for d in config_list]

    with pytest.raises(ValueError, match="All the zip params must have the same length"):
        test_module._resolve_features(config_list)
