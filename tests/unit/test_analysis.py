import pytest

from blueetl import analysis as test_module
from tests.unit.utils import assert_not_duplicates


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {"features": []},
            {"features": []},
        ),
        (
            {
                "features": [
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
                ]
            },
            {
                "features": [
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
                ]
            },
        ),
    ],
)
def test_config__resolve_features(config, expected):
    test_module.MultiAnalyzerConfig._resolve_features(config)
    assert config == expected
    assert_not_duplicates(config)


def test_config__resolve_features_error():
    config = {
        "features": [
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
    }
    with pytest.raises(ValueError, match="All the zip params must have the same length"):
        test_module.MultiAnalyzerConfig._resolve_features(config)
