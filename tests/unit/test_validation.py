import pytest

from blueetl import validation as test_module


def test_read_schema():
    schema = test_module.read_schema("analysis_config")
    assert isinstance(schema, dict)


@pytest.mark.parametrize(
    "analysis_config",
    [
        {
            "version": 2,
            "simulation_campaign": "/path/to/simulation/campaign",
            "output": "/path/to/output",
            "analysis": {},
        },
        {
            "version": 2,
            "simulation_campaign": "/path/to/simulation/campaign",
            "output": "/path/to/output",
            "analysis": {
                "spikes": {
                    "extraction": {
                        "report": {"type": "spikes"},
                        "population": "default",
                        "neuron_classes": {},
                        "windows": {},
                    }
                },
            },
        },
    ],
)
def test_validate_config(analysis_config):
    schema = test_module.read_schema("analysis_config")
    test_module.validate_config(config=analysis_config, schema=schema)


@pytest.mark.parametrize(
    "analysis_config",
    [
        {},
        {
            "version": 2,
            "simulation_campaign": "/path/to/simulation/campaign",
            "output": "/path/to/output",
            "analysis": {
                "spikes": {"extraction": {}},
            },
        },
    ],
)
def test_validate_config_invalid(analysis_config):
    schema = test_module.read_schema("analysis_config")
    with pytest.raises(test_module.ValidationError):
        test_module.validate_config(config=analysis_config, schema=schema)
