import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from blueetl.apps import main as test_module
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml
from blueetl.validation import ValidationError


@pytest.mark.parametrize("clear_cache", [True, False, None])
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("calculate", [True, False])
@pytest.mark.parametrize("extract", [True, False])
@patch(test_module.__name__ + ".run_from_file")
def test_run(mock_run_from_file, tmp_path, extract, calculate, show, clear_cache):
    options_dict = {
        "extract": extract,
        "calculate": calculate,
        "show": show,
        "clear-cache": clear_cache,
    }
    options = [f"--{k}" if v else f"--no-{k}" for k, v in options_dict.items() if v is not None]
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.cli, ["run", analysis_config_file, "-vv", *options])

    assert result.output == ""
    assert result.exit_code == 0
    mock_run_from_file.assert_called_once_with(
        analysis_config_file=analysis_config_file,
        seed=0,
        extract=extract,
        calculate=calculate,
        show=show,
        clear_cache=clear_cache,
        loglevel=logging.DEBUG,
    )


@patch.dict(sys.modules, {"IPython": Mock()})
@patch(test_module.__name__ + ".run_from_file")
def test_run_interactive_success(mock_run_from_file, tmp_path):
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.cli, ["run", analysis_config_file, "--interactive"])

    assert result.output == ""
    assert result.exit_code == 0
    assert mock_run_from_file.call_count == 1
    assert sys.modules["IPython"].embed.call_count == 1


@patch.dict(sys.modules, {"IPython": None})
@patch(test_module.__name__ + ".run_from_file")
def test_run_interactive_failure(mock_run_from_file, tmp_path):
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.cli, ["run", analysis_config_file, "--interactive"])

    assert result.output.strip() == "You need to install IPython to start an interactive session."
    assert result.exit_code == 1
    assert mock_run_from_file.call_count == 1


def test_migrate_config(tmp_path):
    input_config_file = "input_config.yaml"
    output_config_file = "output_config.yaml"
    input_data = {
        "simulation_campaign": "/path/to/config.json",
        "simulations_filter_in_memory": {"simulation_id": 2},
        "output": "output_dir",
        "extraction": {
            "neuron_classes": {"L1_EXC": {"layer": [1], "synapse_class": ["EXC"]}},
            "limit": None,
            "target": None,
            "windows": {"w1": {"bounds": [20, 90], "window_type": "spontaneous"}},
        },
        "analysis": {
            "features": [
                {
                    "type": "multi",
                    "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                    "function": "module.user.function",
                    "params": {"export_all_neurons": True},
                }
            ]
        },
    }
    expected_data = {
        "version": CONFIG_VERSION,
        "simulation_campaign": "/path/to/config.json",
        "simulations_filter_in_memory": {"simulation_id": 2},
        "output": "output_dir",
        "analysis": {
            "spikes": {
                "extraction": {
                    "report": {"type": "spikes"},
                    "neuron_classes": {"L1_EXC": {"layer": [1], "synapse_class": ["EXC"]}},
                    "limit": None,
                    "target": None,
                    "windows": {"w1": {"bounds": [20, 90], "window_type": "spontaneous"}},
                },
                "features": [
                    {
                        "type": "multi",
                        "groupby": ["simulation_id", "circuit_id", "neuron_class", "window"],
                        "function": "module.user.function",
                        "params": {"export_all_neurons": True},
                    }
                ],
            }
        },
    }
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        dump_yaml(input_config_file, input_data)
        result = runner.invoke(
            test_module.cli, ["migrate-config", input_config_file, output_config_file]
        )
        output_data = load_yaml(output_config_file)

    assert (
        result.output.strip() == "The converted configuration has been saved to output_config.yaml."
    )
    assert result.exit_code == 0
    assert output_data == expected_data


@patch(test_module.__name__ + ".validation.validate_config")
def test_validate_config_success(mock_validate_config, tmp_path):
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.cli, ["validate-config", analysis_config_file])

    assert result.output.strip() == "Validation successful."
    assert result.exit_code == 0
    assert mock_validate_config.call_count == 1


@patch(test_module.__name__ + ".validation.validate_config")
def test_validate_config_failure(mock_validate_config, tmp_path):
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    mock_validate_config.side_effect = ValidationError()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.cli, ["validate-config", analysis_config_file])

    assert result.output.strip() == "Validation failed."
    assert result.exit_code == 1
    assert mock_validate_config.call_count == 1
