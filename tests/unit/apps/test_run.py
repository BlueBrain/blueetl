import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from blueetl.apps import run as test_module


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
        result = runner.invoke(test_module.run, [analysis_config_file, "-vv", *options])

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
        result = runner.invoke(test_module.run, [analysis_config_file, "--interactive"])

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
        result = runner.invoke(test_module.run, [analysis_config_file, "--interactive"])

    assert result.output.strip() == "You need to install IPython to start an interactive session."
    assert result.exit_code == 1
    assert mock_run_from_file.call_count == 1
