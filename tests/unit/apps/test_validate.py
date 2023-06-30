from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from blueetl.apps import validate as test_module
from blueetl.validation import ValidationError


@patch(test_module.__name__ + ".validation.validate_config")
def test_validate_config_success(mock_validate_config, tmp_path):
    analysis_config_file = "config.yaml"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        Path(analysis_config_file).write_text("---")
        result = runner.invoke(test_module.validate_config, [analysis_config_file])

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
        result = runner.invoke(test_module.validate_config, [analysis_config_file])

    assert result.output.strip() == "Validation failed."
    assert result.exit_code == 1
    assert mock_validate_config.call_count == 1
