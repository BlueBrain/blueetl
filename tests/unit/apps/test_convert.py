from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from blueetl.apps import convert as test_module


@patch(test_module.__name__ + ".main")
def test_validate_config_success(mock_main, tmp_path):
    input_file = tmp_path / "spikes.csv"
    output_dir = tmp_path / "output"
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_file.write_text("ids times\n1 0.0")
        result = runner.invoke(
            test_module.convert_spikes,
            [
                str(input_file),
                str(output_dir),
                "--node-population",
                "custom",
            ],
        )

    assert result.output.strip() == "Conversion successful."
    assert result.exit_code == 0
    assert mock_main.call_count == 1
