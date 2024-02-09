import textwrap

from click.testing import CliRunner

from blueetl import __version__
from blueetl.apps import main as test_module


def test_version():
    runner = CliRunner()
    result = runner.invoke(test_module.cli, ["--version"])

    assert result.output.strip() == f"cli, version {__version__}"
    assert result.exit_code == 0


def test_help():
    expected_message = """
    Usage: cli [OPTIONS] COMMAND [ARGS]...
    
      The CLI entry point.
    
    Options:
      --version  Show the version and exit.
      --help     Show this message and exit.
    
    Commands:
      run              Run the analysis.
      migrate-config   Migrate a configuration file.
      validate-config  Validate a configuration file.
      convert-spikes   Convert spikes in CSV format.
    """

    runner = CliRunner()
    result = runner.invoke(test_module.cli, ["--help"])

    assert result.output.strip() == textwrap.dedent(expected_message).strip()
    assert result.exit_code == 0
