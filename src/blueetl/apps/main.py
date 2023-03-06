"""BlueETL CLI."""
# pylint: disable=unused-import,import-outside-toplevel
import logging
import sys

import click
import numpy as np
import pandas as pd

from blueetl import __version__, validation
from blueetl.analysis import MultiAnalyzer
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml, setup_logging


class NaturalOrderGroup(click.Group):
    """Click group preserving the order of commands."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return the list of possible commands."""
        return list(self.commands)


@click.group(cls=NaturalOrderGroup)
@click.version_option(__version__)
def cli():
    """The CLI entry point."""


@cli.command()
@click.argument("analysis_config_file", type=click.Path(exists=True))
@click.option("--seed", type=int, default=0, help="Pseudo-random generator seed", show_default=True)
@click.option("--extract", is_flag=True, help="Extract or load from the cache the repository.")
@click.option("--calculate", is_flag=True, help="Calculate or load from the cache the features.")
@click.option("--show", is_flag=True, help="Show repository and features dataframes.")
@click.option("-i", "--interactive", is_flag=True, help="Start an interactive IPython shell.")
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
def run(analysis_config_file, seed, extract, calculate, show, interactive, verbose):
    """Run the analysis."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    setup_logging(loglevel=loglevel)
    np.random.seed(seed)
    ma = MultiAnalyzer.from_file(analysis_config_file)
    if extract:
        ma.extract_repo()
    if calculate:
        ma.calculate_features()
    if show:
        ma.show()
    if interactive:
        try:
            from IPython import embed
        except ImportError:
            click.secho("You need to install IPython to start an interactive session.", fg="red")
            sys.exit(1)
        header = (
            f"BlueETL Interactive Shell {__version__}\n\n"
            f"The loaded MultiAnalyzer instance is available for test in the variable 'ma'."
        )
        embed(header=header, colors="neutral")


@cli.command()
@click.argument("input_config_file", type=click.Path(exists=True))
@click.argument("output_config_file", type=click.Path(exists=False))
def migrate_config(input_config_file, output_config_file):
    """Migrate a configuration file."""
    input_config = load_yaml(input_config_file)
    if input_config.get("version", 0) >= CONFIG_VERSION:
        click.secho("The configuration doesn't need to be converted.", fg="red")
        sys.exit(1)
    output_config = {
        "version": CONFIG_VERSION,
        **{
            k: input_config.pop(k)
            for k in [
                "simulation_campaign",
                "output",
                "simulations_filter",
                "simulations_filter_in_memory",
            ]
            if k in input_config
        },
        "analysis": {
            "spikes": {
                "extraction": {
                    "report": {"type": "spikes"},
                    **input_config.pop("extraction"),
                },
                "features": input_config.pop("analysis", {}).pop("features", []),
                **({"custom": input_config} if input_config else {}),
            }
        },
    }
    validation.validate_config(output_config, schema=validation.read_schema("analysis_config"))
    dump_yaml(output_config_file, output_config, default_style="", default_flow_style=None)
    click.secho(f"The converted configuration has been saved to {output_config_file}.", fg="green")


@cli.command()
@click.argument("analysis_config_file", type=click.Path(exists=True))
def validate_config(analysis_config_file):
    """Validate a configuration file."""
    analysis_config = load_yaml(analysis_config_file)
    try:
        validation.validate_config(
            analysis_config, schema=validation.read_schema("analysis_config")
        )
    except validation.ValidationError:
        click.secho("Validation failed.", fg="red")
        sys.exit(1)
    else:
        click.secho("Validation successful.", fg="green")
