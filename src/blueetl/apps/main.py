"""BlueETL CLI."""
import logging
import sys

import click

from blueetl import __version__, validation
from blueetl.analysis import run_from_file
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml


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
@click.option("--extract/--no-extract", help="Extract (or load from the cache) the repository.")
@click.option("--calculate/--no-calculate", help="Calculate (or load from the cache) the features.")
@click.option("--show/--no-show", help="Show repository and features dataframes.")
@click.option(
    "--clear-cache/--no-clear-cache",
    help="If specified, force clearing or keeping the cache, regardless of the configuration file.",
    default=None,
)
@click.option("-i", "--interactive/--no-interactive", help="Start an interactive IPython shell.")
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
def run(analysis_config_file, seed, extract, calculate, show, clear_cache, interactive, verbose):
    """Run the analysis."""
    # pylint: disable=unused-variable,unused-import,import-outside-toplevel
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    # assign the result to a local variable to make it available in the interactive shell
    ma = run_from_file(  # noqa
        analysis_config_file=analysis_config_file,
        seed=seed,
        extract=extract,
        calculate=calculate,
        show=show,
        clear_cache=clear_cache,
        loglevel=loglevel,
    )
    if interactive:
        # make np and pd immediately available in the interactive shell
        import numpy as np  # noqa
        import pandas as pd  # noqa

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

    def _process_extraction(extraction):
        """Convert the gid key to $gids, inplace."""
        if "target" in extraction:
            extraction["node_set"] = extraction.pop("target")
        for config in extraction["neuron_classes"].values():
            if "gid" in config:
                if "$gids" in config:
                    raise RuntimeError("neuron_classes already contain '$gids'")
                config["$gids"] = config.pop("gid")
            if "$target" in config:
                config["$node_set"] = config.pop("$target")
        extraction["population"] = "default"
        return extraction

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
                    **_process_extraction(input_config.pop("extraction")),
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
