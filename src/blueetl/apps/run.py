"""Run CLI."""

import logging
import sys

import click

from blueetl import __version__
from blueetl.analysis import run_from_file


@click.command()
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
