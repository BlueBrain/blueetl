"""Convert CLI."""

import logging
from pathlib import Path

import click

from blueetl.converters.convert_spikes import main
from blueetl.utils import setup_logging


@click.command()
@click.argument("input-file", type=click.Path(exists=True, path_type=Path))
@click.argument("output-dir", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--node-population",
    help="Name of the node population to create.",
    default="synthetic",
    show_default=True,
)
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
def convert_spikes(input_file, output_dir, node_population, verbose):
    """Convert spikes in CSV format.

    Read INPUT_FILE containing the spikes in CSV format, and write synthetic files to OUTPUT_DIR.

    The input file should contain:

    \b
    - headers: node_ids timestamps (or: ids times)
    - values: space separated
    """
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    setup_logging(loglevel=loglevel, force=True)
    main(
        input_file=input_file,
        output_dir=output_dir,
        node_population=node_population,
    )
    click.secho("Conversion successful.", fg="green")
