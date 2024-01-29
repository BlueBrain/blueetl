"""Validate CLI."""

import sys

import click

from blueetl import validation
from blueetl.utils import load_yaml


@click.command()
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
