"""Main CLI."""

import click

from blueetl import __version__
from blueetl.apps.convert import convert_spikes
from blueetl.apps.migrate import migrate_config
from blueetl.apps.run import run
from blueetl.apps.validate import validate_config


class NaturalOrderGroup(click.Group):
    """Click group preserving the order of commands."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return the list of possible commands."""
        return list(self.commands)


@click.group(cls=NaturalOrderGroup)
@click.version_option(__version__)
def cli():
    """The CLI entry point."""


cli.add_command(run)
cli.add_command(migrate_config)
cli.add_command(validate_config)
cli.add_command(convert_spikes)
