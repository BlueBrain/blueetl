"""Migrate CLI."""
import sys

import click

from blueetl import validation
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml


@click.command()
@click.argument("input_config_file", type=click.Path(exists=True))
@click.argument("output_config_file", type=click.Path(exists=False))
def migrate_config(input_config_file, output_config_file):
    """Migrate a configuration file."""

    def _process_extraction(extraction):
        """Convert the gid key to node_id, inplace."""
        if "target" in extraction:
            extraction["node_set"] = extraction.pop("target")
        for config in extraction["neuron_classes"].values():
            if "gid" in config:
                if "$gids" in config:
                    raise RuntimeError("neuron_classes already contain '$gids'")
                config["$gids"] = config.pop("gid")
            if "$target" in config:
                config["node_set"] = config.pop("$target")
        extraction["population"] = "default"
        return extraction

    input_config = load_yaml(input_config_file)
    if version := input_config.get("version", 0) >= CONFIG_VERSION:
        click.secho(f"The configuration version {version} doesn't need to be converted.", fg="red")
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
