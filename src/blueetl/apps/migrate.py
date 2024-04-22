"""Migrate CLI."""

import sys
from copy import deepcopy

import click

from blueetl import validation
from blueetl.constants import CONFIG_VERSION
from blueetl.utils import dump_yaml, load_yaml

# it should always match CONFIG_VERSION if the script supports the current version
MIGRATION_CONFIG_VERSION = 4


def _safe_set(d, key, value):
    """Set the value in the given dict, or raise if it already exists."""
    if key in d:
        raise RuntimeError(f"The key {key} is already present in {d}")
    d[key] = value


def _rename_key(d, old, new):
    """Rename a key in the given dict, only if the old key already exists."""
    if old in d:
        _safe_set(d, new, d.pop(old))


def _migrate_v1_to_v2(input_config):
    """Migrate the configuration from v1 (BlueETL 0.1.x) to v2 (BlueETL 0.2.x)."""

    def _process_extraction(extraction):
        for neuron_class in extraction["neuron_classes"].values():
            _rename_key(neuron_class, "gid", "$gids")
        return extraction

    input_config = deepcopy(input_config)
    return {
        "version": 2,
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


def _migrate_v2_to_v3(input_config):
    """Migrate the configuration from v2 (BlueETL 0.2.x) to v3 (BlueETL 0.3.x)."""
    output_config = deepcopy(input_config)
    output_config["version"] = 3
    for analysis in output_config["analysis"].values():
        extraction = analysis["extraction"]
        _rename_key(extraction, "target", "node_set")
        for neuron_class in extraction["neuron_classes"].values():
            query = {}
            for k in list(neuron_class):
                if not k.startswith("$"):
                    query[k] = neuron_class.pop(k)
            if query:
                _safe_set(neuron_class, "query", query)
            _rename_key(neuron_class, "$target", "node_set")
            _rename_key(neuron_class, "$limit", "limit")
            _rename_key(neuron_class, "$gids", "node_id")
            _rename_key(neuron_class, "$query", "query")
    return output_config


def _migrate_v3_to_v4(input_config):
    """Migrate the configuration from v3 (BlueETL 0.8.x) to v4 (BlueETL 0.9.x)."""
    output_config = deepcopy(input_config)
    output_config["version"] = 4
    cache_config = output_config.setdefault("cache", {})
    if (value := output_config.pop("output", None)) is not None:
        _safe_set(cache_config, "path", value)
    if (value := output_config.pop("clear_cache", None)) is not None:
        _safe_set(cache_config, "clear", value)
    return output_config


def _sort_root_keys(input_config):
    root_keys = [
        "version",
        "simulation_campaign",
        "simulations_filter",
        "simulations_filter_in_memory",
        "cache",
        "analysis",
        "custom",
    ]
    output_config = {k: input_config.pop(k) for k in root_keys if k in input_config}
    return output_config | input_config


@click.command()
@click.option("--sort/--no-sort", help="Sort the root keys.", default=True, show_default=True)
@click.argument("input_config_file", type=click.Path(exists=True))
@click.argument("output_config_file", type=click.Path(exists=False))
def migrate_config(input_config_file, output_config_file, sort):
    """Migrate a configuration file."""
    config = load_yaml(input_config_file)
    version = config.get("version", 1)
    if not isinstance(version, int) or version < 1 or version > CONFIG_VERSION:
        click.secho(f"The config version {version} isn't supported.", fg="red")
        sys.exit(1)
    if version <= 1:
        config = _migrate_v1_to_v2(config)
    if version <= 2:
        config = _migrate_v2_to_v3(config)
    if version <= 3:
        config = _migrate_v3_to_v4(config)
    if version == CONFIG_VERSION:
        click.secho(f"The config version {version} doesn't need to be migrated.", fg="yellow")
    if sort:
        config = _sort_root_keys(config)
    validation.validate_config(config, schema=validation.read_schema("analysis_config"))
    dump_yaml(output_config_file, config, default_style="", default_flow_style=None)
    click.secho(f"The converted configuration has been saved to {output_config_file}.", fg="green")
