"""Analysis Configuration."""

import logging
from collections.abc import Iterator
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import NamedTuple, Union

from blueetl.config.analysis_model import (
    FeaturesConfig,
    MultiAnalysisConfig,
    TrialStepsConfig,
    WindowConfig,
)
from blueetl.constants import CHECKSUM_SEP
from blueetl.utils import checksum_json, dict_product, load_json
from blueetl.validation import read_schema, validate_config

L = logging.getLogger(__name__)


def _resolve_paths(global_config: MultiAnalysisConfig, base_path: Path) -> None:
    """Resolve any relative path at the top level."""
    base_path = base_path or Path()
    global_config.output = base_path / global_config.output
    global_config.simulation_campaign = base_path / global_config.simulation_campaign


def _resolve_neuron_classes(global_config: MultiAnalysisConfig, base_path: Path):
    """Resolve the relative paths in neuron_classes."""
    for config in global_config.analysis.values():
        for neuron_classes_config in config.extraction.neuron_classes.values():
            if path := neuron_classes_config.node_sets_file:
                path = base_path / path
                neuron_classes_config.node_sets_file = path
                neuron_classes_config.node_sets_checksum = checksum_json(load_json(path))


def _resolve_trial_steps(global_config: MultiAnalysisConfig, base_path: Path):
    """Set trial_steps_config.base_path to the same value as global_config.output.

    In this way, the custom function can use it as the base path to save any figure.
    """
    for config in global_config.analysis.values():
        for trial_steps_config in config.extraction.trial_steps.values():
            trial_steps_config.base_path = str(global_config.output)
            if path := trial_steps_config.node_sets_file:
                path = base_path / path
                trial_steps_config.node_sets_file = path
                trial_steps_config.node_sets_checksum = checksum_json(load_json(path))


def _resolve_windows(global_config: MultiAnalysisConfig) -> None:
    """Calculate the hash of any referenced windows in each single analysis configuration.

    This is needed to invalidate the cache of the windows referring to the original windows.
    """

    class Reference(NamedTuple):
        """Reference attributes."""

        report: str
        window: str

    def _parse_ref(ref: str) -> Reference:
        # example: spikes.extraction.windows.w3 -> Reference(report="spikes", window="w3")
        try:
            report, _extraction, _windows, window = ref.split(".")
            if _extraction != "extraction" or _windows != "windows":
                raise ValueError
        except ValueError as ex:
            raise ValueError(
                "Windows reference is not in the expected format: "
                "<report_name>.extraction.windows.<window_name>"
            ) from ex
        return Reference(report, window)

    def _calculate_checksum(ref: str) -> str:
        reference = _parse_ref(ref)
        try:
            extraction = global_config.analysis[reference.report].extraction
            window = extraction.windows[reference.window]
        except KeyError as ex:
            raise ValueError(f"Windows reference {ref!r} points to nonexistent object") from ex
        assert isinstance(window, WindowConfig), "Invalid referenced window config"
        trial_step = None
        if trial_steps_label := window.trial_steps_label:
            L.debug("Checksum including trial_steps_label %s", trial_steps_label)
            # retrieve the dynamic trial_step config to be included in the checksum
            trial_step = extraction.trial_steps[trial_steps_label]
            assert isinstance(trial_step, TrialStepsConfig), "Invalid referenced trial_steps config"
        return checksum_json(
            [
                window.checksum(),
                trial_step.checksum() if trial_step else "",
            ]
        )

    for analysis_name, analysis_config in global_config.analysis.items():
        windows = analysis_config.extraction.windows
        new_windows: dict[str, Union[str, WindowConfig]] = {}
        for window_name, cfg in windows.items():
            if isinstance(cfg, str):
                checksum = _calculate_checksum(cfg)
                new_windows[window_name] = f"{cfg}{CHECKSUM_SEP}{checksum}"
                L.debug(
                    "Config checksum for analysis %s and window %s: %s",
                    analysis_name,
                    window_name,
                    checksum,
                )
            else:
                new_windows[window_name] = cfg
        analysis_config.extraction.windows = new_windows


def _resolve_features(features_config_list: list[FeaturesConfig]) -> list[FeaturesConfig]:
    def expand_product(params: dict, params_product: dict) -> Iterator[dict]:
        for items in dict_product(params_product):
            new_params = deepcopy(params)
            for key, v, i in items:
                new_params[key] = v
                new_params["__suffix__"] = new_params.get("__suffix__", "") + f"_{i}"
            yield new_params

    def expand_zip(params: dict, params_zip: dict) -> Iterator[dict]:
        if len({len(values) for values in params_zip.values()}) != 1:
            raise ValueError("All the zip params must have the same length")
        for i, zip_values in enumerate(zip(*params_zip.values())):
            new_params = deepcopy(params)
            new_params["__suffix__"] = new_params.get("__suffix__", "") + f"__{i}"
            new_params.update(zip(params_zip.keys(), zip_values))
            yield new_params

    new = []
    for config in features_config_list:
        config = deepcopy(config)
        params_list = [config.params]
        config.params = {}
        if params_product := config.params_product:
            config.params_product = {}
            params_list = list(
                chain.from_iterable(expand_product(p, params_product) for p in params_list)
            )
        if params_zip := config.params_zip:
            config.params_zip = {}
            params_list = list(chain.from_iterable(expand_zip(p, params_zip) for p in params_list))
        for p in params_list:
            new_features_config = deepcopy(config)
            new.append(new_features_config)
            if p:
                # suffix becomes part of the configuration
                new_features_config.suffix += p.pop("__suffix__", "")
                new_features_config.params = p
    return new


def _resolve_analysis_configs(global_config: MultiAnalysisConfig) -> None:
    for name, config in global_config.analysis.items():
        config.output = global_config.output / name
        config.simulations_filter = global_config.simulations_filter
        config.simulations_filter_in_memory = global_config.simulations_filter_in_memory
        config.features = _resolve_features(config.features)


def init_multi_analysis_configuration(global_config: dict, base_path: Path) -> MultiAnalysisConfig:
    """Return a config object from a config dict."""
    validate_config(global_config, schema=read_schema("analysis_config"))
    config = MultiAnalysisConfig(**global_config)
    _resolve_paths(config, base_path=base_path)
    _resolve_neuron_classes(config, base_path=base_path)
    _resolve_trial_steps(config, base_path=base_path)
    _resolve_windows(config)
    _resolve_analysis_configs(config)
    return config
