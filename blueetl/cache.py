"""Cache Manager."""
import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

import pandas as pd

from blueetl import DefaultStore
from blueetl.config.simulations import SimulationsConfig
from blueetl.core.utils import is_subfilter
from blueetl.store.base import BaseStore
from blueetl.utils import checksum_json, dump_yaml, load_yaml

L = logging.getLogger(__name__)

CoupledCache = namedtuple("CoupledCache", ["cached", "actual"])


class CacheManager:
    """Cache Manager."""

    def __init__(
        self,
        analysis_config: Dict,
        simulations_config: SimulationsConfig,
        store_class: Type[BaseStore] = DefaultStore,
    ) -> None:
        """Initialize the object."""
        repo_dir = Path(analysis_config["output"], "repo")
        features_dir = Path(analysis_config["output"], "features")
        config_dir = Path(analysis_config["output"], "config")
        config_dir.mkdir(exist_ok=True, parents=True)

        self._version = 1
        self._repo_store = store_class(repo_dir)
        self._features_store = store_class(features_dir)

        self._cached_analysis_config_path = config_dir / "analysis_config.cached.yaml"
        self._cached_simulations_config_path = config_dir / "simulations_config.cached.yaml"
        self._cached_checksums_path = config_dir / "checksums.cached.yaml"

        self._analysis_configs = CoupledCache(
            cached=self._load_cached_analysis_config(),
            actual=analysis_config,
        )
        self._simulations_configs = CoupledCache(
            cached=self._load_cached_simulations_config(),
            actual=simulations_config,
        )
        self._cached_checksums = self._load_cached_checksums()
        self._initialize_cache()

    def _load_cached_analysis_config(self) -> Optional[Dict]:
        """Load the cached analysis config if it exists."""
        path = self._cached_analysis_config_path
        return load_yaml(path) if path.exists() else None

    def _load_cached_simulations_config(self) -> Optional[SimulationsConfig]:
        """Load the cached simulations config if it exists."""
        path = self._cached_simulations_config_path
        return SimulationsConfig.load(path) if path.exists() else None

    def _load_cached_checksums(self) -> Dict:
        """Load the cached checksums, or return null checksums if the file doesn't exist."""
        path = self._cached_checksums_path
        checksums = None
        if path.exists():
            L.info("Loading cached checksums from %s", path)
            checksums = load_yaml(path)
            if checksums["version"] != self._version:
                L.warning(
                    "Incompatible cache version %s != %s, the cache is being deleted",
                    checksums["version"],
                    self._version,
                )
                checksums = None
        if not checksums:
            checksums = {
                "version": self._version,
                "repo": {
                    "simulations": None,
                    "neurons": None,
                    "neuron_classes": None,
                    "trial_steps": None,
                    "windows": None,
                    "spikes": None,
                },
                "features": {},
            }
        return checksums

    def _dump_analysis_config(self) -> None:
        """Write the cached analysis config to file."""
        path = self._cached_analysis_config_path
        dump_yaml(path, self._analysis_configs.actual)

    def _dump_simulations_config(self) -> None:
        """Write the cached simulations config to file."""
        path = self._cached_simulations_config_path
        dump_yaml(path, self._simulations_configs.actual.to_dict())

    def _dump_cached_checksums(self) -> None:
        """Write the cached checksums to file."""
        path = self._cached_checksums_path
        dump_yaml(path, self._cached_checksums)

    def _invalidate_cached_checksums(self, names: Optional[Set[str]] = None) -> None:
        """Invalidate the checksums for the given names.

        This function takes into consideration the dependencies between the dataframes,
        so the cache of all the dataframes depending on the given dataframes are invalidated,
        without the need to specify all of them in ``names``.

        The special name ``features`` can be specified to invalidate all the features.

        If ``names`` is None or empty, all the checksums are invalidated.

        Args:
            names: set of names to be invalidated.
        """

        def _invalidate_repo(_name):
            self._cached_checksums["repo"][_name] = None

        def _invalidate_features(config_checksum):
            d = self._cached_checksums["features"][config_checksum]
            for _name in d:
                d[_name] = None

        def _invalidate_all_features():
            for config_checksum in self._cached_checksums["features"]:
                _invalidate_features(config_checksum)

        ordered_names = [
            "simulations",
            "neurons",
            "neuron_classes",
            "trial_steps",
            "windows",
            "spikes",
            "features",
        ]
        assert not names or names.issubset(ordered_names), "Invalid names specified."
        invalidated: List[str] = []
        for name in ordered_names:
            if invalidated or not names or name in names:
                if name == "features":
                    _invalidate_all_features()
                else:
                    _invalidate_repo(name)
                invalidated.append(name)
        L.info("Invalidated cache: %s", invalidated)

    def _check_config_cache(self) -> None:
        """Compare the cached and actual configurations.

        Only the cache of incompatible dataframes is invalidated.
        """
        if not self._analysis_configs.cached or not self._simulations_configs.cached:
            self._invalidate_cached_checksums()
            return

        # check the criteria used to filter the simulations
        actual_filter = self._analysis_configs.actual["extraction"].get("simulations_filter", {})
        cached_filter = self._analysis_configs.cached["extraction"].get("simulations_filter", {})
        if not is_subfilter(actual_filter, cached_filter):
            # the filter is less specific, so the cache cannot be used
            self._invalidate_cached_checksums()
            return

        # check the simulations config
        if self._simulations_configs.cached != self._simulations_configs.actual:
            self._invalidate_cached_checksums({"simulations"})
            return

        # check the extraction config for neurons
        if any(
            self._analysis_configs.cached["extraction"][k]
            != self._analysis_configs.actual["extraction"][k]
            for k in ["neuron_classes", "limit", "target"]
        ):
            self._invalidate_cached_checksums({"neurons", "neuron_classes"})
            return

        # check the extraction config for windows and spikes
        if any(
            self._analysis_configs.cached["extraction"][k]
            != self._analysis_configs.actual["extraction"][k]
            for k in ["windows", "trial_steps"]
        ):
            self._invalidate_cached_checksums({"trial_steps", "windows", "spikes"})
            return

        # check the features config
        valid_checksums = set()
        for features_config in self._analysis_configs.actual["analysis"]["features"]:
            config_checksum = checksum_json(features_config)
            if config_checksum in self._cached_checksums["features"]:
                valid_checksums.add(config_checksum)

        # invalidate the invalid features checksums
        for config_checksum, d in self._cached_checksums["features"].items():
            if config_checksum not in valid_checksums:
                for _name in d:
                    d[_name] = None

    def _delete_cached_repo_files(self) -> None:
        """Delete the cached repo files if the checksum is None or different."""
        for name, file_checksum in self._cached_checksums["repo"].copy().items():
            if file_checksum != self._repo_store.checksum(name):
                L.info("Deleting cached repo %s", name)
                self._repo_store.delete(name)
                del self._cached_checksums["repo"][name]

    def _delete_cached_features_files(self) -> None:
        """Delete the cached features files if the checksum is None or different."""
        to_be_deleted = set()
        for config_checksum, checksums_by_name in self._cached_checksums["features"].items():
            for name, file_checksum in checksums_by_name.items():
                if file_checksum != self._features_store.checksum(name):
                    to_be_deleted.add(config_checksum)
                    break
        for config_checksum in to_be_deleted:
            # delete every feature generated with the same configuration
            for name in self._cached_checksums["features"].pop(config_checksum):
                L.info("Deleting invalid cached features %s", name)
                self._features_store.delete(name)

    def _initialize_cache(self) -> None:
        """Initialize the cache."""
        L.info("Initialize cache")
        self._check_config_cache()
        self._delete_cached_repo_files()
        self._delete_cached_features_files()
        self._dump_analysis_config()
        self._dump_simulations_config()

    def load_repo(self, name: str) -> Optional[pd.DataFrame]:
        """Load a specific repo dataframe from the cache.

        Args:
            name: name of the repo dataframe.

        Returns:
            The loaded dataframe, or None if it's not cached.
        """
        L.info("Loading cached %s", name)
        file_checksum = self._cached_checksums["repo"].get(name)
        # the checksums have been checked in _initialize_cache/_delete_cached_repo_files,
        # so they are not calculate again here
        return self._repo_store.load(name) if file_checksum else None

    def dump_repo(self, df: pd.DataFrame, name: str) -> None:
        """Write a specific repo dataframe to the cache.

        Args:
            df: dataframe to be saved.
            name: name of the repo dataframe.
        """
        L.info("Writing cached %s", name)
        self._repo_store.dump(df, name)
        self._cached_checksums["repo"][name] = self._repo_store.checksum(name)
        self._dump_cached_checksums()

    def load_features(self, features_config: Dict) -> Optional[Dict[str, pd.DataFrame]]:
        """Load features dataframes from the cache.

        The cache key is determined by the hash of features_config.

        Args:
            features_config: configuration dict of the features to be loaded.

        Returns:
            Dict of dataframes, or None if they are not cached.
        """
        L.info("Loading cached features")
        config_checksum = checksum_json(features_config)
        cached_dataframes = self._cached_checksums["features"].get(config_checksum, {})
        features = {}
        # the checksums have been checked in _initialize_cache/_delete_cached_features_files,
        # so they are not calculate again here
        for name, file_checksum in cached_dataframes.items():
            assert file_checksum is not None
            features[name] = self._features_store.load(name)
        return features or None

    def dump_features(self, features_dict: Dict[str, pd.DataFrame], features_config: Dict) -> None:
        """Write features dataframes to the cache.

        The cache key is determined by the hash of features_config.

        Args:
            features_dict: dict of features to be written.
            features_config: configuration dict of the features to be written.
        """
        L.info("Writing cached features")
        config_checksum = checksum_json(features_config)
        d = self._cached_checksums["features"][config_checksum] = {}
        for name, feature in features_dict.items():
            self._features_store.dump(feature, name)
            d[name] = self._features_store.checksum(name)
        self._dump_cached_checksums()
