"""Cache Manager."""

import errno
import fcntl
import logging
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Generic, Optional, Protocol, TypeVar

import pandas as pd
from blueetl_core.utils import is_subfilter

from blueetl.campaign.config import SimulationCampaign
from blueetl.config.analysis_model import CacheConfig, FeaturesConfig, SingleAnalysisConfig
from blueetl.store.base import BaseStore
from blueetl.store.feather import FeatherStore
from blueetl.store.parquet import ParquetStore
from blueetl.utils import dump_yaml, load_yaml

L = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT", SingleAnalysisConfig, SimulationCampaign)


@dataclass
class CoupledCache(Generic[ConfigT]):
    """Container of cached and actual configurations."""

    cached: Optional[ConfigT]
    actual: ConfigT


def _raise_if(**attrs):
    """Raise if the decorated method is called and all the attrs are equal to the given values."""

    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if all(getattr(self, attr) == value for attr, value in attrs.items()):
                raise CacheError(
                    f"Method {self.__class__.__name__}.{f.__name__} "
                    f"cannot be called when the attributes are: {attrs}"
                )
            return f(self, *args, **kwargs)

        return wrapper

    return decorator


class CacheError(Exception):
    """Cache error raised when a read-only cache is written."""


class LockManagerProtocol(Protocol):
    """Lock Manager Interface."""

    @property
    def locked(self) -> bool:
        """Return the lock status."""

    def lock(self, mode: int) -> None:
        """Lock."""

    def unlock(self) -> None:
        """Unlock."""


class LockManager:
    """Lock Manager.

    On Linux, the flock call is handled locally, and the underlying filesystem (GPFS) does not get
    any notification that locks are being set. Therefore, GPFS cannot enforce locks across nodes.
    """

    LOCK_EX = fcntl.LOCK_EX  # exclusive lock
    LOCK_SH = fcntl.LOCK_SH  # shared lock

    def __init__(self, path: os.PathLike) -> None:
        """Initialize the object.

        Args:
            path: path to an existing directory to be used for locking.
        """
        self._path = path
        self._fd: Optional[int] = None

    @property
    def locked(self) -> bool:
        """Return True if the lock manager is locking the cache, False otherwise."""
        return self._fd is not None

    def lock(self, mode: int) -> None:
        """Lock the directory.

        Args:
            mode: LockManager.LOCK_EX for exclusive lock, or LockManager.LOCK_SH for shared lock.
        """
        if self._fd is None:
            self._fd = os.open(self._path, os.O_RDONLY)
        try:
            fcntl.flock(self._fd, mode | fcntl.LOCK_NB)
        except OSError as ex:
            os.close(self._fd)
            self._fd = None
            if ex.errno in (errno.EACCES, errno.EAGAIN):
                # the lock cannot be acquired
                raise CacheError(f"Another process is locking {self._path}") from None
            # re-raise any other type of OSError
            raise

    def unlock(self) -> None:
        """Unlock the directory."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class DummyLockManager:
    """Dummy Lock Manager."""

    @property
    def locked(self) -> bool:
        """Always return True."""
        return True

    def lock(self, mode: int) -> None:
        """Pretend to lock."""

    def unlock(self) -> None:
        """Pretend to unlock."""


class CacheManager:
    """Cache Manager."""

    def __init__(
        self,
        cache_config: CacheConfig,
        analysis_config: SingleAnalysisConfig,
        simulations_config: SimulationCampaign,
    ) -> None:
        """Initialize the object.

        Args:
            cache_config: cache configuration dict.
            analysis_config: analysis configuration dict.
            simulations_config: simulations campaign configuration.
        """
        self._output_dir = cache_config.path
        if cache_config.clear:
            self._clear_cache()
        repo_dir = self._output_dir / "repo"
        features_dir = self._output_dir / "features"
        config_dir = self._output_dir / "config"
        for new_dir in repo_dir, features_dir, config_dir:
            new_dir.mkdir(exist_ok=True, parents=True)

        store_classes: dict[str, type[BaseStore]] = {
            "parquet": ParquetStore,
            "feather": FeatherStore,
        }
        store_class = store_classes[cache_config.store_type]

        self.readonly = cache_config.readonly
        self._version = 1
        self._repo_store = store_class(repo_dir)
        self._features_store = store_class(features_dir)

        self._lock_manager: LockManagerProtocol = LockManager(self._output_dir)
        self._lock_manager.lock(mode=LockManager.LOCK_SH if self.readonly else LockManager.LOCK_EX)

        self._cached_analysis_config_path = config_dir / "analysis_config.cached.yaml"
        self._cached_simulations_config_path = config_dir / "simulations_config.cached.yaml"
        self._cached_checksums_path = config_dir / "checksums.cached.yaml"

        self._analysis_configs = CoupledCache[SingleAnalysisConfig](
            cached=self._load_cached_analysis_config(),
            actual=analysis_config,
        )
        self._simulations_configs = CoupledCache[SimulationCampaign](
            cached=self._load_cached_simulations_config(),
            actual=simulations_config,
        )
        self._cached_checksums = self._load_cached_checksums()
        self._initialize_cache()

    def _clear_cache(self):
        """Remove the cache directory if it exists."""
        L.info("Removing cache if it exists: %s", self._output_dir)
        shutil.rmtree(self._output_dir, ignore_errors=True)

    def __getstate__(self) -> dict:
        """Get the object state when the object is pickled."""
        return self.__dict__ | {"_lock_manager": None}

    def __setstate__(self, state: dict) -> None:
        """Set the object state when the object is unpickled."""
        self.__dict__.update(state)
        # The unpickled object must always be readonly, even when the pickled object isn't.
        self.readonly = True
        # A new lock isn't created in the subprocess b/c we want to be able to read the cache.
        self._lock_manager = DummyLockManager()

    @property
    def locked(self) -> bool:
        """Return True if the cache manager is locking the cache, False otherwise."""
        return self._lock_manager.locked

    def close(self) -> None:
        """Close the cache manager and unlock the lock directory.

        After calling this method, the Cache Manager instance shouldn't be used anymore.
        """
        self._lock_manager.unlock()

    @_raise_if(locked=False)
    def to_readonly(self) -> "CacheManager":
        """Return a read-only copy of the object.

        The returned object will raise an exception if any writing method is called.
        """
        obj = deepcopy(self)
        # readonly is always True because of __getstate__ and __setstate__
        assert obj.readonly is True
        return obj

    def _load_cached_analysis_config(self) -> Optional[SingleAnalysisConfig]:
        """Load the cached analysis config if it exists."""
        path = self._cached_analysis_config_path
        return SingleAnalysisConfig.load(path) if path.exists() else None

    def _load_cached_simulations_config(self) -> Optional[SimulationCampaign]:
        """Load the cached simulations config if it exists."""
        path = self._cached_simulations_config_path
        return SimulationCampaign.load(path) if path.exists() else None

    def _load_cached_checksums(self) -> dict:
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
                    "windows": None,
                    "report": None,
                },
                "features": {},
            }
        return checksums

    @_raise_if(readonly=True)
    def _dump_analysis_config(self) -> None:
        """Write the cached analysis config to file."""
        path = self._cached_analysis_config_path
        self._analysis_configs.actual.dump(path)

    @_raise_if(readonly=True)
    def _dump_simulations_config(self) -> None:
        """Write the cached simulations config to file."""
        path = self._cached_simulations_config_path
        self._simulations_configs.actual.dump(path)

    @_raise_if(readonly=True)
    def _dump_cached_checksums(self) -> None:
        """Write the cached checksums to file."""
        path = self._cached_checksums_path
        dump_yaml(path, self._cached_checksums)

    def _invalidate_cached_checksums(self, names: Optional[set[str]] = None) -> None:
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
            "windows",
            "report",
            "features",
        ]
        assert not names or names.issubset(ordered_names), "Invalid names specified."
        invalidated: list[str] = []
        for name in ordered_names:
            if invalidated or not names or name in names:
                if name == "features":
                    _invalidate_all_features()
                else:
                    _invalidate_repo(name)
                invalidated.append(name)
        L.info("Invalid cache: %s", invalidated)

    def _check_config_cache(self) -> bool:
        """Compare the cached and actual configurations.

        Only the cache of incompatible dataframes is invalidated.

        Returns:
            True if the cache is valid and complete, False otherwise.
        """
        if not self._analysis_configs.cached or not self._simulations_configs.cached:
            L.debug("The cached configurations have not been found")
            self._invalidate_cached_checksums()
            return False

        # check the criteria used to filter the simulations
        if not self._is_subfilter(strict=False):
            L.debug("The simulations filter is less specific, so the cache cannot be used")
            self._invalidate_cached_checksums()
            return False

        # check the simulations config
        if self._simulations_configs.cached != self._simulations_configs.actual:
            L.debug("The cached simulation campaign configurations is invalid")
            self._invalidate_cached_checksums({"simulations"})
            return False

        # check the extraction config for changed keys, and invalidate the affected names
        keys_and_affected_names = [
            ({"neuron_classes"}, {"neurons", "neuron_classes"}),
            ({"windows", "trial_steps"}, {"windows", "report"}),
            ({"report"}, {"report"}),
        ]
        for keys, names in keys_and_affected_names:
            if any(
                getattr(self._analysis_configs.cached.extraction, k)
                != getattr(self._analysis_configs.actual.extraction, k)
                for k in keys
            ):
                L.debug("The following cached objects are invalid: %s", names)
                self._invalidate_cached_checksums(names)
                return False

        # check the features config
        valid_checksums = set()
        for features_config in self._analysis_configs.actual.features:
            config_checksum = features_config.checksum()
            if config_checksum in self._cached_checksums["features"] and all(
                self._cached_checksums["features"][config_checksum].values()
            ):
                valid_checksums.add(config_checksum)

        # invalidate the invalid features checksums
        is_valid = True
        for config_checksum, d in self._cached_checksums["features"].items():
            if config_checksum not in valid_checksums:
                is_valid = False
                for _name in d:
                    d[_name] = None

        return is_valid

    def _check_cached_repo_files(self) -> set[str]:
        """Determine the cached repo files to be deleted b/c the checksum is None or different.

        Returns:
            set of repository names to be deleted.
        """
        to_be_deleted = set()
        for name, file_checksum in self._cached_checksums["repo"].items():
            if not file_checksum or file_checksum != self._repo_store.checksum(name):
                to_be_deleted.add(name)
        return to_be_deleted

    @_raise_if(readonly=True)
    def _delete_cached_repo_files(self, to_be_deleted: set[str]) -> None:
        """Delete the given repository files.

        Args:
            to_be_deleted: set of repository names to be deleted.
        """
        for name in to_be_deleted:
            L.info("Deleting cached repo %s", name)
            self._repo_store.delete(name)
            del self._cached_checksums["repo"][name]

    def _check_cached_features_files(self) -> set[str]:
        """Determine the cached features files to be deleted b/c the checksum is None or different.

        Returns:
            set of features checksums to be deleted.
        """
        to_be_deleted = set()
        for config_checksum, checksums_by_name in self._cached_checksums["features"].items():
            for name, file_checksum in checksums_by_name.items():
                if not file_checksum or file_checksum != self._features_store.checksum(name):
                    to_be_deleted.add(config_checksum)
                    break
        return to_be_deleted

    @_raise_if(readonly=True)
    def _delete_cached_features_files(self, to_be_deleted) -> None:
        """Delete the files corresponding to the given features checksums.

        Args:
            to_be_deleted: set of features checksums to be deleted.
        """
        for config_checksum in to_be_deleted:
            # delete every feature generated with the same configuration
            for name in self._cached_checksums["features"].pop(config_checksum):
                L.info("Deleting invalid cached features %s/%s", config_checksum[:8], name)
                self._features_store.delete(name)

    def _initialize_cache(self) -> None:
        """Initialize the cache."""
        L.info("Initialize cache")
        is_config_cache_valid = self._check_config_cache()
        repo_to_be_deleted = self._check_cached_repo_files()
        features_to_be_deleted = self._check_cached_features_files()
        if repo_to_be_deleted:
            self._delete_cached_repo_files(repo_to_be_deleted)
        if features_to_be_deleted:
            self._delete_cached_features_files(features_to_be_deleted)
        if not is_config_cache_valid:
            self._dump_analysis_config()
            self._dump_simulations_config()
            self._dump_cached_checksums()

    @_raise_if(locked=False)
    def is_repo_cached(self, name: str) -> bool:
        """Return whether a specific repo dataframe is present in the cache."""
        # the checksums have been checked in _initialize_cache/_delete_cached_repo_files,
        # so they are not calculate again here
        return bool(
            self._cached_checksums["repo"].get(name) and self._repo_store.path(name).is_file()
        )

    @_raise_if(locked=False)
    def load_repo(self, name: str) -> Optional[pd.DataFrame]:
        """Load a specific repo dataframe from the cache.

        Args:
            name: name of the repo dataframe.

        Returns:
            The loaded dataframe, or None if it's not cached.
        """
        is_cached = self.is_repo_cached(name)
        return self._repo_store.load(name) if is_cached else None

    @_raise_if(readonly=True)
    @_raise_if(locked=False)
    def dump_repo(self, df: pd.DataFrame, name: str) -> None:
        """Write a specific repo dataframe to the cache.

        Args:
            df: dataframe to be saved.
            name: name of the repo dataframe.
        """
        self._repo_store.dump(df, name)
        self._cached_checksums["repo"][name] = self._repo_store.checksum(name)
        self._dump_cached_checksums()

    @_raise_if(locked=False)
    def get_cached_features_checksums(
        self, features_config: FeaturesConfig
    ) -> dict[str, dict[str, str]]:
        """Return the cached features checksums, or an empty dict if the cache doesn't exist."""
        config_checksum = features_config.checksum()
        cached = self._cached_checksums["features"].get(config_checksum, {})
        return cached

    @_raise_if(locked=False)
    def load_features(self, features_config: FeaturesConfig) -> Optional[dict[str, pd.DataFrame]]:
        """Load features dataframes from the cache.

        The cache key is determined by the hash of features_config.

        Args:
            features_config: configuration dict of the features to be loaded.

        Returns:
            Dict of dataframes, or None if they are not cached.
        """
        cached_checksums = self.get_cached_features_checksums(features_config)
        features = {}
        # the checksums have been checked in _initialize_cache/_delete_cached_features_files,
        # so they are not calculate again here
        for name, file_checksum in cached_checksums.items():
            assert file_checksum is not None
            features[name] = self._features_store.load(name)
            assert features[name] is not None
        return features or None

    @_raise_if(readonly=True)
    @_raise_if(locked=False)
    def dump_features(
        self, features_dict: dict[str, pd.DataFrame], features_config: FeaturesConfig
    ) -> None:
        """Write features dataframes to the cache.

        The cache key is determined by the hash of features_config.

        Args:
            features_dict: dict of features to be written.
            features_config: configuration dict of the features to be written.
        """
        config_checksum = features_config.checksum()
        old_checksums = self._cached_checksums["features"].pop(config_checksum, None)
        new_checksums = self._cached_checksums["features"][config_checksum] = {}
        for name, feature in features_dict.items():
            self._features_store.dump(feature, name)
            new_checksums[name] = self._features_store.checksum(name)
        if old_checksums is not None:
            assert (
                len(set(old_checksums).difference(new_checksums)) == 0
            ), "Some features have been found only in the old cached data"
            assert (
                len(set(new_checksums).difference(old_checksums)) == 0
            ), "Some features have been found only in the new cached data"
        self._dump_cached_checksums()

    def _is_subfilter(self, strict: bool) -> bool:
        """Check whether the actual filter is more or less specific than the cached filter.

        Args:
            strict: affects the result only when the two filters Actual and Cached are equal.
                If True, the filter Actual isn't considered a subfilter of Cached.
                If False, the filter Actual is considered a subfilter of Cached.

        Returns:
            True if the actual filter is more specific than the cached filter.
            False if the actual filter is less specific than the cached filter.
        """
        if not self._analysis_configs.cached:
            return False
        actual_filter = self._analysis_configs.actual.simulations_filter
        cached_filter = self._analysis_configs.cached.simulations_filter
        return is_subfilter(actual_filter, cached_filter, strict=strict)

    @_raise_if(locked=False)
    def repo_cache_needs_filter(self, name: str) -> bool:
        """Return True if the cached repo needs to be filtered.

        This happens when the cache is used, but the actual filter
        is more specific than the cached filter.
        """
        return self.is_repo_cached(name) and self._is_subfilter(strict=True)

    @_raise_if(locked=False)
    def features_cache_needs_filter(self, features_config: FeaturesConfig) -> bool:
        """Return True if the cached features need to be filtered.

        This happens when the cache is used, but the actual filter
        is more specific than the cached filter.
        """
        cached_checksums = self.get_cached_features_checksums(features_config)
        return len(cached_checksums) > 0 and self._is_subfilter(strict=True)
