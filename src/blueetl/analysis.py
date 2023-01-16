"""Analysis functions."""
import gc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.config.simulations import SimulationsConfig
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository
from blueetl.types import StrOrPath

L = logging.getLogger(__name__)

VERSION = 2


class MultiAnalyzerConfig:
    """MultiAnalyzerConfig class."""

    @classmethod
    def _check_config_version(cls, version: int) -> None:
        """Check that the version of the configuration is supported."""
        if version != VERSION:
            raise ValueError(f"Only version {VERSION} of the analysis configuration is supported.")

    @classmethod
    def _validate_config(cls, config: Dict) -> None:
        """Validate the configuration."""
        cls._check_config_version(int(config.get("version", 1)))
        # TODO: add validation schema
        config.setdefault("simulations_filter", {})
        config.setdefault("simulations_filter_in_memory", {})

    @staticmethod
    def _resolve_paths(config: Dict, base_path: Path) -> None:
        """Resolve any relative path."""
        config["output"] = base_path / config["output"]
        config["simulation_campaign"] = base_path / config["simulation_campaign"]

    @classmethod
    def global_config(cls, config: Dict, base_path: StrOrPath) -> Dict:
        """Validate and return a copy of the global config dict.

        Args:
            config: non-validated global config dict.
            base_path: base path used to resolve relative paths.

        Returns:
            The validated global config dict.
        """
        config = deepcopy(config)
        cls._validate_config(config)
        cls._resolve_paths(config, base_path=Path(base_path))
        return config

    @classmethod
    def partial_config(cls, global_config: Dict, name: str) -> Dict:
        """Build and return a partial config dict.

        Args:
            global_config: global config dict.
            name: name of the analysis.

        Returns:
            The requested partial config dict.
        """
        config = deepcopy(global_config["analysis"][name])
        config["output"] = global_config["output"] / name
        config["simulations_filter"] = global_config["simulations_filter"]
        config["simulations_filter_in_memory"] = global_config["simulations_filter_in_memory"]
        config.setdefault("features", [])
        return config


class Analyzer:
    """Analyzer class."""

    def __init__(
        self,
        analysis_config: Dict,
        simulations_config: Optional[SimulationsConfig] = None,
        _repo: Optional[Repository] = None,
        _features: Optional[FeaturesCollection] = None,
    ) -> None:
        """Initialize the Analyzer from the given configuration.

        Args:
            analysis_config: analysis configuration.
            simulations_config: simulation campaign configuration.
            _repo: if specified, use it instead of creating a new object. Only for internal use.
            _features: if specified, use it instead of creating a new object. Only for internal use.
        """
        self.analysis_config = analysis_config
        if _repo or _features:
            assert _repo and _features, "Both _repo and _features must be specified."
            self.repo = _repo
            self.features = _features
        else:
            assert simulations_config is not None
            cache_manager = CacheManager(
                analysis_config=analysis_config,
                simulations_config=simulations_config,
            )
            self.repo = Repository(
                simulations_config=simulations_config,
                extraction_config=analysis_config["extraction"],
                cache_manager=cache_manager,
                simulations_filter=self.analysis_config["simulations_filter"],
            )
            self.features = FeaturesCollection(
                features_configs=analysis_config["features"],
                repo=self.repo,
                cache_manager=cache_manager,
            )

    def __enter__(self):
        """Initialize the object when used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the object when used as a context manager."""
        self.close()

    def close(self) -> None:
        """Invalidate and unlock the cache.

        After calling this method, the DataFrames already extracted can still be accessed,
        but it's not possible to extract new data or calculate new features.
        """
        self.repo.cache_manager.close()
        self.features.cache_manager.close()

    def extract_repo(self) -> None:
        """Extract all the repositories dataframes."""
        self.repo.extract()

    def calculate_features(self) -> None:
        """Calculate all the features defined in the configuration."""
        self.features.calculate()
        # FIXME: verify if there are reference cycles that can be removed.
        gc.collect()

    def apply_filter(self, simulations_filter: Optional[Dict[str, Any]] = None) -> "Analyzer":
        """Return a new object where the in memory filter is applied to repo and features.

        Before applying the filter, all the repo dataframes are extracted,
        and all the features dataframes are calculated, if not already done.

        Args:
            simulations_filter: optional simulations filter; if not specified, use
                `simulations_filter_in_memory` from the configuration; if neither is specified,
                return the original object.
        """
        self.extract_repo()
        self.calculate_features()
        if not simulations_filter:
            simulations_filter = self.analysis_config["simulations_filter_in_memory"]
        if not simulations_filter:
            return self
        repo = self.repo.apply_filter(simulations_filter)
        features = self.features.apply_filter(repo)
        return Analyzer(analysis_config=self.analysis_config, _repo=repo, _features=features)

    def show(self):
        """Print all the DataFrames."""
        self.repo.show()
        self.features.show()

    def try_one(self, groupby: List[str]) -> Tuple[NamedTuple, pd.DataFrame]:
        """Return the first key and df when grouping spikes by the given list of columns.

        The returned parameters are the same passed to the feature function defined by the user.

        It should be used only for internal use and debug.

        Args:
            groupby: list of columns to group by.

        Returns:
            The first key and df.
        """
        key, df = next(self.repo.report.df.etl.groupby_iter(groupby))
        return key, df


class MultiAnalyzer:
    """MultiAnalyzer class."""

    def __init__(
        self,
        analysis_config: Dict,
        base_path: StrOrPath = ".",
        _analyzers: Optional[Dict[str, Analyzer]] = None,
    ) -> None:
        """Initialize the MultiAnalyzer from the given configuration.

        Args:
            analysis_config: analysis configuration.
            base_path: base path used to resolve relative paths. If omitted, the cwd is used.
            _analyzers: if specified, use it instead of creating a new dict of analyzers.
                Only for internal use.
        """
        self.analysis_config = MultiAnalyzerConfig.global_config(analysis_config, base_path)
        if _analyzers:
            self._analyzers: Dict[str, Analyzer] = _analyzers
        else:
            simulations_config = SimulationsConfig.load(self.analysis_config["simulation_campaign"])
            self._analyzers = {
                name: Analyzer(
                    analysis_config=MultiAnalyzerConfig.partial_config(self.analysis_config, name),
                    simulations_config=simulations_config,
                )
                for name in self.analysis_config["analysis"]
            }

    @property
    def names(self) -> List[str]:
        """Return the names of all the analyzers."""
        return list(self._analyzers)

    def __getattr__(self, name: str) -> Analyzer:
        """Return an analyzer instance by name.

        Args:
            name: name of the analyzer.
        """
        try:
            return self._analyzers[name]
        except KeyError as ex:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            ) from ex

    def __enter__(self):
        """Initialize the object when used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the object when used as a context manager."""
        self.close()

    def close(self) -> None:
        """Invalidate and unlock the cache.

        After calling this method, the DataFrames already extracted can still be accessed,
        but it's not possible to extract new data or calculate new features.
        """
        for a in self._analyzers.values():
            a.close()

    def extract_repo(self) -> None:
        """Extract all the repositories dataframes for all the analysis."""
        for a in self._analyzers.values():
            a.extract_repo()

    def calculate_features(self) -> None:
        """Calculate all the features defined in the configuration for all the analysis."""
        for a in self._analyzers.values():
            a.calculate_features()

    def apply_filter(self, simulations_filter: Optional[Dict[str, Any]] = None) -> "MultiAnalyzer":
        """Return a new object where the in memory filter is applied to repo and features.

        Before applying the filter, all the repo dataframes are extracted,
        and all the features dataframes are calculated, if not already done.

        Args:
            simulations_filter: optional simulations filter; if not specified, use
                `simulations_filter_in_memory` from the configuration; if neither is specified,
                return the original object.
        """
        self.extract_repo()
        self.calculate_features()
        if not simulations_filter:
            simulations_filter = self.analysis_config["simulations_filter_in_memory"]
        if not simulations_filter:
            return self
        analyzers = {
            name: a.apply_filter(simulations_filter) for name, a in self._analyzers.items()
        }
        return MultiAnalyzer(analysis_config=self.analysis_config, _analyzers=analyzers)

    def show(self):
        """Print all the DataFrames."""
        for name in self.names:
            print("#" * 80)
            print("Analysis:", name)
            getattr(self, name).show()
