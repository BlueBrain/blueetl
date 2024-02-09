"""Analysis functions."""

import gc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd

from blueetl.cache import CacheManager
from blueetl.campaign.config import SimulationCampaign
from blueetl.config.analysis import init_multi_analysis_configuration
from blueetl.config.analysis_model import MultiAnalysisConfig, SingleAnalysisConfig
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository
from blueetl.resolver import AttrResolver, Resolver
from blueetl.types import StrOrPath
from blueetl.utils import load_yaml, setup_logging

L = logging.getLogger(__name__)


class Analyzer:
    """Analyzer class."""

    def __init__(
        self,
        analysis_config: SingleAnalysisConfig,
        repo: Repository,
        features: FeaturesCollection,
    ) -> None:
        """Initialize the object.

        Args:
            analysis_config: analysis configuration.
            repo: Repository instance.
            features: FeaturesCollection instance.
        """
        self._analysis_config = analysis_config
        self._repo = repo
        self._features = features

    @classmethod
    def from_config(
        cls,
        analysis_config: SingleAnalysisConfig,
        simulations_config: SimulationCampaign,
        resolver: Resolver,
        clear_cache: bool = False,
    ) -> "Analyzer":
        """Initialize the Analyzer from the given configuration.

        Args:
            analysis_config: analysis configuration.
            simulations_config: simulation campaign configuration.
            resolver: resolver instance.
            clear_cache: if True, remove any existing cache.
        """
        cache_manager = CacheManager(
            analysis_config=analysis_config,
            simulations_config=simulations_config,
            clear_cache=clear_cache,
        )
        repo = Repository(
            simulations_config=simulations_config,
            extraction_config=analysis_config.extraction,
            cache_manager=cache_manager,
            simulations_filter=analysis_config.simulations_filter,
            resolver=resolver,
        )
        features = FeaturesCollection(
            features_configs=analysis_config.features,
            repo=repo,
            cache_manager=cache_manager,
        )
        return cls(
            analysis_config=analysis_config,
            repo=repo,
            features=features,
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

    @property
    def analysis_config(self) -> SingleAnalysisConfig:
        """Return the wrapped analysis configuration."""
        return self._analysis_config

    @property
    def repo(self) -> Repository:
        """Return the wrapped repository."""
        return self._repo

    @property
    def extraction(self) -> Repository:
        """Return the wrapped repository as an alias."""
        return self._repo

    @property
    def features(self) -> FeaturesCollection:
        """Return the wrapped features."""
        return self._features

    def extract_repo(self) -> None:
        """Extract all the repositories dataframes."""
        self.repo.extract()

    def calculate_features(self) -> None:
        """Calculate all the features defined in the configuration."""
        self.features.calculate()
        gc.collect()

    def apply_filter(self, simulations_filter: Optional[dict[str, Any]] = None) -> "Analyzer":
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
            simulations_filter = self.analysis_config.simulations_filter_in_memory
        if not simulations_filter:
            return self
        repo = self.repo.apply_filter(simulations_filter)
        features = self.features.apply_filter(repo)
        return Analyzer(analysis_config=self.analysis_config, repo=repo, features=features)

    def show(self):
        """Print all the DataFrames."""
        self.repo.show()
        self.features.show()

    def try_one(self, groupby: list[str]) -> tuple[NamedTuple, pd.DataFrame]:
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
        global_config: MultiAnalysisConfig,
        analyzers: Optional[dict[str, Analyzer]] = None,
    ) -> None:
        """Initialize the object.

        Args:
            global_config: analysis configuration.
            analyzers: dict of analyzers, or None to load them from the configuration.
        """
        assert isinstance(global_config, MultiAnalysisConfig)
        self._global_config = global_config
        self._analyzers = self._init_analyzers() if analyzers is None else analyzers

    @classmethod
    def from_config(
        cls,
        global_config: dict,
        base_path: StrOrPath,
        clear_cache: Optional[bool] = None,
    ) -> "MultiAnalyzer":
        """Initialize the MultiAnalyzer from the given configuration.

        Args:
            global_config: analysis configuration.
            base_path: base path used to resolve relative paths in the configuration.
            clear_cache: if True, remove any existing cache; if False, reuse the existing cache;
                if None, use the value from the configuration file.
        """
        global_config = init_multi_analysis_configuration(global_config, Path(base_path))
        if clear_cache is not None:
            global_config.clear_cache = clear_cache
        return cls(global_config=global_config)

    def _init_analyzers(self) -> dict[str, Analyzer]:
        """Load and return a dict of analyzers."""
        resolver = AttrResolver(root=self)
        simulations_config = SimulationCampaign.load(self.global_config.simulation_campaign)
        return {
            name: Analyzer.from_config(
                analysis_config=analysis_config,
                simulations_config=simulations_config,
                resolver=resolver,
                clear_cache=self.global_config.clear_cache,
            )
            for name, analysis_config in self.global_config.analysis.items()
        }

    @classmethod
    def from_file(cls, path: StrOrPath, clear_cache: Optional[bool] = None) -> "MultiAnalyzer":
        """Return a new instance loaded using the given configuration file."""
        return cls.from_config(
            global_config=load_yaml(path),
            base_path=Path(path).parent,
            clear_cache=clear_cache,
        )

    @property
    def global_config(self) -> MultiAnalysisConfig:
        """Return the global config instance."""
        return self._global_config

    @property
    def analyzers(self) -> dict[str, Analyzer]:
        """Return the dict of analyzers."""
        return self._analyzers

    @property
    def names(self) -> list[str]:
        """Return the names of all the analyzers."""
        return list(self.analyzers)

    def __getstate__(self) -> dict:
        """Get the object state when the object is pickled."""
        return {"_global_config": self.global_config, "_analyzers": None}

    def __setstate__(self, state: dict) -> None:
        """Set the object state when the object is unpickled."""
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Analyzer:
        """Return an analyzer instance by name.

        Args:
            name: name of the analyzer.
        """
        try:
            return self.analyzers[name]
        except KeyError as ex:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            ) from ex

    def __dir__(self):
        """Allow autocompletion of dynamic attributes."""
        return list(super().__dir__()) + list(self.analyzers)

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
        for a in self.analyzers.values():
            a.close()

    def extract_repo(self) -> None:
        """Extract all the repositories dataframes for all the analysis."""
        for a in self.analyzers.values():
            a.extract_repo()

    def calculate_features(self) -> None:
        """Calculate all the features defined in the configuration for all the analysis."""
        for a in self.analyzers.values():
            a.calculate_features()

    def apply_filter(self, simulations_filter: Optional[dict[str, Any]] = None) -> "MultiAnalyzer":
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
            simulations_filter = self.global_config.simulations_filter_in_memory
        if not simulations_filter:
            return self
        analyzers = {name: a.apply_filter(simulations_filter) for name, a in self.analyzers.items()}
        return MultiAnalyzer(global_config=deepcopy(self.global_config), analyzers=analyzers)

    def show(self):
        """Print all the DataFrames."""
        for name in self.names:
            print("#" * 80)
            print("Analysis:", name)
            getattr(self, name).show()


def run_from_file(
    analysis_config_file: StrOrPath,
    seed: Optional[int] = 0,
    extract: bool = True,
    calculate: bool = True,
    show: bool = False,
    clear_cache: Optional[bool] = None,
    loglevel: Optional[int] = None,
) -> MultiAnalyzer:
    """Initialize and return the MultiAnalyzer.

    Args:
        analysis_config_file: path to the analysis configuration file.
        seed: if not None, random seed used to select random gids.
        extract: if True, run the extraction of the repository.
        calculate: if True, run the calculation of the features.
        show: if True, show a short representation of all the Pandas DataFrames, mainly for debug.
        clear_cache: if True, remove any existing cache; if False, reuse the existing cache;
            if None, use the value from the configuration file.
        loglevel: if specified, used to set up logging.

    Returns:
        a new MultiAnalyzer instance.
    """
    if loglevel is not None:
        setup_logging(loglevel=loglevel, force=True)
    if seed is not None:
        np.random.seed(seed)
    L.info("MultiAnalyzer configuration: %s", analysis_config_file)
    ma = MultiAnalyzer.from_file(analysis_config_file, clear_cache=clear_cache)
    if extract:
        ma.extract_repo()
    if calculate:
        ma.calculate_features()
    if show:
        ma.show()
    return ma
