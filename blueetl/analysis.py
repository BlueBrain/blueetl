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


class Analyzer:
    """Analyzer class."""

    def __init__(
        self,
        analysis_config: Dict,
        base_path: StrOrPath = ".",
        _repo: Optional[Repository] = None,
        _features: Optional[FeaturesCollection] = None,
    ) -> None:
        """Initialize the Analyzer from the given configuration.

        Args:
            analysis_config: analysis configuration.
            base_path: base path used to resolve relative paths. If omitted, the cwd is used.
            _repo: if specified, use it instead of creating a new object. Only for internal use.
            _features: if specified, use it instead of creating a new object. Only for internal use.
        """
        analysis_config = deepcopy(analysis_config)
        analysis_config = self._validate_config(analysis_config)
        analysis_config = self._resolve_paths(analysis_config, base_path=Path(base_path))
        self.analysis_config = analysis_config
        if _repo or _features:
            assert _repo and _features, "Both _repo and _features must be specified."
            self.repo = _repo
            self.features = _features
        else:
            simulations_config = SimulationsConfig.load(analysis_config["simulation_campaign"])
            cache_manager = CacheManager(
                analysis_config=analysis_config,
                simulations_config=simulations_config,
            )
            self.repo = Repository(
                simulations_config=simulations_config,
                extraction_config=analysis_config["extraction"],
                cache_manager=cache_manager,
                simulations_filter=self.simulations_filter,
            )
            self.features = FeaturesCollection(
                features_configs=analysis_config["analysis"]["features"],
                repo=self.repo,
                cache_manager=cache_manager,
            )

    @property
    def simulations_filter(self) -> Optional[Dict[str, Any]]:
        """Return the simulations_filter from the analysis configuration."""
        return self.analysis_config.get("simulations_filter", {})

    @property
    def simulations_filter_in_memory(self) -> Optional[Dict[str, Any]]:
        """Return the simulations_filter_in_memory from the analysis configuration."""
        return self.analysis_config.get("simulations_filter_in_memory", {})

    @staticmethod
    def _validate_config(analysis_config: Dict) -> Dict:
        """Validate the configuration."""
        # TODO: add a validation schema
        # TODO: remove these temporary checks before the first release
        assert (
            "simulation_ids" not in analysis_config
        ), "The key simulation_ids has been replaced by simulations_filter->simulation_id"
        assert (
            "simulations" not in analysis_config["extraction"]
        ), "The key extraction->simulations has been replaced by the top level simulations_filter"
        analysis_config.setdefault("analysis", {}).setdefault("features", [])
        return analysis_config

    @staticmethod
    def _resolve_paths(config: Dict, base_path: Path) -> Dict:
        """Resolve any relative path."""
        config["output"] = base_path / config["output"]
        config["simulation_campaign"] = base_path / config["simulation_campaign"]
        return config

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
        simulations_filter = simulations_filter or self.simulations_filter_in_memory
        if not simulations_filter:
            return self
        repo = self.repo.apply_filter(simulations_filter)
        features = self.features.apply_filter(repo)
        return Analyzer(analysis_config=self.analysis_config, _repo=repo, _features=features)

    def try_one(self, groupby: List[str]) -> Tuple[NamedTuple, pd.DataFrame]:
        """Return the first key and df when grouping spikes by the given list of columns.

        The returned parameters are the same passed to the feature function defined by the user.

        It should be used only for internal use and debug.

        Args:
            groupby: list of columns to group by.

        Returns:
            The first key and df.
        """
        key, df = next(self.repo.spikes.df.etl.groupby_iter(groupby))
        return key, df
