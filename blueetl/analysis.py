"""Analysis functions."""
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.config.simulations import SimulationsConfig
from blueetl.extract.types import StrOrPath
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository

L = logging.getLogger(__name__)


class Analyzer:
    """Analyzer class."""

    def __init__(self, analysis_config: Dict, base_path: StrOrPath = ".") -> None:
        """Initialize the Analyzer from the given configuration.

        Args:
            analysis_config: analysis configuration.
            base_path: base path used to resolve relative paths. If omitted, the cwd is used.
        """
        analysis_config = deepcopy(analysis_config)
        analysis_config = self._validate_config(analysis_config)
        analysis_config = self._resolve_paths(analysis_config, base_path=Path(base_path))
        simulations_config = SimulationsConfig.load(analysis_config["simulation_campaign"])
        simulations_filter = analysis_config.get("simulations_filter", {})
        cache_manager = CacheManager(
            analysis_config=analysis_config,
            simulations_config=simulations_config,
        )
        self.repo = Repository(
            simulations_config=simulations_config,
            extraction_config=analysis_config["extraction"],
            cache_manager=cache_manager,
            simulations_filter=simulations_filter,
        )
        self.features = FeaturesCollection(
            features_configs=analysis_config["analysis"]["features"],
            repo=self.repo,
            cache_manager=cache_manager,
        )
        self.analysis_config = analysis_config

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

    def extract_repo(self, debug: bool = False) -> None:
        """Extract all the repositories dataframes."""
        self.repo.extract()
        if debug:
            self.repo.print()

    def calculate_features(self, debug: bool = False) -> None:
        """Calculate all the features defined in the configuration."""
        self.features.calculate()
        if debug:
            self.features.print()

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
