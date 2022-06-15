import logging
from pathlib import Path

from blueetl.config.simulations import SimulationsConfig
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository

L = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, analysis_config, base_path=".", use_cache=False):
        assert (
            "simulation_ids" not in analysis_config
        ), "The key simulation_ids has been replaced by extraction->simulations->simulation_id"
        self.analysis_config = self._resolve_paths(analysis_config, base_path=base_path)
        self.repo = Repository(
            simulations_config=SimulationsConfig.load(self.analysis_config["simulation_campaign"]),
            extraction_config=self.analysis_config["extraction"],
            store_dir=Path(self.analysis_config["output"], "repo"),
            use_cache=use_cache,
        )
        self.features = FeaturesCollection(
            features_configs=self.analysis_config["analysis"]["features"],
            repo=self.repo,
            store_dir=Path(self.analysis_config["output"], "features"),
            use_cache=use_cache,
        )

    @staticmethod
    def _resolve_paths(config, base_path):
        config["output"] = Path(base_path, config["output"])
        config["simulation_campaign"] = Path(base_path, config["simulation_campaign"])
        return config

    def extract_repo(self, debug=False):
        self.repo.extract()
        if debug:
            self.repo.print()

    def calculate_features(self, debug=False):
        self.features.calculate()
        if debug:
            self.features.print()

    def try_one(self, groupby):
        """Return the first key and df when grouping spikes by the given list of columns.

        The returned parameters are the same passed to the feature function defined by the user.
        It should be used only for debug.

        Args:
            groupby: list of columns to group by.

        Returns:
            The first key and df.
        """
        key, df = next(self.repo.spikes.df.etl.groupby_iter(groupby))
        return key, df
