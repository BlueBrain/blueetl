import logging
from pathlib import Path

from blueetl.config.simulations import SimulationsConfig
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository

L = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, analysis_config, use_cache=False):
        self.analysis_config = analysis_config
        self.repo = Repository(
            simulations_config=SimulationsConfig.load(analysis_config["simulation_campaign"]),
            extraction_config=analysis_config["extraction"],
            store_dir=Path(self.analysis_config["output"]) / "repo",
            use_cache=use_cache,
        )
        self.features = FeaturesCollection(
            features_configs=analysis_config["analysis"]["features"],
            repo=self.repo,
            store_dir=Path(self.analysis_config["output"]) / "features",
            use_cache=use_cache,
        )

    def extract_repo(self):
        self.repo.extract()
        self.repo.print()

    def calculate_features(self):
        self.features.calculate()
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
        key, df = next(self.repo.spikes.df.etl.grouped_by(groupby))
        return key, df
