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

    def run(self):
        self.repo.extract()
        self.repo.print()
        self.features.calculate()
        self.features.print()
