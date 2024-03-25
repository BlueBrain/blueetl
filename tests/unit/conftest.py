import pytest

from blueetl.campaign.config import SimulationCampaign
from blueetl.config.analysis import init_multi_analysis_configuration
from blueetl.config.analysis_model import FeaturesConfig
from blueetl.features import FeaturesCollection
from blueetl.repository import Repository
from blueetl.utils import load_yaml
from tests.unit.utils import TEST_DATA_PATH, PicklableMock


@pytest.fixture
def lazy_fixture(request):
    """Return a function that returns the actual fixture from its name."""
    return lambda name: request.getfixturevalue(name)


@pytest.fixture
def global_config():
    config_path = TEST_DATA_PATH / "analysis" / "analysis_config_01_relative.yaml"
    return init_multi_analysis_configuration(load_yaml(config_path), config_path.parent)


@pytest.fixture
def repo(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    extraction_config = global_config.analysis["spikes"].extraction
    cache_manager = PicklableMock(
        load_repo=PicklableMock(return_value=None),
        load_features=PicklableMock(return_value=None),
        get_cached_features_checksums=PicklableMock(return_value={}),
    )
    simulations_filter = global_config.simulations_filter
    resolver = PicklableMock()

    return Repository(
        simulations_config=simulations_config,
        extraction_config=extraction_config,
        cache_manager=cache_manager,
        simulations_filter=simulations_filter,
        resolver=resolver,
    )


@pytest.fixture
def features(repo):
    features_configs = [
        FeaturesConfig(
            type="multi",
            groupby=["simulation_id", "circuit_id", "neuron_class", "window"],
            function="blueetl.external.bnac.calculate_features.calculate_features_multi",
            params={"export_all_neurons": True},
            neuron_classes=["L2_X", "L6_Y"],
            windows=["w0", "w1"],
        )
    ]
    return FeaturesCollection(
        features_configs=features_configs,
        repo=repo,
        cache_manager=repo.cache_manager,
    )


@pytest.fixture
def features_with_suffixes(repo):
    features_configs = [
        FeaturesConfig(
            type="multi",
            groupby=["simulation_id", "circuit_id", "neuron_class", "window"],
            function="blueetl.external.bnac.calculate_features.calculate_features_multi",
            params={"export_all_neurons": True},
            neuron_classes=["L2_X", "L6_Y"],
            windows=["w0", "w1"],
            suffix="_0",
        ),
        FeaturesConfig(
            type="multi",
            groupby=["simulation_id", "circuit_id", "neuron_class", "window"],
            function="blueetl.external.bnac.calculate_features.calculate_features_multi",
            params={"export_all_neurons": False},
            neuron_classes=["L2_X", "L6_Y"],
            windows=["w0", "w1"],
            suffix="_1",
        ),
    ]
    return FeaturesCollection(
        features_configs=features_configs,
        repo=repo,
        cache_manager=repo.cache_manager,
    )
