import pytest

from blueetl.campaign.config import SimulationCampaign
from blueetl.config.analysis import init_multi_analysis_configuration
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
    cache_manager = PicklableMock(load_repo=PicklableMock(return_value=None))
    simulations_filter = global_config.simulations_filter
    resolver = PicklableMock()

    repo = Repository(
        simulations_config=simulations_config,
        extraction_config=extraction_config,
        cache_manager=cache_manager,
        simulations_filter=simulations_filter,
        resolver=resolver,
    )

    assert isinstance(repo, Repository)
    assert repo.extraction_config == extraction_config
    assert repo.simulations_config == simulations_config
    assert repo.cache_manager == cache_manager
    assert repo.simulations_filter == simulations_filter
    assert repo.resolver == resolver
    return repo
