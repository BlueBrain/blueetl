import pandas as pd
import pytest

from blueetl import cache as test_module
from blueetl.campaign.config import SimulationCampaign
from tests.unit.utils import assert_frame_equal


@pytest.fixture
def cache_manager(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache
    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    yield instance
    instance.close()


def test_lock_manager_exclusive(tmp_path):
    mode_ex = test_module.LockManager.LOCK_EX
    lock_manager_1 = test_module.LockManager(tmp_path)
    lock_manager_2 = test_module.LockManager(tmp_path)
    lock_manager_1.lock(mode_ex)
    # locking again shouldn't do anything
    lock_manager_1.lock(mode_ex)
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        lock_manager_2.lock(mode_ex)
    lock_manager_1.unlock()
    # unlocking again shouldn't do anything
    lock_manager_1.unlock()
    # now lock_manager_2 can lock and unlock
    lock_manager_2.lock(mode_ex)
    lock_manager_2.unlock()


def test_lock_manager_shared(tmp_path):
    mode_ex = test_module.LockManager.LOCK_EX
    mode_sh = test_module.LockManager.LOCK_SH
    lock_manager_1 = test_module.LockManager(tmp_path)
    lock_manager_2 = test_module.LockManager(tmp_path)
    lock_manager_3 = test_module.LockManager(tmp_path)
    # acquire a shared lock
    lock_manager_1.lock(mode_sh)
    # the lock can be shared
    lock_manager_2.lock(mode_sh)
    # but not acquired exclusively
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        lock_manager_3.lock(mode_ex)
    # not even from an already acquired lock
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        lock_manager_1.lock(mode_ex)
    # unless the other shared locks are released
    lock_manager_2.unlock()
    lock_manager_1.lock(mode_ex)
    lock_manager_1.unlock()


def test_cache_manager_init_and_close(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache

    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    assert instance.locked is True
    assert instance.readonly is False
    instance.close()
    assert instance.locked is False
    assert instance.readonly is False


def test_cache_manager_to_readonly(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache

    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    new_instance = instance.to_readonly()

    assert isinstance(instance, test_module.CacheManager)
    assert isinstance(new_instance, test_module.CacheManager)
    assert new_instance is not instance
    assert instance.readonly is False
    assert new_instance.readonly is True

    instance.close()
    with pytest.raises(
        test_module.CacheError,
        match=(
            "Method CacheManager.to_readonly cannot be called "
            "when the attributes are: {'locked': False}"
        ),
    ):
        instance.to_readonly()


def test_cache_manager_concurrency_is_not_allowed_when_locked(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache

    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    # verify that a new instance cannot be created when the old instance is keeping the lock
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        test_module.CacheManager(
            cache_config=cache_config,
            analysis_config=analysis_config,
            simulations_config=simulations_config,
        )
    # verify that a new instance can be created after closing the old instance
    instance.close()
    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    instance.close()


def test_cache_manager_concurrency_is_allowed_when_readonly(global_config):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache.model_copy(update={"readonly": False})
    cache_config_readonly = global_config.cache.model_copy(update={"readonly": True})

    # init the cache that will be used later
    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    instance.close()

    # use the same cache in multiple cache managers
    instances = [
        test_module.CacheManager(
            cache_config=cache_config_readonly,
            analysis_config=analysis_config,
            simulations_config=simulations_config,
        )
        for _ in range(3)
    ]
    for instance in instances:
        instance.close()


def test_cache_manager_clear_cache(global_config, tmp_path):
    simulations_config = SimulationCampaign.load(global_config.simulation_campaign)
    analysis_config = global_config.analysis["spikes"]
    cache_config = global_config.cache.model_copy(update={"clear": False})
    cache_config_clear = global_config.cache.model_copy(update={"clear": True})

    output = cache_config.path
    sentinel = output / "sentinel"

    assert output.exists() is False
    instance = test_module.CacheManager(
        cache_config=cache_config_clear,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is False
    sentinel.touch()

    # reuse the cache
    instance = test_module.CacheManager(
        cache_config=cache_config,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is True

    # delete the cache
    instance = test_module.CacheManager(
        cache_config=cache_config_clear,
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is False


def test_cache_manager_dump_load_repo(cache_manager):
    assert cache_manager.is_repo_cached("simulations") is False
    assert cache_manager.load_repo("simulations") is None

    simulations_df = pd.DataFrame([{"simulation_id": 0, "circuit_id": 0}])
    cache_manager.dump_repo(simulations_df, name="simulations")

    assert cache_manager.is_repo_cached("simulations") is True

    loaded_simulations_df = cache_manager.load_repo("simulations")

    assert_frame_equal(loaded_simulations_df, simulations_df)


def test_cache_manager_dump_load_features(global_config, cache_manager):
    analysis_config = global_config.analysis["spikes"]
    features_config = analysis_config.features[0]

    assert cache_manager.load_features(features_config) is None

    features_dict = {
        "by_gid": pd.DataFrame([{"simulation_id": 0, "circuit_id": 0, "gid": 0}]),
    }
    cache_manager.dump_features(features_dict, features_config=features_config)

    loaded_features_dict = cache_manager.load_features(features_config)

    assert set(features_dict) == set(loaded_features_dict)
    assert_frame_equal(loaded_features_dict["by_gid"], features_dict["by_gid"])
