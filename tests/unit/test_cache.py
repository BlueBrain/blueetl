from pathlib import Path

import pandas as pd
import pytest

from blueetl import cache as test_module
from blueetl.config.analysis_model import SingleAnalysisConfig
from blueetl.campaign.config import SimulationCampaign


def _get_analysis_config(path):
    """Return an empty config dict."""
    return SingleAnalysisConfig(
        output=Path(path),
        extraction={"report": {"type": "spikes"}, "population": "root__neurons"},
        features=[],
    )


def _get_simulations_config():
    return SimulationCampaign(
        data=pd.DataFrame(
            [
                {"ca": 1.1, "seed": 1, "simulation_path": "uuid/0"},
                {"ca": 1.2, "seed": 1, "simulation_path": "uuid/1"},
            ]
        ),
        name="uuid",
        attrs={
            "path_prefix": "/tmp/path/to/campaign",
            "circuit_config": "/tmp/path/to/circuit_sonata.json",
        },
    )


def test_lock_manager(tmp_path):
    # lock_dir = tmp_path / "lock"
    lock_manager_1 = test_module.LockManager(tmp_path)
    lock_manager_2 = test_module.LockManager(tmp_path)
    lock_manager_1.lock()
    # locking again shouldn't do anything
    lock_manager_1.lock()
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        lock_manager_2.lock()
    lock_manager_1.unlock()
    # unlocking again shouldn't do anything
    lock_manager_1.unlock()
    # now lock_manager_2 can lock and unlock
    lock_manager_2.lock()
    lock_manager_2.unlock()


def test_cache_manager_init_and_close(tmp_path):
    analysis_config = _get_analysis_config(path=tmp_path)
    simulations_config = _get_simulations_config()

    instance = test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    assert instance.locked is True
    assert instance.readonly is False
    instance.close()
    assert instance.locked is False
    assert instance.readonly is False


def test_cache_manager_to_readonly(tmp_path):
    analysis_config = _get_analysis_config(path=tmp_path)
    simulations_config = _get_simulations_config()

    instance = test_module.CacheManager(
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


def test_cache_manager_concurrency_is_not_allowed_when_locked(tmp_path):
    analysis_config = _get_analysis_config(path=tmp_path)
    simulations_config = _get_simulations_config()

    instance = test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )
    # verify that a new instance cannot be created when the old instance is keeping the lock
    with pytest.raises(test_module.CacheError, match="Another process is locking"):
        test_module.CacheManager(
            analysis_config=analysis_config,
            simulations_config=simulations_config,
        )
    # verify that a new instance can be created after closing the old instance
    instance.close()
    test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
    )


def test_cache_manager_clear_cache(tmp_path):
    output = tmp_path / "output"
    sentinel = output / "sentinel"
    analysis_config = _get_analysis_config(path=output)
    simulations_config = _get_simulations_config()

    assert output.exists() is False
    instance = test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
        clear_cache=True,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is False
    sentinel.touch()

    # reuse the cache
    instance = test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
        clear_cache=False,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is True

    # delete the cache
    instance = test_module.CacheManager(
        analysis_config=analysis_config,
        simulations_config=simulations_config,
        clear_cache=True,
    )
    instance.close()
    assert output.exists() is True
    assert sentinel.exists() is False
