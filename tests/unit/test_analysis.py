import pickle
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from blueetl import analysis as test_module
from blueetl.config.analysis_model import MultiAnalysisConfig
from tests.unit.utils import TEST_DATA_PATH


def _prepare_env(path):
    """Copy test data to the given path, and return the path to the analysis config."""
    (path / "circuit").mkdir()
    (path / "analysis").mkdir()
    shutil.copytree(TEST_DATA_PATH / "circuit" / "sonata", path / "circuit" / "sonata")
    shutil.copytree(TEST_DATA_PATH / "simulation_campaign", path / "simulation_campaign")
    return Path(
        shutil.copy(
            TEST_DATA_PATH / "analysis" / "analysis_config_01_relative.yaml",
            path / "analysis",
        )
    )


@pytest.mark.parametrize("skip_features_cache", [True, False, None])
@pytest.mark.parametrize("readonly_cache", [True, False, None])
@pytest.mark.parametrize("clear_cache", [True, False, None])
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("calculate", [True, False])
@pytest.mark.parametrize("extract", [True, False])
@patch.object(test_module.MultiAnalyzer, "from_file")
def test_run_from_file(
    from_file,
    tmp_path,
    extract,
    calculate,
    show,
    clear_cache,
    readonly_cache,
    skip_features_cache,
):
    analysis_config_file = tmp_path / "config.yaml"
    analysis_config_file.write_text("---")

    instance = test_module.run_from_file(
        analysis_config_file=analysis_config_file,
        extract=extract,
        calculate=calculate,
        show=show,
        clear_cache=clear_cache,
        readonly_cache=readonly_cache,
        skip_features_cache=skip_features_cache,
    )

    from_file.assert_called_once_with(
        analysis_config_file,
        extra_params={
            "clear_cache": clear_cache,
            "readonly_cache": readonly_cache,
            "skip_features_cache": skip_features_cache,
        },
    )
    assert instance.extract_repo.call_count == int(extract)
    assert instance.calculate_features.call_count == int(calculate)
    assert instance.show.call_count == int(show)


def test_multi_analyzer_from_file(tmp_path):
    path = _prepare_env(tmp_path)
    with test_module.MultiAnalyzer.from_file(path, extra_params={}) as ma:
        assert isinstance(ma, test_module.MultiAnalyzer)
        assert isinstance(ma.global_config, MultiAnalysisConfig)
        assert isinstance(ma.analyzers, dict)
        assert ma.names == ["spikes"]
        assert isinstance(ma.spikes, test_module.Analyzer)

        filtered = ma.apply_filter()
        assert isinstance(filtered, test_module.MultiAnalyzer)
        assert filtered is ma

        filtered = ma.apply_filter(simulations_filter={"seed": 334630})
        assert isinstance(filtered, test_module.MultiAnalyzer)
        assert filtered is not ma

        dumped = pickle.dumps(ma)
        loaded = pickle.loads(dumped)

        assert isinstance(loaded, test_module.MultiAnalyzer)
