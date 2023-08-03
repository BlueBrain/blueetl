import re
from itertools import chain

import pytest

from tests.functional.utils import TEST_DATA_PATH


def _get_marks(path, match_pattern, skip_pattern):
    """Return the marks for the specified path."""
    if skip_pattern and re.match(skip_pattern, path.name):
        return pytest.mark.skip(reason="File skipped because matching skip_pattern")
    if match_pattern and not re.match(match_pattern, path.name):
        return pytest.mark.skip(reason="File skipped because not matching match_pattern")
    return ()


def _get_analysis_configs(config_path, expected_path, match_pattern, skip_pattern):
    """Return a list of (analysis_config_path, expected_path) to be used as fixtures."""
    analysis_configs = [
        pytest.param(
            path,
            expected_path / path.stem.replace("analysis_config_", "analysis_"),
            marks=_get_marks(path, match_pattern, skip_pattern),
            id=str(path.relative_to(path.parents[2])),
        )
        for path in sorted(config_path.glob("analysis_config_*.yaml"))
    ]
    # ensure that there are config files to test
    assert len(analysis_configs) >= 1, f"No configuration files found in {config_path}"
    return analysis_configs


def pytest_addoption(parser):
    parser.addoption(
        "--analysis-config-dir",
        action="append",
        default=[],
        help="List of directories containing configurations to test",
    )
    parser.addoption(
        "--match-pattern",
        action="store",
        default="",
        help="Pattern matching the configuration files to be tested",
    )
    parser.addoption(
        "--skip-pattern",
        action="store",
        default="",
        help="Pattern matching the configuration files to be skipped",
    )
    parser.addoption(
        "--force-update",
        action="store_true",
        default=False,
        help="Update the expected files",
    )


def pytest_generate_tests(metafunc):
    # See https://docs.pytest.org/en/latest/example/parametrize.html#generating-parameters-combinations-depending-on-command-line
    if "analysis_config_path" in metafunc.fixturenames or "expected_path" in metafunc.fixturenames:
        parents = metafunc.config.getoption("analysis_config_dir") or ["sonata", "bbp"]
        match_pattern = metafunc.config.getoption("match_pattern")
        skip_pattern = metafunc.config.getoption("skip_pattern")
        analysis_configs = list(
            chain.from_iterable(
                _get_analysis_configs(
                    config_path=TEST_DATA_PATH / parent / "config",
                    expected_path=TEST_DATA_PATH / parent / "expected",
                    match_pattern=match_pattern,
                    skip_pattern=skip_pattern,
                )
                for parent in parents
            )
        )
        metafunc.parametrize("analysis_config_path, expected_path", analysis_configs)


def pytest_collection_modifyitems(config, items):
    # see https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    if config.getoption("force_update"):
        # do not skip any test
        return
    marker = pytest.mark.skip(reason="Not overwriting files, use --force-update if needed")
    for item in items:
        if "force_update" in item.keywords:
            item.add_marker(marker)


def pytest_configure(config):
    config.addinivalue_line("markers", "force_update: mark tests as updating the expected files")
