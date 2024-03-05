import pytest

from blueetl.config import analysis_model as test_module
from blueetl.utils import load_yaml
from tests.functional.utils import TEST_DATA_PATH as TEST_DATA_PATH_FUNCTIONAL
from tests.unit.utils import TEST_DATA_PATH as TEST_DATA_PATH_UNIT


@pytest.mark.parametrize(
    "config_file",
    [
        pytest.param(f, id=f"{f.relative_to(f.parents[2])}")
        for base in (
            TEST_DATA_PATH_UNIT / "analysis",
            TEST_DATA_PATH_FUNCTIONAL / "sonata" / "config",
            TEST_DATA_PATH_FUNCTIONAL / "bbp" / "config",
        )
        for f in base.glob("*.yaml")
    ],
)
def test_analysis_model(config_file):
    config_dict = load_yaml(config_file)
    config = test_module.MultiAnalysisConfig.model_validate(config_dict)
    assert isinstance(config, test_module.MultiAnalysisConfig)
