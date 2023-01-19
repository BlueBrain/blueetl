"""Analysis Configuration Models."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra, validator

from blueetl.constants import CONFIG_VERSION
from blueetl.utils import checksum_str, dump_yaml, load_yaml

BaseModelT = TypeVar("BaseModelT", bound="BaseModel")


class BaseModel(PydanticBaseModel):
    """Custom BaseModel."""

    class Config:
        """Custom Model Config."""

        extra = Extra.forbid
        allow_mutation = True
        allow_inf_nan = False
        validate_assignment = True

    def __getitem__(self, name):
        """Allow to get the fields of the model as items."""
        # note: self.__fields__ doesn't contain extra fields even when they are allowed
        return self.__dict__[name]

    def dump(self, path: Path) -> None:
        """Dump the model to file in yaml format."""
        dump_yaml(path, self)

    @classmethod
    def load(cls: Type[BaseModelT], path: Path) -> BaseModelT:
        """Load the model from file in yaml format."""
        return cls.parse_obj(load_yaml(path))

    def checksum(self):
        """Calculate the checksum of the model."""
        return checksum_str(self.json(sort_keys=True))


class ReportConfig(BaseModel):
    """ReportConfig Model."""

    type: str
    name: str = ""


class WindowConfig(BaseModel):
    """WindowConfig Model."""

    initial_offset: float = 0.0
    bounds: Tuple[float, float]
    t_step: float = 0.0
    n_trials: int = 1
    trial_steps_value: float = 0.0
    trial_steps_label: str = ""
    window_type: str = ""


class TrialStepsConfig(BaseModel):
    """TrialStepsConfig Model."""

    class Config(BaseModel.Config):
        """Custom Model Config."""

        extra = Extra.allow

    function: str
    initial_offset: float = 0.0
    bounds: Tuple[float, float]


class ExtractionConfig(BaseModel):
    """ExtractionConfig Model."""

    report: ReportConfig
    neuron_classes: Dict[str, Dict[str, Any]] = {}
    limit: Optional[int] = None
    target: Optional[str] = None
    windows: Dict[str, Union[str, WindowConfig]] = {}
    trial_steps: Dict[str, TrialStepsConfig] = {}


class FeaturesConfig(BaseModel):
    """FeaturesConfig Model.

    Attributes:
        type: 'single' to calculate a single dataframe of features,
            or 'multi' to calculate multiple dataframes of features (parallelized).
        name: name of the features dataframe, used only if type=multi.
        groupby: columns for aggregation.
        function: Function to be executed to calculate the features.
            If type=single, it should return a dict, where each key is a column in the DataFrame.
            If type=multi, it should accept (repo, key, df, params) and return a dict of DataFrames.
        neuron_classes: list of neuron classes to consider, or empty to consider them all.
        windows: list of windows to consider, or empty to consider them all.
        params: optional dict of params that will be passed to the function.
        params_product: optional dict of params that should be combined with itertools.product.
        params_zip: optional dict of params that should be combined with itertools.zip.
        suffix: suffix to be added to the features DataFrames, used only if type=multi.
    """

    type: str
    name: Optional[str] = None
    groupby: List[str]
    function: str
    neuron_classes: List[str] = []
    windows: List[str] = []
    params: Dict[str, Any] = {}
    params_product: Dict[str, Any] = {}
    params_zip: Dict[str, Any] = {}
    suffix: str = ""


class SingleAnalysisConfig(BaseModel):
    """SingleAnalysisConfig Model."""

    output: Optional[Path] = None
    simulations_filter: Dict[str, Any] = {}
    simulations_filter_in_memory: Dict[str, Any] = {}
    extraction: ExtractionConfig
    features: List[FeaturesConfig] = []


class MultiAnalysisConfig(BaseModel):
    """MultiAnalysisConfig Model."""

    version: int
    simulation_campaign: Path
    output: Path
    simulations_filter: Dict[str, Any] = {}
    simulations_filter_in_memory: Dict[str, Any] = {}
    analysis: Dict[str, SingleAnalysisConfig]

    @validator("version")
    def version_match(cls, version):
        """Verify that the config version is supported."""
        if version != CONFIG_VERSION:
            raise ValueError(f"Only config version {CONFIG_VERSION} is supported.")
        return version


if __name__ == "__main__":
    import yaml

    # print the json schema of the model in yaml format,
    # to be used only to update the current schema
    print(yaml.safe_dump(MultiAnalysisConfig.schema(), sort_keys=False))
