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
    """FeaturesConfig Model."""

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
    custom: Dict[str, Any] = {}

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
