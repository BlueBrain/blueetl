"""Analysis Configuration Models."""

import json
from pathlib import Path
from typing import Annotated, Any, Optional, TypeVar, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic.functional_validators import field_validator, model_validator

from blueetl.constants import CONFIG_VERSION
from blueetl.utils import checksum_str, dump_yaml, load_yaml

BaseModelT = TypeVar("BaseModelT", bound="BaseModel")


class BaseModel(PydanticBaseModel):
    """Custom BaseModel."""

    model_config = {
        "extra": "forbid",
        "allow_inf_nan": False,
        "validate_assignment": True,
    }

    def dict(self, *args, by_alias=True, **kwargs):
        """Generate a dictionary representation of the model, using by_alias=True by default."""
        return super().model_dump(*args, by_alias=by_alias, **kwargs)

    def json(self, *args, sort_keys=False, **kwargs):
        """Generate a JSON representation of the model, using by_alias=True by default."""
        # use json.dumps because model_dump_json in pydantic v2 doesn't support sort_keys
        return json.dumps(self.dict(*args, **kwargs), sort_keys=sort_keys)

    def dump(self, path: Path) -> None:
        """Dump the model to file in yaml format."""
        dump_yaml(path, self)

    @classmethod
    def load(cls: type[BaseModelT], path: Path) -> BaseModelT:
        """Load the model from file in yaml format."""
        return cls.model_validate(load_yaml(path))

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
    bounds: tuple[float, float]
    t_step: float = 0.0
    n_trials: int = 0
    trial_steps_value: float = 0.0
    trial_steps_list: list[float] = []
    trial_steps_label: str = ""
    window_type: str = ""

    @model_validator(mode="after")
    def validate_values(self):
        """Validate the values after loading them."""
        if self.trial_steps_list and (self.n_trials or self.trial_steps_value):
            raise ValueError("trial_steps_list cannot be set with n_trials or trial_steps_value")
        if self.n_trials > 1 and not self.trial_steps_value:
            raise ValueError("trial_steps_value cannot be 0 when n_trials > 1")
        return self


class TrialStepsConfig(BaseModel):
    """TrialStepsConfig Model."""

    model_config = {
        **BaseModel.model_config,
        "extra": "allow",
    }
    _forbidden_extra_fields: set[str] = {
        "initial_offset",
    }
    function: str
    bounds: tuple[float, float]
    population: Optional[str] = None
    node_set: Optional[str] = None
    node_sets_file: Optional[Path] = None
    node_sets_checksum: Optional[str] = None  # to invalidate the cache when the file changes
    limit: Optional[int] = None

    @model_validator(mode="after")
    def forbid_fields(self):
        """Verify that the forbidden extra fields have not been specified."""
        if found := self._forbidden_extra_fields.intersection(self.model_extra):
            raise ValueError(f"Forbidden extra fields: {found}")
        return self


class NeuronClassConfig(BaseModel):
    """NeuronClassConfig Model."""

    query: Union[dict[str, Any], list[dict[str, Any]]] = {}
    population: Optional[str] = None
    node_set: Optional[str] = None
    node_sets_file: Optional[Path] = None
    node_sets_checksum: Optional[str] = None  # to invalidate the cache when the file changes
    limit: Optional[int] = None
    node_id: Optional[list[int]] = None


class ExtractionConfig(BaseModel):
    """ExtractionConfig Model."""

    report: ReportConfig
    neuron_classes: dict[str, NeuronClassConfig] = {}
    windows: dict[str, Union[str, WindowConfig]] = {}
    trial_steps: dict[str, TrialStepsConfig] = {}

    @model_validator(mode="before")
    @classmethod
    def propagate_global_values(cls, values):
        """Propagate global values to each dictionary in neuron_classes and trial_steps."""
        for key in ["population", "node_set", "node_sets_file", "limit"]:
            if key in values:
                value = values.pop(key)
                for inner_key in ["neuron_classes", "trial_steps"]:
                    if inner_key in values:
                        for inner in values[inner_key].values():
                            inner.setdefault(key, value)
        return values


class FeaturesConfig(BaseModel):
    """FeaturesConfig Model."""

    id: Annotated[Optional[int], Field(exclude=True)] = None  # do not consider in the checksum
    type: str
    name: Optional[str] = None
    groupby: list[str]
    function: str
    neuron_classes: list[str] = []
    windows: list[str] = []
    params: dict[str, Any] = {}
    params_product: dict[str, Any] = {}
    params_zip: dict[str, Any] = {}
    suffix: str = ""


class SingleAnalysisConfig(BaseModel):
    """SingleAnalysisConfig Model."""

    output: Optional[Path] = None
    simulations_filter: dict[str, Any] = {}
    simulations_filter_in_memory: dict[str, Any] = {}
    extraction: ExtractionConfig
    features: list[FeaturesConfig] = []
    custom: dict[str, Any] = {}

    @field_validator("features")
    @classmethod
    def assign_features_config_id(cls, lst):
        """Assign an incremental id to each FeaturesConfig."""
        for i, item in enumerate(lst):
            item.id = i
        return lst


class MultiAnalysisConfig(BaseModel):
    """MultiAnalysisConfig Model."""

    version: int
    simulation_campaign: Path
    output: Path
    clear_cache: Annotated[bool, Field(exclude=True)] = False  # do not consider in the checksum
    simulations_filter: dict[str, Any] = {}
    simulations_filter_in_memory: dict[str, Any] = {}
    analysis: dict[str, SingleAnalysisConfig]
    custom: dict[str, Any] = {}

    @field_validator("version")
    @classmethod
    def version_match(cls, version):
        """Verify that the config version is supported."""
        if version != CONFIG_VERSION:
            raise ValueError(f"Only config version {CONFIG_VERSION} is supported.")
        return version


if __name__ == "__main__":
    import yaml

    # print the json schema of the model in yaml format,
    # to be used only to update the current schema
    print(yaml.safe_dump(MultiAnalysisConfig.model_json_schema(), sort_keys=False))
