"""Repository."""

import logging
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generic, Optional

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.campaign.config import SimulationCampaign
from blueetl.config.analysis_model import ExtractionConfig
from blueetl.constants import CIRCUIT_ID, SIMULATION_ID, SIMULATION_PATH
from blueetl.extract.base import ExtractorT
from blueetl.extract.compartment_report import CompartmentReport
from blueetl.extract.neuron_classes import NeuronClasses
from blueetl.extract.neurons import Neurons
from blueetl.extract.report import ReportExtractor
from blueetl.extract.simulations import Simulations
from blueetl.extract.soma_report import SomaReport
from blueetl.extract.spikes import Spikes
from blueetl.extract.windows import Windows
from blueetl.resolver import Resolver
from blueetl.utils import timed

L = logging.getLogger(__name__)


class BaseExtractor(ABC, Generic[ExtractorT]):
    """BaseExtractor class."""

    def __init__(self, repo: "Repository") -> None:
        """Initialize the object."""
        self._repo = repo

    @abstractmethod
    def extract_new(self) -> ExtractorT:
        """Instantiate an object from the configuration."""

    @abstractmethod
    def extract_cached(self, df: pd.DataFrame) -> ExtractorT:
        """Instantiate an object from a cached DataFrame."""

    def extract(self, name: str) -> ExtractorT:
        """Return an object extracted from the cache or as new.

        Args:
            name: name of the dataframe.
        """
        with timed(L.info, "Extracting %s", name):
            df = self._repo.cache_manager.load_repo(name)
            if df is not None:
                instance = self.extract_cached(df)
            else:
                instance = self.extract_new()
            assert instance is not None, "The extraction didn't return a valid instance."
            is_cached = instance._cached  # pylint: disable=protected-access
            is_filtered = instance._filtered  # pylint: disable=protected-access
            if not is_cached or is_filtered:
                self._repo.cache_manager.dump_repo(df=instance.to_pandas(), name=name)
            L.info("Extracting %s: cached=%s, filtered=%s", name, is_cached, is_filtered)
            return instance


class SimulationsExtractor(BaseExtractor[Simulations]):
    """SimulationsExtractor class."""

    def extract_new(self) -> Simulations:
        """Instantiate an object from the configuration."""
        return Simulations.from_config(
            config=self._repo.simulations_config,
            query=self._repo.simulations_filter,
        )

    def extract_cached(self, df: pd.DataFrame) -> Simulations:
        """Instantiate an object from a cached DataFrame."""
        return Simulations.from_pandas(df, query=self._repo.simulations_filter, cached=True)


class NeuronsExtractor(BaseExtractor[Neurons]):
    """NeuronsExtractor class."""

    def extract_new(self) -> Neurons:
        """Instantiate an object from the configuration."""
        return Neurons.from_simulations(
            simulations=self._repo.simulations,
            neuron_classes=self._repo.extraction_config.neuron_classes,
        )

    def extract_cached(self, df: pd.DataFrame) -> Neurons:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            selected_sims = self._repo.simulations.df.etl.q(simulation_id=self._repo.simulation_ids)
            query = {CIRCUIT_ID: sorted(set(selected_sims[CIRCUIT_ID]))}
        return Neurons.from_pandas(df, query=query, cached=True)


class NeuronClassesExtractor(BaseExtractor[NeuronClasses]):
    """NeuronClassesExtractor class."""

    def extract_new(self) -> NeuronClasses:
        """Instantiate an object from the configuration."""
        return NeuronClasses.from_neurons(
            neurons=self._repo.neurons, neuron_classes=self._repo.extraction_config.neuron_classes
        )

    def extract_cached(self, df: pd.DataFrame) -> NeuronClasses:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            selected_sims = self._repo.simulations.df.etl.q(simulation_id=self._repo.simulation_ids)
            query = {CIRCUIT_ID: sorted(set(selected_sims[CIRCUIT_ID]))}
        return NeuronClasses.from_pandas(df, query=query, cached=True)


class WindowsExtractor(BaseExtractor[Windows]):
    """WindowsExtractor class."""

    def extract_new(self) -> Windows:
        """Instantiate an object from the configuration."""
        assert self._repo.resolver is not None
        return Windows.from_simulations(
            simulations=self._repo.simulations,
            windows_config=self._repo.extraction_config.windows,
            trial_steps_config=self._repo.extraction_config.trial_steps,
            resolver=self._repo.resolver,
        )

    def extract_cached(self, df: pd.DataFrame) -> Windows:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            query = {SIMULATION_ID: self._repo.simulation_ids}
        return Windows.from_pandas(df, query=query, cached=True)


class SpikesExtractor(BaseExtractor[Spikes]):
    """SpikesExtractor class."""

    def extract_new(self) -> Spikes:
        """Instantiate an object from the configuration."""
        return Spikes.from_simulations(
            simulations=self._repo.simulations,
            neurons=self._repo.neurons,
            windows=self._repo.windows,
            neuron_classes=self._repo.neuron_classes,
            name=self._repo.extraction_config.report.name,
        )

    def extract_cached(self, df: pd.DataFrame) -> Spikes:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            query = {SIMULATION_ID: self._repo.simulation_ids}
        return Spikes.from_pandas(df, query=query, cached=True)


class SomaReportExtractor(BaseExtractor[SomaReport]):
    """SomaReportExtractor class."""

    def extract_new(self) -> SomaReport:
        """Instantiate an object from the configuration."""
        return SomaReport.from_simulations(
            simulations=self._repo.simulations,
            neurons=self._repo.neurons,
            windows=self._repo.windows,
            neuron_classes=self._repo.neuron_classes,
            name=self._repo.extraction_config.report.name,
        )

    def extract_cached(self, df: pd.DataFrame) -> SomaReport:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            query = {SIMULATION_ID: self._repo.simulation_ids}
        return SomaReport.from_pandas(df, query=query, cached=True)


class CompartmentReportExtractor(BaseExtractor[CompartmentReport]):
    """CompartmentReportExtractor class."""

    def extract_new(self) -> CompartmentReport:
        """Instantiate an object from the configuration."""
        return CompartmentReport.from_simulations(
            simulations=self._repo.simulations,
            neurons=self._repo.neurons,
            windows=self._repo.windows,
            neuron_classes=self._repo.neuron_classes,
            name=self._repo.extraction_config.report.name,
        )

    def extract_cached(self, df: pd.DataFrame) -> CompartmentReport:
        """Instantiate an object from a cached DataFrame."""
        query = {}
        if self._repo.simulations_filter:
            query = {SIMULATION_ID: self._repo.simulation_ids}
        return CompartmentReport.from_pandas(df, query=query, cached=True)


class Repository:
    """Repository class."""

    def __init__(
        self,
        simulations_config: SimulationCampaign,
        extraction_config: ExtractionConfig,
        cache_manager: CacheManager,
        simulations_filter: Optional[dict[str, Any]] = None,
        resolver: Optional[Resolver] = None,
    ) -> None:
        """Initialize the repository.

        Args:
            simulations_config: simulation campaign configuration.
            extraction_config: extraction configuration.
            cache_manager: cache manager responsible to load and dump dataframes.
            simulations_filter: optional simulations filter.
            resolver: resolver instance.
        """
        self._extraction_config = extraction_config
        self._simulations_config = simulations_config
        self._cache_manager = cache_manager
        self._simulations_filter = simulations_filter
        self._resolver = resolver
        report_type = extraction_config.report.type
        available_reports: dict[str, type[BaseExtractor]] = {
            "spikes": SpikesExtractor,
            "soma": SomaReportExtractor,
            "compartment": CompartmentReportExtractor,
        }
        self._mapping: dict[str, type[BaseExtractor]] = {
            "simulations": SimulationsExtractor,
            "neurons": NeuronsExtractor,
            "neuron_classes": NeuronClassesExtractor,
            "windows": WindowsExtractor,
            "report": available_reports[report_type],
        }
        self._names = list(self._mapping)

    def __getstate__(self) -> dict:
        """Get the object state when the object is pickled."""
        if not self.is_extracted():
            # ensure that the dataframes are extracted and stored to disk,
            # because we want to be able to use the cached data in the subprocesses.
            L.info("Extracting dataframes before serialization")
            self.extract()
        # Copy the object's state, excluding the unpicklable entries.
        names_set = set(self.names)
        return {k: v for k, v in self.__dict__.items() if k not in names_set}

    def __setstate__(self, state: dict) -> None:
        """Set the object state when the object is unpickled."""
        self.__dict__.update(state)

    @property
    def extraction_config(self) -> ExtractionConfig:
        """Access to the extraction configuration."""
        return self._extraction_config

    @property
    def simulations_config(self) -> SimulationCampaign:
        """Access to the simulation campaign configuration."""
        return self._simulations_config

    @property
    def cache_manager(self) -> CacheManager:
        """Access to the cache manager."""
        return self._cache_manager

    @property
    def simulations_filter(self) -> Optional[dict[str, Any]]:
        """Access to the simulations filter."""
        return self._simulations_filter

    @property
    def names(self) -> list[str]:
        """Return the list of names of the extracted objects."""
        return self._names

    @property
    def resolver(self) -> Optional[Resolver]:
        """Return the resolver."""
        return self._resolver

    @cached_property
    def simulations(self) -> Simulations:
        """Return the Simulations extraction."""
        return self._mapping["simulations"](self).extract(name="simulations")

    @cached_property
    def neurons(self) -> Neurons:
        """Return the Neurons extraction."""
        return self._mapping["neurons"](self).extract(name="neurons")

    @cached_property
    def neuron_classes(self) -> NeuronClasses:
        """Return the NeuronClasses extraction."""
        return self._mapping["neuron_classes"](self).extract(name="neuron_classes")

    @cached_property
    def windows(self) -> Windows:
        """Return the Windows extraction."""
        return self._mapping["windows"](self).extract(name="windows")

    @property
    def spikes(self) -> ReportExtractor:
        """Return the Spikes extraction."""
        warnings.warn(
            "Accessing Repository.spikes is deprecated, please use Repository.report",
            FutureWarning,
            stacklevel=2,
        )
        assert isinstance(self.report, Spikes)
        return self.report

    @cached_property
    def report(self) -> ReportExtractor:
        """Return the Report extraction."""
        return self._mapping["report"](self).extract(name="report")

    @property
    def simulation_ids(self) -> list[int]:
        """Return the list of simulation ids, possibly filtered."""
        return self.simulations.df[SIMULATION_ID].to_list()

    def extract(self) -> None:
        """Extract all the dataframes."""
        for name in self.names:
            getattr(self, name)
        self.check_extractions()

    def is_extracted(self) -> bool:
        """Return True if all the dataframes have been extracted."""
        # note: the cached_property is stored as an attribute after it's accessed
        return all(name in self.__dict__ for name in self.names)

    def check_extractions(self) -> None:
        """Check that all the dataframes have been extracted."""
        if not self.is_extracted():
            raise RuntimeError("Not all the dataframes have been extracted")

    def missing_simulations(self) -> pd.DataFrame:
        """Return a DataFrame with the simulations ignored because of missing spikes.

        Returns:
            pd.DataFrame with the simulation conditions and simulation_path as columns,
                and one record for each ignored and missing simulation.
        """
        all_simulations = self._simulations_config.get()
        extracted_simulations = self.simulations.df[[SIMULATION_PATH]]
        return (
            pd.merge(
                all_simulations,
                extracted_simulations,
                left_on=[SIMULATION_PATH],
                right_on=[SIMULATION_PATH],
                how="left",
                indicator=True,
            )
            .etl.q(_merge="left_only")
            .drop(columns="_merge")
        )

    def show(self) -> None:
        """Print some information about the instance, mainly for debug and inspection."""
        for name in self.names:
            print("~" * 80)
            print("Extraction:", name)
            print(getattr(self, name).df)

    def apply_filter(self, simulations_filter: dict[str, Any]) -> "Repository":
        """Apply the given filter and return a new object."""
        return FilteredRepository(parent=self, simulations_filter=simulations_filter)


class FilteredRepository(Repository):
    """FilteredRepository class."""

    def __init__(self, parent: Repository, simulations_filter: dict[str, Any]) -> None:
        """Initialize the object using the given dict of DataFrames.

        Filtered dataframes are never written to disk.
        """
        super().__init__(
            simulations_config=parent.simulations_config,
            extraction_config=parent.extraction_config,
            cache_manager=parent.cache_manager.to_readonly(),
            simulations_filter=simulations_filter,
        )
        dataframes = {name: getattr(parent, name).df for name in parent.names}
        self._assign_from_dataframes(dataframes)

    def _assign_from_dataframes(self, dicts: dict[str, pd.DataFrame]) -> None:
        """Assign the repository properties from the given dict of DataFrames."""
        for name, df in dicts.items():
            assert name not in self.__dict__
            value = self._mapping[name](self).extract_cached(df)
            setattr(self, name, value)
