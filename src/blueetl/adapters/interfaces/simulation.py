"""Interfaces for Simulation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, Optional, TypeVar

import pandas as pd

from blueetl.adapters.interfaces.circuit import CircuitInterface
from blueetl.utils import CachedPropertyMixIn

SimulationT = TypeVar("SimulationT")


class PopulationSpikesReportInterface(ABC):
    """PopulationSpikesReport Interface."""

    @abstractmethod
    def get(self, group=None, t_start=None, t_stop=None) -> pd.Series:
        """Return the spikes report for the specified group and interval."""


class PopulationReportInterface(ABC):
    """PopulationReport Interface."""

    @abstractmethod
    def get(self, group=None, t_start=None, t_stop=None, t_step=None) -> pd.DataFrame:
        """Return the report for the specified group and interval."""


class SimulationInterface(Generic[SimulationT], CachedPropertyMixIn, ABC):
    """Simulation Interface."""

    def __init__(self, simulation: SimulationT) -> None:
        """Init the simulation interface with the specified simulation."""
        self._simulation = simulation

    @property
    def instance(self) -> SimulationT:
        """Return the wrapped instance."""
        return self._simulation

    @abstractmethod
    def is_complete(self) -> bool:
        """Return True if the simulation is complete, False otherwise."""

    @property
    @abstractmethod
    def circuit(self) -> CircuitInterface:
        """Return the circuit used for the simulation."""

    @property
    @abstractmethod
    def spikes(self) -> Mapping[Optional[str], PopulationSpikesReportInterface]:
        """Return the spikes report as a dict: population -> report."""

    @property
    @abstractmethod
    def reports(self) -> Mapping[str, Mapping[Optional[str], PopulationReportInterface]]:
        """Return the reports as a dict: name -> population -> report."""
