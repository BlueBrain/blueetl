"""Interfaces for Circuit."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Generic, Optional, TypeVar

import numpy as np
import pandas as pd

from blueetl.adapters.interfaces.node_sets import NodeSetsInterface
from blueetl.utils import CachedPropertyMixIn

CircuitT = TypeVar("CircuitT")


class NodePopulationInterface(ABC):
    """NodePopulation Interface."""

    @abstractmethod
    def get(self, group, properties) -> pd.DataFrame:
        """Return a DataFrame of nodes."""

    @abstractmethod
    def ids(self, group) -> np.ndarray:
        """Return an array of node ids."""


class CircuitInterface(Generic[CircuitT], CachedPropertyMixIn, ABC):
    """Circuit Interface."""

    def __init__(self, circuit: CircuitT) -> None:
        """Init the circuit interface with the specified circuit."""
        self._circuit = circuit

    @property
    def instance(self) -> CircuitT:
        """Return the wrapped instance."""
        return self._circuit

    @abstractmethod
    def checksum(self) -> str:
        """Return a checksum of the relevant keys in the circuit configuration."""

    @property
    @abstractmethod
    def nodes(self) -> Mapping[Optional[str], NodePopulationInterface]:
        """Return the nodes as a dict: population -> nodes."""

    @property
    @abstractmethod
    def node_sets_file(self) -> Optional[Path]:
        """Returns the NodeSets file used by the circuit."""

    @property
    @abstractmethod
    def node_sets(self) -> NodeSetsInterface:
        """Returns the NodeSets used by the circuit."""
