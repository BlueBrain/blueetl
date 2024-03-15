"""Base Adapter."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from blueetl.adapters.interfaces.circuit import CircuitInterface
from blueetl.adapters.interfaces.node_sets import NodeSetsInterface
from blueetl.adapters.interfaces.simulation import SimulationInterface

InterfaceT = TypeVar("InterfaceT", CircuitInterface, SimulationInterface, NodeSetsInterface)
BaseAdapterT = TypeVar("BaseAdapterT", bound="BaseAdapter")


class AdapterError(Exception):
    """Generic Adapter Error."""


class BaseAdapter(Generic[InterfaceT], ABC):
    """Base Adapter to be subclassed by other adapters."""

    def __init__(self, _impl: Optional[InterfaceT]) -> None:
        """Init the adapter from the specified implementation."""
        self._impl: Optional[InterfaceT] = _impl

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: Optional[Path]) -> "BaseAdapter":
        """Load and return a new object from file."""

    @property
    def _ensure_impl(self) -> InterfaceT:
        """Return the inner implementation, or raise an error if it doesn't exist."""
        if self._impl is None:
            raise AdapterError("The implementation doesn't exist")
        return self._impl

    def exists(self) -> bool:
        """Return True if the wrapped object exists, False otherwise."""
        return self._impl is not None

    @property
    def instance(self) -> Any:
        """Return the wrapped instance, or None if it doesn't exist."""
        return self._impl.instance if self._impl is not None else None
