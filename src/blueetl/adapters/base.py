"""Base Adapter."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from blueetl.adapters.interfaces.circuit import CircuitInterface
from blueetl.adapters.interfaces.simulation import SimulationInterface
from blueetl.types import StrOrPath

InterfaceT = TypeVar("InterfaceT", CircuitInterface, SimulationInterface)
BaseAdapterT = TypeVar("BaseAdapterT", bound="BaseAdapter")


class AdapterError(Exception):
    """Generic Adapter Error."""


class BaseAdapter(Generic[InterfaceT], ABC):
    """Base Adapter to be subclassed by other adapters."""

    def __init__(self, config: StrOrPath) -> None:
        """Init the adapter from the specified config."""
        self._impl: Optional[InterfaceT] = self._load_impl(str(config))

    @abstractmethod
    def _load_impl(self, config: str) -> Optional[InterfaceT]:
        """Load and return the implementation object, or None if the config file doesn't exist."""

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

    @classmethod
    def from_impl(cls: type[BaseAdapterT], impl: InterfaceT) -> BaseAdapterT:
        """Return a new adapter with the specified implementation."""
        result = object.__new__(cls)
        result._impl = impl
        return result
