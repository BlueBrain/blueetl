"""Resolver."""

from abc import ABC, abstractmethod
from typing import Any

from blueetl.constants import LEVEL_SEP


class ResolverError(Exception):
    """Error raised when the requested reference cannot be resolved."""


class Resolver(ABC):
    """Resolver class."""

    def __init__(self, root: Any) -> None:
        """Initialize the Resolver.

        Args:
            root: referenced root object.
        """
        self._root = root

    def get(self, ref: str) -> Any:
        """Return the object referred by the ref string.

        Args:
            ref: string with attributes separated by dot.
        """
        obj = self._root
        names = ref.split(LEVEL_SEP)
        for name in names:
            try:
                obj = self._get(obj, name)
            except Exception:
                msg = f"Impossible to resolve {LEVEL_SEP.join(names)} at level {name}"
                raise ResolverError(msg) from None
        return obj

    @abstractmethod
    def _get(self, obj: Any, name: str) -> Any:
        """Return the item or attr corresponding to the given name."""


class AttrResolver(Resolver):
    """AttrResolver class.

    Examples:
        >>> from unittest.mock import Mock
        >>> obj = Mock()  # any object with nested attributes
        >>> obj.a.b.c = 123
        >>> resolver = AttrResolver(obj)
        >>> resolver.get("a.b.c")
        123
    """

    def _get(self, obj: Any, name: str) -> Any:
        return getattr(obj, name)


class ItemResolver(Resolver):
    """ItemResolver class.

    Examples:
        >>> obj = {"a": {"b": {"c": 123}}}
        >>> resolver = ItemResolver(obj)
        >>> resolver.get("a.b.c")
        123
    """

    def _get(self, obj: Any, name: str) -> Any:
        return obj[name]
