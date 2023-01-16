"""Resolver."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from blueetl.constants import LEVEL_SEP


class ResolverError(Exception):
    """Error raised when the requested reference cannot be resolved."""


class Resolver(ABC):
    """Resolver class."""

    def __init__(self, root: Any, replace: Optional[Dict[str, str]] = None) -> None:
        """Initialize the Resolver.

        Args:
            root: referenced root object.
            replace: optional dict used to replace the names in the ref string.
        """
        self._root = root
        self._replacement_dict = replace or {}

    def get(self, ref: str, level: Optional[int] = None) -> Any:
        """Return the object referred by the ref string.

        Args:
            ref: string with attributes separated by dot.
            level: if specified, limit the resolution up to the given level.

        Examples:
            - get("a.b.c.d") is resolved as: root -> a -> b -> c -> d
            - get("a.b.c.d", level=0) is resolved as: root
            - get("a.b.c.d", level=1) is resolved as: root -> a
            - get("a.b.c.d", level=-1) is resolved as: root -> a -> b -> c

        Returns:
            the loaded attribute.
        """
        obj = self._root
        names = ref.split(LEVEL_SEP)
        names = names if level is None else names[:level]
        for name in names:
            try:
                obj = self._get(obj, self._replace(name))
            except Exception:
                msg = f"Impossible to resolve {LEVEL_SEP.join(names)} at level {name}"
                raise ResolverError(msg) from None
        return obj

    def _replace(self, name: str) -> str:
        return self._replacement_dict.get(name, name)

    @abstractmethod
    def _get(self, obj: Any, name: str) -> Any:
        """Return the object corresponding to the given name from the object obj."""


class ObjectResolver(Resolver):
    """ObjectResolver class."""

    def _get(self, obj: Any, name: str) -> Any:
        return getattr(obj, name)


class DictResolver(Resolver):
    """DictResolver class."""

    def _get(self, obj: Any, name: str) -> Any:
        return obj[name]
