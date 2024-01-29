"""Base data store."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from blueetl.types import StrOrPath
from blueetl.utils import checksum, resolve_path

L = logging.getLogger(__name__)


class BaseStore(ABC):
    """Abstract class defining a generic file data store.

    It's responsible for reading and writing Pandas DataFrames in a specific serialization format.
    """

    def __init__(self, basedir: StrOrPath) -> None:
        """Initialize the object.

        Args:
            basedir: base directory where the files should be stored.
        """
        self._basedir = resolve_path(basedir)
        L.info("Using class %s with basedir %s", self.__class__.__name__, self.basedir)

    @property
    def basedir(self) -> Path:
        """Return the directory containing the files."""
        return self._basedir

    @property
    @abstractmethod
    def extension(self) -> str:
        """Return the file extension to be used with this specific data store."""

    @abstractmethod
    def dump(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataframe to file, using the given name and the class extension."""

    @abstractmethod
    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataframe from file, using the given name and the class extension."""

    def delete(self, name: str) -> None:
        """Delete the file with the given name and the class extension, if it exists."""
        self.path(name).unlink(missing_ok=True)

    def path(self, name: str) -> Path:
        """Return the full path of the file with the given name and the class extension."""
        return self.basedir / f"{name}.{self.extension}"

    def checksum(self, name: str) -> Optional[str]:
        """Return a checksum of the file, or None if it doesn't exist."""
        path = self.path(name)
        if path.exists():
            return checksum(path)
        return None
