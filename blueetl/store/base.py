import logging
from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional, Union

import pandas as pd

from blueetl.utils import resolve_path

L = logging.getLogger(__name__)


class BaseStore(ABC):
    def __init__(self, basedir: Union[str, PathLike]) -> None:
        self.basedir = resolve_path(basedir)
        self.basedir.mkdir(parents=True, exist_ok=True)
        L.info("Using class %s with basedir %s", self.__class__.__name__, self.basedir)

    @abstractmethod
    def dump(self, df: pd.DataFrame, name: str) -> None:
        pass

    @abstractmethod
    def load(self, name: str) -> Optional[pd.DataFrame]:
        pass
