import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import pandas as pd

L = logging.getLogger(__name__)


class BaseStore(ABC):
    def __init__(self, basedir: Union[str, PathLike]) -> None:
        self.basedir = Path(basedir).resolve()
        self.basedir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def dump(self, df: pd.DataFrame, name: str) -> None:
        pass

    @abstractmethod
    def load(self, name: str) -> Optional[pd.DataFrame]:
        pass
