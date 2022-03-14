from abc import ABC

import pandas as pd

from blueetl.utils import ensure_dtypes


class BaseExtractor(ABC):
    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._df = ensure_dtypes(df)

    @staticmethod
    def _validate(df):
        pass

    @property
    def df(self):
        return self._df

    @classmethod
    def from_pandas(cls, df):
        return cls(df)

    def to_pandas(self):
        return self.df
