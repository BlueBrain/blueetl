from abc import ABC
from typing import List

import pandas as pd

from blueetl.utils import ensure_dtypes


class BaseExtractor(ABC):
    COLUMNS: List[str] = []

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._df = ensure_dtypes(df)

    @classmethod
    def _validate(cls, df):
        cls._validate_columns(df)

    @classmethod
    def _validate_columns(cls, df, allow_missing=False, allow_extra=False):
        # check the names of the columns
        actual = set(df.columns)
        expected = set(cls.COLUMNS)
        if not allow_missing and expected - actual:
            raise ValueError(f"Expected columns not present: {expected - actual}")
        if not allow_extra and actual - expected:
            raise ValueError(f"Additional columns not allowed: {actual - expected}")

    @property
    def df(self):
        return self._df

    @classmethod
    def from_pandas(cls, df):
        return cls(df)

    def to_pandas(self):
        return self.df
