"""Base extractor."""

import logging
from abc import ABC
from typing import Optional, TypeVar

import pandas as pd

from blueetl.utils import ensure_dtypes

L = logging.getLogger(__name__)
ExtractorT = TypeVar("ExtractorT", bound="BaseExtractor")


class BaseExtractor(ABC):
    """Base extractor class."""

    COLUMNS: list[str] = []
    _allow_missing_columns = False
    _allow_extra_columns = False
    _allow_empty_data = False

    def __init__(self, df: pd.DataFrame, cached: bool, filtered: bool) -> None:
        """Initialize the extractor.

        Args:
            df: Pandas DataFrame containing the extracted data.
            cached: True if the data have been extracted from the cache, False otherwise.
            filtered: True if the data have been filtered using a custom query, False otherwise.
        """
        self._cached = cached
        self._filtered = filtered
        self._validate(df)
        self._df = ensure_dtypes(df)

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        """Validate the dataframe.

        It can be overridden if a custom validation is needed.
        """
        cls._validate_data(df)
        cls._validate_columns(df)

    @classmethod
    def _validate_data(cls, df: pd.DataFrame) -> None:
        """Validate the content of the dataframe."""
        if not cls._allow_empty_data and df.empty:
            raise RuntimeError(f"No data extracted to {cls.__name__}")

    @classmethod
    def _validate_columns(cls, df: pd.DataFrame) -> None:
        """Validate the names of the columns of the dataframe."""
        actual = set(df.columns)
        expected = set(cls.COLUMNS)
        if not cls._allow_missing_columns and expected - actual:
            raise ValueError(f"Expected columns not present: {expected - actual}")
        if not cls._allow_extra_columns and actual - expected:
            raise ValueError(f"Additional columns not allowed: {actual - expected}")

    @property
    def df(self) -> pd.DataFrame:
        """Return the internally wrapped dataframe."""
        return self._df

    @classmethod
    def from_pandas(
        cls: type[ExtractorT],
        df: pd.DataFrame,
        query: Optional[dict] = None,
        cached: bool = True,
    ) -> ExtractorT:
        """Return a new object from the given dataframe.

        If a query is specified, it's passed to ``etl.q`` and applied as a filter.

        It can be overridden together with ``to_pandas`` if some columns are not serializable.

        Args:
            df: dataframe to load.
            query: optional filter dictionary, passed to ``etl.q``.
            cached: True if the data is loaded from the cache, False otherwise.

        Returns:
            a new extractor instance.
        """
        original_len = len(df)
        if query:
            L.debug("Filtering dataframe %s by %s", cls.__name__, query)
            df = df.etl.q(query)
            if not any(df.index.names):
                # reset the index to remove any gap
                df = df.reset_index(drop=True)
        filtered = len(df) != original_len
        return cls(df, cached=cached, filtered=filtered)

    def to_pandas(self) -> pd.DataFrame:
        """Return a dataframe that can be serialized and stored to disk.

        It should be possible to call ``from_pandas`` with the returned dataframe to create
        an equivalent object.

        It can be overridden together with ``from_pandas`` if some columns are not serializable.

        Returns:
            serializable dataframe.
        """
        return self.df
