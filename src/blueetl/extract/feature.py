"""Features extractor."""

import logging

import pandas as pd

from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Feature(BaseExtractor):
    """Features extractor class."""

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        # no validation is needed for features
        pass
