"""Features extractor."""

import logging

import pandas as pd

from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Feature(BaseExtractor):
    """Features extractor class."""

    def _validate(self, df: pd.DataFrame) -> None:
        # no validation is needed for features
        pass
