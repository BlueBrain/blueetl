import logging

import pandas as pd

from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Feature(BaseExtractor):
    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        pass
