import logging

from blueetl.extract.base import BaseExtractor

L = logging.getLogger(__name__)


class Feature(BaseExtractor):
    @classmethod
    def _validate(cls, df):
        pass
