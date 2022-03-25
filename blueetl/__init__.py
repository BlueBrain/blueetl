"""blueetl."""

from blueetl.core.etl import register_accessors
from blueetl.store.parquet import ParquetStore as DefaultStore
from blueetl.version import __version__

register_accessors()
