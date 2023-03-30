"""BlueETL initialization and registration of Pandas accessors."""

from blueetl.core.etl import register_accessors
from blueetl.version import __version__  # noqa

register_accessors()
