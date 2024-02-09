"""Bluepy implementation."""

from blueetl.utils import import_optional_dependency

# Immediately raise an error with a custom message if the submodule is imported
import_optional_dependency("bluepy")
