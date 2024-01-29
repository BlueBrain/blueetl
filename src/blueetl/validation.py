"""Validation utilities."""

import importlib.resources
import logging

import jsonschema
import yaml

L = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error."""


def read_schema(schema_name: str) -> dict:
    """Load a schema and return the result as a dictionary."""
    traversable = importlib.resources.files(__package__) / "schemas" / f"{schema_name}.yaml"
    with traversable.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_config(config: dict, schema: dict) -> None:
    """Raise an exception if the configuration is not valid.

    Args:
        config: configuration to be validated.
        schema: json schema.

    Raises:
        ValidationError if the validation failed.
    """
    if config is None:
        L.error("The configuration cannot be empty.")
        raise ValidationError("Empty configuration")
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    validator = cls(schema)
    errors = list(validator.iter_errors(config))
    if errors:
        # Log an error message for each error.
        msg = []
        for n, e in enumerate(errors, 1):
            path = ".".join(str(elem) for elem in ["root"] + list(e.absolute_path))
            msg.append(f"{n}: Failed validating {path}: {e.message}")
        msg = "\n".join(msg)
        L.error("Invalid configuration:\n%s", msg)
        raise ValidationError("Invalid configuration")
