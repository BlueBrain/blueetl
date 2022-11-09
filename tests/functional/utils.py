import contextlib
import os
from contextlib import contextmanager
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parent / "data"
EXPECTED_PATH = TEST_DATA_PATH / "expected"


@contextmanager
def change_directory(path):
    """Context manager to temporarily change the working directory."""
    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@contextlib.contextmanager
def assertion_error_message(msg):
    try:
        yield
    except AssertionError as ex:
        raise AssertionError(f"{msg}\n{ex}") from ex
