import contextlib
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parent / "data"


@contextlib.contextmanager
def assertion_error_message(msg):
    try:
        yield
    except AssertionError as ex:
        raise AssertionError(f"{msg}\n{ex}") from ex
