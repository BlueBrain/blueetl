import os
from contextlib import contextmanager
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parent / "data"
GPFS_DATA_PATH = Path("/gpfs/bbp.cscs.ch/project/proj12/NSE/blueetl/data")


@contextmanager
def change_directory(path):
    """Context manager to temporarily change the working directory."""
    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)
