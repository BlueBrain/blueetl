import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from blueetl import utils as test_module
from blueetl.constants import DTYPES


@pytest.mark.parametrize(
    "x, expected",
    [
        ([], []),
        ((), ()),
        (None, [None]),
        ("", [""]),
        ("a", ["a"]),
        (["a"], ["a"]),
        (("a",), ("a",)),
    ],
)
def test_ensure_list(x, expected):
    result = test_module.ensure_list(x)
    assert result == expected


def test_ensure_dtypes():
    df = pd.DataFrame([{k: i for i, k in enumerate(DTYPES)}])
    df["other"] = 99
    result = test_module.ensure_dtypes(df)
    assert_equal(result.dtypes.to_numpy(), [*list(DTYPES.values()), np.dtype("int64")])


def test_import_by_string():
    name = "json.load"
    result = test_module.import_by_string(name)
    assert result is json.load


def test_resolve_path(tmp_path):
    subdir = Path("subdir")
    my_file = Path("my_file")
    my_symlink = Path("my_symlink")
    non_existent = Path("non_existent")

    tmp_path = tmp_path.resolve()
    subdir_absolute = tmp_path / subdir
    my_file_absolute = tmp_path / subdir / my_file
    my_symlink_absolute = tmp_path / subdir / my_symlink
    non_existent_absolute = tmp_path / subdir / non_existent

    subdir_absolute.mkdir()
    my_file_absolute.touch()
    my_symlink_absolute.symlink_to(my_file)

    result = test_module.resolve_path(tmp_path, subdir, my_file, symlinks=False)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_file, symlinks=True)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_symlink, symlinks=False)
    assert result == my_symlink_absolute

    result = test_module.resolve_path(tmp_path, subdir, my_symlink, symlinks=True)
    assert result == my_file_absolute

    result = test_module.resolve_path(tmp_path, subdir, non_existent, symlinks=False)
    assert result == non_existent_absolute

    result = test_module.resolve_path(tmp_path, subdir, non_existent, symlinks=True)
    assert result == non_existent_absolute
