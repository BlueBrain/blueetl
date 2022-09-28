import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from blueetl.store import parquet as test_module


# fastparquet 0.8.1 fails to write DataFrames with MultiIndexes without names,
# but probably it's not a good idea to use them anyway. See the code at:
# https://github.com/dask/fastparquet/blob/34069fe2a41a7491e5b7b1f1b2cae9c41176f7b8/fastparquet/util.py#L140-L144


@pytest.mark.parametrize(
    "df",
    [
        lazy_fixture("storable_df_with_unnamed_index"),
        lazy_fixture("storable_df_with_named_index"),
        # lazy_fixture("storable_df_with_unnamed_multiindex"),  # failing with fastparquet
        lazy_fixture("storable_df_with_named_multiindex"),
    ],
)
@pytest.mark.parametrize(
    "dump_engine, load_engine",
    [
        ("pyarrow", "pyarrow"),
        ("fastparquet", "fastparquet"),
        # ("pyarrow", "fastparquet"),  # error: column e is loaded as float64 instead of object
        # ("fastparquet", "pyarrow"),  # error: column h is loaded as bytes instead of object
    ],
)
def test_dump_load_roundtrip(tmp_path, df, dump_engine, load_engine):
    name = "myname"
    store = test_module.ParquetStore(tmp_path)
    store._dump_options["engine"] = dump_engine
    store._load_options["engine"] = load_engine

    store.dump(df, name)
    result = store.load(name)

    assert_frame_equal(result, df)
