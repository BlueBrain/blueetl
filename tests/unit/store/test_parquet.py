import pytest
from pandas.testing import assert_frame_equal

from blueetl.store import parquet as test_module


@pytest.mark.parametrize(
    "df",
    [
        "storable_df_with_unnamed_index",
        "storable_df_with_named_index",
        "storable_df_with_named_multiindex",
        # fastparquet 0.8.1 fails to write DataFrames with MultiIndexes without names,
        # but probably it's not a good idea to use them anyway. See the code at:
        # https://github.com/dask/fastparquet/blob/34069fe2a41a7491e5b7b1f1b2cae9c41176f7b8/fastparquet/util.py#L140-L144
        # "storable_df_with_unnamed_multiindex",
    ],
)
@pytest.mark.parametrize(
    "dump_options, load_options",
    [
        # test the configuration actually used
        (None, None),
        # test other possible configurations not used yet
        ({"engine": "pyarrow", "index": None}, {"engine": "pyarrow"}),
        ({"engine": "fastparquet", "index": None}, {"engine": "fastparquet"}),
        pytest.param(
            {"engine": "fastparquet", "index": True},
            {"engine": "fastparquet"},
            marks=pytest.mark.xfail(
                reason="Fails because index names are different ('index', None)",
                raises=AssertionError,
            ),
        ),
        pytest.param(
            {"engine": "pyarrow"},
            {"engine": "fastparquet"},
            marks=pytest.mark.xfail(
                reason="Fails because column e is loaded as float64 instead of object",
                raises=AssertionError,
            ),
        ),
        pytest.param(
            {"engine": "fastparquet"},
            {"engine": "pyarrow"},
            marks=pytest.mark.xfail(
                reason="Fails because column h is loaded as bytes instead of object",
                raises=AssertionError,
            ),
        ),
    ],
)
def test_dump_load_roundtrip(tmp_path, df, dump_options, load_options, lazy_fixture):
    df = lazy_fixture(df)
    name = "myname"
    store = test_module.ParquetStore(tmp_path)
    if dump_options is not None:
        store._dump_options = dump_options
    if load_options is not None:
        store._load_options = load_options

    store.dump(df, name)
    result = store.load(name)

    assert_frame_equal(result, df)
