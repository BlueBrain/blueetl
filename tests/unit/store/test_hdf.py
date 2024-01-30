import pytest
from pandas.testing import assert_frame_equal

from blueetl.store import hdf as test_module


@pytest.mark.parametrize(
    "df",
    [
        "storable_df_with_unnamed_index",
        "storable_df_with_named_index",
        "storable_df_with_unnamed_multiindex",
        "storable_df_with_named_multiindex",
    ],
)
def test_dump_load_roundtrip(tmp_path, df, lazy_fixture):
    df = lazy_fixture(df)
    name = "myname"
    store = test_module.HDFStore(tmp_path)

    store.dump(df, name)
    result = store.load(name)

    assert_frame_equal(result, df)
