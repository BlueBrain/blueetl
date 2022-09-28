import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from blueetl.store import feather as test_module


@pytest.mark.parametrize(
    "df",
    [
        lazy_fixture("storable_df_with_unnamed_index"),
        lazy_fixture("storable_df_with_named_index"),
        lazy_fixture("storable_df_with_unnamed_multiindex"),
        lazy_fixture("storable_df_with_named_multiindex"),
    ],
)
def test_dump_load_roundtrip(tmp_path, df):
    name = "myname"
    store = test_module.FeatherStore(tmp_path)

    store.dump(df, name)
    result = store.load(name)

    assert_frame_equal(result, df)
