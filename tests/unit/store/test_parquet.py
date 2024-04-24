import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyarrow import ArrowNotImplementedError

from blueetl.store import parquet as test_module


@pytest.mark.parametrize(
    "df",
    [
        "storable_df_with_unnamed_index",
        "storable_df_with_named_index",
        "storable_df_with_named_multiindex",
        "storable_df_with_unnamed_multiindex",
    ],
)
def test_dump_load_roundtrip(tmp_path, df, lazy_fixture):
    df = lazy_fixture(df)
    name = "myname"

    store = test_module.ParquetStore(tmp_path)
    store.dump(df, name)
    result = store.load(name)

    assert_frame_equal(result, df)


@pytest.mark.parametrize(
    "df",
    [
        "storable_df_with_unnamed_index",
        "storable_df_with_named_index",
        "storable_df_with_named_multiindex",
        "storable_df_with_unnamed_multiindex",
    ],
)
def test_dump_load_roundtrip_with_inferred_schema(tmp_path, df, lazy_fixture):
    df = lazy_fixture(df)
    df1 = df.copy()
    df2 = df.copy()
    df1 = df1.etl.add_conditions(conditions=["extra_level"], values=[1])
    df2 = df2.etl.add_conditions(conditions=["extra_level"], values=[2])

    df1.insert(loc=0, column="extra_columns", value=[[] for _ in range(len(df1))])
    df2.insert(loc=0, column="extra_columns", value=[[float(i)] for i in range(len(df2))])

    df = pd.concat([df1, df2])

    store = test_module.ParquetStore(tmp_path)
    store.dump(df1, name="01")
    store.dump(df2, name="02")
    result = store.load()

    assert_frame_equal(result, df)

    # in pyarrow 15.0.0, the empty lists in the first DataFrame would cause:
    #
    #   pyarrow.lib.ArrowNotImplementedError:
    #     Unsupported cast from double to null using function cast_null
    #
    # if the schema is not inferred correctly from both the files
    with pytest.raises(ArrowNotImplementedError):
        store.load(schema=None)
