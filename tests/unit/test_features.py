from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd

from blueetl import features as test_module
from blueetl.config.analysis_model import FeaturesConfig
from blueetl.extract.feature import Feature
from blueetl.utils import ensure_dtypes
from tests.unit.utils import assert_frame_equal


def test_concatenated_features():
    base_df = pd.DataFrame(
        {
            "simulation_id": [0, 0, 0, 0],
            "circuit_id": [0, 0, 0, 0],
            "neuron_class": ["EXC", "EXC", "INH", "INH"],
            "window": ["w0", "w1", "w0", "w1"],
            "decay": [12.85, 9.85, 8.60, 13.40],
            "ratio": [0.10, 0.15, 0.50, 0.55],
        }
    ).set_index(["simulation_id", "circuit_id", "neuron_class", "window"])

    parent = MagicMock()
    obj = test_module.ConcatenatedFeatures(parent)
    assert {"params", "aliases", "df"}.isdisjoint(obj.__dict__)

    base_config = FeaturesConfig(
        type="multi",
        groupby=["simulation_id", "circuit_id", "neuron_class", "window"],
        function="mymodule.myfunc",
        neuron_classes=["EXC", "INH"],
        windows=["w0", "w1"],
    )
    size = 2
    for i in range(size):
        df = base_df.copy(deep=True)
        df["decay"] = df["decay"] * (i + 1)
        feature = Feature.from_pandas(df)
        config = deepcopy(base_config)
        config.params = {"common": 123, "nested": {"myparam1": i * 10, "myparam2": [0.0, 0.5]}}
        config.suffix = f"_{i}"
        name = f"myfeature{config.suffix}"
        setattr(parent, name, feature)
        obj.update(feature_name=name, features_config=config)

    expected_params = pd.DataFrame(
        {
            "common": [123, 123],
            "nested.myparam1": [0, 10],
            "nested.myparam2": [[0.0, 0.5], [0.0, 0.5]],
        },
        index=pd.RangeIndex(size, name="params_id"),
    )
    assert_frame_equal(obj.params, expected_params)

    expected_aliases = pd.DataFrame(
        {
            "column": ["nested.myparam1"],
            "alias": ["myparam1"],
        }
    )
    assert_frame_equal(obj.aliases, expected_aliases)

    expected_df = pd.DataFrame(
        {
            "simulation_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "circuit_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "neuron_class": ["EXC", "EXC", "INH", "INH", "EXC", "EXC", "INH", "INH"],
            "window": ["w0", "w1", "w0", "w1", "w0", "w1", "w0", "w1"],
            "decay": [12.85, 9.85, 8.60, 13.40, 25.7, 19.7, 17.2, 26.8],
            "ratio": [0.10, 0.15, 0.50, 0.55, 0.10, 0.15, 0.50, 0.55],
            "params_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "myparam1": [0, 0, 0, 0, 10, 10, 10, 10],
        }
    ).set_index(["simulation_id", "circuit_id", "neuron_class", "window"])
    expected_df = ensure_dtypes(expected_df)
    assert_frame_equal(obj.df, expected_df)

    # test clear_cache
    assert {"params", "aliases", "df"}.issubset(obj.__dict__)
    obj.clear_cache()
    assert {"params", "aliases", "df"}.isdisjoint(obj.__dict__)
    assert_frame_equal(obj.params, expected_params)
    assert_frame_equal(obj.aliases, expected_aliases)
    assert_frame_equal(obj.df, expected_df)
    assert {"params", "aliases", "df"}.issubset(obj.__dict__)

    # test cloning
    parent = MagicMock()
    clone = obj.clone(parent=parent)
    assert clone._parent == parent
    assert clone._configs == obj._configs
    assert clone._configs is not obj._configs


def test_calculate_features(repo):
    groupby = ["simulation_id", "circuit_id", "neuron_class", "window"]
    features_configs_key = test_module.FeaturesConfigKey(
        groupby=groupby,
        neuron_classes=[],
        windows=[],
    )
    features_configs_list = [
        FeaturesConfig(
            type="multi",
            groupby=groupby,
            function="blueetl.external.bnac.calculate_features.calculate_features_multi",
        ),
    ]

    result = test_module.calculate_features(repo, features_configs_key, features_configs_list)
    assert isinstance(result, list)
    assert len(result) == 1

    data = result[0]
    assert isinstance(data, dict)
    assert set(data) == {
        "by_gid",
        "by_gid_and_trial",
        "by_neuron_class",
        "by_neuron_class_and_trial",
        "histograms",
    }
