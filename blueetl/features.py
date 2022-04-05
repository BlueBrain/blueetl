import logging
from collections import defaultdict
from typing import Dict, List, Type

import pandas as pd

from blueetl import DefaultStore
from blueetl.constants import (
    CIRCUIT_ID,
    GID,
    NEURON_CLASS,
    NEURON_CLASS_INDEX,
    SIMULATION_ID,
    TRIAL,
    WINDOW,
)
from blueetl.repository import Repository
from blueetl.store.base import BaseStore
from blueetl.utils import import_by_string, timed

L = logging.getLogger(__name__)


class FeaturesCollection:
    def __init__(
        self,
        features_configs,
        repo,
        store_dir,
        store_class: Type[BaseStore] = DefaultStore,
        use_cache: bool = False,
    ):
        self.features_configs = features_configs
        self.repo = repo
        self.store = store_class(store_dir)
        self.data = {}

    def get(self, name):
        return self.data[name]

    def update(self, iterable):
        self.data.update(iterable)

    def dump(self, features: Dict[str, pd.DataFrame]):
        for df_name, df in features.items():
            L.info("Dumping features %s...", df_name)
            self.store.dump(df, df_name)

    def dump_all(self):
        self.dump(self.data)

    def print(self):
        print("### features")
        for k, v in self.data.items():
            print("#", k)
            print(v)

    def calculate(self):
        for features_config in self.features_configs:
            method = getattr(self, f"_calculate_{features_config['type']}")
            new_features = method(features_config)
            self.data.update(new_features)
            self.dump(new_features)

    def _calculate_single(self, features_config):
        return {
            features_config["name"]: calculate_features_single(
                repo=self.repo,
                features_func=features_config["function"],
                features_groupby=features_config["groupby"],
                features_params=features_config.get("params", {}),
            )
        }

    def _calculate_multi(self, features_config):
        return calculate_features_multi(
            repo=self.repo,
            features_func=features_config["function"],
            features_groupby=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )

    def _calculate_comparison(self, features_config):
        L.warning("Comparison features not implemented")
        return {}


def _get_spikes_for_all_neurons_and_windows(repo: Repository) -> pd.DataFrame:
    """Extend the spikes df to include all the possible neurons and windows, with spikes or not."""
    return (
        repo.neurons.df[[CIRCUIT_ID, NEURON_CLASS, GID, NEURON_CLASS_INDEX]]
        .merge(repo.windows.df[[SIMULATION_ID, CIRCUIT_ID, WINDOW, TRIAL]], how="left")
        .merge(repo.spikes.df, how="left")
    )


@timed(L.info, "Completed calculate_features_single")
def calculate_features_single(
    repo: Repository, features_func: str, features_groupby: List[str], features_params: Dict
) -> pd.DataFrame:
    """Calculate features for the given repository as a single DataFrame.

    Args:
        repo: repository containing spikes.
        features_func: string of the function to be executed to calculate the features.
            The function should return a dict, where each key will be a column in the DataFrame.
        features_groupby: columns for aggregation.
        features_params: generic dict of params that will be passed to the function.

    Returns:
        DataFrame

    """
    func = import_by_string(features_func)
    records = []
    key = None
    main_df = _get_spikes_for_all_neurons_and_windows(repo)
    for key, group_df in main_df.etl.grouped_by(features_groupby):
        record = key._asdict()
        result = func(repo=repo, key=key, df=group_df, params=features_params)
        assert isinstance(result, dict), "The returned object must be a dict"
        record.update(result)
        records.append(record)
    features_df = pd.DataFrame(records)
    if key:
        features_df = features_df.set_index(list(key._fields))
    return features_df


@timed(L.info, "Completed calculate_features_multi")
def calculate_features_multi(
    repo: Repository, features_func: str, features_groupby: List[str], features_params: Dict
) -> Dict[str, pd.DataFrame]:
    """Calculate features for the given repository as a dict of DataFrames.

    Args:
        repo: repository containing spikes.
        features_func: string of the function to be executed to calculate the features.
            The function should return a dict of DataFrames.
        features_groupby: columns for aggregation.
        features_params: generic dict of params that will be passed to the function.

    Returns:
        dict of DataFrames

    """
    func = import_by_string(features_func)
    features_records = defaultdict(list)
    main_df = _get_spikes_for_all_neurons_and_windows(repo)
    for key, group_df in main_df.etl.grouped_by(features_groupby):
        L.info("Calculating features for %s", key)
        record = key._asdict()
        conditions = list(record.keys())
        values = tuple(record.values())
        features_dict = func(repo=repo, key=key, df=group_df, params=features_params)
        assert isinstance(features_dict, dict), "The returned object must be a dict"
        for feature_group, result_df in features_dict.items():
            assert isinstance(result_df, pd.DataFrame), "Each contained object must be a DataFrame"
            # ignore the index if it's unnamed and with one level; this can be useful
            # for example when the returned DataFrame has a RangeIndex to be dropped
            drop = result_df.index.names == [None]
            result_df = result_df.etl.add_conditions(conditions, values, drop=drop)
            features_records[feature_group].append(result_df)
    features = {
        feature_group: pd.concat(df_list) for feature_group, df_list in features_records.items()
    }
    return features
