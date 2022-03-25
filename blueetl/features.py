import logging
from collections import defaultdict
from typing import Dict, List, Type

import pandas as pd

from blueetl import DefaultStore
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

    def dump_all(self):
        raise NotImplementedError

    def print(self):
        print("### features")
        for k, v in self.data.items():
            print("#", k)
            print(v)

    def calculate(self):
        for features_config in self.features_configs:
            method = getattr(self, f"_calculate_{features_config['type']}")
            self.data.update(method(features_config))

    def _calculate_single(self, features_config):
        return {
            features_config["name"]: calculate_features_single(
                repo=self.repo,
                features_func=features_config["function"],
                features_groupy=features_config["groupby"],
                features_params=features_config.get("params", {}),
            )
        }

    def _calculate_multi(self, features_config):
        return calculate_features_multi(
            repo=self.repo,
            features_func=features_config["function"],
            features_groupy=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )

    def _calculate_comparison(self, features_config):
        L.warning("Comparison features not implemented")
        return {}


@timed(L.info, "Completed calculate_features_single")
def calculate_features_single(
    repo: Repository, features_func: str, features_groupy: List[str], features_params: Dict
) -> pd.DataFrame:
    """Calculate features for the given repository as a single DataFrame.

    Args:
        repo: repository containing spikes.
        features_func: string of the function to be executed to calculate the features.
            The function should return a dict, where each key will be a column of the DataFrame.
        features_groupy: columns for aggregation.
        features_params: generic dict of params that will be passed to the function.

    Returns:
        DataFrame

    """
    func = import_by_string(features_func)
    records = []
    key = None
    for key, df in repo.spikes.df.etl.grouped_by(features_groupy):
        record = key._asdict()
        record.update(func(repo=repo, key=key, df=df, params=features_params))
        records.append(record)
    df = pd.DataFrame(records)
    if key:
        df = df.set_index(list(key._fields))
    return df


@timed(L.info, "Completed calculate_features_multi")
def calculate_features_multi(
    repo: Repository, features_func: str, features_groupy: List[str], features_params: Dict
) -> Dict[str, pd.DataFrame]:
    """Calculate features for the given repository as a dict of DataFrames.

    Args:
        repo: repository containing spikes.
        features_func: string of the function to be executed to calculate the features.
            The function should return a dict of DataFrames.
        features_groupy: columns for aggregation.
        features_params: generic dict of params that will be passed to the function.

    Returns:
        dict of DataFrames

    """
    func = import_by_string(features_func)
    features_records = defaultdict(list)
    for key, df in repo.spikes.df.etl.grouped_by(features_groupy):
        L.info("Calculating features for %s", key)
        record = key._asdict()
        conditions = list(record.keys())
        values = tuple(record.values())
        features_dict = func(repo=repo, key=key, df=df, params=features_params)
        for feature_group, result in features_dict.items():
            assert isinstance(result, pd.DataFrame), "The returned object must be a DataFrame"
            # ignore the index if it's unnamed and with one level; this can be useful
            # for example when the returned DataFrame has a RangeIndex to be dropped
            drop = result.index.names == [None]
            result = result.etl.add_conditions(conditions, values, drop=drop)
            features_records[feature_group].append(result)
    features = {
        feature_group: pd.concat(df_list) for feature_group, df_list in features_records.items()
    }
    return features
