import logging
from collections import defaultdict
from functools import partial
from os import PathLike
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Type, Union

import pandas as pd

from blueetl import DefaultStore
from blueetl.constants import (
    CIRCUIT,
    CIRCUIT_ID,
    GID,
    NEURON_CLASS,
    NEURON_CLASS_INDEX,
    SIMULATION,
    SIMULATION_ID,
    TRIAL,
    WINDOW,
)
from blueetl.core.parallel import Task, TaskContext, run_parallel
from blueetl.extract.feature import Feature
from blueetl.repository import Repository
from blueetl.store.base import BaseStore
from blueetl.utils import import_by_string, timed

L = logging.getLogger(__name__)


class FeaturesCollection:
    def __init__(
        self,
        features_configs: List[Dict],
        repo: Repository,
        store_dir: Union[str, PathLike],
        store_class: Type[BaseStore] = DefaultStore,
        use_cache: bool = False,
    ) -> None:
        self._features_configs = features_configs
        self._repo = repo
        self._store = store_class(store_dir)
        self._use_cache = use_cache
        self._data: Dict[str, Feature] = {}

    @property
    def names(self) -> List[str]:
        return sorted(self._data)

    def __getattr__(self, name: str) -> Feature:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    def update(self, mapping: Mapping[str, Feature]) -> None:
        self._data.update(mapping)

    def dump(self, features: Dict[str, Feature]) -> None:
        for name, feature in features.items():
            self._store.dump(feature.df, name)

    def dump_all(self) -> None:
        self.dump(self._data)

    def print(self) -> None:
        print("### features")
        for k, v in self.data.items():
            print("#", k)
            print(v)

    def calculate(self) -> None:
        features_len = len(self._features_configs)
        for n, features_config in enumerate(self._features_configs, start=1):
            features_type = features_config["type"]
            L.info("Calculating features %s/%s [type: %s]", n, features_len, features_type)
            method = getattr(self, f"_calculate_{features_type}")
            new_features = method(features_config)
            self.update(new_features)
            self.dump(new_features)

    def _calculate_single(self, features_config: Dict[str, Any]) -> Dict[str, Feature]:
        df = calculate_features_single(
            repo=self._repo,
            features_func=features_config["function"],
            features_groupby=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )
        return {features_config["name"]: Feature.from_pandas(df)}

    def _calculate_multi(self, features_config: Dict[str, Any]) -> Dict[str, Feature]:
        df_dict = calculate_features_multi(
            repo=self._repo,
            features_func=features_config["function"],
            features_groupby=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )
        return {name: Feature.from_pandas(df) for name, df in df_dict.items()}

    def _calculate_comparison(self, features_config: Dict[str, Any]) -> Dict[str, Feature]:
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
    for key, group_df in main_df.etl.groupy_iter(features_groupby):
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
    repo: Repository,
    features_func: str,
    features_groupby: List[str],
    features_params: Dict,
    jobs: Optional[int] = None,
    backend: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Calculate features in parallel for the given repository as a dict of DataFrames.

    Args:
        repo: repository containing spikes.
        features_func: string of the function to be executed to calculate the features.
            The function should accept (repo, key, df, params) and return a dict of DataFrames.
        features_groupby: columns for aggregation.
        features_params: generic dict of params that will be passed to the function.
        jobs: number of jobs (see run_parallel)
        backend: parallel backend (see run_parallel)

    Returns:
        dict of DataFrames

    """

    def func_wrapper(key: NamedTuple, df: pd.DataFrame):
        # executed in a subprocess
        features_records = {}
        L.debug("Calculating features for %s", key)
        record = key._asdict()
        conditions = list(record.keys())
        values = tuple(record.values())
        features_dict = func(repo=repo, key=key, df=df, params=features_params)
        assert isinstance(features_dict, dict), "The returned object must be a dict"
        for feature_group, result_df in features_dict.items():
            assert isinstance(result_df, pd.DataFrame), "Each contained object must be a DataFrame"
            # ignore the index if it's unnamed and with one level; this can be useful
            # for example when the returned DataFrame has a RangeIndex to be dropped
            drop = result_df.index.names == [None]
            result_df = result_df.etl.add_conditions(conditions, values, drop=drop)
            features_records[feature_group] = result_df
        return features_records

    func = import_by_string(features_func)
    main_df = _get_spikes_for_all_neurons_and_windows(repo)
    # list of dicts: feature_group -> dataframe
    results = main_df.etl.groupby_run_parallel(
        features_groupby, func=func_wrapper, jobs=jobs, backend=backend
    )
    # concatenate all the dataframes having the same feature_group label
    all_features_records = defaultdict(list)
    for result in results:
        for feature_group, tmp_df in result.items():
            all_features_records[feature_group].append(tmp_df)
    # build the final dict of dataframes
    return {
        feature_group: pd.concat(df_list) for feature_group, df_list in all_features_records.items()
    }


def call_by_simulation(
    repo: Repository,
    func: Callable,
    jobs: Optional[int] = None,
    backend: Optional[str] = None,
) -> List[Any]:
    """

    Args:
        repo: repository instance.
        func: callable called for each simulation, accepting
            simulation_index: NamedTuple
            simulation_row: NamedTuple
            simulation_spikes: pd.DataFrame
            simulation_windows: pd.DataFrame
            circuit_neurons: pd.DataFrame
            circuit_neuron_classes: pd.DataFrame
        jobs: number of jobs (see run_parallel)
        backend: parallel backend (see run_parallel)

    Returns:
        list of results
    """

    def tasks_generator():
        for circuit_id in repo.simulations.df[CIRCUIT_ID].unique():
            circuit_neurons = repo.neurons.df.etl.q(circuit_id=circuit_id)
            circuit_neuron_classes = repo.neuron_classes.df.etl.q(circuit_id=circuit_id)
            it = (
                repo.simulations.df.drop(columns=[CIRCUIT, SIMULATION])
                .etl.q(circuit_id=circuit_id)
                .etl.iter()
            )
            for simulation_index, simulation_row in it:
                simulation_id = simulation_row.simulation_id
                simulation_spikes = repo.spikes.df.etl.q(simulation_id=simulation_id)
                simulation_windows = repo.windows.df.etl.q(simulation_id=simulation_id)
                yield Task(
                    partial(
                        func,
                        simulation_index=simulation_index,
                        simulation_row=simulation_row,
                        simulation_spikes=simulation_spikes,
                        simulation_windows=simulation_windows,
                        circuit_neurons=circuit_neurons,
                        circuit_neuron_classes=circuit_neuron_classes,
                    )
                )

    return run_parallel(tasks_generator(), jobs=jobs, backend=backend)
