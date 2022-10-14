"""Features collection."""
import logging
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional

import pandas as pd

from blueetl.cache import CacheManager
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
from blueetl.core.parallel import Task, run_parallel
from blueetl.extract.feature import Feature
from blueetl.repository import Repository
from blueetl.utils import import_by_string, timed, timed_log

L = logging.getLogger(__name__)


class FeaturesCollection:
    """Features collection class."""

    def __init__(
        self,
        features_configs: List[Dict],
        repo: Repository,
        cache_manager: CacheManager,
        _dataframes: Dict[str, pd.DataFrame] = None,
    ) -> None:
        """Initialize the FeaturesCollection from the given list of configurations.

        Args:
            features_configs: list of features configuration dicts.
            repo: Repository instance.
            cache_manager: CacheManager instance.
            _dataframes: DataFrames to be automatically loaded, only for internal use.
        """
        self._features_configs = features_configs
        self._repo = repo
        self._cache_manager = cache_manager
        self._data = self._dataframes_to_features(_dataframes) if _dataframes else {}

    @property
    def names(self) -> List[str]:
        """Return the names of all the calculated features."""
        return sorted(self._data)

    @property
    def cache_manager(self) -> CacheManager:
        """Access to the cache manager."""
        return self._cache_manager

    def __getattr__(self, name: str) -> Feature:
        """Return the features by name.

        It mimics the behaviour of Repository, where each Extractor can be accessed by attribute.

        Args:
            name: name of the features.
        """
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            ) from ex

    def _update(self, mapping: Mapping[str, Feature]) -> None:
        self._data.update(mapping)

    def _print(self) -> None:
        """Print some information about the instance.

        Only for debug and internal use, it may be removed in a future release.
        """
        print("### features")
        for k, v in self._data.items():
            print("#", k)
            print(v)

    def calculate(self) -> None:
        """Calculate all the features based on the configuration."""
        if self._data:
            # Features already calculated
            return
        t_log = timed_log(L.info)
        features_len = len(self._features_configs)
        for n, features_config in enumerate(self._features_configs):
            features = self.cache_manager.load_features(features_config=features_config)
            if features is not None:
                L.info("Processing cached features %s/%s", n + 1, features_len)
                initial_lengths = {name: len(df) for name, df in features.items()}
                features = self._dataframes_to_features(features)
                # the len may have changed because of the filter on simulation ids
                to_be_written = {
                    name
                    for name, feature in features.items()
                    if len(feature.df) != initial_lengths[name]
                }
                is_cached = True
            else:
                L.info("Processing new features %s/%s", n + 1, features_len)
                method = getattr(self, f"_calculate_{features_config['type']}")
                features = method(features_config)
                to_be_written = set(features)
                is_cached = False
            self._update(features)
            if to_be_written:
                # write only the new or filtered dataframes
                self.cache_manager.dump_features(
                    {name: features[name].to_pandas() for name in to_be_written},
                    features_config=features_config,
                )
            t_log(
                "Calculated features %s/%s: cached=%s, modified=%s",
                n + 1,
                features_len,
                is_cached,
                bool(to_be_written),
            )

    def _dataframes_to_features(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Feature]:
        query = {SIMULATION_ID: self._repo.simulation_ids}
        return {name: Feature.from_pandas(df, query=query) for name, df in df_dict.items()}

    def _calculate_single(self, features_config: Dict[str, Any]) -> Dict[str, Feature]:
        df = calculate_features_single(
            repo=self._repo,
            features_func=features_config["function"],
            features_groupby=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )
        name = str(features_config["name"])
        return self._dataframes_to_features({name: df})

    def _calculate_multi(self, features_config: Dict[str, Any]) -> Dict[str, Feature]:
        df_dict = calculate_features_multi(
            repo=self._repo,
            features_func=features_config["function"],
            features_groupby=features_config["groupby"],
            features_params=features_config.get("params", {}),
        )
        return self._dataframes_to_features(df_dict)

    def apply_filter(self, repo: Repository) -> "FeaturesCollection":
        """Apply a filter based on the extracted simulations and return a new object.

        Filtered dataframes are not written to disk.
        """
        dataframes = {name: features.df for name, features in self._data.items()}
        return FeaturesCollection(
            features_configs=self._features_configs,
            repo=repo,
            cache_manager=self.cache_manager.to_readonly(),
            _dataframes=dataframes,
        )


def _get_spikes_for_all_neurons_and_windows(repo: Repository) -> pd.DataFrame:
    """Extend the spikes df to include all the possible neurons and windows, with spikes or not."""
    return (
        repo.neurons.df[[CIRCUIT_ID, NEURON_CLASS, GID, NEURON_CLASS_INDEX]]
        .merge(repo.windows.df[[SIMULATION_ID, CIRCUIT_ID, WINDOW, TRIAL]], how="left")
        .merge(repo.spikes.df, how="left")
    )


@timed(L.info, "Executed calculate_features_single")
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
    for key, group_df in main_df.etl.groupby_iter(features_groupby):
        record = key._asdict()
        result = func(repo=repo, key=key, df=group_df, params=features_params)
        assert isinstance(result, dict), "The returned object must be a dict"
        record.update(result)
        records.append(record)
    features_df = pd.DataFrame(records)
    if key:
        features_df = features_df.set_index(list(key._fields))
    return features_df


@timed(L.info, "Executed calculate_features_multi")
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

    def func_wrapper(key: NamedTuple, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
    """Execute the given function in parallel, one task for each simulation.

    Args:
        repo: repository instance.
        func: callable called for each simulation, accepting
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

    def tasks_generator() -> Iterator[Task]:
        for circuit_id in repo.simulations.df[CIRCUIT_ID].unique():
            circuit_neurons = repo.neurons.df.etl.q(circuit_id=circuit_id)
            circuit_neuron_classes = repo.neuron_classes.df.etl.q(circuit_id=circuit_id)
            it = (
                repo.simulations.df.drop(columns=[CIRCUIT, SIMULATION])
                .etl.q(circuit_id=circuit_id)
                .etl.iter()
            )
            for _, simulation_row in it:
                simulation_id = simulation_row.simulation_id
                simulation_spikes = repo.spikes.df.etl.q(simulation_id=simulation_id)
                simulation_windows = repo.windows.df.etl.q(simulation_id=simulation_id)
                yield Task(
                    partial(
                        func,
                        simulation_row=simulation_row,
                        simulation_spikes=simulation_spikes,
                        simulation_windows=simulation_windows,
                        circuit_neurons=circuit_neurons,
                        circuit_neuron_classes=circuit_neuron_classes,
                    )
                )

    return run_parallel(tasks_generator(), jobs=jobs, backend=backend)
