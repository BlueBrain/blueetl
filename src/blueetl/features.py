"""Features collection."""
import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.config.analysis_model import FeaturesConfig
from blueetl.constants import CIRCUIT_ID, SIMULATION_ID
from blueetl.core.parallel import Task, run_parallel
from blueetl.core.utils import CachedDataFrame
from blueetl.extract.feature import Feature
from blueetl.repository import Repository
from blueetl.utils import ensure_dtypes, import_by_string, timed, timed_log

L = logging.getLogger(__name__)


class FeaturesCollection:
    """Features collection class."""

    def __init__(
        self,
        features_configs: list[FeaturesConfig],
        repo: Repository,
        cache_manager: CacheManager,
        _dataframes: Optional[dict[str, pd.DataFrame]] = None,
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
        self._data: dict[str, Feature] = {}
        if _dataframes:
            self._data = self._dataframes_to_features(_dataframes, config=None)

    @property
    def names(self) -> list[str]:
        """Return the names of all the calculated features."""
        if not self._data:
            self.calculate()
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

    def __dir__(self):
        """Allow autocompletion of dynamic attributes."""
        return list(super().__dir__()) + list(self._data)

    def _update(self, mapping: Mapping[str, Feature]) -> None:
        if overlapping := set(mapping).intersection(self._data):
            raise RuntimeError(
                f"Some features DataFrames have been defined more than once: {sorted(overlapping)}"
            )
        self._data.update(mapping)

    def show(self) -> None:
        """Print some information about the instance, mainly for debug and inspection."""
        for name in self.names:
            print("~" * 80)
            print("Features:", name)
            print(getattr(self, name).df)

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
                features = self._dataframes_to_features(features, config=features_config)
                # the len may have changed because of the filter on simulation ids
                to_be_written = {
                    name
                    for name, feature in features.items()
                    if len(feature.df) != initial_lengths[name]
                }
                is_cached = True
            else:
                L.info("Processing new features %s/%s", n + 1, features_len)
                features = self._calculate_result(features_config)
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

    def _dataframes_to_features(
        self, df_dict: dict[str, pd.DataFrame], config: Optional[FeaturesConfig]
    ) -> dict[str, Feature]:
        query = {SIMULATION_ID: self._repo.simulation_ids}
        result = {}
        for name, df in df_dict.items():
            result[name] = Feature.from_pandas(df, query=query)
            if config is not None:
                # make a copy of the config accessible from the features dataframe attrs
                result[name].df.attrs["config"] = deepcopy(config)
        return result

    def _calculate_result(self, features_config: FeaturesConfig) -> dict[str, Feature]:
        df_dict = calculate_features(repo=self._repo, features_config=features_config)
        return self._dataframes_to_features(df_dict, config=features_config)

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


def _get_tasks_generator(
    neurons_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    report_df: pd.DataFrame,
    func: Callable,
    features_config: FeaturesConfig,
) -> Iterator[Task]:
    def _unique_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        columns = [col for col in columns if col in df]
        return df[columns].drop_duplicates(ignore_index=True)

    def _query(nt: NamedTuple, columns: list[str]) -> dict[str, Any]:
        return {col: getattr(nt, col) for col in columns}

    def _groups(
        neurons_df: pd.DataFrame, windows_df: pd.DataFrame, groupby: list[str]
    ) -> pd.DataFrame:
        """Get all the possible groups and return a sorted merged dataframe."""
        u_neurons = _unique_rows(neurons_df, groupby)
        u_windows = _unique_rows(windows_df, groupby)
        if diff := [col for col in groupby if col not in u_neurons and col not in u_windows]:
            raise ValueError(f"Unknown columns specified in groupby: {diff}")
        if len(u_neurons) > 0 and len(u_windows) == 0:
            result = u_neurons
        elif len(u_neurons) == 0 and len(u_windows) > 0:
            result = u_windows
        else:
            result = pd.merge(left=u_windows, right=u_neurons, how="inner")
        # ensure that the groups are sorted as requested
        return result[groupby].sort_values(groupby, ignore_index=True)

    if features_config.neuron_classes:
        neurons_df = neurons_df.etl.q(neuron_class=features_config.neuron_classes)
    if features_config.windows:
        windows_df = windows_df.etl.q(window=features_config.windows)

    caches = {
        "neurons": CachedDataFrame(neurons_df),
        "windows": CachedDataFrame(windows_df),
        "report": CachedDataFrame(report_df),
    }
    merged = _groups(neurons_df, windows_df, groupby=features_config.groupby)
    # keep the order of the columns because needed for the cache
    columns = {
        "neurons": [col for col in merged if col in neurons_df],
        "windows": [col for col in merged if col in windows_df],
        "report": [col for col in merged if col in report_df],
    }

    # for each group, yield a task that will be called in a subprocess,
    # passing the group key and the filtered DataFrames as parameters
    L.info("Tasks to be executed: %s", len(merged))
    for _, key in merged.etl.iter():
        yield Task(
            partial(
                func,
                key=key,
                neurons_df=caches["neurons"].query(query=_query(key, columns["neurons"])),
                windows_df=caches["windows"].query(query=_query(key, columns["windows"])),
                report_df=caches["report"].query(query=_query(key, columns["report"])),
            )
        )


def _func_wrapper(
    key: NamedTuple,
    neurons_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    report_df: pd.DataFrame,
    repo: Repository,
    features_config: FeaturesConfig,
) -> dict[str, pd.DataFrame]:
    # executed in a subprocess
    L.debug("Calculating features for %s", key)
    merged_df = neurons_df.merge(windows_df, how="left").merge(report_df, how="left")
    # The params dict is deepcopied because it could be modified in the user function.
    # It could happen even with multiprocessing, because joblib may process tasks in batch.
    func = import_by_string(features_config.function)
    features_dict = func(repo=repo, key=key, df=merged_df, params=deepcopy(features_config.params))
    # compatibility with features defined with type=single
    if features_config.type == "single":
        features_dict = {features_config.name: features_dict}
    # verify and process the result
    if not isinstance(features_dict, dict):
        raise ValueError("The user function must return a dict of dataframes")
    features_records = {}
    for feature_group, result_df in features_dict.items():
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError(f"Expected a DataFrame, not {type(result_df).__name__}")
        # ignore the index if it's unnamed and with one level; this can be useful
        # for example when the returned DataFrame has a RangeIndex to be dropped
        drop = result_df.index.names == [None]
        result_df = result_df.etl.add_conditions(conditions=key._fields, values=key, drop=drop)
        features_records[feature_group + features_config.suffix] = result_df
    return features_records


@timed(L.info, "Executed calculate_features_multi")
def calculate_features(
    repo: Repository,
    features_config: FeaturesConfig,
    jobs: Optional[int] = None,
    backend: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Calculate features in parallel for the given repository as a dict of DataFrames.

    Args:
        repo: repository containing spikes.
        features_config: features configuration.
        jobs: number of jobs (see run_parallel)
        backend: parallel backend (see run_parallel)

    Returns:
        dict of DataFrames

    """
    func = partial(_func_wrapper, repo=repo, features_config=features_config)
    tasks_generator = _get_tasks_generator(
        neurons_df=repo.neurons.df,
        windows_df=repo.windows.df,
        report_df=repo.report.df,
        func=func,
        features_config=features_config,
    )
    results = run_parallel(tasks_generator, jobs=jobs, backend=backend)
    # concatenate all the dataframes having the same feature_group label
    all_features_records = defaultdict(list)
    for result in results:
        for feature_group, tmp_df in result.items():
            all_features_records[feature_group].append(tmp_df)
    # build the final dict of dataframes
    return {
        feature_group: ensure_dtypes(pd.concat(df_list))
        for feature_group, df_list in all_features_records.items()
    }


def call_by_simulation(
    simulations: pd.DataFrame,
    dataframes_to_filter: dict[str, pd.DataFrame],
    func: Callable,
    jobs: Optional[int] = None,
    backend: Optional[str] = None,
    how: str = "namedtuple",
) -> list[Any]:
    """Execute the given function in parallel, one task for each simulation.

    Args:
        simulations: DataFrame of simulations.
        dataframes_to_filter: dict of DataFrames to filter by simulation_id and/or circuit_id,
            and passed to each subprocess.
        func: callable called for each simulation, accepting:
            - simulation_row: NamedTuple (or the type specified with the `how` parameter)
            - filtered_dataframes: dict of DataFrames filtered by simulation_id and/or circuit_id
            If the function has other parameters, they can be applied using `functools.partials`,
            and they will be serialized and passed unchanged to the subprocesses.
        jobs: number of jobs (see run_parallel)
        backend: parallel backend (see run_parallel)
        how: format the `simulation_row` parameter passed to the func callback.
            It can be one of "namedtuple", "dict", "series", "dataframe".

    Returns:
        list of results
    """

    def tasks_generator() -> Iterator[Task]:
        # cache the names to avoid looking into df.index.names and df.columns in each iteration
        cached_names = {
            name: set(df).union(df.index.names) for name, df in dataframes_to_filter.items()
        }
        for circuit_id in simulations[CIRCUIT_ID].unique():
            filtered_by_circuit_id = {
                name: df.etl.q(circuit_id=circuit_id) if CIRCUIT_ID in cached_names[name] else df
                for name, df in dataframes_to_filter.items()
            }
            it = simulations.etl.q(circuit_id=circuit_id).etl.iter()
            for _, simulation_row in it:
                simulation_id = simulation_row.simulation_id
                filtered_by_circuit_id_and_simulation_id = {
                    name: df.etl.q(simulation_id=simulation_id)
                    if SIMULATION_ID in cached_names[name]
                    else df
                    for name, df in filtered_by_circuit_id.items()
                }
                yield Task(
                    partial(
                        func,
                        simulation_row=convert_row(simulation_row),
                        filtered_dataframes=filtered_by_circuit_id_and_simulation_id,
                    )
                )

    convert_row = {
        "namedtuple": lambda row: row,
        "dict": lambda row: row._asdict(),
        "series": lambda row: pd.Series(row._asdict()),
        "dataframe": lambda row: pd.DataFrame([row]),
    }[how]
    return run_parallel(tasks_generator(), jobs=jobs, backend=backend)
