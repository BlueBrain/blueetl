"""Features collection."""

import logging
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import pandas as pd
from blueetl_core.parallel import isolated
from blueetl_core.utils import smart_concat

from blueetl.cache import CacheManager
from blueetl.config.analysis_model import FeaturesConfig
from blueetl.constants import SIMULATION_ID
from blueetl.extract.feature import Feature
from blueetl.parallel import merge_filter
from blueetl.repository import Repository
from blueetl.store.parquet import ParquetStore
from blueetl.utils import (
    all_equal,
    ensure_dtypes,
    extract_items,
    get_shmdir,
    import_by_string,
    timed,
)

L = logging.getLogger(__name__)


class ConcatenatedFeatures:
    """ConcatenatedFeatures class.

    It can be used to view as a single DataFrame all the features calculated
    for various combinations of parameters.
    """

    def __init__(
        self,
        parent: "FeaturesCollection",
        configs: Optional[dict[str, FeaturesConfig]] = None,
    ) -> None:
        """Initialize the object."""
        self._parent = parent
        self._configs: dict[str, FeaturesConfig] = configs or {}

    def clone(self, parent: "FeaturesCollection") -> "ConcatenatedFeatures":
        """Return a copy using the same config and the new parent."""
        return self.__class__(parent, deepcopy(self._configs))

    def update(self, feature_name: str, features_config: FeaturesConfig) -> None:
        """Update the list of features and configurations."""
        assert feature_name not in self._configs, f"Duplicate feature_name: {feature_name}"
        self._configs[feature_name] = features_config
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear the cached properties."""
        for key in "params", "aliases", "df":
            self.__dict__.pop(key, None)

    @cached_property
    def params(self) -> pd.DataFrame:
        """Return all the parameters as a DataFrame."""
        return pd.DataFrame(
            [
                dict(extract_items(config.params))
                for params_id, config in enumerate(self._configs.values())
            ],
            index=pd.RangeIndex(len(self._configs), name="params_id"),
        )

    @cached_property
    def aliases(self) -> pd.DataFrame:
        """Return the varying column names used as parameters and their aliases."""
        params_df = self.params
        # drop columns containing only constant values
        columns = [col for col in params_df.columns if not all_equal(params_df[col])]
        # shorten unambiguous column names
        aliases = [col.rsplit(".", maxsplit=1)[-1] for col in columns]
        counter = Counter(aliases)
        return pd.DataFrame(
            [
                {"column": col, "alias": alias if counter[alias] == 1 else col}
                for col, alias in zip(columns, aliases)
            ]
        )

    @cached_property
    def df(self) -> pd.DataFrame:
        """Return the concatenation of features."""
        params_df = self.params[self.aliases["column"]]
        params_df.columns = self.aliases["alias"]
        # add params_id as a column
        params_df = params_df.reset_index()
        # concatenate all the features together
        return smart_concat(
            self._augment_dataframe(self._partial_df(name), params)
            for name, (_, params) in zip(self._configs, params_df.etl.iterdict())
        )

    def _partial_df(self, name: str) -> pd.DataFrame:
        """Return the specified partial DataFrame from the parent."""
        return getattr(self._parent, name).df

    @staticmethod
    def _augment_dataframe(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Return a copy of the DataFrame after adding columns from the given dict."""
        # ensure that a value can be assigned to a cell even when it's a list
        params = {k: [v] * len(df) for k, v in params.items()}
        return df.assign(**params)


@dataclass(frozen=True)
class FeaturesConfigKey:
    """Group of common keys used to build a group of FeaturesConfig.

    FeaturesConfig in the same group are grouped and processed together.

    Warning: the keys are considered immutable, and must not be modified after the object creation.
    """

    groupby: list[str]
    neuron_classes: list[str]
    windows: list[str]

    @classmethod
    def from_config(cls, config: FeaturesConfig) -> "FeaturesConfigKey":
        """Return an instance from the given FeaturesConfig."""
        return cls(
            groupby=config.groupby,
            neuron_classes=config.neuron_classes,
            windows=config.windows,
        )

    def __hash__(self):
        """Return the hash of the object."""
        return hash((tuple(self.groupby), tuple(self.neuron_classes), tuple(self.windows)))


class FeaturesCollection:
    """FeaturesCollection class."""

    def __init__(
        self,
        features_configs: list[FeaturesConfig],
        repo: Repository,
        cache_manager: CacheManager,
    ) -> None:
        """Initialize the FeaturesCollection from the given list of configurations.

        Args:
            features_configs: list of features configuration dicts.
            repo: Repository instance.
            cache_manager: CacheManager instance.
        """
        self._features_configs = features_configs
        self._repo = repo
        self._cache_manager = cache_manager
        self._data: dict[str, Feature] = {}
        self._concatenated_features: dict[str, ConcatenatedFeatures] = {}

    @property
    def names(self) -> list[str]:
        """Return the names of all the calculated features."""
        if not self._data:
            self.calculate()
        return sorted([*self._data, *self._concatenated_features])

    @property
    def cache_manager(self) -> CacheManager:
        """Access to the cache manager."""
        return self._cache_manager

    def __getattr__(self, name: str) -> Union[Feature, ConcatenatedFeatures]:
        """Return the features by name.

        It mimics the behaviour of Repository, where each Extractor can be accessed by attribute.

        Args:
            name: name of the features.
        """
        if name in self._data:
            return self._data[name]
        if name in self._concatenated_features:
            return self._concatenated_features[name]
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    def __dir__(self):
        """Allow autocompletion of dynamic attributes."""
        return list(super().__dir__()) + list(self._data) + list(self._concatenated_features)

    def _update_features(self, mapping: dict[str, Feature]) -> None:
        if overlapping := set(mapping).intersection(self._data):
            raise RuntimeError(
                f"Overlapping features dataframes aren't allowed: {sorted(overlapping)}"
            )
        if overlapping := set(mapping).intersection(self._concatenated_features):
            raise RuntimeError(
                f"Features cannot overlap with concatenated features: {sorted(overlapping)}"
            )
        self._data.update(mapping)

    def _update_concatenated_features(
        self, feature_names: list[str], features_config: FeaturesConfig
    ) -> None:
        if not features_config.suffix:
            # not a concatenated feature, nothing to do
            return
        for name in feature_names:
            basename = name.removesuffix(features_config.suffix)
            if basename in self._data:
                raise RuntimeError(
                    f"Concatenated features {basename} cannot overlap with features dataframes"
                )
            if basename not in self._concatenated_features:
                self._concatenated_features[basename] = ConcatenatedFeatures(self)
            concatenated = self._concatenated_features[basename]
            concatenated.update(feature_name=name, features_config=features_config)

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

        def _process_features(
            features_config: FeaturesConfig, features: dict[str, Feature]
        ) -> None:
            """Update the features of the instance, and write the cache if needed."""
            self._update_features(features)
            self._update_concatenated_features(list(features), features_config)
            to_be_written: dict[str, pd.DataFrame] = {}
            for name, f in features.items():
                if not f._cached or f._filtered:  # pylint: disable=protected-access
                    to_be_written[name] = f.to_pandas()
            if to_be_written:
                with timed(L.info, "Writing cached features"):
                    self.cache_manager.dump_features(to_be_written, features_config=features_config)

        def _log_features(features: dict[str, Feature], n: int, tot: int, features_id) -> None:
            """Log a message about the features being processed."""
            msg = "\n".join(
                # pylint: disable=protected-access
                f"- {name}: cached={f._cached}, filtered={f._filtered}"
                for name, f in features.items()
            )
            L.info("Calculated features %s/%s [id=%s]\n%s", n, tot, features_id, msg)

        def _group_features_by_attributes() -> tuple[
            list[FeaturesConfig],
            dict[FeaturesConfigKey, list[FeaturesConfig]],
        ]:
            cached: list[FeaturesConfig] = []
            groups: dict[FeaturesConfigKey, list[FeaturesConfig]] = defaultdict(list)
            tot = len(self._features_configs)
            for n, features_config in enumerate(self._features_configs, 1):
                L.info("Preprocessing features %s/%s [id=%s]", n, tot, features_config.id)
                if self.cache_manager.get_cached_features_checksums(features_config):
                    # cached, append the features config
                    cached.append(features_config)
                else:
                    # not cached, group the features_config by common keys
                    key = FeaturesConfigKey.from_config(features_config)
                    groups[key].append(features_config)
            return cached, groups

        def _process_cached_features(cached: list[FeaturesConfig]) -> None:
            for n, features_config in enumerate(cached, 1):
                query = None
                if self._repo.cache_manager.features_cache_needs_filter(features_config):
                    query = {SIMULATION_ID: self._repo.simulation_ids}
                df_dict = self.cache_manager.load_features(features_config=features_config)
                features = _calculate_cached(features_config, df_dict, query=query)
                _process_features(features_config, features)
                _log_features(features, n, len(cached), features_config.id)

        def _process_new_features(groups: dict[FeaturesConfigKey, list[FeaturesConfig]]) -> None:
            for num, (features_configs_key, features_configs_list) in enumerate(groups.items(), 1):
                L.info("Considering group: %s/%s, key: %s", num, len(groups), features_configs_key)
                for n, (features_config, features) in enumerate(
                    _calculate_new(self._repo, features_configs_key, features_configs_list), 1
                ):
                    _process_features(features_config, features)
                    _log_features(features, n, len(features_configs_list), features_config.id)

        with timed(L.info, "Step 1: grouping features by attributes"):
            features_configs_cached, features_configs_groups = _group_features_by_attributes()
        with timed(L.info, "Step 2: processing cached features"):
            _process_cached_features(features_configs_cached)
        with timed(L.info, "Step 3: processing new features"):
            _process_new_features(features_configs_groups)
        L.info("Features calculation completed")

    def apply_filter(self, repo: Repository) -> "FeaturesCollection":
        """Apply a filter based on the extracted simulations and return a new object."""
        return FilteredFeaturesCollection(parent=self, repo=repo)


class FilteredFeaturesCollection(FeaturesCollection):
    """FilteredFeaturesCollection class."""

    def __init__(self, parent: FeaturesCollection, repo: Repository) -> None:
        """Init from an existing FeaturesCollection filtered by the simulations ids in repo.

        Filtered dataframes are never written to disk.
        """
        super().__init__(
            features_configs=parent._features_configs,
            repo=repo,
            cache_manager=parent.cache_manager.to_readonly(),
        )
        dataframes = {name: features.df for name, features in parent._data.items()}
        query = {SIMULATION_ID: repo.simulation_ids}
        self._data = _dataframes_to_features(dataframes, config=None, cached=True, query=query)
        self._concatenated_features = self._clone_concatenated_features(
            parent._concatenated_features
        )

    def _clone_concatenated_features(
        self, concatenated_features: dict[str, ConcatenatedFeatures]
    ) -> dict[str, ConcatenatedFeatures]:
        """Return a dict of cloned concatenated_features."""
        return {name: cf.clone(parent=self) for name, cf in concatenated_features.items()}


def _dataframes_to_features(
    df_dict: dict[str, pd.DataFrame],
    config: Optional[FeaturesConfig],
    cached: bool,
    query: Optional[dict],
) -> dict[str, Feature]:
    """Wrap the given DataFrames with Feature objects, and return the resulting dict.

    The DataFrames are filtered by the simulation_ids contained in the internal repo.

    Args:
        df_dict: dict of DataFrames to be wrapped.
        config: config converted to dict and assigned to attrs["config"] for each DataFrame.
        cached: True if the data is loaded from the cache, False otherwise.
        query: optional query that can be used to filter the features dataframes.
    """
    result = {}
    for name, df in df_dict.items():
        result[name] = Feature.from_pandas(df, query=query, cached=cached)
        if config is not None:
            # make a copy of the config accessible from the features dataframe attrs
            result[name].df.attrs["config"] = config.dict()
    return result


def _calculate_cached(
    features_config: FeaturesConfig,
    df_dict: dict[str, pd.DataFrame],
    query: Optional[dict],
) -> dict[str, Feature]:
    """Load cached features from a dict of DataFrames."""
    return _dataframes_to_features(df_dict, config=features_config, cached=True, query=query)


def _calculate_new(
    repo: Repository,
    features_configs_key: FeaturesConfigKey,
    features_configs_list: list[FeaturesConfig],
) -> Iterator[tuple[FeaturesConfig, dict[str, Feature]]]:
    """Calculate new features and yield tuples."""
    results = _calculate_features(
        repo=repo,
        features_configs_key=features_configs_key,
        features_configs_list=features_configs_list,
    )
    assert len(features_configs_list) == len(results)
    for features_config, df_dict in zip(features_configs_list, results):
        features = _dataframes_to_features(
            df_dict, config=features_config, cached=False, query=None
        )
        yield features_config, features


def _user_func_wrapper(
    *,
    task_index: int,
    key: NamedTuple,
    neurons_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    report_df: pd.DataFrame,
    repo: Repository,
    features_config: FeaturesConfig,
    temp_folder: Path,
) -> None:
    """Call the user function for the specified key, and save the resulting DataFrames.

    It should be executed in a subprocess.

    Args:
        task_index: incremental index.
        key: namedtuple specifying the filter.
        neurons_df: filtered neurons DataFrame.
        windows_df:  filtered windows DataFrame.
        report_df:  filtered report DataFrame.
        repo: Repository instance.
        features_config: features configuration.
        temp_folder: path to the shared memory (recommended) or temp directory.
    """
    L.info("Calculating features for %s", key)
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
    for feature_group, result_df in features_dict.items():
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError(f"Expected a DataFrame, not {type(result_df).__name__}")
        # ignore the index if it's unnamed and with one level; this can be useful
        # for example when the returned DataFrame has a RangeIndex to be dropped
        drop = result_df.index.names == [None]
        if features_config.multi_index:
            result_df = result_df.etl.add_conditions(conditions=key._fields, values=key, drop=drop)
        else:
            result_df.reset_index(drop=drop, inplace=True)
            result_df.etl.insert_columns(loc=0, columns=key._fields, values=key)
        # the conversion to the desired dtype here is important to reduce memory usage and cpu time
        result_df = ensure_dtypes(result_df)
        output_dir = temp_folder / f"{feature_group}{features_config.suffix}"
        if not output_dir.is_dir():
            # the directory should be created in the first process
            output_dir.mkdir(parents=True, exist_ok=True)
        ParquetStore(output_dir).dump(result_df, name=f"{task_index:08d}")


def _merge_filter_func(
    *,
    task_index: int,
    key: NamedTuple,
    df_list: list[pd.DataFrame],
    temp_folder: Path,
    repo: Repository,
    features_configs_list: list[FeaturesConfig],
) -> None:
    """Executed in a subprocess, call the wrapper function for each features_config."""
    neurons_df, windows_df, report_df = df_list
    for features_config_index, features_config in enumerate(features_configs_list):
        _user_func_wrapper(
            task_index=task_index,
            key=key,
            neurons_df=neurons_df,
            windows_df=windows_df,
            report_df=report_df,
            repo=repo,
            features_config=features_config,
            temp_folder=temp_folder / str(features_config_index),
        )


def _filter_by_value(df: pd.DataFrame, key: str, value: Any) -> pd.DataFrame:
    """Filter the DataFrame only if the specified value is not None or empty."""
    return df.etl.q({key: value}) if value else df


@isolated
def _merge_filter_wrapper(
    temp_folder: Path,
    repo: Repository,
    features_configs_key: FeaturesConfigKey,
    features_configs_list: list[FeaturesConfig],
) -> None:
    """Execute merge_filter in an isolated subprocess.

    It's faster than running in the main process, for example the subprocess
    can take 167 seconds instead of 266 seconds (needing more investigation).

    With the current approach:

    - repo is pickled and passed to the subprocess
    - the needed dataframes are loaded from the cache in the subprocess
    - this is faster than passing big dataframes as parameters, that should be pickled
      in the main process and unpickled in the subprocess.
    """
    func = partial(
        _merge_filter_func,
        temp_folder=temp_folder,
        features_configs_list=features_configs_list,
        repo=repo,
    )
    merge_filter(
        df_list=[
            _filter_by_value(
                repo.neurons.df,
                key="neuron_class",
                value=features_configs_key.neuron_classes,
            ),
            _filter_by_value(
                repo.windows.df,
                key="window",
                value=features_configs_key.windows,
            ),
            repo.report.df,
        ],
        groupby=features_configs_key.groupby,
        func=func,
    )


def _concatenate_all(temp_folder: Path) -> list[dict[str, pd.DataFrame]]:
    """Concatenate all the dataframes having the same feature_group label.

    The DataFrames are loaded from parquet files contained in a folder structure like:

    /temp_folder
    ├── 0
    │   ├── by_gid
    │   │   └── *.parquet
    │   ├── by_gid_and_trial
    │   │   └── *.parquet
    │   ├── by_neuron_class
    │   │   └── *.parquet
    │   └── by_neuron_class_and_trial
    │       └── *.parquet
    ├── 1
    │   └── other_features
    │       └── *.parquet
    ...
    └── n
        └── other_features
            └── *.parquet

    where each numerical folder corresponds to one entry in features_configs_list,
    and contains one subfolder for each DataFrame to be created.

    Returns:
        list of DataFrames obtained by the concatenation of the partial DataFrames.
    """
    result = []
    # the numerical subdirectories must be processed in ascending order
    for index_path in sorted(temp_folder.iterdir(), key=lambda p: int(p.name)):
        d = {}
        # the dataframes subdirectories can be processed in any order
        for features_path in sorted(index_path.iterdir()):
            df = ParquetStore(features_path).load()
            df = ensure_dtypes(df)
            d[features_path.name] = df
        result.append(d)
    return result


def _calculate_features(
    repo: Repository,
    features_configs_key: FeaturesConfigKey,
    features_configs_list: list[FeaturesConfig],
) -> list[dict[str, pd.DataFrame]]:
    """Calculate features in parallel for the given repository as a dict of DataFrames.

    Args:
        repo: repository containing spikes.
        features_configs_key: common key of the features configurations.
        features_configs_list: list of features configurations.

    Returns:
        list of dicts of features DataFrames, one item for each features_config.
    """
    with tempfile.TemporaryDirectory(prefix="blueetl_", dir=get_shmdir()) as _temp_folder:
        with timed(L.info, "Executing merge_filter"):
            _merge_filter_wrapper(
                temp_folder=Path(_temp_folder),
                repo=repo,
                features_configs_key=features_configs_key,
                features_configs_list=features_configs_list,
            )
        with timed(L.info, "Executing concatenation"):
            return _concatenate_all(temp_folder=Path(_temp_folder))
