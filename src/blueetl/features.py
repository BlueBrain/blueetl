"""Features collection."""
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, Optional, Union

import pandas as pd

from blueetl.cache import CacheManager
from blueetl.config.analysis_model import FeaturesConfig
from blueetl.constants import SIMULATION_ID
from blueetl.core.utils import safe_concat
from blueetl.extract.feature import Feature
from blueetl.parallel import merge_filter
from blueetl.repository import Repository
from blueetl.utils import all_equal, ensure_dtypes, extract_items, import_by_string, timed

L = logging.getLogger(__name__)


class ConcatenatedFeatures:
    """ConcatenatedFeatures class.

    It can be used to view as a single DataFrame all the features calculated
    for various combinations of parameters.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        self._features: list[Feature] = []
        self._configs: list[FeaturesConfig] = []

    def update(self, feature: Feature, features_config: FeaturesConfig) -> None:
        """Update the list of features and configurations."""
        self.clear_cache()
        self._features.append(feature)
        self._configs.append(features_config)

    def clear_cache(self) -> None:
        """Clear the cached properties."""
        for key in "params", "aliases", "df":
            self.__dict__.pop(key, None)

    @cached_property
    def params(self) -> pd.DataFrame:
        """Return all the parameters as a DataFrame."""
        return pd.DataFrame(
            [dict(extract_items(config.params)) for params_id, config in enumerate(self._configs)],
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
        return safe_concat(
            self._augment_dataframe(feature.df, params)
            for feature, (_, params) in zip(self._features, params_df.etl.iterdict())
        )

    @staticmethod
    def _augment_dataframe(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Return a copy of the DataFrame after adding columns from the given dict."""
        # this is done to handle values when they are lists
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
        self._concatenated_features: dict[str, ConcatenatedFeatures] = {}
        if _dataframes:
            self._data = self._dataframes_to_features(_dataframes, config=None, cached=True)

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
        return list(super().__dir__()) + list(self._data)

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
        self, mapping: dict[str, Feature], features_config: FeaturesConfig
    ) -> None:
        if not features_config.suffix:
            # not a concatenated feature, nothing to do
            return
        for name, feature in mapping.items():
            basename = name.removesuffix(features_config.suffix)
            if basename in self._data:
                raise RuntimeError(
                    f"Concatenated features {basename} cannot overlap with features dataframes"
                )
            concatenated = self._concatenated_features.setdefault(basename, ConcatenatedFeatures())
            concatenated.update(feature=feature, features_config=features_config)

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

        def _calculate_cached(
            features_config: FeaturesConfig, df_dict: dict[str, pd.DataFrame]
        ) -> dict[str, Feature]:
            """Load cached features from a dict of DataFrames."""
            return self._dataframes_to_features(df_dict, config=features_config, cached=True)

        def _calculate_new(
            features_configs_key: FeaturesConfigKey, features_configs_list: list[FeaturesConfig]
        ) -> Iterator[tuple[FeaturesConfig, dict[str, Feature]]]:
            """Calculate new features and yield tuples."""
            results = calculate_features(
                repo=self._repo,
                features_configs_key=features_configs_key,
                features_configs_list=features_configs_list,
            )
            assert len(features_configs_list) == len(results)
            for features_config, df_dict in zip(features_configs_list, results):
                features = self._dataframes_to_features(
                    df_dict, config=features_config, cached=False
                )
                yield features_config, features

        def _process_features(
            features_config: FeaturesConfig, features: dict[str, Feature]
        ) -> None:
            """Update the features of the instance, and write the cache if needed."""
            self._update_features(features)
            self._update_concatenated_features(features, features_config)
            to_be_written: dict[str, pd.DataFrame] = {}
            for name, f in features.items():
                if not f._cached or f._filtered:  # pylint: disable=protected-access
                    to_be_written[name] = f.to_pandas()
            if to_be_written:
                self.cache_manager.dump_features(to_be_written, features_config=features_config)

        def _log_features(features: dict[str, Feature], n: int, tot: int, features_id) -> None:
            """Log a message about the features being processed."""
            msg = "\n".join(
                # pylint: disable=protected-access
                f"- {name}: cached={f._cached}, filtered={f._filtered}"
                for name, f in features.items()
            )
            L.info("Calculated features %s/%s [id=%s]\n%s", n, tot, features_id, msg)

        def _group_features_by_attributes() -> (
            tuple[
                list[FeaturesConfig],
                dict[FeaturesConfigKey, list[FeaturesConfig]],
            ]
        ):
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
                df_dict = self.cache_manager.load_features(features_config=features_config)
                features = _calculate_cached(features_config, df_dict)
                _process_features(features_config, features)
                _log_features(features, n, len(cached), features_config.id)

        def _process_new_features(groups: dict[FeaturesConfigKey, list[FeaturesConfig]]) -> None:
            for features_configs_key, features_configs_list in groups.items():
                L.info("Considering: %s", features_configs_key)
                for n, (features_config, features) in enumerate(
                    _calculate_new(features_configs_key, features_configs_list), 1
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

    def _dataframes_to_features(
        self, df_dict: dict[str, pd.DataFrame], config: Optional[FeaturesConfig], cached: bool
    ) -> dict[str, Feature]:
        query = {SIMULATION_ID: self._repo.simulation_ids}
        result = {}
        for name, df in df_dict.items():
            result[name] = Feature.from_pandas(df, query=query, cached=cached)
            if config is not None:
                # make a copy of the config accessible from the features dataframe attrs
                result[name].df.attrs["config"] = config.dict()
        return result

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


def _func_wrapper(
    key: NamedTuple,
    neurons_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    report_df: pd.DataFrame,
    repo: Repository,
    features_config: FeaturesConfig,
) -> dict[str, pd.DataFrame]:
    """Call the user function for the specified key.

    Args:
        key: namedtuple specifying the filter.
        neurons_df: filtered neurons DataFrame.
        windows_df:  filtered windows DataFrame.
        report_df:  filtered report DataFrame.
        repo: Repository instance.
        features_config: features configuration.

    Returns:
        dict of features DataFrames.
    """
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


def calculate_features(
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
        list of dicts of features DataFrames, one item for each features config.
    """

    def _func(key: NamedTuple, df_list: list[pd.DataFrame]) -> list[dict[str, pd.DataFrame]]:
        """Should be called in a subprocess to execute the wrapper function."""
        neurons_df, windows_df, report_df = df_list
        return [
            _func_wrapper(
                key=key,
                neurons_df=neurons_df,
                windows_df=windows_df,
                report_df=report_df,
                repo=repo,
                features_config=features_config,
            )
            for features_config in features_configs_list
        ]

    def _filter_by_value(df: pd.DataFrame, key: str, value: Any) -> pd.DataFrame:
        """Filter the DataFrame only if the specified value is not None or empty."""
        return df.etl.q({key: value}) if value else df

    def _concatenate_all(
        it: Iterator[list[dict[str, pd.DataFrame]]]
    ) -> list[dict[str, pd.DataFrame]]:
        """Concatenate all the dataframes having the same feature_group label.

        Args:
            it: iterator yielding lists of dict of DataFrames, where the number of lists is equal
                to the number of groups determined by features_configs_key, and the number of dicts
                in each list is equal to the number of FeaturesConfig in features_configs_list.

        Returns:
            list of DataFrames obtained by the concatenation of the partial DataFrames.
        """
        tmp_result: list[dict[str, list[pd.DataFrame]]] = [
            defaultdict(list) for _ in range(len(features_configs_list))
        ]
        for n_group, lst in enumerate(it):
            # lst is the list of dicts returned by _func, and it contains one dict for each config
            assert len(lst) == len(tmp_result)
            for n_config, df_dict in enumerate(lst):
                # to concatenate across the groups the DataFrames contained in each dict,
                # append tmp_df to the list holding all the other tmp_df of the same type
                partial_result = tmp_result[n_config]
                for feature_group, tmp_df in df_dict.items():
                    L.debug(
                        "Iterating over group=%s, config=%s, feature_group=%s",
                        n_group,
                        n_config,
                        feature_group,
                    )
                    partial_result[feature_group].append(tmp_df)
        # finally, build the dicts of DataFrames in a single pd.concat operation
        return [
            {
                feature_group: ensure_dtypes(pd.concat(df_list))
                for feature_group, df_list in dct.items()
            }
            for dct in tmp_result
        ]

    key = features_configs_key
    return _concatenate_all(
        merge_filter(
            df_list=[
                _filter_by_value(repo.neurons.df, "neuron_class", value=key.neuron_classes),
                _filter_by_value(repo.windows.df, "window", value=key.windows),
                repo.report.df,
            ],
            groupby=key.groupby,
            func=_func,
            parallel=True,
        )
    )
