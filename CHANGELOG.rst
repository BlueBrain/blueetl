Changelog
=========

Version 0.13.4
--------------

Bug Fixes
~~~~~~~~~

- In soma and compartment reports, add trial number in the extracted dataframe, as in the spikes report.
- In soma and compartment reports, make the exported time relative to the offset, as in the spikes report.

Version 0.13.3
--------------

Bug Fixes
~~~~~~~~~

- Create ``.zenodo.json`` as a workaround to publish the release on Zenodo.

Version 0.13.2
--------------

- Technical release.

Version 0.13.1
--------------

- Technical release.

Version 0.13.0
--------------

Improvements
~~~~~~~~~~~~

- When loading the cache, do not fail if the original simulation campaign and related files are missing [NSETM-2336].


Version 0.12.1
--------------

Bug Fixes
~~~~~~~~~

- Fix functional tests.


Version 0.12.0
--------------

Breaking changes
~~~~~~~~~~~~~~~~

- Ensure that some methods and functions are called with named parameters. In particular, the following public methods require named parameters:

  - ``blueetl.analysis.run_from_file()``
  - ``blueetl.extract.report.ReportExtractor.from_simulations()``
  - ``blueetl.repository.Repository.__init__()``
  - ``blueetl.repository.FilteredRepository.__init__()``

Version 0.11.4
--------------

Bug Fixes
~~~~~~~~~

- Fix tests with libsonata 0.1.28.
- Skip simulations with empty path in the simulation campaign configuration (fix regression in blueetl 0.8.0).


Version 0.11.3
--------------

Bug Fixes
~~~~~~~~~

- Update expected results in functional tests, because the output of statistics.Complexity changed in elephant >= 0.13.0.
- Fix warning when building docs, and include ``blueetl.analysis`` only.

Version 0.11.2
--------------

Bug Fixes
~~~~~~~~~

- Fix version number in the analysis config created by ``blueetl convert-spikes``.


Version 0.11.1
--------------

Bug Fixes
~~~~~~~~~

- Revert raising an error if the env variable SHMDIR isn't set. Log a more detailed warning instead.


Version 0.11.0
--------------

Improvements
~~~~~~~~~~~~

- Allow to access the cache config and the output path of individual analyses with ``analysis_config.cache`` and ``analysis_config.output``, as a shortcut to ``analysis_config.cache.path``.
- Raise an error if the env variable ``SHMDIR`` isn't set, instead of logging a warning.


Version 0.10.1
--------------

Bug Fixes
~~~~~~~~~

- Fix link to the example configuration in the documentation.


Version 0.10.0
--------------

New Features
~~~~~~~~~~~~

- Add ``multi_index`` option to the features configuration, to decide whether ``reset_index()`` should be applied to the features DataFrames.
  Note: the features cache will be rebuilt, although the resulting DataFrames are unchanged (because the default value of the new option is ``True``).


Version 0.9.1
-------------

Improvements
~~~~~~~~~~~~

- Improve performance of report extraction and features calculation by writing partial DataFrames to the shared memory (or temp directory, if shared memory is not available).
  Both the used memory and the execution time should be lower than before, when processing large DataFrames.
- Use zstd compression instead of snappy when writing parquet files.
- When ``repo`` is pickled, extract the DataFrames only if they aren't already stored in the cache.
- Remove fastparquet extra dependency.

Version 0.9.0
-------------

New Features
~~~~~~~~~~~~

- Add ``cache.readonly``, to be able to use an existing cache without exclusive locking [NSETM-2310].
- Add ``cache.store_type``, to change the file format (experimental).
- Add ``cache.skip_features``, to skip writing the features DataFrames [NSETM-2312].

Deprecations
~~~~~~~~~~~~

- Deprecate ``output``, use ``cache.path`` instead.
- Deprecate ``clear_cache``, use ``cache.clear`` instead.


Version 0.8.3
-------------

Bug Fixes
~~~~~~~~~

- Ensure that invalid cache entries are always deleted from ``checksums.cached.yaml``.

Improvements
~~~~~~~~~~~~

- Filter features only when ``apply_filter`` is called to save some time, but ensure that repo and features are filtered when they are loaded from the cache using a more restrictive filter.
- Improve logging in ``blueetl.utils.timed()``.
- Improve tests coverage.
- Split migration documentation.
- Add ``--sort/--no-sort`` option to ``blueetl migrate-config`` to sort the root keys of the converted configuration.

Version 0.8.2
-------------

Bug Fixes
~~~~~~~~~

- If ``node_id`` in the neuron_classes configuration is set to an empty list, it's now considered as an empty selection instead of selecting all the neurons.

Version 0.8.1
-------------

Improvements
~~~~~~~~~~~~

- Add configuration examples 11 and 12 to the documentation.
- Improve tests coverage.

Bug Fixes
~~~~~~~~~

- Fix method ``Repository.missing_simulations()``.

Version 0.8.0
-------------

New Features
~~~~~~~~~~~~

- Support custom node_sets_file in extraction, neuron_classes and trial_steps [NSETM-2226]


Version 0.7.1
-------------

Bug Fixes
~~~~~~~~~

- Fix expected data in functional tests.

Version 0.7.0
-------------

New Features
~~~~~~~~~~~~

- Allow to specify ``trial_steps_label`` to calculate the dynamic offset of trial steps [NSETM-2281]


Version 0.6.0
-------------

New Features
~~~~~~~~~~~~

- Allow to specify ``trial_steps_list`` instead of ``trial_steps_value`` and ``n_trials`` [NSETM-2280]

Bug Fixes
~~~~~~~~~

- Temporarily disable ``trial_steps_label`` [NSETM-2281]

Improvements
~~~~~~~~~~~~

- Add tests for Python 3.12.
- Remove brion dependency in tests.


Version 0.5.0
-------------

New Features
~~~~~~~~~~~~

- Add CLI to convert and import inferred spikes in CSV format.


Version 0.4.4
-------------

Improvements
~~~~~~~~~~~~

- Support relative paths in the simulation campaign config.
- Add a simple simulation campaign using a subsampled circuit, to run the Jupyter notebooks in the documentation.

Version 0.4.3
-------------

Improvements
~~~~~~~~~~~~

- Update DOI.

Version 0.4.2
-------------

Improvements
~~~~~~~~~~~~

- Fix docs build in rtd.
- Update badges.
- Conditionally skip tests requiring bluepy.

Version 0.4.1
-------------

- First public release.

Version 0.4.0
-------------

New Features
~~~~~~~~~~~~

- Extend the API of SimulationCampaign (previously SimulationsConfig) to open simulation campaigns.

Breaking changes
~~~~~~~~~~~~~~~~

- Rename SimulationsConfig to SimulationCampaign.


Version 0.3.0
-------------

New Features
~~~~~~~~~~~~
- Support SONATA simulation campaigns, circuits, and reports using bluepysnap.

Breaking changes
~~~~~~~~~~~~~~~~
- Simulation campaigns, circuits, and reports using BlueConfig format aren't supported anymore.
- The analysis configuration accepts ``population`` and ``node_set``, instead of ``target``.
- In the ``neuron_classes`` definition, the query parameters must be moved to ``query``, ``$limit`` must be renamed to ``limit``, ``$gids`` to ``node_id``.
- The function ``blueetl.core.utils.safe_concat`` has been renamed to ``smart_concat``.
- The module ``blueetl.core`` has been moved to a separate package, ``blueetl-core``.

Improvements
~~~~~~~~~~~~
- The function ``blueetl.core.utils.smart_concat`` uses ``copy=False`` by default, and accepts dictionaries as ``pd.concat`` does.
- All the internal calls to ``pd.concat`` are redirected to ``smart_concat``.


Version 0.2.3
-------------

Improvements
~~~~~~~~~~~~
- Improve performance of etl.add_conditions.


Version 0.2.2
-------------

Bug Fixes
~~~~~~~~~
- Ensure that the package can be installed and used without optional dependencies.


Version 0.2.1
-------------

Improvements
~~~~~~~~~~~~
- Support Pandas 2.0.
  Changed in Pandas 2.0.0: Index can hold all numpy numeric dtypes (except float16).
  Previously only int64/uint64/float64 dtypes were accepted.

Version 0.2.0
-------------

New Features
~~~~~~~~~~~~
- Add MultiAnalyzer class to support multiple reports [NSETM-2015]
- Allow to resolve windows by reference [NSETM-2015]
- Support combination of parameters in features configuration [NSETM-2091]
- Allow to access the concatenation of features dataframes using the basename [NSETM-2149]
- Add analysis configuration model and validation [NSETM-2099]
- Add blueetl CLI [NSETM-2115]
- Add blueetl.analysis.run_from_file [NSETM-2151]
- Improve performance of report extraction [NSETM-2116]
- Improve performance of features calculation [NSETM-2116]
- Process features in group when possible.
- Add `_cached` and `_filtered` private attributes to `BaseExtractor`.
- Add `clear_cache` parameter to `run_from_file` and to the configuration schema [NSETM-2150]
- Allow etl.q to support regular expressions [NSETM-2170]
- Allow etl.q to accept a list of query dicts [NSETM-2162]
- Allow neuron_classes configuration to be defined as a list of query dicts [NSETM-2163]

Breaking changes
~~~~~~~~~~~~~~~~
- The previous analysis configuration format has been replaced by the version 2 to support multiple reports.
- After applying a filter, the indices of the repo DataFrames are reset to remove any gap.
- The function ``call_by_simulation`` has been refactored and moved into ``blueetl.parallel``.
- In neuron_classes configuration, ``gid`` has been renamed to ``$gids``.
- Require Python >= 3.9.


Version 0.1.2
-------------

Improvements
~~~~~~~~~~~~
- Raise an exception if there are multiple features dataframes with the same name.
- Enforce the correct dtype in the features dataframes.
- Add ``dtypes`` parameter to ``ETLBaseAccessor.add_conditions``.
- Add ``dtypes`` and ``astype`` methods to ``ETLIndexAccessor``.
- Support filtering by windows or neuron classes for each features configuration [NSETM-2085]

Bug Fixes
~~~~~~~~~
- Deepcopy the params dict passed to the user func.


Version 0.1.1
-------------

New Features
~~~~~~~~~~~~
- Ignore simulations for which BlueConfig no longer exists [NSETM-1967]
- Add optional in-memory filter [NSETM-1965]
- Support subtargets per neuron class [NSETM-2004]

Improvements
~~~~~~~~~~~~
- Add env variable ``BLUEETL_SUBPROCESS_LOGGING_LEVEL`` to set a logging level in subprocesses.
- Improve log of execution times.
- Improve performances of ``etl.q`` when only a single condition is specified.
- Lock the cache used by the Analyzer instance [NSETM-1971]
- Make the function `call_by_simulation` more flexible.

Bug Fixes
~~~~~~~~~
- Reset the index in the simulations dataframe after applying filters.
- Ensure that RangeIndex is converted to Int64Index in MultiIndexes with Pandas 1.5.0,
  see https://issues.apache.org/jira/browse/ARROW-17806.

Version 0.1.0
-------------

First release including:

- Core Transformations
- Simulation Campaign Configuration
- Spike Analysis (Repository Extraction and Features Collection)
