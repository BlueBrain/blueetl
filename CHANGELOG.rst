Changelog
=========

Version 0.2.0
-------------

New Features
~~~~~~~~~~~~
- Add MultiAnalyzer class to support multiple reports [NSETM-2015]
- Allow to resolve windows by reference [NSETM-2015]
- Support combination of parameters in features configuration [NSETM-2091]
- Add analysis configuration model and validation [NSETM-2099]
- Add blueetl CLI [NSETM-2115]
- Add blueetl.analysis.run_from_file [NSETM-2151]
- Improve performance of report extraction [NSETM-2116]
- Improve performance of features calculation [NSETM-2116]
- Process features in group when possible.
- Add `cached` and `filtered` attributes to `BaseExtractor`.
- Add `clear_cache` parameter to `run_from_file` and to the configuration schema [NSETM-2150]

Breaking changes
~~~~~~~~~~~~~~~~
- The previous analysis configuration format has been replaced by the version 2 to support multiple reports.
- After applying a filter, the indices of the repo DataFrames are reset to remove any gap.
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
