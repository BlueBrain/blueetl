Changelog
=========

Version 0.1.2
-------------

Improvements
~~~~~~~~~~~~
- Raise an exception if there are multiple features dataframes with the same name.
- Enforce the correct dtype in the features dataframes.
- Add ``dtypes`` parameter to ``ETLBaseAccessor.add_conditions``.
- Add ``dtypes`` and ``astype`` methods to ``ETLIndexAccessor``.

Bug Fixes
~~~~~~~~~
- Deepcopy the params dict passed to the user func.


Version 0.1.1
-------------

New Features
~~~~~~~~~~~~
- Ignore simulations for which BlueConfig no longer exists [NSETM-1967]
- Add optional in-memory filter [NSETM-1965]

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
