Changelog
=========

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

Bug Fixes
~~~~~~~~~
- Reset the index in the simulations dataframe after applying filters.


Version 0.1.0
-------------

First release including:

- Core Transformations
- Simulation Campaign Configuration
- Spike Analysis (Repository Extraction and Features Collection)
