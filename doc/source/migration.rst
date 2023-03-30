Migration
=========


From 0.1.x to 0.2.x
-------------------

BlueETL 0.2 introduces some breaking changes to support the analysis of multiple reports.

In this document it's described what should be done to adapt the existing configurations and user code.

Note that there aren't breaking changes in the core functionalities.


Configuration
~~~~~~~~~~~~~

Automatic migration
...................

You can automatically migrate a configuration file executing in a virtualenv with blueetl installed:

.. code-block::

    blueetl migrate-config INPUT_CONFIG_FILE OUTPUT_CONFIG_FILE

However, you may need to manually copy any commented lines from the old configuration, or they will be lost.


Manual migration
................

If you prefer to migrate the configuration manually instead, follow these steps:

1. The specification ``version: 2`` should be added at the top level of the file.
2. The section ``extraction`` should be moved to ``analysis.spikes.extraction``.
3. The section ``analysis.features`` should be moved to ``analysis.spikes.features``.
4. Any custom key should be moved into an optional dict: ``custom`` if the parameters are global, or ``analysis.spikes.custom`` if the parameters are specific to the spikes analysis.
5. The following sub-section should be added to ``analysis.spikes.extraction``:

.. code-block:: yaml

    report:
      type: spikes

You can see an example of configuration in the old and new format here:

- https://bbpgitlab.epfl.ch/nse/blueetl/-/blob/blueetl-v0.1.2/tests/functional/data/analysis_config_01.yaml
- https://bbpgitlab.epfl.ch/nse/blueetl/-/blob/blueetl-v0.2.0.dev0/tests/functional/data/analysis_config_01.yaml

Analysis
~~~~~~~~

Initialization
..............

Instead of code like this:

.. code-block:: python

    import logging
    import numpy as np
    import yaml
    from blueetl.analysis import Analyzer

    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    config = yaml.safe_load("analysis_config.yaml")
    a = Analyzer(config)
    a.extract_repo()
    a.calculate_features()


you can use this:

.. code-block:: python

    from blueetl.analysis import run_from_file

    ma = run_from_file("analysis_config.yaml", loglevel="INFO")
    a = ma.spikes

where ``ma`` is an instance of ``MultiAnalyzer`` and ``a`` is an instance of ``SingleAnalyzer``.

If you need to work with multiple analysis, using the instance of ``MultiAnalyzer`` may be more convenient.


Deprecation of the ``spikes`` attribute
.......................................

When accessing the ``spikes`` DataFrame with:

.. code-block:: python

    a.repo.spikes.df

you should use instead the generic ``report`` attribute, valid for any type of report:

.. code-block:: python

    a.repo.report.df

The old name `spikes` is kept for backward compatibility, but it should be considered deprecated and it will be removed later.


Accessing the custom configuration
..................................

If you stored any custom configuration, you can get the values from the dictionaries:

- ``ma.global_config.custom``
- ``ma.spikes.analysis_config.custom``


Using ``call_by_simulation``
............................

The function ``call_by_simulation`` has been moved from ``bluepy.features`` to ``bluepy.parallel``.
