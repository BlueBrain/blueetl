Migration
=========


From 0.2.x to 0.3.x
-------------------

BlueETL 0.3.x introduces some breaking changes to support the analysis of SONATA simulation campaigns, along with BBP simulation campaigns.

It automatically uses Blue Brain SNAP or BluePy to open simulations, circuits, and reports.

In this section it's described what should be done to adapt the existing configurations and user code.


Configuration
~~~~~~~~~~~~~

Automatic migration
...................

You can automatically migrate a configuration from 0.2.x to 0.3.x, executing in a virtualenv with blueetl installed:

.. code-block::

    blueetl migrate-config INPUT_CONFIG_FILE OUTPUT_CONFIG_FILE

However, you may need to manually:

- copy any commented lines from the old configuration, or they will be lost.
- verify the names of the node_sets, because they are copied from the existing target names.


Manual migration
................

If you prefer to migrate the configuration manually instead, follow these steps:

1. The specification ``version: 2`` should be replaced by ``version: 3`` at the top level of the file.
2. Only when opening SONATA simulation campaigns, the key ``population``, containing the name of the node population to be analyzed, must be present in any section ``analysis.<name>.extraction``.
3. In the ``neuron_classes definition``, the parameters of the query not starting with ``$`` must be moved into a sub-dictionary under the key ``query``.
4. If present, any key ``$query`` must be replaced with ``query``.
5. If present, any key ``$target`` or ``target`` must be replaced with ``node_set``, regardless of whether it is a SONATA or a BBP simulation campaign.
   The value of ``node_set`` is treated as ``target`` when opening BlueConfig simulations.
   Pleas ensure that the specified name exists as a node_set (in case of SONATA simulations), or target (in case of BlueConfig simulations).
6. If present, any key ``$limit`` must be replaced with ``limit``.
7. If present, any key ``$gids`` must be replaced with ``node_id``.


You can see an example of configuration in the new format here:

- https://github.com/BlueBrain/blueetl/blob/blueetl-v0.3.0/tests/functional/data/sonata/config/analysis_config_01.yaml


From 0.1.x to 0.2.x
-------------------

BlueETL 0.2 introduces some breaking changes to support the analysis of multiple reports.

In this section it's described what should be done to adapt the existing configurations and user code.

Note that there aren't breaking changes in the core functionalities.


Configuration
~~~~~~~~~~~~~

Automatic migration
...................

You can automatically migrate a configuration from 0.1.x to 0.3.x using the command line.

See the section above for more details.


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

- https://github.com/BlueBrain/blueetl/blob/blueetl-v0.1.2/tests/functional/data/analysis_config_01.yaml
- https://github.com/BlueBrain/blueetl/blob/blueetl-v0.2.0/tests/functional/data/analysis_config_01.yaml


Output directory
................

If the output directory defined in the configuration is a relative path, it's now relative to the configuration file itself.

Previously, using a relative path was not recommended, since it was relative to the working directory.

This means that you can decide to always set ``output: output``, and have a directory layout like the following:

.. code-block::

    analyses
    ├── analysis_config_01
    │   ├── config.yaml
    │   ├── output
    │   │   └── spikes
    │   │       ├── config
    │   │       │   ├── analysis_config.cached.yaml
    │   │       │   ├── checksums.cached.yaml
    │   │       │   └── simulations_config.cached.yaml
    │   │       ├── features
    │   │       │   ├── baseline_PSTH.parquet
    │   │       │   ├── decay.parquet
    │   │       │   ├── latency.parquet
    │   │       └── repo
    │   │           ├── neuron_classes.parquet
    │   │           ├── neurons.parquet
    │   │           ├── report.parquet
    │   │           ├── simulations.parquet
    │   │           ├── trial_steps.parquet
    │   │           └── windows.parquet
    ├── analysis_config_02
    │   ├── config.yaml
    │   ├── output
    │   │   └── spikes
    ...


Analysis
~~~~~~~~

Initialization
..............

Instead of code like this:

.. code-block:: python

    import logging
    import numpy as np
    from blueetl.analysis import Analyzer
    from blueetl.utils import load_yaml

    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    config = load_yaml("analysis_config.yaml")
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


Deprecation of spikes
.....................

Instead of accessing the ``spikes`` DataFrame with:

.. code-block:: python

    a.repo.spikes.df

you should use the generic ``report`` attribute, valid for any type of report:

.. code-block:: python

    a.repo.report.df

The old name `spikes` is kept for backward compatibility, but it should be considered deprecated and it will be removed later.


Accessing the custom config
...........................

If you stored any custom configuration, you can get the values from the dictionaries:

- ``ma.global_config.custom``
- ``ma.spikes.analysis_config.custom``


Using call_by_simulation
........................

The function ``call_by_simulation`` has been moved from ``bluepy.features`` to ``bluepy.parallel``.
