Usage
=====

Core Transformations
--------------------

To use the Core Transformations provided by the ``.etl`` accessor with any Pandas DataFrame or Series, it's enough to import BlueETL and call the desired methods.

For example:

.. code-block:: python


    import blueetl
    import pandas as pd

    df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
    df = df.etl.q(a=1)


See the Jupyter notebook :doc:`/notebooks/01_core_transformations` for more information and examples.


Analysis of reports
-------------------

Basic usage
+++++++++++

To perform the analysis of reports across multiple simulations in a given simulation campaign, a configuration file needs to be provided.

The configuration file should specify in particular:

- ``simulation_campaign``: path to the json configuration file of the Simulation Campaign produced by bbp-workflow.
- ``output``: path to the output directory where the results are stored. It's also used as cache, and existing stale files may be automatically deleted.
- ``analysis``: dictionary containing a key for each report to be analyzed.

and for each report:

- ``extraction``: configuration dictionary used to extract data from the report, by window and neuron class.
- ``features``: list of configuration dictionaries used to calculate the features.

See the :ref:`ref-configuration` page for full reference and examples.

A simple way to initialize a MultiAnalyzer object from the configuration file in your code is:

.. code-block:: python

    from blueetl.analysis import run_from_file

    ma = run_from_file("analysis_config.yaml", loglevel="INFO")

The code above will automatically execute the extraction of the report data, and the calculation of the features.

If you prefer to execute the `extraction` and `calculation` steps manually, you could use instead:

.. code-block:: python

    from blueetl.analysis import run_from_file

    ma = run_from_file("analysis_config.yaml", loglevel="INFO", extract=False, calculate=False)

You can also specify other parameters:

- ``seed`` (int): to set a specific seed, or ``None`` if you don't want to initialize the random number generator used to select random neurons.
- ``clear_cache`` (bool): ``True`` or ``False`` to force clearing or keeping any existing cache, regardless of the value in the configuration file.
- ``show`` (bool): ``True`` to print a short representation of all the DataFrames, sometimes useful for a quick inspection.

If not already done automatically with the initialization code above, you can execute the `extraction` of the data from the report and the `calculation` of the features with:

.. code-block:: python

    ma.extract_repo()
    ma.calculate_features()


In case of spikes report, the resulting dataframes will be accessible as:

.. code-block:: python

    ma.spikes.repo.simulations.df
    ma.spikes.repo.neurons.df
    ma.spikes.repo.neuron_classes.df
    ma.spikes.repo.trial_steps.df
    ma.spikes.repo.windows.df
    ma.spikes.repo.report.df

    ma.spikes.features.<custom_name_1>.df
    ma.spikes.features.<custom_name_2>.df
    ...


The list of the available names of the reports can be obtained with:

.. code-block:: python

    ma.names

The list of the available names of the dataframes can be obtained with:

.. code-block:: python

    ma.spikes.repo.names
    ma.spikes.features.names


Command Line Interface
++++++++++++++++++++++

BlueETL includes a simple CLI providing a few subcommands:

.. command-output:: blueetl --help

To extract and calculate features without writing additional code, you can use the ``run`` subcommand:

.. command-output:: blueetl run --help

To validate the configuration file without running the analysis, you can use the ``validate-config`` subcommand:

.. command-output:: blueetl validate-config --help

To migrate an old configuration, you can use the ``migrate-config`` subcommand:

.. command-output:: blueetl migrate-config --help


Output and caching
++++++++++++++++++

The extracted dataframes are saved into the configured output directory.

.. warning:: It is important to understand the caching strategy. The cache can be manually deleted to ensure that everything is recalculated from scratch.

The dataframes are automatically loaded and used as cache if the MultiAnalyzer object is recreated using the same configuration,
or they may be automatically deleted and rebuilt if the configuration has changed.

If only some parts of the configuration have changed, only the invalid dataframes are deleted and rebuilt.

In particular, given this ordered list of extracted dataframes:

#. ``simulations``
#. ``neurons``
#. ``neuron_classes``
#. ``trial_steps``
#. ``windows``
#. ``report``
#. all the features dataframes

these rules apply:

* If the Simulation Campaign configuration specified by ``simulation_campaign`` changed, all the dataframes are rebuilt.
* If any of ``neuron_classes``, ``limit``, ``target`` changed in the ``extraction`` section of the configuration, then the ``neurons`` dataframe and all the following are rebuilt.
* If any of ``windows`` and ``trial_steps`` changed in the ``extraction`` section of the configuration, then the ``trial_steps`` dataframe and all the following are rebuilt.
* If a feature configuration changed in the ``features`` section of the configuration, then the corresponding dataframes are rebuilt.
* If a feature configuration has been removed from the ``features`` section of the configuration, then the corresponding dataframes are deleted.
* If a feature configuration is unchanged, then the corresponding dataframes are loaded from the cache, regardless of any change in the python function.

  Because of this, **if you changed the logic of the function, you may need to manually delete the cached dataframes**.

When ``simulations_filter`` is specified in the configuration:

* If the new filter is narrower or equal to the filter used to generate the old cache, then the old cache is used to produce the new filtered dataframes, and the cache is replaced if different.
* If the new filter is broader than the filter used to generate the old cache, then the old cache is deleted and rebuilt.

Examples of narrower and broader filters:

* the filter ``{"key": 1}`` is narrower than ``{"key": [1, 2]}``
* the filter ``{"key": {"lt": 3}}`` is narrower than ``{"key": {"lt": 4}}``
* the filter ``{"key": {"le": 3, "ge": 1}}`` is narrower than ``{"key": {"le": 4}}``
