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


Spikes Analysis
---------------

To perform the analysis of the spikes across multiple simulations in a given simulation campaign, a configuration file needs to be provided.

It should specify in particular:

- ``simulation_campaign``: path to the json configuration file of the Simulation Campaign produced by bbp-workflow.
- ``output``: path to the output directory where the results are stored. It's also used as cache, and existing stale files may be deleted.
- ``extraction``: configuration dictionary needed to extract spikes by window and neuron class.
- ``analysis.features``: list of configuration dictionaries needed to calculate the features.

Example:

.. code-block:: yaml

    ---
    simulation_campaign: /path/to/simulation/campaign/config.json
    output: /path/to/output/directory
    extraction:
      neuron_classes:
        L1_EXC: { layer: [ 1 ], synapse_class: [ EXC ] }
        L23_EXC: { layer: [ 2, 3 ], synapse_class: [ EXC ] }
        L4_EXC: { layer: [ 4 ], synapse_class: [ EXC ] }
        L5_EXC: { layer: [ 5 ], synapse_class: [ EXC ] }
        L6_EXC: { layer: [ 6 ], synapse_class: [ EXC ] }
        L1_INH: { layer: [ 1 ], synapse_class: [ INH ] }
        L23_INH: { layer: [ 2, 3 ], synapse_class: [ INH ] }
        L4_INH: { layer: [ 4 ], synapse_class: [ INH ] }
        L5_INH: { layer: [ 5 ], synapse_class: [ INH ] }
        L6_INH: { layer: [ 6 ], synapse_class: [ INH ] }
      limit: null
      target: hex0
      windows:
        w1: { bounds: [ 2000, 7000 ], window_type: spontaneous }
        w2: { bounds: [ 0, 100 ], initial_offset: 7000, n_trials: 3, trial_steps_value: 1000 }
        w3: { bounds: [ 0, 25 ], initial_offset: 7000, n_trials: 5, trial_steps_label: ts1 }
      trial_steps:
        ts1:
          function: blueetl.external.bnac.calculate_trial_step.onset_from_spikes
          initial_offset: 7000
          bounds: [ -50, 25 ]
          pre_window: [ -50, 0 ]
          post_window: [ 0, 25 ]
          smoothing_width: 1.5
          histo_bins_per_ms: 5
          threshold_std_multiple: 4
          ms_post_offset: 1
          fig_paths: [ ]
    analysis:
      features:
        - type: multi
          groupby: [ simulation_id, circuit_id, neuron_class, window ]
          function: blueetl.external.bnac.calculate_features.calculate_features_multi
          params: { export_all_neurons: true }


If needed, a ``simulations_filter`` can be specified at the top level of the configuration file.
In this way, the simulations loaded from the campaign can be filtered by any attribute used in the campaign, or by ``simulation_id``.
The syntax of the filter is the same supported by the ``etl.q()`` method, and the simulations are filtered as a Pandas dataframe.

Example:

.. code-block:: yaml

    simulations_filter:
      ca: 1.0
      depol_stdev_mean_ratio: 0.45
      fr_scale: 0.4
      vpm_pct: 2.0

The Analyzer can be initialized with:

.. code-block:: python

    import logging
    import numpy as np
    from blueetl.analysis import Analyzer
    from blueetl.utils import load_yaml

    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    analysis_config_file = "/path/to/analysis-config.yaml"
    analysis_config = load_yaml(analysis_config_file)
    a = Analyzer(analysis_config)


To run the extraction of the spikes and the calculation of the features:

.. code-block:: python

    a.extract_repo()
    a.calculate_features()


The resulting dataframes will be accessible as:

.. code-block:: python

    a.repo.simulations.df
    a.repo.neurons.df
    a.repo.neuron_classes.df
    a.repo.trial_steps.df
    a.repo.windows.df
    a.repo.spikes.df

    a.features.<custom_name_1>.df
    a.features.<custom_name_2>.df
    ...


The list of the available dataframes names can be obtained with:

.. code-block:: python

    a.repo.names
    a.features.names


Extraction configuration
++++++++++++++++++++++++

The ``extraction`` configuration should specify:

* ``neuron_classes`` (dict): dictionary ``neuron_class_label->dict_of_properties``, used to filter the neurons.

* ``limit`` (int): optional limit to the number of extracted neurons for each neuron class. If specified and not `null`, the neuron are chosen randomly.

* ``target`` (str): optional target used to filter the neurons.

* ``windows`` (dict): dictionary of windows, used to decide which spikes to consider.

* ``trial_steps`` (dict): dictionary of trial steps referenced by the windows.

..
    TODO: add more details about the target and windows configurations.


Features configuration
++++++++++++++++++++++

The ``features`` key in the ``analysis`` section of the configuration contains a list of features dictionaries.

Each dictionary should contain:

* ``type`` (str): type of computation. Valid values are:

  * ``multi``: if the configured function produces multiple dataframes of features; features are calculated in parallel subprocesses.
  * ``single``: if the configured function produces a single dataframe of features; features are calculated in a single process.

  Using ``type=multi`` may speed up the performance of the calculation.

* ``groupby`` (list of str): list of columns of the ``spikes`` dataframe to group by.
  Valid item values are: ``simulation_id``, ``circuit_id``, ``window``, ``trial``, ``neuron_class``, ``gid``.

* ``function`` (str): name of the function that should be called for each group of spikes.

  The function should accept the parameters ``repo, key, df, params``, and it should return:

  * if ``type=multi``, a dictionary ``dataframe_name->dataframe``, that will be used to produce multiple final DataFrames.
  * if ``type=single``, a dictionary ``feature_name->number``, where each key will be a column in the final features DataFrame.

* ``params`` (dict): arbitrary configuration parameters that will be passed to the specified function.

* ``name`` (str): only in case of ``type=single``, the name of the features DataFrame to be created.


Output and caching
++++++++++++++++++

The extracted dataframes are saved into the configured output directory.

.. warning:: It is important to understand the caching strategy.

The dataframes are automatically loaded and used as cache if the Analyzer object is recreated using the same configuration,
or they may be automatically deleted and rebuilt if the configuration has changed.

If only some parts of the configuration have changed, only the invalid dataframes are deleted and rebuilt.

In particular, given this ordered list of extracted dataframes:

#. ``simulations``
#. ``neurons``
#. ``neuron_classes``
#. ``trial_steps``
#. ``windows``
#. ``spikes``
#. all the features dataframes

these rules apply:

* If the Simulation Campaign configuration specified by ``simulation_campaign`` changed, all the dataframes are rebuilt.
* If any of ``neuron_classes``, ``limit``, ``target`` changed in the ``extraction`` section of the configuration, then the ``neurons`` dataframe and all the following are rebuilt.
* If any of ``windows`` and ``trial_steps`` changed in the ``extraction`` section of the configuration, then the ``trial_steps`` dataframe and all the following are rebuilt.
* If a feature configuration changed in the ``analysis`` section of the configuration, then the corresponding dataframes are rebuilt.
* If a feature configuration has been removed from the ``analysis`` section of the configuration, then the corresponding dataframes are deleted.
* If a feature configuration is unchanged, then the corresponding dataframes are loaded from the cache, regardless of any change in the python function.
  Because of this, you may need to manually delete the dataframes, or change any function parameter in the configuration to invalidate the cache.

When ``simulations_filter`` is specified in the configuration:

* If the new filter is narrower or equal to the filter used to generate the old cache, then the old cache is used to produce the new filtered dataframes, and the cache is replaced if different.
* If the new filter is broader than the filter used to generate the old cache, then the old cache is deleted and rebuilt.

Examples:

* the filter ``{"key": 1}`` is narrower than ``{"key": [1, 2]}``
* the filter ``{"key": {"lt": 3}}`` is narrower than ``{"key": {"lt": 4}}``
* the filter ``{"key": {"le": 3, "ge": 1}}`` is narrower than ``{"key": {"le": 4}}``
