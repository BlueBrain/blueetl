Features
========

Features configuration
----------------------

When defining the configuration of features according to the specifications in the :ref:`ref-configuration` page,
it's possible to specify some parameters to be passed to the user defined function as in this example:

.. code-block:: yaml

    ...
    features:
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123


In some particular cases, it may be desirable to call the same custom function several times, passing different combinations of parameters.

In these cases, it's possible to define the optional configuration keys ``params_product`` and ``params_zip``.

These parameters will be automatically expanded to build the final dict of parameters to be passed to the custom function,
using respectively ``itertools.product`` and ``zip`` from the Python standard library.

For example, using ``params_product``:

.. code-block:: yaml

    ...
    features:
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
        params_product:
          ratio: [ 0.25, 0.50, 0.75 ]
          nested:
            - { bin_size: 1, offset: -6 }
            - { bin_size: 2, offset: -6 }

they are automatically expanded to 3 * 2 = 6 combinations of parameters:

.. code-block:: yaml

    ...
    features:
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.25
          nested: { bin_size: 1, offset: -6 }
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.50
          nested: { bin_size: 2, offset: -6 }
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.75
          nested: { bin_size: 1, offset: -6 }
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.25
          nested: { bin_size: 2, offset: -6 }
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.50
          nested: { bin_size: 1, offset: -6 }
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          ratio: 0.75
          nested: { bin_size: 2, offset: -6 }


Similarly, using ``params_zip``:

.. code-block:: yaml

    ...
    features:
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
        params_zip:
          param1: [ 10, 20 ]
          param2: [ 11, 21 ]

they are automatically expanded to 2 combinations of parameters:

.. code-block:: yaml

    ...
    features:
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          param1: 10
          param2: 11
      - type: multi
        groupby: [ simulation_id, circuit_id, neuron_class, window ]
        function: custom_module.custom_function
        params:
          custom_param: 123
          param1: 20
          param2: 21


In more complex cases, ``params_product`` and ``params_zip`` can be combined together.


Features access
---------------

After the features have been calculated, it's possible to access the underlying Pandas DataFrames as shown in the usage section.

For example:

.. code-block:: python

    ma.spikes.features.<custom_name>.df

However, when ``params_product`` or ``params_zip`` have been defined in the configuration, a suffix is automatically added to the custom name, so they can be accessed as in this example:

.. code-block:: python

    ma.spikes.features.<custom_name>_0.df
    ma.spikes.features.<custom_name>_1.df
    ...

or, depending on the number of variable parameters:

.. code-block:: python

    ma.spikes.features.<custom_name>_0_0.df
    ma.spikes.features.<custom_name>_0_1.df
    ...

You can check the parameters used to build each feature reading the ``config`` key in the ``df.attrs`` dictionary:

.. code-block:: python

    ma.spikes.features.<custom_name>_0_0.df.attrs["config"]

The list of names including the suffixes can be obtained with:

.. code-block:: python

    ma.spikes.features.names

A DataFrame obtained as the result of the concatenation of the partial DataFrames can be accessed using just the custom name, without suffixes:

.. code-block:: python

    ma.spikes.features.<custom_name>.df
    ma.spikes.features.<custom_name>.params
    ma.spikes.features.<custom_name>.aliases

In the example above:

- ``df`` returns a cached DataFrame of the concatenated partial DataFrames, including additional columns for the varying parameters.
- ``params`` returns a cached DataFrame of all the parameters.
- ``aliases`` returns a cached DataFrame of the varying parameters and their aliases (the shortened names, if they are nested in a dict).

To free memory, the cache can be cleared with:

.. code-block:: python

    ma.spikes.features.<custom_name>.clear_cache()


See the Jupyter notebook :doc:`/notebooks/02_features_basics` for an example.
