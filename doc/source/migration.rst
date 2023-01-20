Migration
=========


From 0.1.x to 0.2.x
-------------------

BlueETL 0.2 introduces some breaking changes to support the analysis of multiple reports.

In this document it's described what should be done to adapt the existing configurations and user code.

Note that there aren't breaking changes in the core functionalities.


Configuration
~~~~~~~~~~~~~

1. The specification ``version: 2`` should be added at the top level of the file.
2. The section ``extraction`` should be moved to ``analysis.spikes.extraction``.
3. The section ``analysis.features`` should be moved to ``analysis.spikes.features``.
4. Any custom key should be moved into an optional top level dict named ``custom``.
5. The following sub-section should be added to ``analysis.spikes.extraction``:

.. code-block:: yaml

    report:
      type: spikes

You can see an example of configuration in the old and new format here:

- https://bbpgitlab.epfl.ch/nse/blueetl/-/blob/blueetl-v0.1.2/tests/functional/data/analysis_config_01.yaml
- https://bbpgitlab.epfl.ch/nse/blueetl/-/blob/blueetl-v0.2.0.dev0/tests/functional/data/analysis_config_01.yaml

Analysis
~~~~~~~~

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


you should use something like:

.. code-block:: python

    import logging
    import numpy as np
    from blueetl.analysis import MultiAnalyzer

    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    ma = MultiAnalyzer.from_file("analysis_config.yaml")
    a = ma.spikes

where ``ma`` is an instance of ``MultiAnalyzer`` and ``a`` is an instance of ``SingleAnalyzer``.

If you need to work with multiple analysis, using the instance of ``MultiAnalyzer`` may be more convenient.

When accessing the ``spikes`` DataFrame with:

.. code-block:: python

    a.repo.spikes.df

you should use instead the generic ``report`` attribute, valid for any type of report:

.. code-block:: python

    a.repo.report.df

The old name `spikes` is kept for backward compatibility, but it should be considered deprecated and it will be removed later.

Lastly, if you stored any custom configuration, you can get the values from the dictionary ``ma.global_config.custom``.
