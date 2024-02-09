Installation
============

The package can be installed in an existing python virtual environment with::

    pip install -U blueetl

If you need all the optional dependencies, you can install the full package with::

    pip install -U 'blueetl[all]'

If you want to try the latest code not release yet, you can use respectively::

    pip install 'git+https://github.com/BlueBrain/blueetl.git@main#egg=blueetl'

or::

    pip install 'blueetl[all] @ git+https://github.com/BlueBrain/blueetl.git@main#egg=blueetl'

where ``@main`` can be omitted when requesting the default branch, or can be replaced with the desired git branch.

.. warning:: When installing from a git repository, it's necessary to first uninstall any pre-existing version running: ``pip uninstall blueetl``.
