Installation
============

The full package can be installed in an existing python virtual environment with::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'blueetl[all]'

If you need instead only the core transformations to extend Pandas with ``.etl`` methods, use::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'blueetl'

If you want to try the latest code not release yet, you can use::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'git+ssh://git@bbpgitlab.epfl.ch/nse/blueetl.git@main#egg=blueetl[all]'

or::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'git+ssh://git@bbpgitlab.epfl.ch/nse/blueetl.git@main#egg=blueetl'

where ``@main`` can be omitted when requesting the default branch, or can be replaced with the desired branch.
