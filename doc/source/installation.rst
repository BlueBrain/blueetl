Installation
============

The package including only the core transformations can be installed in an existing python virtual environment with::

    PIP_INDEX_URL=https://bbpteam.epfl.ch/repository/devpi/simple \
    pip install blueetl

Instead, if you need to use the analysis features, you need to install the full package with::

    PIP_INDEX_URL=https://bbpteam.epfl.ch/repository/devpi/simple \
    pip install 'blueetl[all]'

If you want to try the latest code not release yet, you can use respectively::

    PIP_INDEX_URL=https://bbpteam.epfl.ch/repository/devpi/simple \
    pip install 'git+ssh://git@bbpgitlab.epfl.ch/nse/blueetl.git@main#egg=blueetl'

or::

    PIP_INDEX_URL=https://bbpteam.epfl.ch/repository/devpi/simple \
    pip install 'git+ssh://git@bbpgitlab.epfl.ch/nse/blueetl.git@main#egg=blueetl[all]'

where ``@main`` can be omitted when requesting the default branch, or can be replaced with the desired git branch.
