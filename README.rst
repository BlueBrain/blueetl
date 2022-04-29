blueetl
=======

Multiple simulations analysis tool.


Installation (WIP)
------------------

In a virtualenv, to install the full package, execute::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'blueetl[all]'

If instead you need only the core transformations to extend Pandas with `.etl` methods, use::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ 'blueetl'

WIP


Examples (WIP)
--------------

You need to import blueetl to use the Pandas extension ``.etl``:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import blueetl

    np.random.seed(0)
    df = pd.DataFrame(
        {
            "a": np.random.randint(10, size=10),
            "b": np.random.randint(10, size=10),
            "c": np.random.rand(10) * 10,
            "i0": np.arange(10) // 5,
            "i1": np.arange(10) % 5,
        }
    )
    df = df.set_index(['i0', 'i1'])

Result::

           a  b         c
    i0 i1
    0  0   5  7  9.255966
       1   0  6  0.710361
       2   3  8  0.871293
       3   3  8  0.202184
       4   7  1  8.326198
    1  0   9  6  7.781568
       1   3  7  8.700121
       2   5  7  9.786183
       3   2  8  7.991586
       4   4  1  4.614794


Select by index or columns:

.. code-block:: python

    df.etl.q(i0=1, i1=[1, 2, 3], b=7)

Result::

           a  b         c
    i0 i1
    1  1   3  7  8.700121
       2   5  7  9.786183

Select by range:

.. code-block:: python

    df.etl.q(c={"gt": 2, "lt": 9})

Result::

           a  b         c
    i0 i1
    0  4   7  1  8.326198
    1  0   9  6  7.781568
       1   3  7  8.700121
       3   2  8  7.991586
       4   4  1  4.614794


See also the jupyter notebooks in the ``notebooks`` directory.
