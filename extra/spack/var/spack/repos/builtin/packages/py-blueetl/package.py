# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


# replace all 'x-y' with 'xY' (e.g. 'Py-morph-tool' -> 'PyMorphTool')
class Py-blueetl(PythonPackage):
    """TODO: write something meaningful here"""

    homepage = "https://bbpteam.epfl.ch/documentation/projects/blueetl"
    git      = "git@bbpgitlab.epfl.ch:nse/blueetl.git"

    version('develop', branch='master')
    version('0.1.0.dev0', tag='blueetl-v0.1.0.dev0')

    depends_on('py-setuptools', type='build')  # type=('build', 'run') if specifying entry points in 'setup.py'

    # for all 'foo>=X' in 'install_requires' and 'extra_requires':
    # depends_on('py-foo@<min>:')
