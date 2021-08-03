#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

# read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()

VERSION = imp.load_source("", "blueetl/version.py").__version__

setup(
    name="blueetl",
    author="bbp-ou-nse",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="TODO: write something meaningful here",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/blueetl",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "git@bbpgitlab.epfl.ch:nse/blueetl.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        "numpy>=1.19.4",
        "pandas>=1.2.5",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "spa": [
            "simProjectAnalysis @ git+ssh://git@bbpgitlab.epfl.ch/conn/personal/reimann/bbp-analysis-framework.git@newbluepy#egg=simProjectAnalysis",
            "progressbar>=2.5",  # needed by simProjectAnalysis
            "interval>=1.0.0",  # to avoid a DependencyWarning in NeuroTools
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
