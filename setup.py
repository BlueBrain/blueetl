#!/usr/bin/env python
import importlib.util
from pathlib import Path

from setuptools import find_packages, setup

README = Path("README.rst").read_text(encoding="utf-8")

spec = importlib.util.spec_from_file_location(
    "blueetl.version",
    "blueetl/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

REQUIREMENTS = {
    "core": [
        "numpy>=1.19.4",
        "pandas>=1.2.5,<2",
        "pyyaml>=5.4.1",
        "joblib>=1.1.0",
    ],
    "extra": [
        "tables>=3.6.1",  # needed by pandas to read and write hdf files
        "pyarrow>=7,<9",  # needed by pandas to read and write feather or parquet files
        "xarray>=0.18.0",
        "bluepy>=2.4",
    ],
    "bnac": [
        "seaborn>=0.11.2",
        "scipy>=1.8.0",
        "matplotlib>=3.4.3",
    ],
    "bluecv": [
        "elephant>=0.10.0",
        "quantities>=0.13.0",
    ],
    "docs": [
        "sphinx",
        "sphinx-bluebrain-theme",
        "myst-nb",
    ],
}

setup(
    name="blueetl",
    author="bbp-ou-nse",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Multiple simulations analysis tool",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/blueetl",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "git@bbpgitlab.epfl.ch:nse/blueetl.git",
    },
    license="BBP-internal-confidential",
    install_requires=REQUIREMENTS["core"],
    packages=find_packages(),
    python_requires=">=3.8",
    extras_require={
        "docs": REQUIREMENTS["docs"],
        "bnac": REQUIREMENTS["extra"] + REQUIREMENTS["bnac"],
        "bluecv": REQUIREMENTS["extra"] + REQUIREMENTS["bluecv"],
        "all": REQUIREMENTS["extra"] + REQUIREMENTS["bnac"] + REQUIREMENTS["bluecv"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
