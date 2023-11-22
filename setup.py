#!/usr/bin/env python
import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages, setup

README = Path("README.rst").read_text(encoding="utf-8")

spec = importlib.util.spec_from_file_location(
    "blueetl.version",
    "src/blueetl/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

REQUIREMENTS = {
    "base": [
        # requirements for base functionalities
        "blueetl-core>=0.1.0",
        "bluepysnap>=1.0.7",
        "click>=8",
        "jsonschema>=4.0",
        "numpy>=1.19.4",
        "pandas>=1.3.0",
        "pyarrow>=7",  # needed by pandas to read and write feather or parquet files
        "pydantic>=2",
        "pyyaml>=5.4.1",
        "xarray>=0.18.0",
    ],
    "extra": [
        # extra requirements that may be dropped at some point
        "bluepy>=2.5.2",
        "fastparquet>=0.8.3,!=2023.1.0",  # needed by pandas to read and write parquet files
        "orjson",  # faster json decoder used by fastparquet
        "tables>=3.6.1",  # needed by pandas to read and write hdf files
    ],
    "external": [
        # external requirements needed to run custom code in blueetl.external submodule
        "elephant>=0.10.0",
        "matplotlib>=3.4.3",
        "quantities>=0.13.0",
        "scipy>=1.8.0",
        "seaborn>=0.11.2",
    ],
    "docs": [
        "sphinx",
        "sphinx-bluebrain-theme",
        "sphinx-jsonschema",
        "sphinxcontrib-programoutput",
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
        "Source": "https://github.com/BlueBrain/blueetl.git",
    },
    license="BBP-internal-confidential",
    install_requires=REQUIREMENTS["base"],
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    extras_require={
        "docs": REQUIREMENTS["docs"],
        "extra": REQUIREMENTS["extra"],
        "external": REQUIREMENTS["external"],
        "all": REQUIREMENTS["extra"] + REQUIREMENTS["external"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "blueetl=blueetl.apps.main:cli",
        ],
    },
)
