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
    "core": [
        "numpy>=1.19.4",
        "pandas>=1.3.0",
        "pyyaml>=5.4.1",
        "joblib>=1.1.0",
        "packaging>=21.3",
    ],
    "extra": [
        "tables>=3.6.1",  # needed by pandas to read and write hdf files
        "pyarrow>=7",  # needed by pandas to read and write feather or parquet files
        "fastparquet>=0.8.3,!=2023.1.0",  # needed by pandas to read and write parquet files
        "orjson",  # faster json decoder used by fastparquet
        "xarray>=0.18.0",
        "bluepysnap>=1.0.7",
        "pydantic>=1.10,<2",
        "jsonschema>=4.0",
        "click>=8",
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
        "Source": "git@bbpgitlab.epfl.ch:nse/blueetl.git",
    },
    license="BBP-internal-confidential",
    install_requires=REQUIREMENTS["core"],
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    extras_require={
        "docs": REQUIREMENTS["docs"],
        "extra": REQUIREMENTS["extra"],
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "blueetl=blueetl.apps.main:cli",
        ],
    },
)
