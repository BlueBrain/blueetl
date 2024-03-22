# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import subprocess

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "blueetl"

# The short X.Y version
version = importlib.metadata.version("blueetl")

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx-jsonschema",
    "sphinxcontrib.programoutput",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx-bluebrain-theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_theme_options = {
    "metadata_distribution": "blueetl",
}

html_title = "blueetl"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

autoclass_content = "both"

autodoc_mock_imports = ["xarray", "bluepy"]

# autosummary settings
autosummary_generate = True

nb_execution_show_tb = True
nb_execution_timeout = 60
nb_execution_excludepatterns = []

# generate the link to the notebooks on GitHub
_base_url = "https://github.com/BlueBrain/blueetl"
_git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
extlinks = {
    "notebooks_source": (
        f"{_base_url}/blob/{_git_commit}/doc/source/notebooks/%s.ipynb",
        "Notebook: %s",
    )
}
