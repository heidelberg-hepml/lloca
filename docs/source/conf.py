# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../.."))

project = "LLoCa"
copyright = "2026, Jonas Spinner, Luigi Favaro"
author = "Jonas Spinner, Luigi Favaro"

# Version comes from setuptools-scm via the installed package; falls back for uninstalled checkouts.
try:
    release = _pkg_version("lloca")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]
autosummary_generate = True
autoclass_content = "both"
autodoc_class_signature = "mixed"

autodoc_default_options = {
    "members": True,
    "inherited-members": False,
}
templates_path = ["_templates"]
napoleon_custom_sections = [
    ("Parameters auto-set by LLoCa", "params_style"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = ["xformers"]
