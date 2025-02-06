import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scope'
copyright = '2024, University of Warwick'
author = 'Dmitrii Kolotkov, Weijie Gu, Sergey Belov, Valery Nakariakov'
release = '20/01/2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',         # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',        # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',        # Add links to source code
    'sphinx_rtd_theme'            # Use the Read the Docs theme
    ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
