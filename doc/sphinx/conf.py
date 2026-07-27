# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys, re

# adding a source path for documentation contained in python script docstrings
sys.path.insert(0, os.path.join('..', '..', 'scripts', 'python', 'stellar'))

# a little something extra to remove default values of args in python method signatures (cleaner visuals)
# this might need to be removed/adjusted later on once we document the main c/c++ codebase.
def remove_default_values(app, what, name, obj, options, signature, return_annotation):
    if signature:
        # strips "=something" up to the next comma or closing paren
        signature = re.sub(r'=\s*[^,)]+', '', signature)
    return signature, return_annotation

def setup(app):
    app.connect('autodoc-process-signature', remove_default_values)

project = "Phoebus"
copyright = "2024, Triad National Security"
author = "The Phoebus Team"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_multiversion",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
]

# ensuring return types are always inline (rather than a separate field) for numpy-style docstrings
napoleon_use_rtype = False

# adding mock imports so autodoc doesn't crash on external dependencies/libraries
autodoc_mock_imports = [
    "numpy",
    "astropy",
    "scipy",
    "pandas",
    "singularity_eos",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# configuration for sphinx_multiversion
smv_remote_whitelist = r"^(origin)$"

# Display todos by setting to True
todo_include_todos = True

# baseurl for sitemap
html_baseurl = "https://lanl.github.io/phoebus"
