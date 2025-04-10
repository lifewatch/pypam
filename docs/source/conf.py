# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
import warnings
from importlib.machinery import SourceFileLoader

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "pypam"
copyright = "2023, Clea Parcerisas"
author = "Clea Parcerisas"

# The full version, including alpha/beta/rc tags
# pypam_version = SourceFileLoader('pypam.version',
#                                 '../pypam/version.py').load_module()

# release = pypam_version.__version__
# version = pypam_version.__version__
release = "0.2.0"
version = "0.2.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    # 'sphinx_autodoc_typehints',  # Automatically document param types (less noise in class signature)
    "sphinx.ext.githubpages",
    "numpydoc",  # docstring examples
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True

# -- Example Gallery --
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "_auto_examples",  # path to where to save gallery generated output,
    "capture_repr": (),
    # define which output is captured https://sphinx-gallery.github.io/stable/configuration.html#capture-repr
    "ignore_repr_types": r"matplotlib[text, axes]",
}
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

numpydoc_show_class_members = False

# napoleon_numpy_docstring = True

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
# html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no class summary, inherit base class summary

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
import sphinx_rtd_theme

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# html_logo = './logo/your_logo.png'
# html_theme_options = {'logo_only': True}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True
# If false, no index is generated.
html_use_index = True
html_use_modindex = True
