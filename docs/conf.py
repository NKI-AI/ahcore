#!/usr/bin/env python
# coding=utf-8
#
# Ahcore documentation build configuration file.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import ast

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from typing import Dict, List  # noqa: E402

import ahcore  # noqa: E402

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
curpath = os.path.dirname(__file__)
sys.path.append(os.path.join(curpath, "ext"))

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
    "doi_role",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "myst_parser",
]

# Do not copy prompts in code.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Do not add class members in generated docs
numpydoc_show_class_members = False

# Add class content from main and derived classes
autoclass_content = "both"

# build the templated autosummary files
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "ahcore"
copyright = "2024, ahcore contributors"
author = "AI for Oncology"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.

with open("../ahcore/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees: List[str] = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# ------------------------------------------------------------------------
# Sphinx-gallery configuration
# ------------------------------------------------------------------------

from packaging.version import parse  # noqa: E402

v = parse(release)
if v.release is None:
    raise ValueError("Ill-formed version: {!r}. Version should follow " "PEP440".format(version))

if v.is_devrelease:
    binder_branch = "main"
else:
    major, minor = v.release[:2]
    binder_branch = "v{}.{}.x".format(major, minor)


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "repository_url": "https://github.com/NKI-AI/ahcore",
    "path_to_docs": "/docs",
    "repository_branch": "main",
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "single_page": False,
    "use_fullscreen_button": False,
    "home_page_in_toc": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "ahcoredoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements: Dict[str, str] = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "ahcore.tex", "Ahcore Documentation", "Jonas Teuwen", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "ahcore", "Ahcore Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "ahcore",
        "Ahcore Documentation",
        author,
        "ahcore",
        "Computational pathology models.",
        "Miscellaneous",
    ),
]


# ----------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------

import inspect  # noqa: E402
from os.path import dirname, relpath  # noqa: E402


# Function courtesy of NumPy to return URLs containing line numbers
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except:  # noqa
            return None

    # Strip decorators which would resolve to the source of the decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except:  # noqa
        fn = None
    if not fn:
        return None

    try:
        source, start_line = inspect.getsourcelines(obj)
    except:  # noqa
        linespec = ""
    else:
        stop_line = start_line + len(source) - 1
        linespec = f"#L{start_line}-L{stop_line}"

    fn = relpath(fn, start=dirname(ahcore.__file__))

    if "dev" in ahcore.__version__:
        return "https://github.com/NKI-AI/ahcore/blob/" "main/ahcore/%s%s" % (
            fn,
            linespec,
        )
    return "https://github.com/NKI-AI/ahcore/blob/" "v%s/ahcore/%s%s" % (
        ahcore.__version__,
        fn,
        linespec,
    )
