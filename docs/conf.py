"""Docs config."""

import os
import sys
from importlib import metadata

project = "focalpy"
author = "Martí Bosch"

release = metadata.version("focalpy")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_typehints = "description"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/martibosch/focalpy",
}

# add module to path
sys.path.insert(0, os.path.abspath(".."))

# do NOT execute notebooks
nbsphinx_execute = "never"

# no prompts in rendered notebooks
# https://github.com/microsoft/torchgeo/pull/783
html_static_path = ["_static"]
html_css_files = ["notebooks.css"]
