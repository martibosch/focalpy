"""Docs config."""

import dataclasses
import os
import sys
from importlib import metadata

from sphinx.builders.latex import transforms
from sphinxcontrib.bibtex import plugin as sphinxcontrib_bibtex_plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

project = "focalpy"
author = "Martí Bosch"

release = metadata.version("focalpy")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

autodoc_typehints = "description"


# citation styles
def bracket_style() -> BracketStyle:
    """Bracket style."""
    return BracketStyle(
        left="(",
        right=")",
    )


@dataclasses.dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    """Custom reference style."""

    bracket_parenthetical: BracketStyle = dataclasses.field(
        default_factory=bracket_style
    )
    bracket_textual: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_author: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_label: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_year: BracketStyle = dataclasses.field(default_factory=bracket_style)


sphinxcontrib_bibtex_plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", MyReferenceStyle
)


# work-around to get LaTeX references at the same place as HTML
# see https://github.com/mcmtroffaes/sphinxcontrib-bibtex/issues/156
class DummyTransform(transforms.BibliographyTransform):
    """Dummy transform."""

    def run(self, **kwargs):
        """Run."""
        pass


transforms.BibliographyTransform = DummyTransform

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year_round"


# add module to path
sys.path.insert(0, os.path.abspath(".."))

# do NOT execute notebooks
nbsphinx_execute = "never"

# theme
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/martibosch/focalpy",
}

# no prompts in rendered notebooks
# https://github.com/microsoft/torchgeo/pull/783
html_static_path = ["_static"]
html_css_files = ["notebooks.css"]
