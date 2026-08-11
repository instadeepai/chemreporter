import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath(".."))

from database_schema_doc import render_database_schema_markdown

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ChemReporter'
copyright = '2026, InstaDeep Ltd'
author = 'InstaDeep Ltd'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []
default_role = "code"

# Render Google-style "Attributes:" docstring sections as inline :ivar: fields
# instead of separate `.. attribute::` directives — the latter collide with
# the `py:attribute` objects autodoc already generates for the same fields
# via `:members: :undoc-members:` on the config_schemas automodule.
napoleon_use_ivar = True

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = f"{project} docs"

_SCHEMA_MARKER = "<!-- schema-fields -->"


def _inject_database_schema(_app, docname: str, source: list[str]) -> None:
    """Replace the schema marker in database_schema.md with generated field docs."""
    if docname != "reference/database_schema":
        return

    rendered = render_database_schema_markdown()
    text = source[0]
    if _SCHEMA_MARKER in text:
        source[0] = text.replace(_SCHEMA_MARKER, rendered)
    else:
        source[0] = f"{text}\n\n{rendered}"


def setup(app):
    """Register Sphinx event hooks.

    Returns:
        Extension metadata for Sphinx.
    """
    app.connect("source-read", _inject_database_schema)
    return {"version": "0.1", "parallel_read_safe": True}
