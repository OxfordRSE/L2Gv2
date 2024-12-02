# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "l2gv2"
copyright = "2024"
author = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_markdown_builder", "sphinx.ext.autodoc", "sphinx.ext.autosummary"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "optuna",
    "autograd",
    "pymanopt",
    "tqdm",
    "torch",
    "numba",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "community",
    "torch_geometric",
    "torch_scatter",
    "local2global",
    "raphtory",
    "networkx",
    "matplotlib",
    "nfts",
    "polars",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
