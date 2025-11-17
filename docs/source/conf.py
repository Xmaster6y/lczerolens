# Configuration file for the Sphinx documentation builder.

import os

import lczerolens

# Project Information
project = "lczerolens"
copyright = "2024, Yoann Poupart"
author = "Yoann Poupart"


# General Configuration
extensions = [
    # 'sphinx.ext.autosectionlabel',
    "sphinx.ext.autodoc",  # Auto documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # View code in the browser
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Boostrap design components
    "nbsphinx",  # Jupyter notebook support
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore
fixed_sidebar = True


# HTML Output Options

# See https://sphinx-themes.org/ for more
html_theme = "pydata_sphinx_theme"
html_title = "lczerolens"
html_logo = "_static/images/lczerolens-logo.svg"
html_static_path = ["_static"]

html_favicon = "_static/images/favicon.ico"
html_show_sourcelink = False

# Define the json_url for our version switcher.
json_url = "https://lczerolens.readthedocs.io/en/latest/_static/switcher.json"


version_match = os.environ.get("READTHEDOCS_VERSION")
release = lczerolens.__version__
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev"
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 2,
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Xmaster6y/lczerolens",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/e7vhrTsjnt",
            "icon": "fa-brands fa-discord",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pydata-sphinx-theme",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "show_version_warning_banner": True,
    "navbar_center": ["version-switcher", "navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}
html_sidebars = {"about": [], "start": []}

html_context = {"default_mode": "auto"}

html_css_files = [
    "css/custom.css",
    "css/nbsphinx.css",
]

# Nbsphinx
nbsphinx_execute = "auto"
nbsphinx_allow_errors = True

# Autoapi
autoapi_dirs = ["../../src"]
autoapi_root = "api"
autoapi_keep_files = False
autodoc_typehints = "description"
