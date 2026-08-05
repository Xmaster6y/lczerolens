# Configuration file for the Sphinx documentation builder.

import importlib
import os
from pathlib import Path
import sys

# The maintained notebooks reuse the executable fixture from ``examples``.
# Make that repository-local companion importable in Sphinx and in the kernel
# subprocesses started by nbsphinx.
REPOSITORY_ROOT = str(Path(__file__).parents[2])
sys.path.insert(0, REPOSITORY_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (REPOSITORY_ROOT, os.environ.get("PYTHONPATH"))))

lczerolens = importlib.import_module("lczerolens")

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
    "nbsphinx",  # Executable first-class notebook pages
    "autoapi.extension",
    "sphinx_llm.txt",
]

templates_path = ["_templates"]
fixed_sidebar = True
exclude_patterns = ["**/*.nbconvert.ipynb"]

# These notebooks are deliberately small and hermetic. Executing them in the
# docs build keeps the rendered walkthroughs aligned with the public API; the
# integration tier independently executes the same files.
nbsphinx_execute = "always"
nbsphinx_allow_errors = False
nbsphinx_timeout = 60


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
            "url": "https://pypi.org/project/lczerolens/",
            "icon": "fa-brands fa-python",
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
]

# Autoapi
autoapi_dirs = ["../../src"]
autoapi_root = "api"
# AutoAPI sources are generated for the build and must not become checkout
# artifacts.  Keeping them also made a documentation build leave ``api/``
# untracked.
autoapi_keep_files = False
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autodoc_typehints = "description"
