# Configuration file for the Sphinx documentation builder.

# Project Information
project = "lczerolens"
copyright = "2024, Yoann Poupart"
author = "Yoann Poupart"


# General Configuration
extensions = [
    "sphinx.ext.autodoc",  # Auto documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Boostrap design components
    "nbsphinx",  # Jupyter notebook support
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
html_theme_options = {
    "show_nav_level": 2,
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
    ],
}

html_context = {"default_mode": "auto"}

html_css_files = [
    "css/custom.css",
]
