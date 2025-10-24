import bblean

# General variables used in |substitutions|
project = "BitBIRCH-Lean"
copyright = "2025  The Miranda-Quintana Lab and other BitBirch developers"
author = "The Miranda-Quintana Lab and other BitBirch developers"
_version = ".".join(bblean.__version__.split(".")[:2])
version = f"{_version} (dev)" if "dev" in bblean.__version__ else _version
release = bblean.__version__

# Common substitutions used throughout the docs
rst_epilog = """
..  |X| replace:: Fingerprints
"""
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # For autogen python module docs
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For google-style docstr
    "sphinx_design",  # For grid directive
    "nbsphinx",
]
# Autosummary
templates_path = ["_templates"]
autosummary_ignore_module_all = False  # Respect <module>.__all__
# Extensions config
# autodoc
autodoc_typehints_format = "short"  # Avoid qualified names in autodoc types
autodoc_typehints = "description"  # Write types in description, not in signature
autodoc_typehints_description_target = "documented"  # Only write type for docum. params
autodoc_inherit_docstrings = True  # Docstring of supercls is used by dflt
autodoc_default_options = {
    "members": None,  # This means "True"
    "member-order": "bysource",  # Document in the same order as python source code
}
# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
}


# General sphinx config
# nitpicky = True  # Fail if refs can't be resolved TODO re-enable and fix invalid refs
default_role = "any"  # Behavior of `inline-backticks`, try to link to "anything"
pygments_style = "sphinx"  # Code render style
master_doc = "index"  # Default, Main toctree
source_suffix = {".rst": "restructuredtext"}  # Default, Suffix of files

# Python-domain sphinx config
python_use_unqualifierd_type_names = True  # Try to dequa py obj names if resolveable
python_display_short_literal_types = True  # show literals as a | b | ...

# HTML config
html_title = f"{project} v{version} Manual"
html_static_path = ["_static"]  # Static html resources
html_css_files = ["style.css"]  # Overrides for theme style sheet
html_theme = "pydata_sphinx_theme"
html_use_modindex = True
html_domain_indices = False
html_copy_source = False
html_file_suffix = ".html"
htmlhelp_basename = "bblean-docs"

# PyData Theme config
# Primary HTML sidebar (left)
html_sidebars = {
    "index": [],
    "installing": [],
    "user-guide": ["sidebar-nav-bs"],
    "linux_memory_setup": [],
    "api-reference": ["sidebar-nav-bs"],
    "api_autogen/*": ["sidebar-nav-bs"],
    "publications": [],
}
html_theme_options = {
    "show_toc_level": 1,  # default is 2?
    "primary_sidebar_end": [],
    "navbar_center": ["navbar-nav"],
    # Secondary HTML sidebar (right)
    "secondary_sidebar_items": {
        "index": [],
        "installing": [],
        "user-guide": [],
        "linux_memory_setup": [],
        "api-reference": ["page-toc"],
        "_api_autogen/*": ["page-toc"],
        "publications": [],
    },
    # Misc
    "github_url": "https://github.com/mqcomplab/bblean",
    "icon_links": [],
    "logo": {
        "image_light": "_static/logo-light-bw.svg",
        "image_dark": "_static/logo-dark-bw.svg",
    },
    "show_version_warning_banner": True,
}

# Other: info, tex, man
latex_documents = [
    (
        master_doc,
        "BitBIRCH-Lean.tex",
        "BitBIRCH-Lean Documentation",
        "BitBIRCH-Lean developers",
        "manual",
    ),
]
man_pages = [(master_doc, "bblean", "BitBIRCH-Lean Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "BitBIRCH-Lean",
        "BitBIRCH-Lean Documentation",
        author,
        "BitBIRCH-Lean",
        "Clustering of huge molecular libraries",
        "Miscellaneous",
    ),
]
