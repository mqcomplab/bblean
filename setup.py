import setuptools
from pathlib import Path
import tomli

import pybind11

root = Path(__file__).parent
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

name = pyproject["project"]["name"]

# setuptools paths must be relative
ext_sources = [str((Path(name) / "csrc" / "similarity.cpp"))]
ext_modules = [
    setuptools.Extension(
        ".".join((name, "_cpp_similarity")),
        ext_sources,
        include_dirs=[pybind11.get_include()],
        language="c++",
        # TODO: Check how to optimize flags
        # NOTE: Compile with -DDEBUG_LOGS=1 to get some debug info from the extensions
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-ftree-vectorize",
            "-march=nocona",
            "-mtune=haswell",
            "-fopenmp",
            "-mpopcnt",
        ],
    ),
]

setuptools.setup(
    name=name,
    use_scm_version=True,
    build_requires=["setuptools_scm"],
    ext_modules=ext_modules,
)
