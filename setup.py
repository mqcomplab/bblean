import os
import platform
from pathlib import Path
import tomllib

import setuptools

root = Path(__file__).parent
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

name = pyproject["project"]["name"]

_setup_kwargs = dict(
    name=name,
    use_scm_version=True,
    build_requires=["setuptools_scm"],
    long_description_content_type="text/markdown",
)

X86 = platform.machine().lower() in ["x86_64", "amd64", "i386", "x86", "i686"]
APPLE_SILICON = platform.machine().lower() in ["arm64", "aarch64"]
ARM = platform.machine().lower().startswith("arm") and not APPLE_SILICON

# Build C++ extensions (recommended)
extra_compile_args = []
build_extensions = os.getenv("BITBIRCH_PY_ONLY") is None
build_portable = os.getenv("BITBIRCH_BUILD_PORTABLE") is not None

if build_extensions:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, WIN

    # setuptools paths must be relative
    ext_sources = [str((Path(name) / "csrc" / "similarity.cpp"))]
    if not WIN:
        extra_compile_args.append("-O3")  # -O3 includes -ftree-vectorize
    if not WIN:
        if X86:
            if build_portable:
                extra_compile_args.extend(
                    ["-march=nocona", "-mtune=haswell", "-mpopcnt"]
                )
            else:
                extra_compile_args.extend(
                    ["-march=native", "-mtune=native", "-mpopcnt"]
                )
        elif APPLE_SILICON:
            if not build_portable:
                extra_compile_args.extend(["-mcpu=native"])
    if os.getenv("BITBIRCH_BUILD_CUSTOM_FLAGS"):
        # Override defaults and allow user to specify all flags, useful for attempting
        # to compile with MSVC
        extra_compile_args = []

    if os.getenv("BITBIRCH_DEBUG_EXT"):
        extra_compile_args.append("-fopt-info-vec-all")  # print loop vectorization info
    _setup_kwargs["ext_modules"] = [
        Pybind11Extension(
            ".".join((name, "_cpp_similarity")),
            ext_sources,
            include_dirs=[pybind11.get_include()],
            language="c++",
            cxx_std=17,
            extra_compile_args=extra_compile_args,
        ),
    ]

setuptools.setup(**_setup_kwargs)
