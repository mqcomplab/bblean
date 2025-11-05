import os
import platform
from pathlib import Path
import tomli

import setuptools

root = Path(__file__).parent
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

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
if os.getenv("BITBIRCH_BUILD_CPP"):
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, WIN

    # setuptools paths must be relative
    ext_sources = [str((Path(name) / "csrc" / "similarity.cpp"))]
    if not WIN:
        extra_compile_args.append("-O3")  # -O3 includes -ftree-vectorize
    if not WIN:
        if X86:
            if os.getenv("BITBIRCH_BUILD_NATIVE"):
                extra_compile_args.extend(
                    ["-march=native", "-mtune=native", "-mpopcnt"]
                )
            else:
                extra_compile_args.extend(
                    ["-march=nocona", "-mtune=haswell", "-mpopcnt"]
                )
        elif APPLE_SILICON:
            if os.getenv("BITBIRCH_BUILD_NATIVE"):
                extra_compile_args.extend(["-mcpu=native"])
    if os.getenv("BITBIRCH_BUILD_CUSTOM_FLAGS"):
        # Override defaults and allow user to specify all flags, useful for attempting
        # to compile with MSVC
        extra_compile_args = []

    # NOTE: Compile with -DDEBUG_LOGS=1 to get some debug info from the extensions
    if os.getenv("BITBIRCH_DEBUG_EXT"):
        extra_compile_args.append("-fopt-info-vec-all")  # print loop vectorization info
        extra_compile_args.append("-DDEBUG_LOGS=1")
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
