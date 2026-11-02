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

    if os.getenv("BITBIRCH_DEBUG_EXT"):
        extra_compile_args.append("-fopt-info-vec-all")  # print loop vectorization info
    # The BitBirch tree must give the same results as the python tree, so
    # multiply-adds must not be fused (GCC fuses them by default when the target
    # has FMA instructions, e.g. on aarch64 or with -march=native)
    if not WIN:
        extra_compile_args.append("-ffp-contract=off")
    _setup_kwargs["ext_modules"] = [
        Pybind11Extension(
            ".".join((name, module)),
            # setuptools paths must be relative
            [str((Path(name) / "csrc" / source))],
            include_dirs=[pybind11.get_include()],
            language="c++",
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            depends=[str((Path(name) / "csrc" / "kernels.hpp"))],
        )
        for module, source in [
            ("_cpp_similarity", "similarity.cpp"),
            ("_cpp_bitbirch", "bitbirch.cpp"),
        ]
    ]

setuptools.setup(**_setup_kwargs)
