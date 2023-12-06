#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import getpass
import os
import re
import subprocess
import sys
from datetime import datetime
from string import Template

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", targetdir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.targetdir = targetdir


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary 'native' libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = (
            int(os.environ.get("DEBUG", False)) if self.debug is None else self.debug
        )
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:
            # Single config generators are handled 'normally'
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)
        # copy *.so to target dir
        so_path = os.path.abspath(self.get_ext_fullpath(ext.name))
        self.copy_file(so_path, os.path.join(extdir, ext.targetdir))


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# Read Version
with open(os.path.join(ROOT_PATH, "version.txt"), "r") as rf:
    version = rf.readline().strip("\n").strip()

with open(os.path.join(ROOT_PATH, "openasce/version.py"), "w") as wf:
    content = Template(
        """
# The file is generated by setup and don't modify manually

__version__ = "${VERSION}"

class VersionInfo:
    BUILD_DATE = "${BUILD_DATE}"
    BUILD_VERSION = "${VERSION}"
    BUILD_USER = "${BUILD_USER}"
    log_once = False

    def info(self):
        if self.log_once:
            return
        self.log_once = True
        import sys
        print(f\"\"\"
============ OpenASCE ReleaseInfo ==========
BUILD_VERSION:{VersionInfo.BUILD_VERSION}
BUILD_DATE:{VersionInfo.BUILD_DATE}
BUILD_USER:{VersionInfo.BUILD_USER}
===========================================================\"\"\",
            file=sys.stderr, flush=True)
    """
    ).substitute(
        VERSION=version,
        BUILD_USER=getpass.getuser(),
        BUILD_DATE=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    wf.write(content)

options = {}
if list(filter(lambda x: "--with-cpp" == x.strip().lower(), sys.argv)):
    sys.argv = list(filter(lambda x: "--with-cpp" != x.strip().lower(), sys.argv))
    options.update(
        {
            "cmdclass": {"build_ext": CMakeBuild},
            "ext_modules": [
                CMakeExtension(
                    "gbct_utils",
                    sourcedir=os.path.join(ROOT_PATH, "cpp", "gbct_utils"),
                    targetdir=os.path.join(ROOT_PATH, "openasce", "inference", "tree"),
                )
            ],
        }
    )

current_packages = find_packages(
    where=".",
    exclude=[
        "configs",
        "configs.*",
        "*_test.py",
        "tests",
        "tests.*",
        "test",
        "test.*",
        "*.tests",
        "*.tests.*",
        "*.pyc",
    ],
)
current_package_data = {"openasce": ["inference/tree/*.so"]}

if list(filter(lambda x: "macosx" in x.strip().lower(), sys.argv)):
    # Don't support the causal tree in Macos for now
    current_packages = list(
        filter(lambda x: "tree" not in x.strip().lower(), current_packages)
    )
    current_package_data = {}

setup(
    name="openasce",
    version=version,
    description="Open All-Scale Causal Engine",
    long_description="OpenASCE (Open All-Scale Casual Engine) is a Python package for end-to-end large-scale causal learning. It provides causal discovery, causal effect estimation and attribution algorthms all in one package.",
    author="Ant Group",
    license="Apache",
    url="https://github.com/Open-All-Scale-Causal-Engine/OpenASCE",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
    ],
    packages=current_packages,
    package_data=current_package_data,
    include_package_data=True,
    install_requires=[
        r.strip()
        for r in open(ROOT_PATH + "/requirements.txt", "r")
        if not r.strip().startswith("#")
    ],
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.11",
    **options,
)
