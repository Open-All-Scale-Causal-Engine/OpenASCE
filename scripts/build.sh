#!/usr/bin/env bash

set -e

function gbct_utils_build() {
    pushd "${PROJECT_DIR}"/cpp/gbct_utils

    \rm -rf CMakeCache.txt
    \rm -fr CMakeFiles
    \rm -fr Makefile
    \rm -fr cmake_install.cmake
    LOG_FILE=`pwd`/build.log

    cmake . 2>&1 | tee -a ${LOG_FILE}
    make clean 2>&1 | tee -a ${LOG_FILE}
    make -j8 2>&1 | tee -a ${LOG_FILE}

    \rm -rf CMakeCache.txt
    \rm -fr CMakeFiles
    \rm -fr Makefile
    \rm -fr cmake_install.cmake

    popd
}

function pkg_build() {
    pushd "${PROJECT_DIR}"

    mv -f "${PROJECT_DIR}"/cpp/gbct_utils/gbct_utils.*.so "${PROJECT_DIR}"/openasce/inference/tree
    \rm -rf dist
    \rm -rf build
    \rm -rf openasce.egg-info
    LOG_FILE=`pwd`/package.log
    echo > ${LOG_FILE}

    python setup.py bdist_wheel --python-tag=cp311 --plat-name macosx_10_9_x86_64 2>&1 | tee -a ${LOG_FILE}
    python setup.py bdist_wheel --python-tag=cp311 --plat-name manylinux1_x86_64 2>&1 | tee -a ${LOG_FILE}

    \rm -rf build
    \rm -rf openasce.egg-info
    \rm -fr "${PROJECT_DIR}"/openasce/inference/tree/gbct_utils.*.so

    popd
}

function pkg_install() {
    pushd "${PROJECT_DIR}"

    whl_src=`find dist -type f -name "*manylinux1_x86_64.whl"`
    pip install --upgrade --no-deps --force-reinstall ${whl_src}

    popd
}

SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
PROJECT_DIR="$SCRIPT_DIR"/..

gbct_utils_build

pkg_build

if [[ $1 == "dev" ]]; then
    pkg_install
fi
