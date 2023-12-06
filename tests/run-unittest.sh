#!/bin/sh

function info() {
    echo -e "[Info] $1"
}

function error() {
    echo -e "[Error] $1"
}

function gbct_utils_build() {
    pushd "$CPP_DIR/gbct_utils" &> /dev/null

    \rm -rf CMakeCache.txt
    \rm -fr CMakeFiles
    \rm -fr Makefile
    \rm -fr cmake_install.cmake
    LOG_FILE=`pwd`/build.log
    echo > ${LOG_FILE}

    cmake . 2>&1 | tee -a ${LOG_FILE}
    make clean 2>&1 | tee -a ${LOG_FILE}
    make 2>&1 | tee -a ${LOG_FILE}

    \rm -rf CMakeCache.txt
    \rm -fr CMakeFiles
    \rm -fr Makefile
    \rm -fr cmake_install.cmake

    popd &> /dev/null
}

function prepare_gbct_utils() {
    pushd "$PROJECT_DIR" &> /dev/null

    echo ${PROJECT_DIR}
    so_file=`find "${PROJECT_DIR}"/openasce/inference/tree -type f | grep "gbct_utils" | grep ".so"`
    if [ "x${so_file}" == "x" ] || [ ! -f ${so_file} ]; then
        # No gbct_utils so
        info "No gbct_utils lib and begin to build."
        gbct_utils_build
        mv -f "${CPP_DIR}"/gbct_utils/gbct_utils.*.so "${PROJECT_DIR}"/openasce/inference/tree
        so_file=`find "${PROJECT_DIR}"/openasce/inference/tree -type f | grep "gbct_utils" | grep ".so"`
        if [ "x${so_file}" == "x" ] || [ ! -f ${so_file} ]; then
            error "Fail to build gbct_utils lib"
            exitcode=1
        else
            info "Succeed to build gbct_utils lib"
        fi
    else
        info "gbct_utils lib has been available"
    fi

    popd &> /dev/null
}

TEST_DIR=$(cd "$(dirname $0)" && pwd)
PROJECT_DIR="${TEST_DIR}"/..
CPP_DIR="$PROJECT_DIR"/cpp

pushd "$PROJECT_DIR" &> /dev/null
wget http://arks-model.oss-cn-hangzhou-zmf.aliyuncs.com/067303/alps/interp/model/kept_external_libs.tar.gz -O external_libs.tar.gz
tar xvfz external_libs.tar.gz
popd &> /dev/null

exitcode=0
prepare_gbct_utils
if [ ${exitcode} -ne 0 ]; then
    error "gbct_utils lib is not ready"
    exit ${exitcode}
fi

pushd "${PROJECT_DIR}" &> /dev/null

files=`find openasce tests -type f -name "*_test.py"`
TEST_RESULT_DIR="${PROJECT_DIR}"/tests/testresult
test -d ${TEST_RESULT_DIR} && rm -rf ${TEST_RESULT_DIR}
mkdir -p ${TEST_RESULT_DIR}

excluded_files=()
if [[ $# -gt 0 ]] && [[ x$1 == 'xtiny' ]];
then
    excluded_files=(
        # Poor test server makes the case run so slowly, disable parts to speed up.
        openasce/discovery/regression_discovery/regression_discovery_test.py
        openasce/attribution/attribution_test.py
    )
fi

exitcode=0
succ_case_count=0
fail_case_count=0
skip_case_count=0
for file in ${files}
do
    # echo ${file}
    file_path=$(dirname ${file})
    file_name=$(basename ${file})
    # echo ${file_path}
    name=`echo ${file} | tr '/' '.'`
    len=${#name}-3    # The length of suffix '.py' is 3
    name=${name:0:len}
    # echo ${name}
    log_name="${name}.log"
    logfile_name="$TEST_RESULT_DIR"/${log_name}
    # echo ${logfile_name}

    excluded=false
    for excluded_file in "${excluded_files[@]}"; do
        # echo ${excluded_file}
        if [[ $file == *"$excluded_file"* ]]; then
            excluded=true
            break
        fi
    done
    if $excluded; then
        info "Skipping test: [${name}]"
        let skip_case_count+=1
        continue
    fi

    info "Run test: "${name}
    echo ${file}
    # echo ${logfile_name}
    python -m unittest discover -s ${file_path} -p ${file_name} > ${logfile_name} 2>&1
    RET=${PIPESTATUS[0]}
    # echo $RET
    if [[ $RET -ne 0 ]]
    then
        exitcode=1
        let b+=2
        mv -f ${logfile_name} "${logfile_name}.err"
        error "Failed test: [${name}], log file: ${logfile_name}.err.\n python -m unittest discover -s ${file_path} -p ${file_name} > ${logfile_name} 2>&1"
        let fail_case_count+=1
    else
        info "Successful test: [${name}], log file: ${logfile_name}"
        let succ_case_count+=1
    fi
done

info "Success: ${succ_case_count} \nFailure: ${fail_case_count} \nSkip: ${skip_case_count}"
popd &> /dev/null
exit ${exitcode}
