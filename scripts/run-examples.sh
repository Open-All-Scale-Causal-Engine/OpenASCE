#!/usr/bin/env bash

set +e

function info() {
    echo -e "[Info] $1"
}

function error() {
    echo -e "[Error] $1"
}

SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
PROJECT_DIR="$SCRIPT_DIR"/..
EXAMPLE_DIR="$PROJECT_DIR"/examples

Python_Package="openasce"

pip show ${Python_Package} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    error "${Python_Package} is unavailable."
    exit 1
fi

EXAMPLES_RESULT_DIR="$EXAMPLE_DIR"/example_results
test -d ${EXAMPLES_RESULT_DIR} && rm -rf ${EXAMPLES_RESULT_DIR}
mkdir -p ${EXAMPLES_RESULT_DIR}

pushd "${EXAMPLE_DIR}" &> /dev/null
excluded_files=(
    # Exclude files
    # e.g. openasce/discovery/xxx.py
    # openasce/inference/t_model/t_model_rossi_test.py
)

exitcode=0
succ_example_count=0
fail_example_count=0
example_files=`find . -type f -name *.py`
# info "${example_files}"
for file in ${example_files}
do
    file_path=$(dirname ${file})
    file_name=$(basename ${file})
    name=`echo ${file} | tr '/' '.'`
    len=${#name}-3    # The length of suffix '.py' is 3
    name=${name:0:len}
    log_name="${name}.log"
    logfile_name="${EXAMPLES_RESULT_DIR}"/${log_name}

    excluded=false
    for excluded_file in "${excluded_files[@]}"; do
        if [[ $file == *"$excluded_file"* ]]; then
            excluded=true
            break
        fi
    done
    if $excluded; then
        continue
    fi

    info "Run example: ${name}, file: ${file}"
    pushd "${file_path}" &> /dev/null
    python ${file_name} 2>&1 | tee ${logfile_name}
    RET=${PIPESTATUS[0]}
    echo ${RET}
    if [[ $RET -ne 0 ]]
    then
        exitcode=1
        mv -f ${logfile_name} "${logfile_name}.err"
        error "Failed example: [${name}], log file: [${logfile_name}.err], Executive: [python ${file} > ${logfile_name} 2>&1]"
        let fail_example_count+=1
    else
        info "Successful test: [${name}], log file: ${logfile_name}"
        let succ_example_count+=1
    fi
    popd &> /dev/null
done

info "Success: ${succ_example_count} \nFailure: ${fail_example_count}"
popd &> /dev/null
exit ${exitcode}
