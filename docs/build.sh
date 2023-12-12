function build_api_doc()
{
    pushd "${DOCS_DIR}"
    pip install -r ../requirements.txt
    sphinx-apidoc -f --module-first -o source ${PROJECT_DIR}/openasce \
        ${PROJECT_DIR}/openasce/inference/*_test* \
        ${PROJECT_DIR}/openasce/inference/learner/*_test* \
        ${PROJECT_DIR}/openasce/inference/tree/*_test* \
        ${PROJECT_DIR}/openasce/discovery/regression_discovery/*_test* \
        ${PROJECT_DIR}/openasce/discovery/search_discovery/*_test* \
        ${PROJECT_DIR}/openasce/attribution/*_test* \
        ${PROJECT_DIR}/openasce/extension/*_test* \
        ${PROJECT_DIR}/openasce/core/*_test*
    make clean
    make html
    popd
}

DOCS_DIR=$(cd "$(dirname $0)" && pwd)
PROJECT_DIR="$DOCS_DIR"/..

build_api_doc
