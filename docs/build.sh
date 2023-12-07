function build_api_doc()
{
    pushd "${DOCS_DIR}"
    pip install -r ../requirements.txt
    sphinx-apidoc -f --module-first -o source ${PROJECT_DIR}/openasce
    make clean
    make html
    popd
}

DOCS_DIR=$(cd "$(dirname $0)" && pwd)
PROJECT_DIR="$DOCS_DIR"/..

build_api_doc
