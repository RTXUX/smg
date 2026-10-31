#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_wheel_image.sh BASE_IMAGE IMAGE_TAG [docker build options...]

Build SMG, gRPC proto, and servicer wheels from this checkout in independent
builder stages, then install them into BASE_IMAGE and tag the result IMAGE_TAG.
The base image must provide Python >=3.10 and pip, and use glibc >=2.28.
Its startup command is preserved. Run this script from any directory.

Examples:
  scripts/build_wheel_image.sh vllm/vllm-openai:your-tag vllm-smg:local
  scripts/build_wheel_image.sh your-engine:base your-engine:smg \
    --build-arg RUNTIME_PYTHON=/opt/venv/bin/python --build-arg CARGO_BUILD_JOBS=32
  scripts/build_wheel_image.sh python:3.12-slim-bookworm smg:local --progress=plain

Additional options are passed directly to docker build, e.g. --platform,
--no-cache, --build-arg BUILD_PROFILE=ci, or --target wheels --output type=local,dest=dist.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 || -z $1 || -z $2 || $1 == -* || $2 == -* ]]; then
    usage >&2
    exit 2
fi

base_image=$1
image_tag=$2
shift 2
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

exec docker build \
    --file "${repo_root}/docker/Dockerfile.wheels" \
    --build-arg "BASE_IMAGE=${base_image}" \
    --tag "${image_tag}" \
    --push \
    "$@" \
    "${repo_root}"
