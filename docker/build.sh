#!/bin/bash
set -euo pipefail

variant="${1:-ros}"

build_noros() {
  docker build \
    -t neurosim:noros \
    -f docker/Dockerfile.noros \
    .
}

case "${variant}" in
  ros)
    build_noros
    docker build \
      --build-arg BASE_IMAGE=neurosim:noros \
      -t neurosim:ros \
      -t neurosim:latest \
      -f docker/Dockerfile \
      .
    ;;
  noros|no-ros)
    build_noros
    ;;
  *)
    echo "Usage: bash docker/build.sh [ros|noros]"
    exit 1
    ;;
esac
