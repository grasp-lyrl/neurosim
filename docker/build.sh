#!/bin/bash
set -euo pipefail

variant="${1:-ros}"

case "${variant}" in
  ros)
    docker build \
      -t neurosim:ros \
      -t neurosim:latest \
      -f docker/Dockerfile \
      .
    ;;
  noros|no-ros)
    docker build \
      -t neurosim:noros \
      -f docker/Dockerfile.noros \
      .
    ;;
  *)
    echo "Usage: bash docker/build.sh [ros|noros]"
    exit 1
    ;;
esac
