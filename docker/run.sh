#!/bin/bash
set -euo pipefail

variant="${1:-ros}"

case "${variant}" in
    ros)
        image="neurosim:ros"
        ;;
    noros|no-ros)
        image="neurosim:noros"
        ;;
    *)
        echo "Usage: bash docker/run.sh [ros|noros]"
        exit 1
        ;;
esac

xhost +local:root
trap 'xhost -local:root' EXIT

docker run --privileged \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e DISPLAY \
        -v /dev:/dev \
        -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
        -v /home/${USER}/.Xauthority:/home/${USER}/.Xauthority \
        -v ~/.ssh:/root/.ssh \
        -v "${PWD}":/home/${USER}/neurosim \
        -it --rm --ipc=host --net=host --gpus all \
        --name "neurosim-${variant}" \
        --workdir /home/${USER} \
        "${image}" \
        /bin/bash
