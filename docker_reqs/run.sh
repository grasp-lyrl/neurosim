#!/bin/bash
xhost +
docker run --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY \
    -v /dev:/dev \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -v /home/${USER}/.Xauthority:/home/${USER}/.Xauthority \
    -v ~/.ssh:/root/.ssh \
    -v $(pwd):/home/${USER}/neurosim \
    -it --rm --ipc=host --net=host --gpus all \
    --name neurosim \
    --workdir /home/${USER} \
    neurosim:latest \
    /bin/bash
