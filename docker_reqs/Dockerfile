FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV LANG=en_US.UTF-8
ENV TZ=America/New_York

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-c"]

RUN apt update \
    && apt install -y --no-install-recommends apt-transport-https ca-certificates wget curl nano \
    lsb-release locales git git-lfs build-essential pkg-config libjpeg-dev libglm-dev libgl1-mesa-glx \
    libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libpng-dev libglfw3-dev libx11-dev libomp-dev tmux \
    cmake python3-pip python3-dev python3-setuptools libeigen3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean \    
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8 \
    && python3 -m pip install pip==25.1.1 

RUN echo '"\e[A": history-search-backward' > /root/.inputrc \
    && echo '"\e[B": history-search-forward' >> /root/.inputrc \
    && echo 'set completion-ignore-case on' >> /root/.inputrc \
    && echo 'set show-all-if-ambiguous on' >> /root/.inputrc \
    && echo 'set enable-keypad on' >> /root/.inputrc

# Install ROS2 Humble
ARG ROS_APT_SOURCE_VERSION=1.1.0
ARG UBUNTU_CODENAME=jammy

RUN apt update \
    && apt install -y software-properties-common \
    && add-apt-repository universe \
    && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && apt update \
    && apt upgrade -y \
    && apt install ros-humble-desktop -y \
    && echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc \
    && apt install -y python3-rosdep python3-colcon-common-extensions libgflags-dev python3-vcstool ros-dev-tools ros-humble-ament-* \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc \
    && rosdep init \
    && rosdep update

# Install Conda
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda \
    && rm Miniforge3-$(uname)-$(uname -m).sh \
    && /opt/conda/bin/conda clean -ya \
    && echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc \
    && /opt/conda/bin/conda init bash

COPY docker_reqs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
