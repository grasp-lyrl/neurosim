FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=America/New_York \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        nano \
        lsb-release \
        locales \
        git \
        git-lfs \
        build-essential \
        pkg-config \
        cmake \
        python3-pip \
        python3-dev \
        python3-setuptools \
        libjpeg-dev \
        libglm-dev \
        libgl1-mesa-glx \
        libegl1-mesa-dev \
        mesa-utils \
        xorg-dev \
        freeglut3-dev \
        libpng-dev \
        libglfw3-dev \
        libx11-dev \
        libxkbcommon-x11-0 \
        libomp-dev \
        libeigen3-dev \
        tmux \
        software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && python3 -m pip install --no-cache-dir pip==25.1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN echo '"\e[A": history-search-backward' > /root/.inputrc \
    && echo '"\e[B": history-search-forward' >> /root/.inputrc \
    && echo 'set completion-ignore-case on' >> /root/.inputrc \
    && echo 'set show-all-if-ambiguous on' >> /root/.inputrc \
    && echo 'set enable-keypad on' >> /root/.inputrc

ARG ROS_APT_SOURCE_VERSION=1.1.0
ARG UBUNTU_CODENAME=jammy

# Install ROS2 Humble and tooling.
RUN add-apt-repository universe \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        python3-rosdep \
        python3-colcon-common-extensions \
        python3-vcstool \
        ros-dev-tools \
        ros-humble-ament-* \
        libgflags-dev \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# Build C++ dependencies required by the ROS2 ZMQ bridge:
# 1) libzmq: ZeroMQ core C/C++ transport library.
# 2) cppzmq: C++ header-only bindings used by bridge code.
# 3) nlohmann/json: JSON serialization used in bridge messages.
RUN set -eux; \
    build_jobs="$(nproc)"; \
    cd /tmp; \
    # 1) libzmq
    wget -q https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz; \
    tar -xzf zeromq-4.3.5.tar.gz; \
    cmake -S zeromq-4.3.5 -B zeromq-4.3.5/build -DCMAKE_BUILD_TYPE=Release; \
    cmake --build zeromq-4.3.5/build --parallel "${build_jobs}"; \
    cmake --install zeromq-4.3.5/build; \
    rm -rf zeromq-4.3.5 zeromq-4.3.5.tar.gz; \
    # 2) cppzmq
    wget -q -O cppzmq-4.11.0.tar.gz https://github.com/zeromq/cppzmq/archive/refs/tags/v4.11.0.tar.gz; \
    tar -xzf cppzmq-4.11.0.tar.gz; \
    cmake -S cppzmq-4.11.0 -B cppzmq-4.11.0/build -DCPPZMQ_BUILD_TESTS=OFF; \
    cmake --install cppzmq-4.11.0/build; \
    rm -rf cppzmq-4.11.0 cppzmq-4.11.0.tar.gz; \
    # 3) nlohmann/json
    wget -q https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz; \
    tar -xzf v3.12.0.tar.gz; \
    cmake -S json-3.12.0 -B json-3.12.0/build -DCMAKE_BUILD_TYPE=Release; \
    cmake --build json-3.12.0/build --parallel "${build_jobs}"; \
    cmake --install json-3.12.0/build; \
    rm -rf json-3.12.0 v3.12.0.tar.gz

# Install Miniforge and create the neurosim environment.
RUN curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh \
    && conda clean -ya

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "export ROS_LOCALHOST_ONLY=1" >> /root/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /root/.bashrc \
    && echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate neurosim" >> /root/.bashrc

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/neurosim
COPY . /opt/neurosim

# Install neurosim in editable mode into the dedicated conda env.
RUN conda create -n neurosim python=3.10 cmake=3.14.0 pip==25.1.1 -y \
    && conda run -n neurosim pip install -e . -v \
    && conda clean -ya
