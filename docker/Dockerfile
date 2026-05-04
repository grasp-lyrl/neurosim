ARG BASE_IMAGE=neurosim:noros
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROS_APT_SOURCE_VERSION=1.1.0
ARG UBUNTU_CODENAME=jammy

# Install ROS2 Humble and tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository universe \
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

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "export ROS_LOCALHOST_ONLY=1" >> /root/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /root/.bashrc

WORKDIR /opt/neurosim
