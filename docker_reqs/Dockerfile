FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV TZ=America/New_York
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt update && apt install -y --no-install-recommends apt-transport-https ca-certificates wget curl nano \
    lsb-release locales git git-lfs build-essential pkg-config libjpeg-dev libglm-dev libgl1-mesa-glx \
    libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libpng-dev libglfw3-dev libx11-dev libomp-dev tmux \
    cmake virtualenv python3-pip python3-dev python3-setuptools \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean \    
    # && curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    # && sh ~/miniconda.sh -b -p /opt/conda \
    # && rm ~/miniconda.sh \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV PATH=/opt/conda/bin:$PATH

RUN echo '"\e[A": history-search-backward' > /root/.inputrc \
    && echo '"\e[B": history-search-forward' >> /root/.inputrc \
    && echo 'set completion-ignore-case on' >> /root/.inputrc \
    && echo 'set show-all-if-ambiguous on' >> /root/.inputrc \
    && echo 'set enable-keypad on' >> /root/.inputrc \
    # # Setting up conda environment
    # && conda update -n base -c defaults conda \
    # && conda init bash \
    # && conda create -n habitat python=3.10 cmake=3.22 \
    # && echo "conda activate habitat" >> ~/.bashrc
    # Setting up a virtualenv
    && mkdir -p ~/colcon_venv/src \
    && virtualenv -p python3 ~/colcon_venv/habitat \
    && touch ~/colcon_venv/habitat/COLCON_IGNORE \
    && echo "source ~/colcon_venv/habitat/bin/activate" >> ~/.bashrc \
    && echo "export PYTHONPATH=/root/colcon_venv/habitat/lib/python3.10/site-packages:$PYTHONPATH" >> ~/.bashrc

# SHELL ["conda", "run", "-n", "habitat", "/bin/bash", "-c"]

COPY ./habitat-sim /tmp/habitat-sim
COPY ./rotorpy /tmp/rotorpy
COPY ./docker_reqs/10_nvidia.json /usr/share/glvnd/egl_vendor.d

SHELL ["/bin/bash", "-c"]

RUN source ~/colcon_venv/habitat/bin/activate \
    && cd /tmp/habitat-sim \
    && pip install -r requirements.txt \
    && rm -rf build \
    && python setup.py install --with-cuda --bullet --headless \
    && ln -s ~/colcon_venv/habitat/lib/python3.10/site-packages/habitat_sim-0.3.3-py3.10-linux-x86_64.egg/habitat_sim \
    ~/colcon_venv/habitat/lib/python3.10/site-packages/habitat_sim \
    && cd /tmp/rotorpy \
    && pip install .

# Install basic requirements
RUN source ~/colcon_venv/habitat/bin/activate \
    && pip install torch==2.8.0.dev20250408+cu126 torchvision==0.22.0.dev20250408+cu126 \
    torchaudio==2.6.0.dev20250408+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 \
    && pip install pyyaml==6.0.2 matplotlib numpy==1.26.4 h5py empy==3.3.4 catkin_pkg lark

# Install ROS2 Humble
RUN apt update && apt install -y software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update && apt upgrade -y && apt install -y ros-humble-ros-base ros-dev-tools \
    && echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc \
    && sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update && apt install python3-colcon-common-extensions \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc \
    && mkdir /workspace \
    && rosdep init \
    && rosdep update

WORKDIR /workspace

CMD [ "/bin/bash" ]
