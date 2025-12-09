[![Neurosim Banner](assets/neurosim_banner.jpg)](assets/neurosim_banner.jpg)

**A Fast Pythonic Simulator for Neuromorphic Perception on Multirotors**

## 📋 Table of Contents

- [Installation](#🚀-installation)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Conda](#option-2-conda)
- [Usage](#🎯-usage)
- [Troubleshooting](#-troubleshooting)

---


## 🚀 Installation

### Option 1: Docker (Recommended)

Docker provides a consistent environment with all dependencies pre-configured, including CUDA, ROS2 Humble, and ZMQ libraries.

#### Prerequisites
- Docker with NVIDIA GPU support ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker))
- NVIDIA drivers installed on host (nvcc 12.9+)

#### Build the Docker Image (Dockerhub coming soon!)

```bash
# Clone the repository
git clone https://github.com/grasp-lyrl/neurosim.git
cd neurosim

# Build the Docker image (takes ~15-20 minutes)
bash docker/build.sh
```

#### Run the Container

```bash
# Launch the container with GPU and display support
bash docker/run.sh
```

This will:
- Mount the current directory to `/home/${USER}/neurosim` inside the container
- Enable GPU access with all CUDA capabilities
- Forward X11 display for GUI applications
- Set up shared networking and IPC

#### Inside the Container

```bash
# Navigate to the workspace
cd neurosim

# Create and activate conda environment
conda create -n neurosim python=3.10 cmake=3.14.0 pip==25.1.1 -y
conda activate neurosim

# Install neurosim
pip install -e . -v

# Download example scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/

# Run the simulator
python test_sim.py --settings configs/skokloster-castle-settings.yaml --display --world_rate 750
```

---

### Option 2: Conda

#### System Requirements

- **OS:** Ubuntu 22.04/24.04 (tested)
- **Compiler:** GCC 11.4.0+
- **CMake:** 3.14.0+
- **CUDA:** 12.2 / 12.4 / 12.6 / 12.9 / 13.0 (tested)
- **Python:** 3.10

#### Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev \
    mesa-utils xorg-dev freeglut3-dev
```

#### Install Neurosim

```bash
# Create conda environment
conda create -n neurosim python=3.10 cmake=3.14.0 pip==25.1.1 -y
conda activate neurosim

# Install neurosim in editable mode
pip install -e . -v
```

---

## 🎯 Usage

### Download Example Data

```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### Run the Simulator

```bash
python test_sim.py --settings configs/skokloster-castle-settings.yaml --display --world_rate 750
```

## Compilation Issues

- If compilation crashes due to high memory usage or CPU load, manually set `self.parallel=4` inside `setup.py` to limit parallel jobs.

- `pip==25.3` breaks installation due to changes in the build isolation process. Use `pip==25.1.1` as specified.


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is enabled by the following amazing open-source projects:
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [ZeroMQ](https://zeromq.org)
- [RotorPy](https://github.com/spencerfolk/rotorpy)
