# Neurosim
A habitat sim based full functional quadrotor simulator with event camera support. real. time.

## Build from source

### Install Dependencies

```
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```

### Install Neurosim
```
conda create -n neurosim python=3.10 cmake=3.14.0 pip==25.1.1 -y
conda activate neurosim
pip install -e . -v
```

Tested on `Ubuntu 22.04: gcc 11.4.0, cmake 3.14.0, nvcc 12.9 | 12.6 | 12.4 | 12.2`

### Compilation issues
- If compilation of habitat-sim crashes due to high memory consumption or cpu load, manually set `self.parallel=4` inside `setup.py`.
- `pip==25.3` breaks the installation due to changes in the build isolation process. Use `pip==25.1.1` for now.


## Usage Example

### Download example data
```
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### Run simulator
```
python test_sim.py --settings configs/skokloster-castle-settings.yaml --display --world_rate 750
```