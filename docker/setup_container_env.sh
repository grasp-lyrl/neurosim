#!/usr/bin/env bash
# Reinstall the container-side dependencies that are NOT baked into the
# neurosim:noros image and are therefore lost every time the container is
# recreated.
#
#   docker exec -w /home/odexter/neurosim <container> bash docker/setup_container_env.sh
#
# Idempotent: safe to re-run. Takes ~10-20 min, almost all of it the two CUDA
# extension builds.
set -euo pipefail

PY=/opt/conda/envs/neurosim/bin/python
PIP="${PY} -m pip install --no-cache-dir"
WORKSPACE=/home/odexter/neurosim

# GPU arch to compile the CUDA extensions for. Detected rather than hardcoded:
# the project has already moved between an RTX A5000 host (sm_86) and a
# Tesla V100 host (sm_70), and an extension built for the wrong arch either
# fails to load or silently falls back.
ARCH=$(${PY} -c 'import torch; m=torch.cuda.get_device_capability(0); print(f"{m[0]}.{m[1]}")')
echo "== building CUDA extensions for sm_${ARCH/./} ($(${PY} -c 'import torch;print(torch.cuda.get_device_name(0))'))"

# torchvision MUST match torch exactly. 0.28.0 against torch 2.9.1 fails at
# import with "operator torchvision::nms does not exist". EfficientNet-B0 --
# the event backbone in every v16+ config -- comes from torchvision, so
# nothing with event_backbone: efficientnet_b0 runs without this.
# Pin torch as well as torchvision. The driver here (535.x) supports CUDA 12.x
# only, so a cu130 torch is unusable; installing them together stops pip from
# resolving one against the other's index.
${PIP} 'torch==2.9.1' 'torchvision==0.24.1' \
    --index-url https://download.pytorch.org/whl/cu128

# tensorboard is not optional: every config sets a tensorboard_log, so SB3
# raises ImportError from _setup_learn and the run dies AFTER building all
# its envs -- several minutes in, with a traceback that names logging rather
# than the missing package.
${PIP} 'stable_baselines3==2.9.0' 'sb3_contrib==2.9.0' wandb tensorboard

# The image ships neurosim_cu_esim 0.1, which predates the API the repo calls:
# no `mode` kwarg ("single"/"multi") and no DVSVoltmeterSimulator. Every config
# with `backend: cuda` dies at env construction with
# "EventSimulator.__init__() got an unexpected keyword argument 'mode'"
# until this is reinstalled from source over the image's copy.
if [ ! -d "${WORKSPACE}/deps/neurosim_cu_esim" ]; then
    git clone https://github.com/grasp-lyrl/neurosim_cu_esim.git \
        "${WORKSPACE}/deps/neurosim_cu_esim"
fi
# --no-deps is NOT optional. neurosim_cu_esim depends on `torch` unpinned, so a
# plain (re)install resolves it to the latest wheel -- observed pulling
# torch 2.11.0+cu130 over the pinned 2.9.1+cu128. cu130 needs driver >= 580;
# this host has 535.309.01, so every torch CUDA call then died with the
# thoroughly misleading "The NVIDIA driver on your system is too old
# (found version 12090)". Keep --no-deps here and in any similar build.
TORCH_CUDA_ARCH_LIST="${ARCH}" ${PIP} --no-build-isolation --no-deps \
    --force-reinstall "${WORKSPACE}/deps/neurosim_cu_esim"

# vid2e ESIM backend (event_sim backend: vid2e). Not used by the velocity_dodge
# configs, which are all backend: cuda, but cheap to keep working.
TORCH_CUDA_ARCH_LIST="${ARCH}" ${PIP} --no-build-isolation --no-deps \
    "${WORKSPACE}/src/neurosim/core/event_sim/cu_rpg_vid2e_esim"

# Habitat scene meshes. data/ is gitignored, so it does not survive a move
# between machines. ~101 MB.
if [ ! -e "${WORKSPACE}/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb" ]; then
    ${PY} -m habitat_sim.utils.datasets_download \
        --uids habitat_test_scenes --data-path "${WORKSPACE}/data"
    # datasets_download writes an ABSOLUTE symlink, which is broken on the host
    # whenever the host path differs from the in-container path.
    ln -sfn ../versioned_data/habitat_test_scenes \
        "${WORKSPACE}/data/scene_datasets/habitat-test-scenes"
fi

# Prefetch ImageNet B0 weights so the first training run does not race on them.
${PY} - <<'PYEOF'
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print("efficientnet_b0 ImageNet weights cached")
PYEOF

# Everything the container writes is root-owned. Hand it back to the invoking
# host user -- NOT to uid 1000, which was odexter on the old host but is a
# different user here.
if [ -n "${HOST_UID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" "${WORKSPACE}/data" "${WORKSPACE}/deps" \
        "${WORKSPACE}/outputs" 2>/dev/null || true
fi

echo "== verifying"
PYTHONPATH="${WORKSPACE}/src" ${PY} - <<'PYEOF'
import torch, torchvision, stable_baselines3, sb3_contrib, esim_cuda
assert torch.cuda.is_available(), (
    "torch cannot see the GPUs. If this says the driver is too old, torch was "
    "resolved to a cu13 wheel -- reinstall the cu128 pin above."
)
torch.cuda.init()
from neurosim_cu_esim import EventSimulator, DVSVoltmeterSimulator
import inspect
assert "mode" in inspect.signature(EventSimulator.__init__).parameters, \
    "neurosim_cu_esim is still the stale image copy (no `mode` kwarg)"
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("sb3", stable_baselines3.__version__, "| contrib", sb3_contrib.__version__)
print("neurosim_cu_esim OK (mode kwarg + voltmeter present)")
print("cuda OK:", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0))
PYEOF
echo "== container env ready"
