# UE5 + Neurosim: setup & test guide (Ubuntu 22.04)

End-to-end instructions for standing up the Unreal Engine 5 visual backend from
nothing on a fresh Ubuntu 22.04 machine, plus the two test paths (editor
game-mode for iteration; packaged build for Docker / production).

Rough timings on a modern workstation:
- Prebuilt UE5 install: ~20 min (mostly download).
- Source build of UE5: 1-2 hours, ~150 GB disk.
- Plugin compile into a project: 2-10 min.
- Packaging a minimal map: ~5 min.

---

## 0. System prerequisites

```bash
sudo apt update
sudo apt install -y \
    build-essential clang lld cmake git git-lfs curl \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libxkbcommon0 libxcb1 libx11-6 libxcursor1 libxrandr2 libxi6 \
    libsm6 libice6 libglu1-mesa libgl1 \
    libfontconfig1 libfreetype6 \
    python3 python3-pip python3-venv \
    dotnet-sdk-8.0

git lfs install
```

NVIDIA driver ≥ 535 (check `nvidia-smi`; `vulkaninfo --summary` should list your
GPU under "GPU id"). Older drivers miss `VK_KHR_external_memory_fd` features we
need in Phase 2.

Disk: keep ~200 GB free on the drive where you put UE and your project.

---

## 1. Get access to Epic's UE5 repository

Unreal Engine source is hosted on GitHub in a private repo. You need to link
your Epic Games account to GitHub and accept an invite before you can clone
anything or download prebuilt binaries.

1. Make an Epic Games account: https://www.epicgames.com/account.
2. Make or sign in to a GitHub account.
3. Go to https://www.unrealengine.com/en-US/ue-on-github, sign in with Epic,
   paste your GitHub username, accept the terms.
4. GitHub sends you an email invite to the `EpicGames` organisation; accept it.
5. Verify: https://github.com/EpicGames/UnrealEngine should now load.

---

## 2. Install UE5 — fast path (prebuilt Linux binary)

Epic publishes "Installed Build" tarballs for Linux on the releases page. This
is the fastest way; skip to §3 if you do this.

1. https://github.com/EpicGames/UnrealEngine/releases
2. Scroll to a 5.3.x (or 5.4.x) release; download
   `Linux_Unreal_Engine_5.X.Y.zip` (~30 GB).
3. Extract:
   ```bash
   mkdir -p /opt/UnrealEngine
   cd /opt/UnrealEngine
   unzip /path/to/Linux_Unreal_Engine_5.3.2.zip
   # You should now have /opt/UnrealEngine/Linux_Unreal_Engine_5.3.2/Engine/...
   ```
4. Smoke test:
   ```bash
   /opt/UnrealEngine/Linux_Unreal_Engine_5.3.2/Engine/Binaries/Linux/UnrealEditor
   ```
   The editor window should appear. Close it.

Export a convenience var you'll use below:
```bash
export UE_ROOT=/opt/UnrealEngine/Linux_Unreal_Engine_5.3.2
echo "export UE_ROOT=$UE_ROOT" >> ~/.bashrc
```

## 2b. Install UE5 — slow path (source build)

Only if the prebuilt isn't available or you need local engine modifications.

```bash
cd /opt
git clone --depth=1 -b 5.3 https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh                     # downloads binaries, patches headers (~30 min)
./GenerateProjectFiles.sh      # ~2 min
make UnrealEditor -j$(nproc)   # ~60-120 min, uses ~100 GB of build artefacts
export UE_ROOT=/opt/UnrealEngine
echo "export UE_ROOT=$UE_ROOT" >> ~/.bashrc
```

---

## 3. Create a UE project

Open the editor once:
```bash
$UE_ROOT/Engine/Binaries/Linux/UnrealEditor
```

In the Project Browser:
- **New Project** → **Games** → **Blank** (C++ template; Blueprint-only is fine
  too, but C++ makes plugin recompilation cleaner).
- **Target Platform: Desktop**, **Quality: Maximum**, **Starter Content: yes**
  (gives you a floor, walls, simple textured assets to render).
- Project name: `NeurosimSim` (anywhere you like, e.g. `~/UEProjects/NeurosimSim`).
- Click **Create**.

The editor will generate the project and compile it. First compile is a couple
of minutes.

Keep note of the project path; below I'll call it `$NSIM_PROJECT`:
```bash
export NSIM_PROJECT=$HOME/UEProjects/NeurosimSim
echo "export NSIM_PROJECT=$NSIM_PROJECT" >> ~/.bashrc
```

Close the editor.

---

## 4. Install the NeurosimBridge plugin

```bash
mkdir -p $NSIM_PROJECT/Plugins
cp -r /path/to/neurosim/external/neurosim_ue_plugin $NSIM_PROJECT/Plugins/NeurosimBridge
```

Regenerate project files and rebuild:
```bash
cd $NSIM_PROJECT
$UE_ROOT/Engine/Build/BatchFiles/Linux/GenerateProjectFiles.sh -project="$NSIM_PROJECT/NeurosimSim.uproject" -game -engine
$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh NeurosimSimEditor Linux Development -Project="$NSIM_PROJECT/NeurosimSim.uproject"
```

If the plugin compiles, the build output ends with `Build succeeded`. If it
fails, the first errors will be in the plugin's `.cpp` files — most likely API
drift between UE versions. Open an issue on the Neurosim repo with the exact
error.

Open the editor again; **Edit → Plugins → Project → Simulation** should list
**Neurosim Bridge** as enabled. If it isn't enabled, tick the box, restart the
editor.

---

## 5. Build the minimal test map

Inside the editor:

1. **File → New Level → Empty Level**. This is our canonical smoke-test scene.
2. In the **Place Actors** panel:
   - Drop a **Directional Light** (sun).
   - Drop a **Sky Light** (fills in ambient).
   - Drop a **Sky Atmosphere** (gives the black viewport a sky).
   - Drop a **Cube** from Basic; scale it to `(50, 50, 0.1)` — this is the floor.
   - Drop 2-3 more **Cubes** at distinct positions; give each a different
     material from Starter Content (`M_Brick_Clay_New`, `M_Metal_Gold`, ...).
3. **File → Save Current Level As…** →  path: `Content/NeurosimTest/Maps/Minimal`.
   Final package path inside the editor is `/Game/NeurosimTest/Maps/Minimal`.
4. **Edit → Project Settings → Maps & Modes**:
   - Set **Editor Startup Map** and **Game Default Map** to `Minimal`.
5. Save project (Ctrl+S).

---

## 6. Iteration loop — run in editor game-mode (recommended for dev)

This is the fast path while developing the plugin: no packaging needed. The
editor binary can run *as* a game client with the plugin active.

Terminal A (UE):
```bash
$UE_ROOT/Engine/Binaries/Linux/UnrealEditor \
    "$NSIM_PROJECT/NeurosimSim.uproject" \
    -game \
    -NeurosimSocket=/tmp/neurosim-ue.sock \
    -graphicsadapter=0 \
    -log
```
You should see `NeurosimBridge: server listening on /tmp/neurosim-ue.sock` in
the log after PostEngineInit.

Terminal B (Python smoke test):
```bash
cd /path/to/neurosim
source /path/to/neurosim-env/bin/activate       # conda / venv — your choice
python - <<'PY'
from neurosim.core.visual_backend.unreal.transport import UnrealTransport

t = UnrealTransport("/tmp/neurosim-ue.sock", connect_timeout_s=30)
t.connect()
print("handshake:", t.call("handshake", client_version="0.1.0", gpu_id=0))
print("set_pose :", t.call("set_agent_pose",
                           position=[0.0, 1.5, -3.0],      # y-up metres
                           rotation=[1.0, 0.0, 0.0, 0.0])) # [w,x,y,z]
print("get_pose :", t.call("get_agent_pose"))
print("shutdown:", t.call("shutdown"))
PY
```

If you had a viewport open you'd see the camera teleport. Expected output:
handshake returns `{"server_version": "0.1.0", "pid": <n>, "sensors": {}}`,
set/get pose round-trips (give or take floating-point error from the Habitat ⇄
Unreal conversion), shutdown returns ok and the UE process exits.

### Testing the full wrapper

```python
from neurosim.core.visual_backend.factory import create_visual_backend

sim = create_visual_backend({
    "backend_type": "unreal",
    "ue_executable": f"{os.environ['UE_ROOT']}/Engine/Binaries/Linux/UnrealEditor",
    "scene": f"{os.environ['NSIM_PROJECT']}/NeurosimSim.uproject",
    "gpu_id": 0,
    "default_agent": 0,
    "agent_height": 1.5,
    "agent_radius": 0.1,
    "socket_path": "/tmp/neurosim-ue.sock",
    "sensors": {
        "rgb": {"type": "color", "width": 512, "height": 512, "hfov": 90,
                "zfar": 100.0, "position": [0,0,0], "orientation": [0,0,0]},
        "dep": {"type": "depth", "width": 512, "height": 512, "hfov": 90,
                "zfar": 100.0, "position": [0,0,0], "orientation": [0,0,0]},
    },
})

import numpy as np
sim.update_agent_state(np.array([0.0, 1.5, -3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
color = sim.render_color("rgb")   # Phase 1: zeros tensor on CUDA
depth = sim.render_depth("dep")   # Phase 1: zeros tensor on CUDA
print(color.shape, color.dtype, color.device)
print(depth.shape, depth.dtype, depth.device)
sim.close()
```

The color/depth tensors are all zeros in Phase 1 — real pixels come in Phase 2.
What this test proves is: UE launches, plugin starts, socket connects, pose
round-trips, render-frame RPC works end to end, shutdown is clean.

> **Note on `ue_executable`.** When you point it at `UnrealEditor` and pass a
> `.uproject` as the scene, UE does the right thing because we pass `-game`
> via the editor-mode command line. For a packaged build (§7) `ue_executable`
> becomes the `NeurosimSim.sh` launcher and `scene` becomes a `/Game/...`
> package path. The process manager is agnostic about which of these you hand
> it — whatever goes in `ue_executable` is what gets spawned.

---

## 7. Packaging for production / Docker

The packaged build runs without an editor install, doesn't need the source
tree, and is what the Dockerfile expects.

From the command line:
```bash
$UE_ROOT/Engine/Build/BatchFiles/RunUAT.sh BuildCookRun \
    -project="$NSIM_PROJECT/NeurosimSim.uproject" \
    -noP4 -platform=Linux -clientconfig=Development \
    -cook -allmaps -build -stage -pak -archive \
    -archivedirectory="$NSIM_PROJECT/Packaged"
```

On success you get:
```
$NSIM_PROJECT/Packaged/LinuxNoEditor/
├── Engine/
│   └── ... (shipped engine binaries, shader libraries)
├── NeurosimSim/
│   ├── Binaries/Linux/NeurosimSim-Linux-Development
│   ├── Content/Paks/NeurosimSim-LinuxNoEditor.pak
│   └── Plugins/NeurosimBridge/...
└── NeurosimSim.sh          # launcher entry point
```

Total size ~2-4 GB.

Smoke-test the packaged build standalone:
```bash
$NSIM_PROJECT/Packaged/LinuxNoEditor/NeurosimSim.sh \
    -NeurosimSocket=/tmp/neurosim-ue.sock -graphicsadapter=0 -nosound -unattended
```
Then run the Python client from §6 against it.

---

## 8. What to put in `ue_packaged/` for the Docker build

The Dockerfile in [`docker/Dockerfile.unreal`](../docker/Dockerfile.unreal)
does `COPY ue_packaged/ /opt/ue5/`. You stage the packaged output there from
§7:

```bash
cd /path/to/neurosim
cp -r $NSIM_PROJECT/Packaged/LinuxNoEditor ./ue_packaged
# Now ./ue_packaged/NeurosimSim.sh exists
```

Then edit the Dockerfile's `ENV NEUROSIM_UE_EXE` to match your launcher name
(default was `/opt/ue5/NeurosimSim.sh`; change `NeurosimSim` if you named the
project differently).

Build:
```bash
docker build -f docker/Dockerfile.unreal -t neurosim-ue:dev .
docker run --rm -it --gpus all --ipc=host -v /tmp:/tmp neurosim-ue:dev
```

For headless hosts (no X server): add `-RenderOffScreen` to the UE command
line. The simplest place is to wire it in via `settings["extra_args"]` through
`UnrealProcess(extra_args=...)`, or hardcode it inside the container's
entrypoint script if you always run headless.

---

## 9. Running the tests

Pure-Python unit tests (fast, no UE, no GPU):
```bash
cd /path/to/neurosim
python -m pytest tests/test_unreal_transport.py tests/test_unreal_wrapper.py -v
```

Integration tests against a real UE (Tier 3, not yet written in-repo; follow
the §6 smoke-test pattern for now). These will land alongside Phase 2 once we
have real render-target fds to assert on.

---

## Troubleshooting

**`NeurosimBridge: -NeurosimSocket=<path> not provided; server disabled.`**
You didn't pass the flag, or the UE launcher isn't forwarding extra args.
Packaged builds forward args correctly; make sure you're actually running
`NeurosimSim.sh` and not bypassing it.

**`bind(...) failed: Address already in use`**
A previous UE run left the socket file. `rm /tmp/neurosim-ue.sock` and retry;
the plugin also tries to `unlink()` stale sockets at startup but can fail if
permissions differ between runs.

**Plugin compile errors about `VulkanRHI` or `IVulkanDynamicRHI`**
Ignore until Phase 2 — the relevant lines are currently commented out in
`NeurosimBridge.Build.cs`. If you uncommented them early, they require
`-IncludeEditorModules=false` in the build args to avoid a circular editor
dependency in some UE versions.

**`vulkaninfo` works on the host but fails inside Docker**
The container is missing the NVIDIA Vulkan ICD loader. Double-check
`docker/10_nvidia.json` is present and that you ran `docker run` with
`--gpus all`.

**`UE exited during startup with code 3`**
Almost always a missing/invalid `-scene` package path, or a plugin that
failed to load. Rerun without the `-unattended` flag and scroll the log.
