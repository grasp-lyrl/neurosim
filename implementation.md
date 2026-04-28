# UE5 Visual Backend — Implementation Plan

## Goal & non-goals

Add an Unreal Engine 5 visual backend to Neurosim that conforms to
[`VisualBackendProtocol`](src/neurosim/core/visual_backend/base.py).

**In scope (v1):**
- Color + depth rendering, returned as **torch CUDA tensors** with no GPU→CPU→GPU roundtrip.
- Linux only, Docker-based deployment.
- UE5 (5.3+), Vulkan RHI.
- Floating-camera Pawn (no collisions, no physics).
- One UE process per scene. `reconfigure()` = kill + respawn.

**Explicit non-goals (v1):**
- No event camera, optical flow, corners, edges, grayscale, navmesh. Stubs raise `NotImplementedError`.
- No dynamic obstacles, no physics stepping.
- No hot scene reload.
- No CPU fallback — if GPU interop fails on a host, the backend fails loudly.

## Architecture

Two processes on the same GPU, two channels between them:

```
┌─────────────── Python (Neurosim) ───────────────┐    ┌──────────── UE5 process ────────────┐
│  UnrealWrapper(VisualBackendProtocol)           │    │  Game + Render threads              │
│   ├─ transport.py                               │    │  NeurosimBridge plugin              │
│   │   JSON-over-AF_UNIX control plane ──────────┼───►│   ├─ JSON socket server             │
│   │   (set_pose, render, open_level, shutdown)  │    │   ├─ FloatingCameraPawn             │
│   ├─ cuda_interop.py                            │    │   └─ RenderTargetExporter           │
│   │   dma-buf fd → cudaExternalMemory → torch   │◄───┼──   (Vulkan external-memory fds)    │
│   │   (VK_KHR_external_memory_fd import)        │    │                                     │
│   └─ process_manager.py                         │    │                                     │
└─────────────────────────────────────────────────┘    └─────────────────────────────────────┘
       fds passed once at handshake via AF_UNIX SCM_RIGHTS; RTs are written in place every step
```

### Control plane: length-prefixed JSON over AF_UNIX

One `SOCK_STREAM` socket at `$XDG_RUNTIME_DIR/neurosim-ue-<pid>.sock`. Framing:
`uint32 big-endian length | utf-8 json body`. Request shape:
`{"id": <int>, "method": "<name>", "params": {...}}`. Response mirrors the id.

Chosen over gRPC/protobuf because:
- UE's build system does not play nicely with gRPC's third-party deps.
- Control plane is low-frequency: pose updates + render signals, not pixels.
- JSON keeps the plugin C++ small and debuggable with `socat`.

When pose-update rate crosses ~1 kHz or we need structured types, revisit (msgpack or flatbuffers, not gRPC).

### Data plane: Vulkan external memory → CUDA

The only path that keeps pixels on the GPU on Linux:

1. Plugin creates two `UTextureRenderTarget2D`s — RGBA8 color, R32F linear depth.
2. We reach into the Vulkan RHI and re-allocate their `VkImage` backing memory with
   `VkExternalMemoryImageCreateInfo{ handleType: OPAQUE_FD_BIT }` and
   **`VK_IMAGE_TILING_LINEAR`** (opaque tiling is not readable by CUDA as a plain buffer).
3. `vkGetMemoryFdKHR` once at startup → two file descriptors.
4. Descriptors + metadata (w, h, row pitch, size, format) are passed to Python over the
   control socket via `SCM_RIGHTS`.
5. Python imports via `cudaImportExternalMemory` + `cudaExternalMemoryGetMappedBuffer`,
   wraps the device pointer as a zero-copy torch tensor.
6. Per-frame synchronization via an imported Vulkan timeline semaphore
   (`VK_KHR_external_semaphore_fd` → `cudaImportExternalSemaphore`). v1 can start with a
   simpler "render blocks until complete" protocol (RenderFrame RPC returns only after
   `FRHICommandListImmediate::GetRenderThread()` has flushed) and upgrade to semaphores
   in Phase 3 if profiling shows the command-buffer flush is a bottleneck.

Known risks to verify **early** (Phase 2 spike):
- `VK_KHR_external_memory_fd` + linear-tiled R32F / RGBA8 export on the target NVIDIA
  driver (≥ 535 recommended).
- CUDA and UE must pick the same GPU. Wire `settings["gpu_id"]` → UE
  `-graphicsadapter=N` and torch `cuda:N`.
- If `VK_IMAGE_TILING_LINEAR` is unsupported for a format, fall back to a compute-shader
  blit into a linear staging image before export.

## File layout

```
implementation.md                                        # this file
src/neurosim/core/visual_backend/
    factory.py                                           # + "unreal" branch
    unreal_wrapper.py                                    # UnrealWrapper: VisualBackendProtocol
    unreal/
        __init__.py
        process_manager.py                               # popen/teardown of UE process
        transport.py                                     # JSON-over-AF_UNIX client
        cuda_interop.py                                  # dma-buf fd → torch CUDA tensor (Phase 2)

external/neurosim_ue_plugin/
    NeurosimBridge.uplugin
    README.md
    Source/NeurosimBridge/
        NeurosimBridge.Build.cs
        Public/
            NeurosimBridge.h
            FloatingCameraPawn.h
        Private/
            NeurosimBridgeModule.cpp
            FloatingCameraPawn.cpp
            JsonSocketServer.{h,cpp}                     # Phase 1
            RenderTargetExporter.{h,cpp}                 # Phase 2

docker/
    Dockerfile.unreal                                    # UE5 runtime container

tests/
    test_unreal_transport.py                             # Phase 1: fake-server tests
    test_unreal_wrapper.py                               # Phase 1: protocol stubs
    test_unreal_integration.py                           # Phase 2+: gated on UE exe
```

## Wire protocol (v1 methods)

All requests return `{"id": <int>, "result": {...}}` or `{"id": <int>, "error": "..."}`.

| Method | Params | Result |
|---|---|---|
| `handshake` | `{client_version, gpu_id}` | `{server_version, pid, sensors: {name: {w, h, format, fd_index}}}` — fds sent out-of-band via SCM_RIGHTS |
| `open_level` | `{level_path}` | `{ok: true}` |
| `set_agent_pose` | `{position: [x,y,z], rotation: [w,x,y,z]}` (Habitat conventions: y-up, meters) | `{ok: true}` |
| `get_agent_pose` | `{}` | `{position, rotation}` |
| `render_frame` | `{}` | `{ok: true}` — returns only after GPU work is complete |
| `shutdown` | `{}` | `{ok: true}` then server exits |

UE's native axes (z-up, cm, left-handed) are converted inside the plugin.

## Settings contract

```python
{
  "backend_type": "unreal",
  "scene": "/Game/NeurosimTest/Maps/Minimal",
  "ue_executable": "/opt/ue5/NeurosimSim.sh",
  "gpu_id": 0,
  "agent_height": 1.5,
  "agent_radius": 0.1,
  "default_agent": 0,
  "seed": 324,
  "sensors": {
    "rgb": {"type": "color", "width": 512, "height": 512, "hfov": 90,
            "zfar": 100.0, "position": [0,0,0], "orientation": [0,0,0]},
    "dep": {"type": "depth", "width": 512, "height": 512, "hfov": 90,
            "zfar": 100.0, "position": [0,0,0], "orientation": [0,0,0]},
  },
}
```

Fields present in Habitat settings but ignored by the UE backend (documented as such):
`enable_physics`, `physics_config_file`, `scene_dataset_config_file`, `frustum_culling`,
`enable_hbao`, `dynamic_obstacles`, `agent_max_climb`, `agent_max_slope`.

## Scenes

- **v1 rapid-iteration scene**: empty UE5 level with StarterContent — ground plane, 2-3
  textured cubes, directional light, skylight. Lives in a separate
  `NeurosimUEProject` repo, packaged to `/opt/ue5/NeurosimSim.sh`.
- **v1.25**: UE5 ThirdPerson template map, character actor removed.
- **v2 outdoor**: Epic's free "Electric Dreams Environment" or "City Sample". Swap via
  the `open_level` RPC. No Python change needed.
- **later**: Quixel Megascans forest, Matterport via Datasmith.

## Docker

Base image pins a recent Vulkan + NVIDIA runtime. UE packaged build is baked as a layer
(built once on a dev machine, copied in).

```
nvidia/cuda:12.8.0-runtime-ubuntu22.04
  + mesa-vulkan-drivers, libvulkan1
  + UE5 packaged build at /opt/ue5/
  + python venv + neurosim
  + entrypoint invokes neurosim; UE spawned as child
```

Runtime requires `--gpus all`, `--ipc=host` (for AF_UNIX fd passing), and
`/tmp:/tmp` (socket path).

## Testing strategy

Three tiers, cleanly separated:

**Tier 1 — Pure Python unit tests (CI-safe, no UE):**
- `test_unreal_transport.py`: spin up a fake JSON server in-process, exercise
  `UnrealTransport.call(...)` for every method, verify framing + id correlation +
  error handling.
- `test_unreal_wrapper.py`: `UnrealWrapper` against a fake transport; verify it satisfies
  `VisualBackendProtocol` (duck-typed) and that v2 sensors raise `NotImplementedError`
  with a clear message.
- `test_unreal_cuda_interop.py` (Phase 2): allocate a real Vulkan external-memory image
  via a tiny test binary, pass fd, assert torch sees a known pattern.

**Tier 2 — UE plugin Automation tests (local, not CI):**
- `FloatingCameraPawnTest`: set pose, read back transform, assert axis conversion round-trip.
- `RenderTargetExporterTest`: export fd, reopen in same process, blit, assert pixels.

**Tier 3 — End-to-end integration (gated on `NEUROSIM_UE_EXE` env var):**
- `test_unreal_integration.py::test_color_shape_and_dtype`
- `test_unreal_integration.py::test_depth_metric_correctness` — place a cube at known
  distance, assert depth tensor reads that distance within tolerance.
- `test_unreal_integration.py::test_speed_512` — 1000 frames, assert ≥ 120 FPS, assert
  `torch.cuda.memory_allocated` does not grow after warmup.

Mark Tier 3 with `@pytest.mark.integration` and skip when `NEUROSIM_UE_EXE` is unset.

## Phased delivery

Each phase is a mergeable PR with its own tests.

| Phase | Scope | Ends when |
|---|---|---|
| **0. Scaffolding** | UE5 plugin skeleton (`.uplugin` + empty module), Python package layout, factory wiring, Dockerfile, `implementation.md`. | `python -c "from neurosim.core.visual_backend.factory import create_visual_backend"` imports cleanly; UE plugin metadata parses (validated by UE team's build). |
| **1. Control plane** | `process_manager.py`, `transport.py`, JSON protocol, fake-server tests, `UnrealWrapper.__init__` + `update_agent_state` + `render_color`/`render_depth` as stubs that call the transport (return placeholder tensors on CPU for now). UE-side `JsonSocketServer` + pose plumbing in `FloatingCameraPawn`. | Tier 1 tests green. Tier 3 `test_set_get_pose_roundtrip` passes against real UE. |
| **2. GPU transport — color** | `RenderTargetExporter` for color RT, fd handshake, `cuda_interop.py`, `render_color` returns real CUDA tensor. | Speed test ≥ 120 FPS for 512×512 color. |
| **3. GPU transport — depth** | Depth post-process material, second RT, `render_depth` returns metric float32. | Depth metric correctness test passes. |
| **4. Packaging & polish** | `reconfigure()` = restart, scene swap via `open_level`, second map verified, README in `external/neurosim_ue_plugin/`. | Two different maps render identically through the same Python API. |
| **5. (later)** | Collisions, dynamic obstacles, forest scene, event cam — re-evaluate transport budget. | — |

Time estimate for one engineer with UE basics: Phase 0+1 ≈ 1 week; Phase 2 ≈ 1-2 weeks
(Vulkan interop is the uncertainty); Phase 3 ≈ 2 days; Phase 4 ≈ 3 days.

Strongly recommended: a 2-day spike at the start of Phase 2 that proves
`VK_KHR_external_memory_fd` → CUDA interop on your exact driver + GPU, before
committing further.

## Conventions & deliberate simplifications

- **Coordinate frames.** Python uses Habitat conventions (y-up, meters,
  `[w, x, y, z]` quaternion). The plugin does the conversion to UE's native frame
  (z-up, cm, left-handed, `FQuat(x,y,z,w)`). Upstream trajectory code is untouched.
- **No retries.** If the UE process dies, `UnrealWrapper` raises. The caller decides
  whether to retry.
- **No shared-memory buffer pool.** One RT per sensor, written in place. Simpler; fine
  until someone needs double-buffering.
- **No plugin-side logging of frames.** All observability is via UE's own logs; Python
  side logs only control-plane events.
