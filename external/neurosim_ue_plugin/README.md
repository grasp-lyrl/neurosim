# Neurosim UE5 Bridge Plugin

Server-side half of Neurosim's Unreal Engine 5 visual backend. See
[`implementation.md`](../../implementation.md) for the full architecture.

## What's here (Phase 0 + 1)

- `NeurosimBridge.uplugin` — plugin descriptor (Linux-only, runtime, PostEngineInit).
- `Source/NeurosimBridge/` — C++ module:
  - `NeurosimBridgeModule.cpp` — reads `-NeurosimSocket=<path>` from the UE command
    line; if set, starts the JSON socket server.
  - `JsonSocketServer.{h,cpp}` — AF_UNIX + length-prefixed JSON; dispatches to
    handlers for `handshake`, `set_agent_pose`, `get_agent_pose`, `open_level`,
    `render_frame`, `shutdown`. Handlers that touch `UWorld` bounce to the game
    thread and block on an `FEvent`.
  - `FloatingCameraPawn.{h,cpp}` — disembodied camera pawn. Owns the
    Habitat ⇄ Unreal coordinate conversion (y-up m ⇄ z-up cm, quaternion order).

Not yet implemented (later phases):
- Vulkan external-memory render-target export (Phase 2).
- Depth post-process material emitting linear meters (Phase 3).

## Building

You need a source build of UE5 (5.3 or later) on Linux. Binary installs of UE5
on Linux currently don't include `UnrealBuildTool` in a usable state for
third-party plugins.

```bash
# Clone the plugin into your UE project
cp -r neurosim/external/neurosim_ue_plugin /path/to/MyProject/Plugins/NeurosimBridge

# From your project directory
/path/to/UnrealEngine/Engine/Build/BatchFiles/Linux/RunUAT.sh BuildPlugin \
    -Plugin="$(pwd)/Plugins/NeurosimBridge/NeurosimBridge.uplugin" \
    -Package="$(pwd)/Plugins/NeurosimBridge/Built" \
    -TargetPlatforms=Linux \
    -Rocket
```

## Packaging a headless build

```bash
/path/to/UnrealEngine/Engine/Build/BatchFiles/Linux/RunUAT.sh BuildCookRun \
    -project="$(pwd)/MyProject.uproject" \
    -noP4 -platform=Linux -clientconfig=Development \
    -cook -allmaps -build -stage -pak -archive \
    -archivedirectory="$(pwd)/Packaged" \
    -server=false -nullrhi=false
```

Then `Packaged/LinuxNoEditor/MyProject.sh` is the executable Neurosim spawns.

## Launching manually (smoke test)

```bash
./MyProject.sh \
    -NeurosimSocket=/tmp/neurosim-ue.sock \
    -graphicsadapter=0 \
    -nosound -unattended
```

Then from another shell:

```bash
python -c "
from neurosim.core.visual_backend.unreal.transport import UnrealTransport
t = UnrealTransport('/tmp/neurosim-ue.sock')
t.connect()
print(t.call('handshake', client_version='0.1.0', gpu_id=0))
print(t.call('set_agent_pose', position=[1,0,0], rotation=[1,0,0,0]))
print(t.call('get_agent_pose'))
t.call('shutdown')
"
```
