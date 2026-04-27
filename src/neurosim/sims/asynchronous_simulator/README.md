# Asynchronous simulator (Cortex)

Split-process loop: **simulator** (dynamics + Habitat + sensors), **controller** (trajectory + control), optional **visualizer** (Rerun). Transport is the standalone **Cortex** package.

## Cortex in this folder

- **Discovery** (required): run **`cortex-discovery`** in a separate terminal (installed with the Cortex package; do not use `python -m cortex...` for discovery). Nodes fail fast if the daemon is not reachable. Default socket: `ipc:///tmp/cortex/discovery.sock` (same as `cortex.discovery.daemon.DEFAULT_DISCOVERY_ADDRESS`).
- **API**: `cortex.core.node.Node` with `create_publisher` / `create_subscriber` / `create_timer`.
- **Topics** (see `cortex_io.py`):
  - `state` / `control` — `DictMessage`
  - `events/<uuid>`, `imu/<uuid>` — `MultiArrayMessage` / `DictMessage`
  - `color/<uuid>`, `depth/<uuid>` — `ArrayMessage`

## Run (example)

From the **repo root**, with the **neurosim** conda env active and scene data available (same expectations as other Habitat configs).

Settings file (repo root): `configs/apartment_1-settings.yaml` — full `simulator`, `visual_backend`, `dynamics`, `controller`, and `trajectory`.

1. **Discovery** (terminal 1):

   ```bash
   cortex-discovery
   ```

2. **Simulator** (terminal 2):

   ```bash
   python -m neurosim.sims.asynchronous_simulator.simulator_node \
     --settings configs/apartment_1-settings.yaml
   ```

3. **Visualizer** (terminal 3 — opens Rerun):

   ```bash
   python -m neurosim.sims.asynchronous_simulator.visualizer_node \
     --settings configs/apartment_1-settings.yaml
   ```

4. **Controller** (terminal 4):

   ```bash
   python -m neurosim.sims.asynchronous_simulator.controller_node \
     --settings configs/apartment_1-settings.yaml
   ```


Override discovery with `--discovery-address <ipc://...>` on each node if needed.
