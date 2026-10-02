"""Bake a SceneSmith scene's lighting into its textures and export a Habitat GLB.

    blender -b <scene>/combined_house/house.blend -P scripts/scenesmith_bake_glb.py \
        -- --out data/scenesmith/Room_scene_042.glb --samples 192 --max-res 2048
"""

import argparse
import math
import os
import sys
import tempfile
import time

import bpy
from mathutils import Vector

FLOOR_MAX_THICKNESS = 0.25
FLOOR_MIN_AREA = 1.0
PLANAR_MAX_POLYS = 64
AREA_PER_LIGHT = 4.5
PROBE_MIN_LUMA = 0.05
MAX_LIGHT_GRID = 5
CEILING_DROP = 0.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="scenesmith_bake_glb")
    ap.add_argument("--out", required=True, help="output .glb path")
    ap.add_argument("--samples", type=int, default=192, help="Cycles bake samples")
    ap.add_argument("--max-res", type=int, default=2048, help="per-object texture cap")
    ap.add_argument("--px-per-m", type=float, default=1024.0, help="texel density")
    ap.add_argument("--watts-per-m2", type=float, default=14.0, help="room light power")
    ap.add_argument(
        "--ambient",
        type=float,
        default=0.25,
        help="world light strength; only reaches rooms through windows",
    )
    ap.add_argument(
        "--target-exposure",
        type=float,
        default=0.62,
        help="mean luminance each scene is exposed to (0 disables)",
    )
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--no-ceiling", action="store_true")
    ap.add_argument("--no-bake", action="store_true", help="geometry/ceiling only")
    return ap.parse_args(argv)


def world_bounds(objects):
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in objects:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            lo = Vector(min(lo[i], w[i]) for i in range(3))
            hi = Vector(max(hi[i], w[i]) for i in range(3))
    return lo, hi


def find_floors(meshes, scene_lo, scene_hi):
    floors = []
    for o in meshes:
        lo, hi = world_bounds([o])
        thickness = hi[2] - lo[2]
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        near_bottom = lo[2] - scene_lo[2] < 0.3
        if thickness <= FLOOR_MAX_THICKNESS and area >= FLOOR_MIN_AREA and near_bottom:
            floors.append((lo, hi))
    return merge_overlapping(floors)


def merge_overlapping(boxes):
    merged = [(lo.copy(), hi.copy()) for lo, hi in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                (alo, ahi), (blo, bhi) = merged[i], merged[j]
                if (
                    alo[0] < bhi[0]
                    and ahi[0] > blo[0]
                    and alo[1] < bhi[1]
                    and ahi[1] > blo[1]
                ):
                    for k in range(3):
                        alo[k] = min(alo[k], blo[k])
                        ahi[k] = max(ahi[k], bhi[k])
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def room_ceiling_z(meshes, lo, hi, scene_hi) -> float:
    tops = []
    for o in meshes:
        olo, ohi = world_bounds([o])
        thin = min(ohi[0] - olo[0], ohi[1] - olo[1]) < 0.35
        tall = (ohi[2] - olo[2]) > 1.2
        overlaps = (
            olo[0] < hi[0] and ohi[0] > lo[0] and olo[1] < hi[1] and ohi[1] > lo[1]
        )
        if thin and tall and overlaps:
            tops.append(ohi[2])
    return (max(tops) if tops else scene_hi[2]) - CEILING_DROP


def add_ceilings(scene, meshes) -> int:
    scene_lo, scene_hi = world_bounds(meshes)
    floors = find_floors(meshes, scene_lo, scene_hi)
    if not floors:
        floors = [(scene_lo, scene_hi)]

    mat = bpy.data.materials.new("CeilingMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.84, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9

    made = 0
    for i, (lo, hi) in enumerate(floors):
        z = room_ceiling_z(meshes, lo, hi, scene_hi)
        pad = 0.02
        mesh = bpy.data.meshes.new(f"Ceiling{i}")
        verts = [
            (lo[0] - pad, lo[1] - pad, z),
            (hi[0] + pad, lo[1] - pad, z),
            (hi[0] + pad, hi[1] + pad, z),
            (lo[0] - pad, hi[1] + pad, z),
        ]
        mesh.from_pydata(verts, [], [(0, 3, 2, 1)])
        mesh.update()
        mesh.materials.append(mat)
        # a fresh UV layer is all zeros; the bake would collapse into one texel
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for loop_index, uv in zip(
            mesh.polygons[0].loop_indices, ((0, 0), (0, 1), (1, 1), (1, 0))
        ):
            uv_layer.data[loop_index].uv = uv
        obj = bpy.data.objects.new(f"Ceiling{i}", mesh)
        scene.collection.objects.link(obj)
        meshes.append(obj)
        made += 1
    print(f"added {made} ceiling(s)", flush=True)
    return made


def add_lights(scene, meshes, watts_per_m2: float, ambient: float) -> None:
    scene_lo, scene_hi = world_bounds(meshes)
    floors = find_floors(meshes, scene_lo, scene_hi) or [(scene_lo, scene_hi)]
    n = 0
    for lo, hi in floors:
        area = max((hi[0] - lo[0]) * (hi[1] - lo[1]), 0.5)
        ceiling_z = room_ceiling_z(meshes, lo, hi, scene_hi)
        side = max(1, min(MAX_LIGHT_GRID, round((area / AREA_PER_LIGHT) ** 0.5)))
        per_light = watts_per_m2 * area / (side * side)
        for ix in range(side):
            for iy in range(side):
                fx = (ix + 1) / (side + 1)
                fy = (iy + 1) / (side + 1)
                data = bpy.data.lights.new(name=f"CeilLight{n}", type="AREA")
                data.energy = per_light
                data.size = 0.5
                data.color = (1.0, 0.95, 0.88)
                obj = bpy.data.objects.new(f"CeilLight{n}", data)
                obj.location = (
                    lo[0] + fx * (hi[0] - lo[0]),
                    lo[1] + fy * (hi[1] - lo[1]),
                    ceiling_z - 0.12,
                )
                scene.collection.objects.link(obj)
                n += 1
        print(
            f"  room {area:.1f}m2 -> {side}x{side} x {per_light:.0f}W at z={ceiling_z:.2f}",
            flush=True,
        )
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.35, 0.38, 0.45, 1.0)
    bg.inputs[1].default_value = ambient
    print(f"added {n} area lights", flush=True)


def auto_exposure(scene, meshes, target: float) -> float:
    scene_lo, scene_hi = world_bounds(meshes)
    floors = find_floors(meshes, scene_lo, scene_hi) or [(scene_lo, scene_hi)]
    lo, hi = max(floors, key=lambda b: (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]))
    spots = [
        (0.5, 0.5),
        (0.3, 0.3),
        (0.7, 0.3),
        (0.3, 0.7),
        (0.7, 0.7),
    ]
    positions = [
        (lo[0] + fx * (hi[0] - lo[0]), lo[1] + fy * (hi[1] - lo[1]), scene_lo[2] + 1.35)
        for fx, fy in spots
    ]

    cam_data = bpy.data.cameras.new("ExposureProbe")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(90)
    cam = bpy.data.objects.new("ExposureProbe", cam_data)
    scene.collection.objects.link(cam)
    previous_camera = scene.camera
    scene.camera = cam

    keep = (
        scene.cycles.samples,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.filepath,
        scene.render.image_settings.file_format,
    )
    scene.cycles.samples = 24
    scene.render.resolution_x, scene.render.resolution_y = 160, 120
    scene.view_settings.exposure = 0.0

    per_spot = []
    with tempfile.TemporaryDirectory(prefix="scenesmith_probe_") as probe_dir:
        for spot_index, position in enumerate(positions):
            cam.location = position
            views = []
            for i in range(2):
                cam.rotation_euler = (math.radians(90), 0, math.radians(180 * i))
                path = os.path.join(probe_dir, f"probe{spot_index}_{i}.png")
                scene.render.filepath = path
                scene.render.image_settings.file_format = "PNG"
                bpy.ops.render.render(write_still=True)
                image = bpy.data.images.load(path)
                pixels = list(image.pixels)
                views.append(sum(pixels[0::4]) / max(len(pixels) // 4, 1))
                bpy.data.images.remove(image)
            per_spot.append(sum(views) / len(views))

    usable = sorted(m for m in per_spot if m > PROBE_MIN_LUMA)
    measured = usable[len(usable) // 2] if usable else 0.0
    if len(usable) < len(per_spot):
        print(
            f"exposure probe: {len(per_spot) - len(usable)}/{len(per_spot)} spots "
            "were inside geometry, ignored",
            flush=True,
        )

    (
        scene.cycles.samples,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.filepath,
        scene.render.image_settings.file_format,
    ) = keep
    scene.camera = previous_camera
    bpy.data.objects.remove(cam, do_unlink=True)

    if measured <= PROBE_MIN_LUMA:
        print("exposure probe saw only black; leaving exposure at 0", flush=True)
        return 0.0
    stops = math.log2(max(target, 1e-4) / measured)
    stops = max(-3.0, min(3.0, stops))
    scene.view_settings.exposure = stops
    print(
        f"auto exposure: probe {measured:.3f} -> target {target:.2f} ({stops:+.2f} stops)",
        flush=True,
    )
    return stops


def setup_cycles(scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.bake.margin = 16
    scene.render.bake.margin_type = "ADJACENT_FACES"
    scene.render.bake.use_clear = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "CUDA"


def prepare_bake_uvs(obj) -> None:
    src = obj.data.uv_layers.active or (
        obj.data.uv_layers[0] if len(obj.data.uv_layers) else None
    )
    uvs = [layer.uv for layer in obj.data.uv_layers[src.name].data] if src else []
    tiled = any(
        u[0] < -0.001 or u[0] > 1.001 or u[1] < -0.001 or u[1] > 1.001 for u in uvs
    )
    if tiled or not uvs:
        uv = obj.data.uv_layers.get("bake") or obj.data.uv_layers.new(name="bake")
        obj.data.uv_layers.active = uv
        if len(obj.data.polygons) <= PLANAR_MAX_POLYS:
            planar_cube_uvs(obj)
        else:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            # a larger island_margin collapses every island on a high-poly mesh
            bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.002)
            bpy.ops.object.mode_set(mode="OBJECT")
    else:
        obj.data.uv_layers.new(name="bake")
        for i, layer in enumerate(obj.data.uv_layers[src.name].data):
            obj.data.uv_layers["bake"].data[i].uv = layer.uv
        obj.data.uv_layers.active = obj.data.uv_layers["bake"]


def planar_cube_uvs(obj) -> None:
    mesh = obj.data
    uv_layer = mesh.uv_layers.get("bake") or mesh.uv_layers.new(name="bake")
    mesh.uv_layers.active = uv_layer

    coords = [v.co for v in mesh.vertices]
    lo = [min(c[i] for c in coords) for i in range(3)]
    hi = [max(c[i] for c in coords) for i in range(3)]
    span = [max(hi[i] - lo[i], 1e-6) for i in range(3)]

    inset = 0.02
    for poly in mesh.polygons:
        normal = poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        other = [i for i in range(3) if i != axis]
        col, row = axis, (0 if normal[axis] >= 0 else 1)
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            u = (co[other[0]] - lo[other[0]]) / span[other[0]]
            v = (co[other[1]] - lo[other[1]]) / span[other[1]]
            uv_layer.data[loop_index].uv = (
                (col + inset + u * (1 - 2 * inset)) / 3.0,
                (row + inset + v * (1 - 2 * inset)) / 2.0,
            )


def texture_size(obj, px_per_m: float, cap: int) -> int:
    sx, sy, _ = obj.matrix_world.to_scale()
    area = sum(p.area for p in obj.data.polygons) * abs(sx * sy)
    want = max(256.0, (area**0.5) * px_per_m)
    return min(1 << max(8, min(11, int(want - 1).bit_length())), cap)


def bake_object(obj, args, tmp_dir: str) -> bool:
    if not obj.data.materials or all(m is None for m in obj.data.materials):
        return False

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bakes write per object, so shared data would share the bake
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    for i, mat in enumerate(obj.data.materials):
        if mat is not None and mat.users > 1:
            obj.data.materials[i] = mat.copy()

    prepare_bake_uvs(obj)
    size = texture_size(obj, args.px_per_m, args.max_res)
    image = bpy.data.images.new(f"bake_{obj.name}", width=size, height=size)
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.label = "BAKE_TARGET"
        mat.node_tree.nodes.active = node
        for other in mat.node_tree.nodes:
            other.select = other is node

    bpy.ops.object.bake(type="COMBINED")

    # bake writes linear; save_render applies the view transform
    toned_path = os.path.join(tmp_dir, f"{abs(hash(obj.name))}.png")
    image.save_render(filepath=toned_path)
    toned = bpy.data.images.load(toned_path)
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.label == "BAKE_TARGET":
                node.image = toned
    obj["bake_image"] = toned.name
    obj.select_set(False)
    return True


def wire_baked_materials(obj) -> None:
    if not obj.get("bake_image"):
        return
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        tree = mat.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        tex = next((n for n in tree.nodes if n.label == "BAKE_TARGET"), None)
        if bsdf is None or tex is None:
            continue
        for link in list(tree.links):
            if link.to_node is bsdf and link.to_socket.name == "Base Color":
                tree.links.remove(link)
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        # else glTF export binds UV0
        uv_node = tree.nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "bake"
        tree.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
    for layer in [l for l in obj.data.uv_layers if l.name != "bake"]:
        obj.data.uv_layers.remove(layer)
    obj.data.uv_layers.active = obj.data.uv_layers["bake"]
    obj.data.uv_layers["bake"].active_render = True


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH" and len(o.data.polygons)]
    print(f"objects={len(scene.objects)} meshes={len(meshes)}", flush=True)
    missing = [i.name for i in bpy.data.images if i.source == "FILE" and not i.has_data]
    if missing:
        print(f"WARNING: {len(missing)} textures failed to load: {missing[:5]}")

    if not args.no_ceiling:
        add_ceilings(scene, meshes)

    if not args.no_bake:
        add_lights(scene, meshes, args.watts_per_m2, args.ambient)
        setup_cycles(scene, args.samples)
        if args.target_exposure > 0:
            auto_exposure(scene, meshes, args.target_exposure)
        bpy.ops.object.select_all(action="DESELECT")
        started = time.time()
        done = 0
        with tempfile.TemporaryDirectory(prefix="scenesmith_bake_") as tmp_dir:
            for obj in meshes:
                if bake_object(obj, args, tmp_dir):
                    done += 1
                if done and done % 25 == 0:
                    print(
                        f"baked {done}/{len(meshes)} in {time.time() - started:.0f}s",
                        flush=True,
                    )
            for obj in meshes:
                wire_baked_materials(obj)
            print(f"baked {done} objects in {time.time() - started:.0f}s", flush=True)

            os.makedirs(
                os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True
            )
            export(args)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        export(args)


def export(args) -> None:
    # +Y-up renders the room 90 deg rolled in Habitat
    bpy.ops.export_scene.gltf(
        filepath=args.out,
        export_format="GLB",
        export_apply=True,
        export_yup=False,
        export_cameras=False,
        export_lights=False,
        export_image_format="JPEG",
        export_jpeg_quality=args.jpeg_quality,
    )
    print(f"wrote {args.out}", flush=True)


main()
