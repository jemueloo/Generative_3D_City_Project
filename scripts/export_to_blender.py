"""
Import a saved CityGrid into Blender, build 3D geometry, and export a
watertight STL suitable for 3D printing.

    blender --background --python scripts/export_to_blender.py -- data/city.npz

Drop --background to have the GUI open with the result already loaded.

Outputs, next to the input .npz:
    <name>.blend   the Blender scene (terrain + buildings, coloured)
    <name>.stl     a printable mesh

NOTE: numpy only. Blender does not bundle scipy, so importing it here would
fail with ModuleNotFoundError.
"""
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("usage: blender --background --python export_to_blender.py -- <city.npz>")

CITY_PATH = argv[0]

CELL_SIZE = 1.0             # world size of one grid cell
HEIGHT_SCALE = 8.0          # terrain_height is 0-1, so scale it up to be visible
BUILDING_HEIGHT_SCALE = 1.0

# --- printability knobs -------------------------------------------------
BASE_THICKNESS = 2.0        # solid slab under the lowest point of the terrain
BUILDING_EMBED = 0.3        # how far each building sinks into the ground
# -----------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citygen import CityGrid, LAND_BUILDING, LAND_GREEN, LAND_ROAD  # noqa: E402

COLORS = {
    "ground":          (0.78, 0.72, 0.55),
    "green":           (0.30, 0.55, 0.25),
    "road":            (0.15, 0.15, 0.16),
    "water":           (0.10, 0.35, 0.55),
    "building_short":  (0.55, 0.55, 0.55),
    "building_mid":    (0.70, 0.62, 0.45),
    "building_tall":   (0.35, 0.42, 0.52),
}


def get_or_create_material(name: str):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b = COLORS[name]
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    mat.diffuse_color = (r, g, b, 1.0)
    return mat


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


# --------------------------------------------------------------------
# terrain -- a closed SOLID, not an open sheet
# --------------------------------------------------------------------

def build_terrain(city: CityGrid):
    """
    Top surface + vertical skirt around the perimeter + flat bottom cap.

    The original version built only the top surface: a zero-thickness sheet
    with an open boundary all the way around, which no slicer can print.
    The skirt and cap turn it into a closed volume.
    """
    size = city.size
    mesh = bpy.data.meshes.new("TerrainMesh")
    bm = bmesh.new()
    slot_names = ["ground", "water", "green", "road"]

    # top surface vertices, created once and shared by every face that
    # touches them
    verts = np.empty((size, size), dtype=object)
    zmin = float("inf")
    for y in range(size):
        for x in range(size):
            z = 0.0 if city.is_water[y, x] else float(city.terrain_height[y, x]) * HEIGHT_SCALE
            zmin = min(zmin, z)
            verts[y, x] = bm.verts.new((x * CELL_SIZE, y * CELL_SIZE, z))

    base_z = zmin - BASE_THICKNESS

    for y in range(size - 1):
        for x in range(size - 1):
            face = bm.faces.new((verts[y, x], verts[y, x + 1],
                                 verts[y + 1, x + 1], verts[y + 1, x]))
            if city.is_water[y, x]:
                face.material_index = slot_names.index("water")
            elif city.land_use[y, x] == LAND_GREEN:
                face.material_index = slot_names.index("green")
            elif city.land_use[y, x] == LAND_ROAD:
                face.material_index = slot_names.index("road")
            else:
                face.material_index = slot_names.index("ground")

    # walk the boundary once, in order, without repeating the corners
    perimeter = []
    for x in range(size):
        perimeter.append((0, x))
    for y in range(1, size):
        perimeter.append((y, size - 1))
    for x in range(size - 2, -1, -1):
        perimeter.append((size - 1, x))
    for y in range(size - 2, 0, -1):
        perimeter.append((y, 0))

    # one bottom vertex per perimeter position, created ONCE and reused by
    # both the skirt and the cap. Creating them twice would leave the mesh
    # looking closed while actually being split along that seam -- the
    # classic silent watertightness bug.
    bottom_verts = [bm.verts.new((x * CELL_SIZE, y * CELL_SIZE, base_z))
                    for (y, x) in perimeter]

    n = len(perimeter)
    for i in range(n):
        j = (i + 1) % n
        ty, tx = perimeter[i]
        uy, ux = perimeter[j]
        f = bm.faces.new((verts[ty, tx], verts[uy, ux],
                          bottom_verts[j], bottom_verts[i]))
        f.material_index = slot_names.index("ground")

    # single n-gon across the whole base
    cap = bm.faces.new(tuple(reversed(bottom_verts)))
    cap.material_index = slot_names.index("ground")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    for name in slot_names:
        mesh.materials.append(get_or_create_material(name))

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------
# buildings -- closed boxes, sunk into the terrain
# --------------------------------------------------------------------

def build_buildings(city: CityGrid):
    size = city.size
    mesh = bpy.data.meshes.new("BuildingsMesh")
    bm = bmesh.new()
    slot_names = ["building_short", "building_mid", "building_tall"]

    building_cells = np.argwhere(city.land_use == LAND_BUILDING)
    heights = city.building_height[city.land_use == LAND_BUILDING]
    heights = heights[heights > 0]
    if len(heights) > 0:
        low_cut, high_cut = np.percentile(heights, [40, 75])
    else:
        low_cut, high_cut = 0.0, 0.0

    for y, x in building_cells:
        h = float(city.building_height[y, x]) * BUILDING_HEIGHT_SCALE
        if h <= 0:
            continue

        # Start from the LOWEST of the four terrain corners around this cell,
        # then sink a little further. Using terrain_height[y, x] alone (the
        # original behaviour) leaves boxes hovering on sloped ground, which
        # prints as a floating block or a hairline crack at the base.
        y1 = min(y + 1, size - 1)
        x1 = min(x + 1, size - 1)
        corner_min = min(float(city.terrain_height[y, x]), float(city.terrain_height[y, x1]),
                         float(city.terrain_height[y1, x]), float(city.terrain_height[y1, x1]))
        base_z = corner_min * HEIGHT_SCALE - BUILDING_EMBED
        top_z = float(city.terrain_height[y, x]) * HEIGHT_SCALE + h

        cx, cy = x * CELL_SIZE, y * CELL_SIZE
        half = CELL_SIZE * 0.4

        if h < low_cut:
            mat_idx = slot_names.index("building_short")
        elif h < high_cut:
            mat_idx = slot_names.index("building_mid")
        else:
            mat_idx = slot_names.index("building_tall")

        bottom = [
            bm.verts.new((cx - half, cy - half, base_z)),
            bm.verts.new((cx + half, cy - half, base_z)),
            bm.verts.new((cx + half, cy + half, base_z)),
            bm.verts.new((cx - half, cy + half, base_z)),
        ]
        top = [
            bm.verts.new((cx - half, cy - half, top_z)),
            bm.verts.new((cx + half, cy - half, top_z)),
            bm.verts.new((cx + half, cy + half, top_z)),
            bm.verts.new((cx - half, cy + half, top_z)),
        ]
        new_faces = [bm.faces.new(bottom), bm.faces.new(top[::-1])]
        for i in range(4):
            j = (i + 1) % 4
            new_faces.append(bm.faces.new((bottom[i], bottom[j], top[j], top[i])))
        for face in new_faces:
            face.material_index = mat_idx

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    for name in slot_names:
        mesh.materials.append(get_or_create_material(name))

    obj = bpy.data.objects.new("Buildings", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------
# STL export
# --------------------------------------------------------------------

def export_stl(path, verbose=False):
    """
    Blender renamed the STL exporter between versions:
        Blender  < 4.2   bpy.ops.export_mesh.stl
        Blender >= 4.2   bpy.ops.wm.stl_export

    Both names resolve as attributes whether or not the operator actually
    exists -- bpy.ops hands back a stub that only raises when called -- so
    the only way to tell is to try one. We therefore attempt each in turn
    and stay QUIET about the ones that aren't there: a fallback working as
    designed is not an error, and printing "failed" for it just alarms
    people. Only a total failure gets reported.

    Set verbose=True if you want to see which operator actually ran.
    """
    path = str(Path(path).resolve())   # operators reject relative paths
    attempts = []

    for label, op in (("wm.stl_export", getattr(bpy.ops.wm, "stl_export", None)),
                      ("export_mesh.stl", getattr(getattr(bpy.ops, "export_mesh", None), "stl", None))):
        if op is None:
            attempts.append(f"{label}: not available")
            continue
        try:
            op(filepath=path)
            print(f"exported STL -> {path}")
            if verbose:
                print(f"  (used bpy.ops.{label})")
            return True
        except Exception as exc:
            attempts.append(f"{label}: {type(exc).__name__}")

    print("STL export failed. Tried -- " + "; ".join(attempts))
    print("Export manually in Blender: File > Export > Stl (.stl)")
    return False


def main():
    city = CityGrid.load(CITY_PATH)
    print(f"loaded {CITY_PATH}")
    print(f"  stats: {city.stats()}")

    clear_scene()
    build_terrain(city)
    build_buildings(city)

    out_blend = str(Path(CITY_PATH).with_suffix(".blend").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"saved {out_blend}")

    export_stl(Path(CITY_PATH).with_suffix(".stl"))


if __name__ == "__main__":
    main()