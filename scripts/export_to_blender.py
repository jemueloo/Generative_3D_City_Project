"""
Import a saved CityGrid into Blender and build 3D geometry from it.

Must be run with Blender's own Python, not your venv:

    blender --background --python scripts/export_to_blender.py -- data/city.npz

Drop --background to have the GUI open with the result already loaded.

Builds a "Terrain" mesh coloured by land use, and a "Buildings" mesh of one
box per building cell coloured by height tier, then saves a .blend next to
the input .npz.

NOTE: numpy only. Blender does not bundle scipy, so importing it here would
fail with ModuleNotFoundError.
"""
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np

# Blender passes its own args first, then "--", then ours.
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("usage: blender --background --python export_to_blender.py -- <city.npz>")

CITY_PATH = argv[0]

CELL_SIZE = 1.0             # world size of one grid cell
HEIGHT_SCALE = 8.0          # terrain_height is 0-1, so scale it up to be visible
BUILDING_HEIGHT_SCALE = 1.0

# Load citygen/city_grid.py DIRECTLY by file path instead of `from citygen import ...`.
# Importing the citygen package runs citygen/__init__.py, which imports terrain.py
# and buildings.py, and those need scipy -- which doesn't exist inside Blender.
# city_grid.py only needs numpy, so loading it on its own works fine.
import importlib.util  # noqa: E402

_CITY_GRID_PATH = Path(__file__).resolve().parent.parent / "citygen" / "city_grid.py"
_spec = importlib.util.spec_from_file_location("city_grid", _CITY_GRID_PATH)
_city_grid = importlib.util.module_from_spec(_spec)
sys.modules["city_grid"] = _city_grid   # dataclasses looks the module up by name
_spec.loader.exec_module(_city_grid)

CityGrid = _city_grid.CityGrid
LAND_BUILDING = _city_grid.LAND_BUILDING
LAND_GREEN = _city_grid.LAND_GREEN
LAND_ROAD = _city_grid.LAND_ROAD

# name -> (R, G, B), each 0-1. Edit these to change the palette.
COLORS = {
    "ground":          (0.78, 0.72, 0.55),
    "green":           (0.30, 0.55, 0.25),
    "road":            (0.15, 0.15, 0.16),
    "water":           (0.10, 0.35, 0.55),
    "building_short":  (0.55, 0.55, 0.55),
    "building_mid":    (0.70, 0.62, 0.45),
    "building_tall":   (0.35, 0.42, 0.52),
}


# --------------------------------------------------------------------
# GIVEN -- helpers
# --------------------------------------------------------------------

def get_or_create_material(name: str):
    """Fetch a material by name, creating it from COLORS if it doesn't exist."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b = COLORS[name]
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    mat.diffuse_color = (r, g, b, 1.0)   # also used by solid viewport shading
    return mat


def clear_scene():
    """Delete everything, so re-running doesn't stack cities on top of each other."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


# --------------------------------------------------------------------
# YOUR CODE -- terrain mesh
# --------------------------------------------------------------------

def build_terrain(city: CityGrid):
    """One vertex per grid cell, joined into quads, coloured by land use."""
    size = city.size
    mesh = bpy.data.meshes.new("TerrainMesh")
    bm = bmesh.new()

    # material slots, in a fixed order -- face.material_index is an index
    # into THIS list, so the order here is what the indices below mean
    slot_names = ["ground", "water", "green", "road"]

    verts = np.empty((size, size), dtype=object)
    for y in range(size):
        for x in range(size):
            # TODO 1: the height of this vertex.
            #   - water cells sit flat at 0.0, so lakes are level
            #   - everything else: city.terrain_height[y, x] * HEIGHT_SCALE
            #     (the heightmap is 0-1; without the scale the terrain is
            #     invisibly flat next to buildings that are 8 units tall)
            if city.is_water[y, x]:
                z = 0.0
            else:
                z = float(city.terrain_height[y, x]) * HEIGHT_SCALE

            verts[y, x] = bm.verts.new((x * CELL_SIZE, y * CELL_SIZE, z))

    for y in range(size - 1):
        for x in range(size - 1):
            # TODO 2: make a quad from this vertex and its three neighbours.
            # bm.faces.new(...) takes a tuple of vertices IN ORDER around the
            # face -- going around the loop, not diagonally across it:
            #   (y, x) -> (y, x+1) -> (y+1, x+1) -> (y+1, x)
            # Listing them in the wrong order gives you a bow-tie face.
            face = bm.faces.new((verts[y, x], verts[y, x + 1],
                                 verts[y + 1, x + 1], verts[y + 1, x]))

            # TODO 3: colour the face by what's on that cell.
            # Set face.material_index to the position of the right name in
            # slot_names, using slot_names.index("...").
            # Priority: water first, then green, then road, else ground.
            if city.is_water[y, x]:
                face.material_index = slot_names.index("water")
            elif city.land_use[y, x] == LAND_GREEN:
                face.material_index = slot_names.index("green")
            elif city.land_use[y, x] == LAND_ROAD:
                face.material_index = slot_names.index("road")
            else:
                face.material_index = slot_names.index("ground")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    for name in slot_names:
        mesh.materials.append(get_or_create_material(name))

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------
# YOUR CODE -- buildings mesh
# --------------------------------------------------------------------

def build_buildings(city: CityGrid):
    """One box per building cell, all merged into a single mesh object."""
    mesh = bpy.data.meshes.new("BuildingsMesh")
    bm = bmesh.new()

    slot_names = ["building_short", "building_mid", "building_tall"]

    # Height tiers from THIS city's own distribution, so the colouring
    # adapts instead of using fixed cutoffs that would look wrong on a very
    # short or very tall city.
    building_cells = np.argwhere(city.land_use == LAND_BUILDING)
    heights = city.building_height[city.land_use == LAND_BUILDING]
    heights = heights[heights > 0]
    if len(heights) > 0:
        low_cut, high_cut = np.percentile(heights, [40, 75])
    else:
        low_cut, high_cut = 0.0, 0.0

    for y, x in building_cells:
        base_z = float(city.terrain_height[y, x]) * HEIGHT_SCALE
        h = float(city.building_height[y, x]) * BUILDING_HEIGHT_SCALE
        if h <= 0:
            continue

        cx, cy = x * CELL_SIZE, y * CELL_SIZE
        half = CELL_SIZE * 0.4   # < 0.5 leaves a gap so neighbours don't fuse

        # TODO 4: pick the colour tier for this building.
        # h < low_cut            -> "building_short"
        # low_cut <= h < high_cut -> "building_mid"
        # otherwise               -> "building_tall"
        # Again use slot_names.index("...").
        if h < low_cut:
            mat_idx = slot_names.index("building_short")
        elif h < high_cut:
            mat_idx = slot_names.index("building_mid")
        else:
            mat_idx = slot_names.index("building_tall")

        # TODO 5: build the box.
        # `bottom` is given. Make `top` the same four corners but at
        # base_z + h. Keep the SAME corner order in both lists -- the side
        # walls below pair them up by index, so a different order twists
        # the box.
        bottom = [
            bm.verts.new((cx - half, cy - half, base_z)),
            bm.verts.new((cx + half, cy - half, base_z)),
            bm.verts.new((cx + half, cy + half, base_z)),
            bm.verts.new((cx - half, cy + half, base_z)),
        ]
        top = [
            bm.verts.new((cx - half, cy - half, base_z + h)),
            bm.verts.new((cx + half, cy - half, base_z + h)),
            bm.verts.new((cx + half, cy + half, base_z + h)),
            bm.verts.new((cx - half, cy + half, base_z + h)),
        ]

        # GIVEN: close the box. top[::-1] reverses the roof's winding order
        # so its normal points up instead of down.
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
# GIVEN -- main
# --------------------------------------------------------------------

def main():
    city = CityGrid.load(CITY_PATH)
    print(f"loaded {CITY_PATH}")
    print(f"  stats: {city.stats()}")

    clear_scene()
    build_terrain(city)
    build_buildings(city)

    # .resolve() matters: save_as_mainfile rejects relative paths
    out_path = str(Path(CITY_PATH).with_suffix(".blend").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
