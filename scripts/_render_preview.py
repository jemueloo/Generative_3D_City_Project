"""
Opens a .blend, frames the scene with a camera, and renders a PNG with the
Workbench engine.

    blender --background --python scripts/_render_preview.py -- in.blend out.png
"""
import sys
from pathlib import Path
import math

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: ... -- <in.blend> <out.png>")

blend_in = str(Path(argv[0]).resolve())
png_out = str(Path(argv[1]).resolve())

bpy.ops.wm.open_mainfile(filepath=blend_in)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh objects found in the .blend")

xs, ys, zs = [], [], []
for o in meshes:
    for corner in o.bound_box:
        wc = o.matrix_world @ type(o.location)(corner)
        xs.append(wc.x); ys.append(wc.y); zs.append(wc.z)

cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
cz = (min(zs) + max(zs)) / 2
extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

cam_data = bpy.data.cameras.new("PreviewCam")
cam = bpy.data.objects.new("PreviewCam", cam_data)
bpy.context.collection.objects.link(cam)

dist = extent * 1.5
cam.location = (cx + dist * 0.75, cy - dist * 0.75, cz + dist * 0.75)
cam.rotation_euler = (math.radians(55), 0.0, math.radians(45))
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"   # stable across Blender versions
try:
    scene.display.shading.color_type = "MATERIAL"   # show our material colours
    scene.display.shading.light = "STUDIO"
except Exception as e:
    print("couldn't set workbench shading options:", e)

scene.render.resolution_x = 1200
scene.render.resolution_y = 800
scene.render.film_transparent = False
scene.render.filepath = png_out

bpy.ops.render.render(write_still=True)
print("rendered", png_out)