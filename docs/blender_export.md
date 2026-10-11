# Blender Export: from .npz to .blend

> Replace each TODO with your own explanation.

## 1. The .npz city file

The .npz file is a compressed archive of the CityGrid data structure, which contains the terrain heightmap, water mask, building heights, and land use information. This file serves as the input for the Blender export script, allowing us to reconstruct the city in 3D.

## 2. Grid cells to 3D vertices

Each cell (y,x) becomes one vertex at (x * CELL_SIZE, y * CELL_SIZE, z), where z is the height of that cell. Water cells sit flat at 0.0, while other cells use city.terrain_height[y, x] * HEIGHT_SCALE to scale the heightmap from 0-1 to a visible range.

## 3. Vertices to faces

Every 2x2 block of neighboring vertices is joined into one quad, giving (size-1)^2 faces. Each face is coloured by the land use of that cell, with water taking priority over green, which takes priority over road, and everything else is ground.

## 4. Buildings

Each building cell becomes a box, with its base at the terrain height and its top at terrain + building_height. The boxes are merged into one mesh, and each is coloured by its height tier (short, mid, tall) based on the city's own distribution of building heights.

## 5. Materials and colors

A mesh holds several materials in slots, and each face's material_index points at one slot, so coloring by land use means setting that index per face. Building colour tiers use percentiles of the city's own building heights, so the colors adapt to the city instead of using fixed cutoffs.

## 6. Running the export

Run blender --background --python scripts/export_to_blender.py -- data/week3_city.npz, which saves a .blend next to the .npz.--background runs Blender without a window and -- separates Blender's own arguments from the script's, so the file path isn't treated as a Blender option

## 7. Gotchas

save_as_mainfile rejects relative paths, so the script uses .resolve() to make the output path absolute. Blender does not bundle scipy, so importing it inside the script fails -- the script loads city_grid.py directly instead of importing the citygen package.
