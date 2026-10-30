# Generative 3D City Generator

A pipeline that procedurally generates cities as data grids and turns them into colored, 3D-printable models in Blender. Built with the Michigan Data Science Team (MDST). **Status: in progress.** The generation and export pipeline works end to end, and a text-to-parameter ML model is being developed.

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/cd4ec8e7-afa3-4948-a106-228dc6bbcdba" />

## Overview

Every city is represented as a fixed-size grid (128×128) with four channels:

| Channel | Type | Meaning |
|---|---|---|
| `terrain_height` | float32 | Ground elevation |
| `is_water` | bool | Lakes, rivers, harbors |
| `building_height` | float32 | Building height (0 = no building) |
| `land_use` | uint8 | Building / road / green space / empty ground |

The grid is saved as a `.npz` file, which every stage of the pipeline reads or writes.

<img width="2700" height="750" alt="image" src="https://github.com/user-attachments/assets/a2b4b1a7-129b-4a21-a861-9fcbd606d795" />

## Pipeline

```
generate_city.py  ->  mycity.npz  ->  visualize_2d.py  ->  mycity_preview.png
                          |
                          +-> export_to_blender.py (runs inside Blender) -> mycity.blend + mycity.stl
```

1. **Terrain generation** (`citygen/terrain.py`): fractal noise shaped by an archetype (flat, hills), with water marked below the water level.
2. **Building generation** (`citygen/buildings.py`): cluster centers, urban intensity falloff, no buildings on water or steep slopes, then roads with gaps and green space.
3. **2D preview** (`scripts/visualize_2d.py`): four panels for terrain, water, building height, and land use.
4. **Blender export** (`scripts/export_to_blender.py`): builds terrain and building meshes, colors them by land use and height, and saves a `.blend` and a printable `.stl`.

## Quick start

Run all commands from the repo root.

```bash
# 1. Set up
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate a city
python scripts/generate_city.py --size 128 --terrain hills --city polycentric \
  --clusters 3 --density 0.6 --cluster-tightness 0.6 \
  --green-amount 0.25 --road-continuity 0.7 --seed 42 --out data/mycity.npz

# 3. Preview in 2D
python scripts/visualize_2d.py data/mycity.npz --out data/mycity_preview.png

# 4. Export to Blender (requires Blender 3.x or 4.x)
blender --background --python scripts/export_to_blender.py -- data/mycity.npz

# 5. Open the model (switch viewport shading to Material Preview)
blender data/mycity.blend
```

On macOS, Blender is not on your PATH by default. Add an alias:

```bash
echo 'alias blender="/Applications/Blender.app/Contents/MacOS/Blender"' >> ~/.zshrc
source ~/.zshrc
```

## Generator options

| Flag | What it controls |
|---|---|
| `--size` | Grid size (default 128) |
| `--terrain` | Terrain archetype (`flat`, `hills`) |
| `--city` | City archetype (e.g. `polycentric`, `linear`) |
| `--clusters` | Number of city centers |
| `--density` | Overall building density |
| `--cluster-tightness` | How tightly buildings pack around centers |
| `--green-amount` | Share of land used for green space |
| `--road-continuity` | How unbroken the road network is |
| `--seed` | Random seed for reproducible cities |

## Project structure

```
citygen/
  city_grid.py      # CityGrid: the four-array data contract, save/load/stats
  terrain.py        # generate_terrain(...)
  buildings.py      # generate_buildings(...)
scripts/
  generate_city.py      # CLI: terrain -> buildings -> .npz
  visualize_2d.py       # 2D preview
  export_to_blender.py  # runs inside Blender: .npz -> .blend + .stl
docs/
  blender_export.md     # how a .npz becomes a .blend
```

`export_to_blender.py` runs in Blender's own Python, which has numpy but not scipy. That is why `citygen/__init__.py` only imports terrain and buildings when scipy is available.

## Roadmap

- [x] Procedural terrain and building generators
- [x] `.npz` city format (`CityGrid`)
- [x] 2D preview
- [x] Blender export with colored materials and printable STL
- [x] Road-continuity metric
- [ ] Synthetic training dataset (thousands of randomized cities)
- [ ] Two-stage conditioned ML model: a terrain model feeds a building model so water and slopes suppress buildings coherently
- [ ] Text-to-parameter model (natural-language description to generator parameters)
- [ ] Visual interface
- [ ] Final expo demo

## Tech stack

Python · NumPy · SciPy · Matplotlib · Blender (bpy) · Jupyter · Git/GitHub
*Planned: PyTorch (ML models), Streamlit (interface)*

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'citygen'` | Run from the repo root |
| `No module named 'numpy'` / `scipy` | Activate the venv and run `pip install -r requirements.txt` |
| `No module named 'scipy'` inside Blender | Use the fixed `citygen/__init__.py`; don't install scipy into Blender |
| Blender quits without saving | Add `--` before the `.npz` path |
| Model looks all grey | Switch viewport shading to Material Preview |

## Credits

Built with the Michigan Data Science Team (MDST), University of Michigan.
