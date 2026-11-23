# Generative 3D Cities

A pipeline that procedurally generates cities as data grids and turns them into colored, 3D-printable models in Blender. **Status: built. The demo video link is the one thing still to add.** You can describe a city in plain English, or set its parameters by hand, in a web app (`app.py`). A trained model turns the sentence into generator settings, and the result comes out as a plan, an interactive 3D view, and downloadable PNG, STL and GLB files.

<img width="1200" height="800" alt="Earlier city previews" src="https://github.com/user-attachments/assets/cd4ec8e7-afa3-4948-a106-228dc6bbcdba" />

Current 3D look. Roofs, a window grid, parks, and water. Commit `docs/CityPhoto.png` with this README or the picture will not show.

<img width="1100" height="720" alt="Generated coastal city" src="docs/CityPhoto.png" />

## Overview

Every city is represented as a fixed-size grid (128×128 by default) with four channels:

| Channel | Type | Meaning |
|---|---|---|
| `terrain_height` | float32 | Ground elevation |
| `is_water` | bool | Lakes, rivers, harbors |
| `building_height` | float32 | Building height (0 = no building) |
| `land_use` | uint8 | Building / road / green space / empty ground |

The grid is saved as a `.npz` file, which every stage of the pipeline reads or writes.

<img width="2700" height="750" alt="Pipeline diagram" src="https://github.com/user-attachments/assets/a2b4b1a7-129b-4a21-a861-9fcbd606d795" />

## Pipeline

```
generate_city.py  ->  mycity.npz  ->  visualize_2d.py  ->  mycity_preview.png
                          |
                          +-> export_to_blender.py (runs inside Blender) -> mycity.blend + mycity.stl

"a dense city in the hills..."  ->  text2city model  ->  10 values  ->  generate_city.py command  ->  .npz
                                    (Week 5)                            (Week 6: infer.py)

app.py (Week 7): Prompt mode or Manual mode  ->  the pipeline above  ->  preview + 3D view + downloads
```

1. **Terrain generation** (`citygen/terrain.py`): fractal noise shaped by an archetype, with water marked below the water level. Terrains are flat, hills, mountains, valley, coastal harbor, peninsula, and a winding river that enters on one edge and leaves on another.
2. **Building generation** (`citygen/buildings.py`): cluster centers, urban intensity falloff, no buildings on water or steep slopes, then roads with gaps and green space. Inside a building cluster the street grid is tighter, and those streets can cut through the blocks.
3. **2D preview** (`scripts/visualize_2d.py`): one top-down plan. `--panels` still draws the four heatmaps (terrain, water, building height, land use).
4. **Blender export** (`scripts/export_to_blender.py`): builds terrain and building meshes, colors them by land use and height, and saves a `.blend` and a printable `.stl`.
5. **Text-to-parameters** (`text2city/`): a trained model reads a sentence and predicts the generator's settings.
6. **Sentence to city** (`text2city/infer.py`): turns the prediction into a `generate_city.py` command, prints it, and runs it.
7. **Web app** (`app.py`): a Streamlit interface over all of the above. The browser model adds pitched roofs on low buildings and a simple window grid on the taller ones.

## Quick start

Run all commands from the repo root.

```bash
# 1. Set up
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate a city
python scripts/generate_city.py --size 128 --terrain hills --city polycentric \
  --clusters 3 --density 0.6 --cluster-tightness 0.6 \
  --green-amount 0.25 --road-continuity 0.7 --seed 42 --out data/mycity.npz

# A river city. The channel bends; it is not a straight cut.
python scripts/generate_city.py --size 128 --terrain river --city linear \
  --clusters 3 --density 0.5 --seed 7 --out data/river.npz

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

### Generate a city from a sentence

This needs a trained model first (see [Text-to-parameter model](#text-to-parameter-model-week-5)).

```bash
python text2city/infer.py "a river city, long and narrow, with many parks" --preview
```

It prints the 10 predicted values and the exact `generate_city.py` command they turn into, runs it, and saves `data/from_text.npz` plus a 2D preview.

### Run the web app

```bash
pip install streamlit trimesh
streamlit run app.py
```

Then open `http://localhost:8501` if the browser doesn't open on its own. Press `Ctrl+C` in the terminal to stop it.

## Web app (Week 7)

`app.py` is a Streamlit app that puts the whole pipeline behind one page.

**Prompt mode**
- Type a description, or click one of the example prompts.
- The app runs `text2city/infer.py` and builds the city.
- **New variation** re-runs the same sentence with a new seed.
- **Edit in manual mode** copies the settings it used into the manual controls.

**Manual mode**
- Environment: terrain type (dropdown), relief, water level (or auto), green amount, green override, road continuity.
- City: city type (dropdown), clusters, density, cluster tightness, and a seed with a random-seed button.
- **Generate** runs `scripts/generate_city.py` with those values.

**For every city**
- A top-down plan.
- A draggable, colored 3D view in the browser.
- Downloads for the PNG, the printable STL, and the GLB.
- A "rebuild it from a terminal" command that reproduces the city exactly.
- A gallery in the sidebar of all the cities from the session, plus a grid-size selector (64, 96, 128 or 160).

Every city is saved in `data/gallery/` as `.npz`, `.png`, `.glb` and `.stl`.

**Notes**
- The printable STL comes from `scripts/export_to_blender.py`, so the app needs to find Blender. It checks the `BLENDER` environment variable, then `blender` on your PATH, then the default macOS location. On macOS:
  ```bash
  export BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
  ```
  If Blender isn't found, the app still works and falls back to a basic STL that isn't guaranteed printable. Leave that box unchecked if you only want the browser model.
- The 3D view loads its viewer from the internet, so the browser needs a connection. Large grids make the 3D file heavy, so try 64 or 96 if it is slow.
- The terrain and city dropdowns come from `citygen`, so any archetype you add to `terrain.py` shows up automatically.
- If you add terrain types, regenerate the dataset and retrain the text model, because the stale-checkpoint check includes the terrain list. River was already in that list when the current model was trained, so the new river shape does not require a retrain.

## Generator options

| Flag | What it controls |
|---|---|
| `--size` | Grid size (default 128) |
| `--terrain` | `flat`, `hills`, `mountains`, `valley`, `coastal_harbor`, `peninsula`, `river` |
| `--city` | City archetype (`monocentric`, `polycentric`, `linear`) |
| `--clusters` | Number of city centers |
| `--density` | Overall building density |
| `--cluster-tightness` | How tightly buildings pack around centers |
| `--relief` | How much the terrain height varies |
| `--water-level` | How much of the map is water |
| `--green-amount` | Share of land used for green space |
| `--green-override` | Let green space replace buildings and roads |
| `--road-continuity` | How unbroken the road network is. 0 removes every street, including the denser streets inside clusters |
| `--seed` | Random seed for reproducible cities |

## Text-to-parameter model (Week 5)

A small model that reads a plain-English description and predicts the generator's 10 parameters: terrain, city type, number of clusters, density, cluster tightness, relief, water level, green amount, green override, and road continuity. Parameters the sentence doesn't mention keep their defaults.

**How it works**

- **Synthetic data** (`generate_dataset.py`): builds 60,000 sentences from randomized parameter settings. Each sentence mentions only some of the parameters, the way a real person would, and the rest stay at their defaults. The five prompts in the app are held out of the CSV.
- **Targets** (`dataset.py`): turns the CSV into training targets: class indices for categorical parameters, 0–1 rescaled values for continuous ones, and 0/1 flags for binary ones.
- **Model** (`model.py`): a frozen sentence embedding (`all-MiniLM-L6-v2`) feeds a small shared network with one output head per parameter.
- **Training** (`train.py`): cross-entropy for categorical heads, mean squared error for continuous heads, and binary cross-entropy for the binary head.
- **Evaluation** (`evaluate.py`): two tests. The first uses held-out synthetic sentences. The second uses hand-written sentences with different wording, to check that the model learned meaning and didn't just memorize the phrase templates. The hand-written check, which includes the app prompts, landed at about 85%.
- **Stale-checkpoint guard** (`freshness.py`): the checkpoint stores a fingerprint of the dataset, the model code, and the terrain and city choices from `citygen`. If any of them change after training, `evaluate.py` and `infer.py` warn that the checkpoint is stale.

**Build the model** (from the repo root)

```bash
python text2city/generate_dataset.py --n 60000 --seed 0
python text2city/train.py --embedder sentence-transformer
python text2city/evaluate.py
```

Training writes the model to `text2city/checkpoint/` (`model.pt` and `embedder.txt`). `data/` and `text2city/checkpoint/` are not tracked in git, so everyone rebuilds them with the commands above. The CSV in this repo was already generated at 60,000 rows.

**What we found**

- Held-out synthetic sentences score close to perfect, since they come from the same phrase banks the model trained on.
- The hand-written sentences are the real test. About 85% of those checks passed. The misses are where the wording was not in the phrase banks. The fix is to add phrases, then regenerate and retrain. A bigger dataset from the same banks would not help.

## Sentence to city (Week 6)

`text2city/infer.py` connects the model to the generator.

```bash
python text2city/infer.py "a quiet town on flat land" --dry-run      # print the command only
python text2city/infer.py "a dense city in the hills with three business hubs" --seed 7 --preview
```

| Flag | What it does |
|---|---|
| `--seed` | Same seed and same sentence give the same city (a random one is picked and printed if you leave it out) |
| `--size` | Grid size (default 128) |
| `--out` | Where to save the `.npz` (default `data/from_text.npz`) |
| `--preview` | Also save a 2D PNG next to the `.npz` |
| `--dry-run` | Print the command and stop |
| `--allow-stale` | Use a stale checkpoint anyway |

**Design: the command line is the hand-off.** `infer.py` never calls the generator directly. It predicts the 10 values, converts them to keyword arguments with `to_generator_kwargs()` in `schema.py`, converts those to flags with `kwargs_to_cli_args()`, and prints the resulting `generate_city.py` command as a receipt before running it. If a city looks wrong, copy the printed command, change a number, and run it yourself. That shows which half is wrong: if the predicted numbers are wrong, the model is to blame; if the numbers are right but the city isn't, the generator is. The web app uses the same idea with its "rebuild it from a terminal" line.

**Does the city match the sentence?** Week 5 checked whether the model predicts the right numbers. Week 6 checks the city itself. The notebook measures each generated city (water share, plus building, road and green share of land cells) and compares pairs of sentences that should differ, such as "extremely dense" against "sparse", averaged over three seeds. Each miss is labeled as a model miss or a generator miss.

## Project structure

```
app.py                  # Streamlit web app (Prompt mode + Manual mode)
citygen/
  city_grid.py          # CityGrid: the four-array data contract, save/load/stats
  terrain.py            # generate_terrain(...)
  buildings.py          # generate_buildings(...)
text2city/
  schema.py             # the 10 parameters, and to_generator_kwargs()
  embedders.py          # sentence embedding (and TF-IDF fallback)
  freshness.py          # stale-checkpoint check
  generate_dataset.py   # synthetic sentence + parameter dataset
  dataset.py            # CSV -> training targets
  model.py              # shared trunk + one head per parameter
  train.py              # training loop
  evaluate.py           # synthetic and hand-written tests
  infer.py              # sentence -> generate_city.py command -> city
  checkpoint/           # trained model (not tracked in git)
scripts/
  generate_city.py      # CLI: terrain -> buildings -> .npz
  visualize_2d.py       # 2D preview
  city_look.py          # browser mesh: roofs, windows, trees, water sheet
  export_to_blender.py  # runs inside Blender: .npz -> .blend + .stl
notebooks/
  ...                   # weekly notebooks
data/
  gallery/              # cities made in the web app (not tracked in git)
docs/
  city_preview.png      # the 3D picture at the top of this README
  mdst-f26-poster.png   # the expo poster
  blender_export.md     # how a .npz becomes a .blend
```

`export_to_blender.py` runs in Blender's own Python, which has numpy but not scipy. That is why `citygen/__init__.py` only imports terrain and buildings when scipy is available.

## Demo video

Paste the link here when the recording is up:

<!-- https:// -->

## Poster

<img width="1600" alt="MDST Fall 2026 poster" src="docs/mdst-f26-poster.png" />

## Roadmap

- [x] Procedural terrain and building generators
- [x] Seven terrains, including a winding river from edge to edge
- [x] `.npz` city format (`CityGrid`)
- [x] 2D preview
- [x] Blender export with colored materials and printable STL
- [x] Road-continuity metric
- [x] Denser streets inside building clusters
- [x] Browser buildings with pitched roofs and a simple window grid
- [x] Synthetic training dataset (60,000 sentences)
- [x] Text-to-parameter model (natural-language description to generator parameters)
- [x] Sentence to city: `infer.py` runs the model and the generator end to end
- [x] Visual interface: Streamlit web app with Prompt mode, Manual mode, 3D view, and downloads
- [x] Buildings already follow the terrain: water and steep slopes get no buildings
- [ ] Final expo demo video (paste the link under Demo video)

## Tech stack

Python · NumPy · SciPy · Matplotlib · PyTorch · scikit-learn · sentence-transformers · Streamlit · trimesh · Blender (bpy) · Jupyter · Git/GitHub

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'citygen'` | Run from the repo root |
| `No module named 'numpy'` / `scipy` | Activate the venv and run `pip install -r requirements.txt` |
| `No module named 'scipy'` inside Blender | Use the fixed `citygen/__init__.py`; don't install scipy into Blender |
| Blender quits without saving | Add `--` before the `.npz` path |
| Model looks all grey | Switch viewport shading to Material Preview |
| `No module named 'torch'` / `sentence_transformers` | Activate the venv and run `pip install torch scikit-learn sentence-transformers` |
| `No checkpoint at ...` from `infer.py` | Train one: `python text2city/generate_dataset.py` then `python text2city/train.py` |
| STALE warning from `evaluate.py` or `infer.py` | The dataset, model code, or terrain list changed after training. Re-run `generate_dataset.py` and `train.py` |
| `streamlit: command not found` | Activate the venv, then run `pip install streamlit trimesh` |
| `streamlit run app.py` says the file doesn't exist | Run it from the repo root, where `app.py` lives |
| The web page is blank | `app.py` is empty or unsaved. Check with `wc -l app.py`, paste the code again, and save |
| "Generation failed" in the app | Read the error shown. It usually means a script uses a different flag name than `app.py` expects |
| The STL is not printable from the app | Set the `BLENDER` environment variable (see Web app notes) |
| The prompt box is black with a red outline | Add `.streamlit/config.toml` with `base = "light"` and restart Streamlit |

## Credits

Built with the Michigan Data Science Team (MDST), University of Michigan.
