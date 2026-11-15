"""City Generator web app.  Run from the repo root:  streamlit run app.py

Prompt mode : sentence -> text2city/infer.py -> city
Manual mode : sliders  -> scripts/generate_city.py -> city
Every city is saved in data/gallery/ (npz grid, 2D png, 3D glb, printable stl).
"""
import base64, os, random, re, shlex, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT / "data" / "gallery"
GALLERY.mkdir(parents=True, exist_ok=True)

try:  # choices come from your own terrain.py / buildings.py
    from citygen import TERRAIN_ARCHETYPES, CITY_ARCHETYPES
    TERRAINS, CITIES = list(TERRAIN_ARCHETYPES), list(CITY_ARCHETYPES)
except Exception:
    TERRAINS, CITIES = ["flat", "hills"], ["monocentric", "polycentric", "linear"]

EXAMPLES = [
    "A harbour city with high mountains on a straight line",
    "A hillside city with lots of green space",
    "A dense polycentric city with four business hubs",
    "A sparse little town on flat land with patchy roads",
    "A long linear city stretched along one road",
]

# ---------------------------------------------------------------- helpers
def run(cmd, timeout=900):
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def new_stem(label):
    nums = [int(m.group(1)) for f in GALLERY.glob("city_*.npz")
            if (m := re.match(r"city_(\d+)_", f.name))]
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "city"
    return GALLERY / f"city_{(max(nums) + 1 if nums else 1):03d}_{slug}"


def parse_flags(text):
    """Pull '--flag value' pairs out of the generate_city.py command infer.py prints."""
    line = next((l for l in text.splitlines() if "generate_city.py" in l), "")
    try:
        toks = shlex.split(line.replace("$", " "))
    except ValueError:
        return {}
    flags, i = {}, 0
    while i < len(toks):
        if toks[i].startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[toks[i][2:]] = toks[i + 1]; i += 1
            else:
                flags[toks[i][2:]] = True
        i += 1
    return flags


def title_from(flags, fallback):
    if not flags:
        return fallback
    t, c, k = flags.get("terrain", "?"), flags.get("city", "?"), flags.get("clusters")
    s = f"{t.capitalize()} terrain, {c} city" + (f", {k} cluster{'s' if str(k) != '1' else ''}" if k else "")
    return s


# ------------------------------------------------------------ 3D geometry
def _boxes(x, y, z0, z1, rgb, w=1.0, g=0.0):
    k = len(x)
    x0, x1, y0, y1 = x + g, x + w - g, y + g, y + w - g
    corners = np.stack([
        np.stack([x0, y0, z0], 1), np.stack([x1, y0, z0], 1),
        np.stack([x1, y1, z0], 1), np.stack([x0, y1, z0], 1),
        np.stack([x0, y0, z1], 1), np.stack([x1, y0, z1], 1),
        np.stack([x1, y1, z1], 1), np.stack([x0, y1, z1], 1)], 1)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    faces = (f[None] + (np.arange(k) * 8)[:, None, None]).reshape(-1, 3)
    rgba = np.concatenate([np.asarray(rgb, np.uint8), np.full((k, 1), 255, np.uint8)], 1)
    return corners.reshape(-1, 3), faces, np.repeat(rgba, 8, axis=0)


def city_mesh(npz_path):
    """Grid -> coloured box mesh (vertices, faces, per-vertex RGBA). y-up for glTF."""
    d = np.load(npz_path)
    th = d["terrain_height"].astype(float)
    water = d["is_water"].astype(bool)
    bh = d["building_height"].astype(float)
    lu = d["land_use"]
    n = th.shape[0]
    ground = 1.0 + (th - th.min()) / (np.ptp(th) + 1e-9) * (n * 0.10)
    if water.any():
        ground[water] = ground.min() - 0.4
    ground[(lu == 3) & ~water] += 0.15
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    X, Y, G = xs.ravel(), ys.ravel(), ground.ravel()
    col = np.tile(np.array([222, 205, 170], np.uint8), (n * n, 1))     # empty ground
    col[(lu == 2).ravel()] = (62, 62, 68)                               # roads
    col[(lu == 3).ravel()] = (112, 172, 92)                             # parks
    col[water.ravel()] = (70, 140, 210)                                 # water
    parts = [_boxes(X, Y, np.zeros_like(G), G, col)]
    b = (bh > 0) & ~water
    if b.any():
        heights = bh[b]
        lo, hi = np.percentile(heights, [40, 75])
        bc = np.where(heights[:, None] < lo, np.array([172, 172, 178]),
             np.where(heights[:, None] < hi, np.array([214, 190, 140]), np.array([120, 140, 175])))
        gb = ground[b]
        parts.append(_boxes(xs[b], ys[b], gb, gb + heights, bc, g=0.08))
    verts, faces, cols, off = [], [], [], 0
    for v, f, c in parts:
        verts.append(v); faces.append(f + off); cols.append(c); off += len(v)
    v = np.concatenate(verts); f = np.concatenate(faces); c = np.concatenate(cols)
    v = v[:, [0, 2, 1]].copy(); v[:, 2] *= -1          # z-up -> y-up (a rotation)
    return v, f, c


def export_models(npz, glb, stl, use_blender):
    """GLB for the browser viewer; STL from Blender (closed base) or a fallback."""
    import trimesh
    v, f, c = city_mesh(npz)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=c)
    mesh.export(glb)
    blender = next((p for p in (os.environ.get("BLENDER"), shutil.which("blender"),
                    "/Applications/Blender.app/Contents/MacOS/Blender")
                    if p and Path(p).exists()), None)
    if use_blender and blender:
        run([blender, "--background", "--python", "scripts/export_to_blender.py", "--", npz])
        if Path(stl).exists():
            return "blender"
    mesh.export(stl)
    return "fallback"


def viewer(glb, height=430):
    b64 = base64.b64encode(Path(glb).read_bytes()).decode()
    components.html(
        '<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>'
        f'<model-viewer src="data:model/gltf-binary;base64,{b64}" camera-controls auto-rotate '
        'shadow-intensity="0.6" exposure="1.1" camera-orbit="35deg 60deg 120%" '
        f'style="width:100%;height:{height}px;background:#f7f6f2;border-radius:0"></model-viewer>',
        height=height + 10)


# ------------------------------------------------------------ build a city
def build(label, cmd, stem, prompt=None, use_blender=False, log=""):
    npz, png, glb, stl = (stem.with_suffix(s) for s in (".npz", ".png", ".glb", ".stl"))
    code, out = run(cmd)
    if code != 0 or not npz.exists():
        st.session_state.last_error = out[-2000:]
        return False
    st.session_state.pop("last_error", None)
    flags = parse_flags(out) or parse_flags(" ".join(map(str, cmd)))
    run([sys.executable, "scripts/visualize_2d.py", npz, "--out", png])
    how = export_models(npz, glb, stl, use_blender)
    rebuild = " ".join(shlex.quote(str(c)) for c in cmd).replace(sys.executable, "python")
    rebuild = rebuild.replace(str(ROOT) + "/", "")
    st.session_state.cities.append(dict(
        title=title_from(flags, label), seed=flags.get("seed", "?"), npz=str(npz),
        png=str(png), glb=str(glb), stl=str(stl), stl_how=how, flags=flags,
        prompt=prompt, rebuild=rebuild, log=log or out))
    st.session_state.pending_sel = len(st.session_state.cities) - 1
    return True


def gen_prompt(text, seed, size, use_blender):
    stem = new_stem(text)
    cmd = [sys.executable, "text2city/infer.py", text, "--seed", seed,
           "--size", size, "--out", stem.with_suffix(".npz")]
    with st.spinner("Reading your description and building the city..."):
        build(text, cmd, stem, prompt=text, use_blender=use_blender)


def to_manual(flags):
    s = st.session_state
    s.mode = "Manual mode"
    for key, flag, cast in [("m_terrain", "terrain", str), ("m_city", "city", str),
                            ("m_clusters", "clusters", int), ("m_density", "density", float),
                            ("m_tight", "cluster-tightness", float), ("m_relief", "relief", float),
                            ("m_green", "green-amount", float), ("m_road", "road-continuity", float),
                            ("m_seed", "seed", int)]:
        if flag in flags:
            try: s[key] = cast(flags[flag])
            except (ValueError, TypeError): pass
    s.m_green_override = bool(flags.get("green-override", False))
    s.m_water_auto = "water-level" not in flags
    if "water-level" in flags:
        try: s.m_water = float(flags["water-level"])
        except (ValueError, TypeError): pass


# ------------------------------------------------------------------- UI
st.set_page_config(page_title="Build your city", layout="wide")

S = st.session_state
S.setdefault("cities", [])
if "pending_sel" in S:
    S.sel = S.pop("pending_sel")
for k, v in dict(mode="Prompt mode", prompt="", m_terrain=TERRAINS[0], m_city=CITIES[0],
                 m_clusters=3, m_density=0.6, m_tight=0.6, m_relief=1.0, m_green=0.25,
                 m_road=0.7, m_seed=42, m_green_override=False, m_water_auto=True,
                 m_water=0.2).items():
    S.setdefault(k, v)

st.markdown("""
<style>
:root {
  --bg: #f3f1ec;
  --paper: #f7f6f2;
  --ink: #1a1916;
  --muted: #5f5c56;
  --line: #c8c3b8;
  --rule: #1a1916;
}
html, body, .stApp, [class*="css"] {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: var(--ink);
  background: var(--bg);
}
.stApp { background: var(--bg); }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
  background: transparent !important;
}
#MainMenu, footer { visibility: hidden; }
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3.5rem;
  max-width: 1100px;
}
[data-testid="stSidebar"] {
  background: var(--paper) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebarContent"] { padding-top: 1.1rem; }
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3,
[data-testid="stMarkdown"] p,
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p {
  color: var(--ink);
}

.hero { margin: 0 0 1.1rem; max-width: 40rem; }
.eyebrow {
  margin: 0 0 0.35rem;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.hero h1 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: 500;
  font-size: 2.4rem;
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 0.45rem;
  color: var(--ink);
}
.deck { margin: 0; color: var(--muted); font-size: 1rem; line-height: 1.45; }
.mode-tabs, .composer-anchor, .suggest-anchor { display: none; }

.element-container:has(.mode-tabs) + .element-container [data-testid="stRadio"] [role="radiogroup"],
[data-testid="stElementContainer"]:has(.mode-tabs) + [data-testid="stElementContainer"] [data-testid="stRadio"] [role="radiogroup"] {
  display: grid !important;
  grid-template-columns: 1fr 1fr;
  width: min(420px, 100%);
  gap: 0 !important;
  padding: 0;
  border: 1px solid var(--rule);
  background: var(--paper);
  border-radius: 0;
}
.element-container:has(.mode-tabs) + .element-container [data-testid="stRadio"] label,
[data-testid="stElementContainer"]:has(.mode-tabs) + [data-testid="stElementContainer"] [data-testid="stRadio"] label {
  margin: 0 !important;
  padding: 0.55rem 0.4rem !important;
  border-radius: 0;
  justify-content: center;
  background: transparent;
  border-right: 1px solid var(--rule);
  opacity: 1 !important;
}
.element-container:has(.mode-tabs) + .element-container [data-testid="stRadio"] label *,
[data-testid="stElementContainer"]:has(.mode-tabs) + [data-testid="stElementContainer"] [data-testid="stRadio"] label * {
  color: var(--ink) !important;
  opacity: 1 !important;
  font-weight: 500;
}
.element-container:has(.mode-tabs) + .element-container [data-testid="stRadio"] label:last-child,
[data-testid="stElementContainer"]:has(.mode-tabs) + [data-testid="stElementContainer"] [data-testid="stRadio"] label:last-child {
  border-right: none;
}
.element-container:has(.mode-tabs) + .element-container [data-testid="stRadio"] label > div:first-child > div,
[data-testid="stElementContainer"]:has(.mode-tabs) + [data-testid="stElementContainer"] [data-testid="stRadio"] label > div:first-child > div {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}

.element-container:has(.composer-anchor) + .element-container [data-testid="stHorizontalBlock"],
[data-testid="stElementContainer"]:has(.composer-anchor) + [data-testid="stElementContainer"] [data-testid="stHorizontalBlock"] {
  align-items: stretch;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 0;
  padding: 0;
  gap: 0;
}
[data-testid="stTextInput"] input {
  background: var(--paper) !important;
  border: none !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
  height: 48px !important;
  font-size: 1rem !important;
  padding: 0 12px !important;
}
[data-testid="stTextInput"] input:focus { outline: none !important; box-shadow: none !important; }
[data-testid="stTextInput"] input::placeholder { color: #8a867e !important; }

button, [data-testid="stFormSubmitButton"] button {
  border-radius: 0 !important;
  box-shadow: none !important;
}
button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button {
  background: #1a1916 !important;
  color: #f7f6f2 !important;
  border: 1px solid #1a1916 !important;
  border-radius: 0 !important;
  font-weight: 500 !important;
  min-height: 48px;
  box-shadow: none !important;
}
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span,
[data-testid="stBaseButton-primary"] div,
[data-testid="stBaseButton-primaryFormSubmit"] p,
[data-testid="stBaseButton-primaryFormSubmit"] span,
[data-testid="stBaseButton-primaryFormSubmit"] div,
[data-testid="stFormSubmitButton"] button p,
[data-testid="stFormSubmitButton"] button span,
[data-testid="stFormSubmitButton"] button div {
  color: #f7f6f2 !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: var(--paper) !important;
  color: #1a1916 !important;
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
  text-align: left !important;
  box-shadow: none !important;
}
button[kind="secondary"] p,
button[kind="secondary"] span,
button[kind="secondary"] div,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] span,
[data-testid="stBaseButton-secondary"] div {
  color: #1a1916 !important;
}
button:disabled { opacity: 0.4 !important; }

.suggest-head {
  display: flex;
  justify-content: space-between;
  margin: 0.7rem 0 0.3rem;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.suggest-head b { font-weight: 500; color: var(--muted); }
.element-container:has(.suggest-anchor) + .element-container button,
.element-container:has(.suggest-anchor) + .element-container + .element-container button,
.element-container:has(.suggest-anchor) + .element-container + .element-container + .element-container button,
.element-container:has(.suggest-anchor) + .element-container + .element-container + .element-container + .element-container button,
.element-container:has(.suggest-anchor) + .element-container + .element-container + .element-container + .element-container + .element-container button,
[data-testid="stElementContainer"]:has(.suggest-anchor) + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.suggest-anchor) + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.suggest-anchor) + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.suggest-anchor) + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.suggest-anchor) + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] + [data-testid="stElementContainer"] button {
  width: 100%;
  justify-content: flex-start !important;
  padding: 0.7rem 0.8rem !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--line) !important;
  border-radius: 0 !important;
}

.manual-lead { margin: 0.35rem 0 0.7rem; color: var(--muted); }
.panel-title {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: 500;
  font-size: 1rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin: 0 0 0.45rem;
  color: var(--ink);
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  background: var(--paper) !important;
  color: var(--ink) !important;
  border-color: var(--line) !important;
  border-radius: 0 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] { background: var(--paper) !important; border-radius: 0 !important; }
[data-baseweb="menu"] li { color: var(--ink) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--ink) !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

.side-brand { margin-bottom: 0.9rem; }
.side-brand strong { display: block; font-weight: 500; color: var(--ink); }
.side-brand span { color: var(--muted); font-size: 13px; }
[data-testid="stSidebar"] h3 { font-weight: 500; font-size: 0.95rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border: 1px solid var(--line);
  background: transparent !important;
  border-radius: 0;
  padding: 0.4rem 0.55rem;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label * {
  color: var(--ink) !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  border-color: var(--line);
  background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child > div {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}

.result-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin: 1.4rem 0 0.7rem;
  padding-top: 0.8rem;
  border-top: 1px solid var(--rule);
}
.result-head h2 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: 500;
  font-size: 1.45rem;
  margin: 0;
  color: var(--ink);
}
.seed {
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
}
.block-label {
  display: flex;
  justify-content: space-between;
  margin: 0 0 0.35rem;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.block-label b { font-weight: 500; color: var(--ink); }

[data-testid="stImage"] {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 0;
  padding: 8px;
}
[data-testid="stImage"] img { border-radius: 0; }
iframe { border-radius: 0 !important; }

[data-testid="stDownloadButton"] button {
  border-radius: 0 !important;
  background: #f7f6f2 !important;
  color: #1a1916 !important;
  border: 1px solid #1a1916 !important;
  letter-spacing: 0.04em;
  font-weight: 500 !important;
  box-shadow: none !important;
}
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span,
[data-testid="stDownloadButton"] button div {
  color: #1a1916 !important;
}
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stJson"] {
  background: var(--paper) !important;
  color: var(--ink) !important;
  border-radius: 0 !important;
  border: 1px solid var(--line) !important;
}
[data-testid="stExpander"] details {
  background: transparent;
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
}
[data-testid="stAlert"] {
  background: var(--paper);
  color: var(--ink);
  border: 1px solid var(--rule);
  border-radius: 0;
}
.empty-stage {
  margin-top: 1.2rem;
  min-height: 220px;
  border: 1px solid var(--line);
  background: var(--paper);
  display: grid;
  place-items: center;
  text-align: center;
  padding: 2rem 1rem;
}
.empty-stage strong {
  display: block;
  font-weight: 500;
  font-size: 1.25rem;
  color: var(--ink);
}
.empty-stage span { color: var(--muted); }
@media (max-width: 800px) {
  .hero h1 { font-size: 1.9rem; }
  .result-head { flex-direction: column; align-items: flex-start; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div class="side-brand"><strong>City generator</strong>'
        '<span>Saved cities</span></div>',
        unsafe_allow_html=True)
    st.subheader("Your cities")
    if S.cities:
        S.setdefault("sel", len(S.cities) - 1)
        st.radio("Gallery", range(len(S.cities)), key="sel", label_visibility="collapsed",
                 format_func=lambda i: f"City {i + 1}: {S.cities[i]['title']}")
    else:
        st.caption("Nothing yet. Generate one!")
    st.markdown("**Grid size**")
    st.caption("Bigger = more detail, slower and larger files")
    size = st.radio("Grid size", [64, 96, 128, 160], horizontal=True, label_visibility="collapsed")
    use_blender = st.checkbox("Printable STL via Blender (slower)", value=True)
    st.caption("Every city is saved in data/gallery/: the 2D preview, the grid, "
               "a printable .stl and a coloured .glb.")

st.markdown(
    '<div class="hero">'
    '<p class="eyebrow">City generator</p>'
    '<h1>Build your city</h1>'
    '<p class="deck">Type a place, or set the dials yourself. You get a plan, '
    'a model you can orbit, and files you can keep.</p>'
    '</div>',
    unsafe_allow_html=True)

st.markdown('<div class="mode-tabs"></div>', unsafe_allow_html=True)
st.radio("Mode", ["Prompt mode", "Manual mode"], horizontal=True, key="mode",
         label_visibility="collapsed")

if S.get("last_error"):
    st.error("Generation failed. Last output:")
    st.code(S.last_error)

if S.mode == "Prompt mode":
    st.markdown('<div class="composer-anchor"></div>', unsafe_allow_html=True)
    with st.form("prompt_form", border=False):
        q, go = st.columns([5, 1])
        with q:
            st.text_input("Describe a city", key="prompt", label_visibility="collapsed",
                          placeholder="e.g. a dense city in the hills with three business hubs")
        with go:
            submitted = st.form_submit_button("Generate", type="primary",
                                              use_container_width=True)
    if submitted and S.prompt.strip():
        gen_prompt(S.prompt.strip(), random.randint(1, 9999), size, use_blender)
        st.rerun()
    st.markdown(
        '<div class="suggest-head"><b>Suggested</b><span>fills the prompt</span></div>'
        '<div class="suggest-anchor"></div>',
        unsafe_allow_html=True)
    for ex in EXAMPLES:
        st.button(ex, key=f"ex_{ex}", on_click=lambda e=ex: S.update(prompt=e))
else:
    st.markdown(
        '<p class="manual-lead">Same generator, set by hand. Terrain on the left, city on the right.</p>',
        unsafe_allow_html=True)
    e, c = st.columns(2)
    with e:
        st.markdown('<div class="panel-title">Environment</div>', unsafe_allow_html=True)
        st.selectbox("Terrain type", TERRAINS, key="m_terrain")
        st.slider("Relief (how hilly)", 0.0, 2.0, key="m_relief")
        st.checkbox("Water level: auto (terrain default)", key="m_water_auto")
        st.slider("Water level", 0.0, 0.6, key="m_water", disabled=S.m_water_auto)
        st.slider("Green amount", 0.0, 1.0, key="m_green")
        st.checkbox("Green override (parks replace buildings)", key="m_green_override")
        st.slider("Road continuity (0 patchy, 1 full grid)", 0.0, 1.0, key="m_road")
    with c:
        st.markdown('<div class="panel-title">City</div>', unsafe_allow_html=True)
        st.selectbox("City type", CITIES, key="m_city")
        st.slider("Clusters", 1, 6, key="m_clusters")
        st.slider("Density", 0.0, 1.0, key="m_density")
        st.slider("Cluster tightness", 0.0, 1.0, key="m_tight")
        st.number_input("Seed", 0, 99999, key="m_seed")
        st.button("Random seed", on_click=lambda: S.update(m_seed=random.randint(1, 9999)))
    if st.button("Generate", type="primary", key="gen_manual"):
        label = f"Manual {S.m_terrain} {S.m_city}"
        stem = new_stem(label)
        cmd = [sys.executable, "scripts/generate_city.py", "--size", size,
               "--terrain", S.m_terrain, "--city", S.m_city, "--clusters", S.m_clusters,
               "--density", S.m_density, "--cluster-tightness", S.m_tight,
               "--relief", S.m_relief, "--green-amount", S.m_green,
               "--road-continuity", S.m_road, "--seed", S.m_seed,
               "--out", stem.with_suffix(".npz")]
        if not S.m_water_auto:
            cmd += ["--water-level", S.m_water]
        if S.m_green_override:
            cmd += ["--green-override"]
        with st.spinner("Generating..."):
            build(label, cmd, stem, use_blender=use_blender)
        st.rerun()

if S.cities:
    city = S.cities[min(S.get("sel", 0), len(S.cities) - 1)]
    _t = str(city["title"]).replace("&", "&").replace("<", "<")
    _sd = str(city["seed"]).replace("&", "&").replace("<", "<")
    st.markdown(
        '<div class="result-head"><h2>' + _t + '</h2>'
        '<span class="seed">Seed ' + _sd + '</span></div>',
        unsafe_allow_html=True)
    plan, stage = st.columns([3, 2])
    with plan:
        st.markdown('<div class="block-label"><b>Plan</b><span>2D grid</span></div>',
                    unsafe_allow_html=True)
        if Path(city["png"]).exists():
            st.image(city["png"], use_container_width=True)
        st.caption("Rebuild it from a terminal:")
        st.code(city["rebuild"], language="bash")
        with st.expander("Parameters used / generator output"):
            st.json(city["flags"]) if city["flags"] else None
            st.code(city["log"][-1500:])
    with stage:
        st.markdown('<div class="block-label"><b>Model</b><span>drag to rotate</span></div>',
                    unsafe_allow_html=True)
        if Path(city["glb"]).exists():
            viewer(city["glb"])
        d1, d2, d3 = st.columns(3)
        for col, key, mime in ((d1, "png", "image/png"), (d2, "stl", "model/stl"),
                               (d3, "glb", "model/gltf-binary")):
            p = Path(city[key])
            if p.exists():
                col.download_button(key.upper(), p.read_bytes(), file_name=p.name, mime=mime,
                                    key=f"dl_{key}_{p.name}")
        if city["stl_how"] == "fallback" and use_blender:
            st.caption("Blender wasn't found, so this STL is a basic export. "
                       "Set the BLENDER environment variable to its path for the printable one.")
        a, b = st.columns(2)
        if city["prompt"] and a.button("New variation (new seed)"):
            gen_prompt(city["prompt"], random.randint(1, 9999), size, use_blender)
            st.rerun()
        b.button("Edit in manual mode", type="primary", on_click=to_manual, args=(city["flags"],))
else:
    st.markdown(
        '<div class="empty-stage"><div><strong>The grid is clear.</strong>'
        '<span>Describe a city, or open manual mode. The plan and the model land here.</span>'
        '</div></div>',
        unsafe_allow_html=True)
