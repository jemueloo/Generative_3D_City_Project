"""Turn a CityGrid .npz into one coloured mesh for the browser viewer.

Ground is a surface, not a stack of boxes. Neighbouring buildings of similar
height become one footprint. Low blocks get a pitched roof, tall ones a
setback. Parks get a few cones. Water is a flat sheet.
"""
from __future__ import annotations

import numpy as np

# land_use codes, duplicated here so this file imports with numpy only
# (Blender does not ship scipy, and citygen pulls it in).
_EMPTY, _ROAD, _GREEN = 0, 2, 3

_GROUND = {
    _EMPTY: np.array([214, 196, 158], np.uint8),
    _ROAD: np.array([52, 54, 58], np.uint8),
    _GREEN: np.array([74, 146, 70], np.uint8),
}
_WATER = np.array([52, 122, 184], np.uint8)
_WALL = {
    "short": np.array([198, 170, 142], np.uint8),
    "mid": np.array([132, 138, 146], np.uint8),
    "tall": np.array([74, 92, 110], np.uint8),
}
_ROOF = np.array([138, 74, 58], np.uint8)
_GLASS = np.array([186, 206, 216], np.uint8)
_WINDOW = np.array([226, 236, 242], np.uint8)
_TRUNK = np.array([94, 66, 42], np.uint8)
_LEAF = np.array([42, 108, 54], np.uint8)


def terrain_z(th: np.ndarray, water: np.ndarray, lu: np.ndarray) -> np.ndarray:
    """Visual height of each cell. Water is one flat level. Roads sit slightly lower."""
    n = th.shape[0]
    z = 1.0 + (th - th.min()) / (np.ptp(th) + 1e-9) * (n * 0.10)
    z = z.astype(np.float64)
    z[(lu == _ROAD) & ~water] -= 0.10
    z[(lu == _GREEN) & ~water] += 0.06
    if water.any():
        shore = np.zeros_like(water)
        shore[1:] |= water[:-1]
        shore[:-1] |= water[1:]
        shore[:, 1:] |= water[:, :-1]
        shore[:, :-1] |= water[:, 1:]
        shore &= ~water
        level = float(np.median(z[shore]) - 0.08) if shore.any() else float(z.min())
        z[water] = level
    return z


def building_footprints(bh: np.ndarray, water: np.ndarray, ground: np.ndarray):
    """Rectangles of similar-height neighbours. Each is (r, c, h, w, z0, z1, tier).

    r, c are the array indices of the corner (row, column). h grows along rows,
    w along columns. z is in the same units as ``ground``.
    """
    sel = (bh > 0) & ~water
    if not np.any(sel):
        return []
    heights = bh[sel]
    lo, hi = np.percentile(heights, [40, 75])
    bins = np.zeros(bh.shape, np.int16)
    bins[sel] = 1 + np.floor(np.clip(bh[sel], 0, None) / 2.0).astype(np.int16)
    claimed = np.zeros(bh.shape, dtype=bool)
    n = bh.shape[0]
    out = []
    rs, cs = np.nonzero(bins)
    for r, c in zip(rs.tolist(), cs.tolist()):
        if claimed[r, c]:
            continue
        b = bins[r, c]
        w = 1
        while c + w < n and bins[r, c + w] == b and not claimed[r, c + w]:
            w += 1
        h = 1
        while r + h < n and np.all(bins[r + h, c:c + w] == b) and not np.any(claimed[r + h, c:c + w]):
            h += 1
        claimed[r:r + h, c:c + w] = True
        block_h = bh[r:r + h, c:c + w]
        block_g = ground[r:r + h, c:c + w]
        height = float(block_h[block_h > 0].mean())
        z0 = float(block_g.min()) - 0.06
        z1 = float(block_g.max()) + height
        if height < lo:
            tier = "short"
        elif height < hi:
            tier = "mid"
        else:
            tier = "tall"
        out.append((int(r), int(c), int(h), int(w), z0, z1, tier))
    return out


def _pack(parts):
    verts, faces, cols, off = [], [], [], 0
    for v, f, c in parts:
        if len(v) == 0:
            continue
        verts.append(v)
        faces.append(f + off)
        cols.append(c)
        off += len(v)
    v = np.concatenate(verts)
    f = np.concatenate(faces)
    c = np.concatenate(cols)
    v = v[:, [0, 2, 1]].copy()
    v[:, 2] *= -1
    return v, f, c


def _surface(z, lu, water):
    n = z.shape[0]
    r = np.arange(n - 1)
    cc, rr = np.meshgrid(r, r)
    rr = rr.ravel()
    cc = cc.ravel()
    i00 = rr * n + cc
    i10 = i00 + 1
    i01 = i00 + n
    i11 = i01 + 1
    gy, gx = np.mgrid[0:n, 0:n]
    verts = np.stack([gy.ravel().astype(float), gx.ravel().astype(float), z.ravel()], 1)
    f1 = np.stack([i00, i11, i10], 1)
    f2 = np.stack([i00, i01, i11], 1)
    faces = np.concatenate([f1, f2], 0)
    shade = 0.78 + 0.28 * (z - z.min()) / (np.ptp(z) + 1e-9)
    col = np.empty((n, n, 3), np.float64)
    col[:] = _GROUND[_EMPTY]
    col[lu == _ROAD] = _GROUND[_ROAD]
    col[lu == _GREEN] = _GROUND[_GREEN]
    col[water] = _WATER
    col *= shade[:, :, None]
    rgba = np.concatenate([
        np.clip(col, 0, 255).astype(np.uint8).reshape(-1, 3),
        np.full((n * n, 1), 255, np.uint8),
    ], 1)
    # skirt so the edge isn't a paper-thin sheet
    parts = [(verts, faces, rgba)]
    skirt_v, skirt_f = [], []
    def edge(a, b):
        skirt_v.append((a[0], a[1], a[2]))
        skirt_v.append((b[0], b[1], b[2]))
        skirt_v.append((b[0], b[1], 0.0))
        skirt_v.append((a[0], a[1], 0.0))
        base = len(skirt_v) - 4
        skirt_f.append((base, base + 1, base + 2))
        skirt_f.append((base, base + 2, base + 3))
    for i in range(n - 1):
        edge(verts[i], verts[i + 1])                          # row 0
        edge(verts[(n - 1) * n + i + 1], verts[(n - 1) * n + i])
        edge(verts[(i + 1) * n], verts[i * n])                # col 0
        edge(verts[i * n + (n - 1)], verts[(i + 1) * n + (n - 1)])
    sv = np.array(skirt_v, np.float64)
    sf = np.array(skirt_f, np.int64)
    sc = np.tile(np.array([186, 168, 136, 255], np.uint8), (len(sv), 1))
    parts.append((sv, sf, sc))
    return parts


def _one_gable(x0, x1, y0, y1, z0, z1, rise):
    """Closed block with a ridge along its longer side. 10 verts."""
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    walls = [(0, 2, 1), (0, 3, 2),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    if (x1 - x0) >= (y1 - y0):
        my = 0.5 * (y0 + y1)
        verts += [(x0, my, z1 + rise), (x1, my, z1 + rise)]  # 8, 9
        roof = [(4, 5, 9), (4, 9, 8),   # slope toward y0
                (7, 8, 9), (7, 9, 6),   # slope toward y1
                (4, 8, 7),              # gable at x0
                (5, 6, 9)]              # gable at x1
    else:
        mx = 0.5 * (x0 + x1)
        verts += [(mx, y0, z1 + rise), (mx, y1, z1 + rise)]  # 8 at y0, 9 at y1
        roof = [(4, 8, 7), (7, 8, 9),   # slope toward x0
                (5, 6, 9), (5, 9, 8),   # slope toward x1
                (4, 5, 8),              # gable at y0
                (6, 7, 9)]              # gable at y1
    return verts, walls + roof


def _push_windows(x0, y0, x1, y1, z0, z1, acc, rows_cap):
    """A plain grid of window boxes, slightly proud of each wall."""
    xs, ys, ws, ds, za, zb = acc
    width, depth, height = x1 - x0, y1 - y0, z1 - z0
    if height < 0.8 or width < 0.35 or depth < 0.35:
        return
    rows = int(np.clip(round(height / 0.95), 1, rows_cap))
    proud = 0.045

    def emit(x, y, w, d, z_a, z_b):
        xs.append(x); ys.append(y); ws.append(w); ds.append(d); za.append(z_a); zb.append(z_b)

    z_margin = min(height * 0.12, 0.35)
    z_span = max(height - 2 * z_margin, 0.2)
    z_step = z_span / rows
    pane_h = z_step * 0.58

    def cols(span):
        return int(np.clip(round(span / 0.75), 1, 4))

    ny = cols(depth)
    y_margin = depth * 0.14
    y_span = max(depth - 2 * y_margin, depth * 0.4)
    y_step = y_span / ny
    pane_w = y_step * 0.55
    for i in range(ny):
        y = y0 + y_margin + i * y_step + (y_step - pane_w) * 0.5
        for r in range(rows):
            z = z0 + z_margin + r * z_step
            emit(x0 - proud, y, proud, pane_w, z, z + pane_h)
            emit(x1, y, proud, pane_w, z, z + pane_h)

    nx = cols(width)
    x_margin = width * 0.14
    x_span = max(width - 2 * x_margin, width * 0.4)
    x_step = x_span / nx
    pane_d = x_step * 0.55
    for i in range(nx):
        x = x0 + x_margin + i * x_step + (x_step - pane_d) * 0.5
        for r in range(rows):
            z = z0 + z_margin + r * z_step
            emit(x, y0 - proud, pane_d, proud, z, z + pane_h)
            emit(x, y1, pane_d, proud, z, z + pane_h)


def _buildings(prints, gap=0.07):
    gab_v, gab_f, box_x, box_y, box_z0, box_z1, box_w, box_d, box_rgb = [], [], [], [], [], [], [], [], []
    top_x, top_y, top_z0, top_z1, top_w, top_d = [], [], [], [], [], []
    win = ([], [], [], [], [], [])
    for r, c, h, w, z0, z1, tier in prints:
        x0, x1 = r + gap, r + h - gap
        y0, y1 = c + gap, c + w - gap
        if x1 <= x0 or y1 <= y0:
            continue
        if tier == "short":
            rise = min(0.55 * min(x1 - x0, y1 - y0), 2.0)
            verts, faces = _one_gable(x0, x1, y0, y1, z0, z1, rise)
            base = len(gab_v)
            gab_v.extend(verts)
            gab_f.extend([(base + a, base + b, base + c) for a, b, c in faces])
            _push_windows(x0, y0, x1, y1, z0, z1, win, rows_cap=2)
        elif tier == "tall" and min(h, w) >= 2:
            span = z1 - z0
            cut = z0 + span * 0.70
            box_x.append(x0); box_y.append(y0); box_z0.append(z0); box_z1.append(cut)
            box_w.append(x1 - x0); box_d.append(y1 - y0); box_rgb.append(_WALL["tall"])
            inset = 0.16 * min(x1 - x0, y1 - y0)
            top_x.append(x0 + inset); top_y.append(y0 + inset)
            top_z0.append(cut - 0.04); top_z1.append(z1)
            top_w.append((x1 - x0) - 2 * inset); top_d.append((y1 - y0) - 2 * inset)
            _push_windows(x0, y0, x1, y1, z0, cut, win, rows_cap=6)
            _push_windows(x0 + inset, y0 + inset, x1 - inset, y1 - inset, cut, z1, win, rows_cap=4)
        else:
            box_x.append(x0); box_y.append(y0); box_z0.append(z0); box_z1.append(z1)
            box_w.append(x1 - x0); box_d.append(y1 - y0); box_rgb.append(_WALL[tier])
            _push_windows(x0, y0, x1, y1, z0, z1, win, rows_cap=5 if tier == "tall" else 3)
    parts = []
    if gab_v:
        gv = np.array(gab_v, np.float64)
        gf = np.array(gab_f, np.int64)
        gc = np.zeros((len(gv), 4), np.uint8)
        gc[:, :3] = _WALL["short"]
        gc[:, 3] = 255
        # ridge vertices are indices 8 and 9 of every 10
        ridge = np.zeros(len(gv), bool)
        ridge[8::10] = True
        ridge[9::10] = True
        gc[ridge, :3] = _ROOF
        parts.append((gv, gf, gc))
    if box_x:
        parts.append(_rect_boxes(
            np.array(box_x), np.array(box_y), np.array(box_w), np.array(box_d),
            np.array(box_z0), np.array(box_z1), np.array(box_rgb)))
    if top_x:
        parts.append(_rect_boxes(
            np.array(top_x), np.array(top_y), np.array(top_w), np.array(top_d),
            np.array(top_z0), np.array(top_z1), np.tile(_GLASS, (len(top_x), 1))))
    if win[0]:
        parts.append(_rect_boxes(
            np.array(win[0]), np.array(win[1]), np.array(win[2]), np.array(win[3]),
            np.array(win[4]), np.array(win[5]), np.tile(_WINDOW, (len(win[0]), 1))))
    return parts


def _rect_boxes(x, y, w, d, z0, z1, rgb):
    """Axis-aligned boxes. x, y are the min corner; w, d are the sizes."""
    k = len(x)
    if k == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 4), np.uint8)
    x1, y1 = x + w, y + d
    corners = np.stack([
        np.stack([x, y, z0], 1), np.stack([x1, y, z0], 1),
        np.stack([x1, y1, z0], 1), np.stack([x, y1, z0], 1),
        np.stack([x, y, z1], 1), np.stack([x1, y, z1], 1),
        np.stack([x1, y1, z1], 1), np.stack([x, y1, z1], 1)], 1)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    faces = (faces[None] + (np.arange(k) * 8)[:, None, None]).reshape(-1, 3)
    rgba = np.concatenate([np.asarray(rgb, np.uint8), np.full((k, 1), 255, np.uint8)], 1)
    return corners.reshape(-1, 3), faces, np.repeat(rgba, 8, axis=0)


def _trees(lu, water, z):
    cells = np.argwhere((lu == _GREEN) & ~water)
    if len(cells) == 0:
        return []
    rng = np.random.default_rng(int(z.sum() * 10) % (2 ** 31))
    if len(cells) > 420:
        cells = cells[rng.choice(len(cells), 420, replace=False)]
    r = cells[:, 0].astype(float)
    c = cells[:, 1].astype(float)
    jitter = rng.uniform(0.28, 0.72, size=(len(cells), 2))
    cx, cy = r + jitter[:, 0], c + jitter[:, 1]
    zb = z[cells[:, 0], cells[:, 1]]
    trunk_h = rng.uniform(0.25, 0.45, size=len(cells))
    leaf_h = rng.uniform(1.0, 2.1, size=len(cells))
    leaf_r = rng.uniform(0.22, 0.40, size=len(cells))
    parts = [_rect_boxes(cx - 0.055, cy - 0.055, np.full(len(cx), 0.11), np.full(len(cx), 0.11),
                         zb, zb + trunk_h, np.tile(_TRUNK, (len(cx), 1)))]
    parts.append(_cones(cx, cy, zb + trunk_h * 0.7, leaf_r, leaf_h, _LEAF))
    return parts


def _cones(cx, cy, zb, radius, height, rgb, sides=6):
    k = len(cx)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([
        cx[:, None] + radius[:, None] * np.cos(ang),
        cy[:, None] + radius[:, None] * np.sin(ang),
        np.broadcast_to(zb[:, None], (k, sides)).copy(),
    ], 2)
    apex = np.stack([cx, cy, zb + height], 1)
    verts = np.concatenate([ring, apex[:, None, :]], 1).reshape(-1, 3)
    side = [(i, (i + 1) % sides, sides) for i in range(sides)]
    bottom = [(0, i, (i + 1) % sides) for i in range(1, sides - 1)]
    local = np.array(side + bottom, np.int64)
    faces = (local[None] + (np.arange(k) * (sides + 1))[:, None, None]).reshape(-1, 3)
    rgba = np.tile(np.append(np.asarray(rgb, np.uint8), 255), (len(verts), 1))
    return verts, faces, rgba


def city_mesh(npz_path):
    """Vertices, faces, per-vertex RGBA. Y-up, for glTF."""
    d = np.load(npz_path)
    th = d["terrain_height"].astype(float)
    water = d["is_water"].astype(bool)
    bh = d["building_height"].astype(float)
    lu = d["land_use"]
    z = terrain_z(th, water, lu)
    parts = _surface(z, lu, water)
    parts += _buildings(building_footprints(bh, water, z))
    parts += _trees(lu, water, z)
    return _pack(parts)
