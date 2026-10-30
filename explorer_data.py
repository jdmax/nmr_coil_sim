"""
explorer_data.py -- turn a coil description into the data the explorer pages draw.

A coil is described by a small JSON-able dict ("spec"), e.g.

    {"type": "saddle", "name": "Saddle 120°, on cell",
     "radius": 8.5, "length": 50, "arc_deg": 120, "turns": 1}

and a cell by {"diam": 15.0, "length": 50.0}. compute_entry(spec, cell) returns
field maps (z-averaged B_T^2 over the cross-section, B_T^2 on the cell surface),
the wire geometry and the circuit numbers, packed for JavaScript.

Used by build_explorer.py / build_web_data.py (full resolution, run offline) and by
the GitHub Pages front end, where it runs in Pyodide (QUALITY["web"]). Only numpy is
needed here.

Coil types and their parameters (mm, degrees):
    spiral          r_in, r_out, turns, height        flat pancake coil in a plane
                                                      `height` above the cell axis
    loop            radius, height, pair (bool),      circular loop(s) above (and
                    turns                             below, if pair) the cell
    rect_loop       width, height_dim, standoff,      flat rectangle in the plane
                    turns                             y = standoff, long side on z
Turns on loop and rect_loop are stacked 0.6 mm apart along the normal.
    saddle          radius, length, arc_deg, turns
    alderman_grant  radius, length, strip_deg, filaments
    birdcage        radius, length, rungs
All single-sided coils sit above the cell (+y); volume coils make their field along y.
"""
from __future__ import annotations

import base64
import hashlib
import json
import numpy as np
import coilsim as cs

F_P, F_D = 212.6e6, 32.6e6     # proton and deuteron Larmor frequencies at 5 T

QUALITY = {
    # pixel/nz: z-averaged map; nphi/nzs/ncap: surface map; grid: volume grid for metrics
    "full": dict(pixel=0.15, nz=21, nphi=72, nzs=36, ncap=8, grid=(30, 72, 25)),
    "web": dict(pixel=0.25, nz=11, nphi=48, nzs=24, ncap=6, grid=(15, 36, 11)),
}

TYPES = {
    "spiral": dict(label="Flat spiral above the cell",
                   params=[("r_in", "Inner radius", "mm", 7.0), ("r_out", "Outer radius", "mm", 13.0),
                           ("turns", "Turns", "", 3), ("height", "Plane height above axis", "mm", 16.0)]),
    "loop": dict(label="Circular loop(s) above the cell",
                 params=[("radius", "Loop radius", "mm", 10.0), ("height", "Plane height above axis", "mm", 16.0),
                         ("pair", "Second loop below, in series", "bool", 0), ("turns", "Turns", "", 1)]),
    "rect_loop": dict(label="Flat rectangular loop above the cell",
                      params=[("width", "Length along beam", "mm", 40.0), ("height_dim", "Width across", "mm", 15.0),
                              ("standoff", "Plane height above axis", "mm", 12.0), ("turns", "Turns", "", 1)]),
    "saddle": dict(label="Saddle (volume)",
                   params=[("radius", "Radius", "mm", 8.5), ("length", "Length", "mm", 50.0),
                           ("arc_deg", "Window arc", "deg", 120.0), ("turns", "Turns", "", 1)]),
    "alderman_grant": dict(label="Alderman–Grant (volume)",
                           params=[("radius", "Radius", "mm", 8.5), ("length", "Length", "mm", 50.0),
                                   ("strip_deg", "Strip width", "deg", 60.0), ("filaments", "Filaments per strip", "", 5)]),
    "birdcage": dict(label="Birdcage, uniform mode (volume)",
                     params=[("radius", "Radius", "mm", 8.5), ("length", "Length", "mm", 50.0),
                             ("rungs", "Rungs", "", 8)]),
}


def make_coil(spec: dict) -> cs.Coil:
    t = spec["type"]
    name = spec.get("name") or TYPES[t]["label"]
    up = (0, 0, 1)
    if t == "spiral":
        c = cs.spiral(spec["r_in"], spec["r_out"], spec["turns"], center=(0, spec["height"], 0),
                      normal=(0, 1, 0))
    elif t == "loop":
        n = int(spec.get("turns", 1))
        c = cs.circle_loop(spec["radius"], center=(0, spec["height"], 0), normal=(0, 1, 0), turns=n)
        if spec.get("pair"):
            c = c + cs.circle_loop(spec["radius"], center=(0, -spec["height"], 0), normal=(0, 1, 0), turns=n)
    elif t == "rect_loop":
        c = cs.rect_loop(spec["width"], spec["height_dim"], center=(0, spec["standoff"], 0),
                         normal=(0, 1, 0), width_dir=(0, 0, 1), turns=int(spec.get("turns", 1)))
    elif t == "saddle":
        c = cs.saddle(spec["radius"], spec["length"], spec["arc_deg"], turns=int(spec.get("turns", 1)))
        c = c.rotated(up, 90)
    elif t == "alderman_grant":
        c = cs.alderman_grant(spec["radius"], spec["length"], spec["strip_deg"],
                              n_filaments=int(spec.get("filaments", 5))).rotated(up, 90)
    elif t == "birdcage":
        c = cs.birdcage(spec["radius"], spec["length"], int(spec["rungs"])).rotated(up, 90)
    else:
        raise ValueError(f"unknown coil type {t!r}")
    return c.renamed(name)


def presets(diam: float, length: float = 50.0) -> list[dict]:
    """The coils of the RGC study, scaled to a cell of the given diameter."""
    Rc = diam / 2
    h = Rc + 8.5            # RGC coil plane: 16 mm above the axis for the 15 mm cell (fitted)
    on_cell, on_bath = Rc + 1.0, Rc + 4.5
    return [
        dict(key="rgc_d", group="RGC (as run)", type="spiral", name="RGC deuteron spiral",
             r_in=7.0, r_out=13.0, turns=3, height=h,
             note=f"3-turn flat spiral, r 7–13 mm, plane {h:.1f} mm above the axis. Fitted to the "
                  "CLAS Note Fig. 8 fractions (0.56 / 0.43) for the 15 mm cell, not measured."),
        dict(key="rgc_p", group="RGC (as run)", type="loop", name="RGC proton loop",
             radius=10.0, height=h, pair=0,
             note="Single circular turn, r 10 mm, same plane as the spiral. Stands in for the "
                  "teardrop-shaped proton loop."),
        dict(key="pair", group="Cheap upgrade", type="loop", name="Loop pair, above + below",
             radius=10.0, height=h, pair=1,
             note="Two single loops, above and below the cell, in series so their fields add."),
        dict(key="rect_in", group="Embedded", type="rect_loop", name="Rectangle 6 × 20 mm, in cell",
             width=20.0, height_dim=6.0, standoff=0.0,
             note="Flat single-turn rectangle in the cell's mid-plane (y = 0), 20 mm along the beam "
                  "and 6 mm across, embedded in the material at the center of the cell."),
        dict(key="rect3_in", group="Embedded", type="rect_loop", name="Rectangle 6 × 20 mm, 3 turns, in cell",
             width=20.0, height_dim=6.0, standoff=0.0, turns=3,
             note="The same embedded rectangle with 3 turns stacked 0.6 mm apart across the mid-plane. "
                  "Turns raise L about as N² at nearly the same η, so more of the tank's energy sits in the material."),
        dict(key="sad_c", group="Volume coils", type="saddle", name="Saddle 120°, on cell",
             radius=on_cell, length=length, arc_deg=120.0, turns=1,
             note=f"Single-turn saddle, 120° windows, on the cell wall (r {on_cell:.1f} mm)."),
        dict(key="sad_b", group="Volume coils", type="saddle", name="Saddle 120°, on bath",
             radius=on_bath, length=length, arc_deg=120.0, turns=1,
             note=f"Single-turn saddle, 120° windows, on the bath (r {on_bath:.1f} mm)."),
        dict(key="ag_c", group="Volume coils", type="alderman_grant", name="Alderman–Grant, on cell",
             radius=on_cell, length=length, strip_deg=60.0, filaments=5,
             note=f"Alderman–Grant, 60° strips as 5 filaments, guard rings at the cell ends "
                  f"(r {on_cell:.1f} mm). A solid sleeve would block the microwaves."),
        dict(key="bc8_c", group="Volume coils", type="birdcage", name="Birdcage 8-rung, on cell",
             radius=on_cell, length=length, rungs=8,
             note=f"8-rung birdcage in its uniform mode (r {on_cell:.1f} mm). Tuned by capacitors "
                  "in its own rungs or rings."),
        dict(key="bc16_b", group="Volume coils", type="birdcage", name="Birdcage 16-rung, on bath",
             radius=on_bath, length=length, rungs=16,
             note=f"16-rung birdcage in its uniform mode (r {on_bath:.1f} mm). Tuned by capacitors "
                  "in its own rungs or rings."),
    ]


def _b64f(a):
    return base64.b64encode(np.asarray(a, np.float32).ravel().tobytes()).decode()


def _b64u8(a):
    return base64.b64encode(np.asarray(a, np.uint8).ravel().tobytes()).decode()


def compute_entry(spec: dict, cell: dict, quality: str = "full") -> dict:
    """Field maps, wires and circuit numbers for one coil in one cell."""
    q = QUALITY[quality]
    Rc = cell["diam"] / 2
    nr, nphi, nz = q["grid"]
    c3 = cs.Cell(radius=Rc, length=cell["length"], nr=nr, nphi=nphi, nz=nz)
    coil = make_coil(spec)
    m = cs.zavg_map(coil, c3, pixel=q["pixel"], nz=q["nz"])
    pts, w = c3.grid()
    B = coil.field(pts)                       # one volume evaluation, shared below
    norm = float(np.sum(w * cs._bt2(B)) / w.sum())
    s = cs.surface_map(coil, c3, nphi=q["nphi"], nz=q["nzs"], ncap=q["ncap"], norm=norm)
    L = coil.inductance()
    rep = cs.sampling_report(coil, c3, raster_radius=max(Rc - 1.5, 0.5), freq=F_P,
                             _cache=dict(B=B, L=L))
    entry = dict(
        key=spec.get("key") or "user_" + hashlib.md5(
            (json.dumps(spec, sort_keys=True) + json.dumps(cell, sort_keys=True)).encode()).hexdigest()[:10],
        name=coil.name, group=spec.get("group", "Your coils"),
        note=spec.get("note", ""), spec={k: v for k, v in spec.items() if k not in ("note", "group")},
        L_nH=L * 1e9,
        C_p_pF=1e12 / ((2 * np.pi * F_P) ** 2 * L),
        C_d_pF=1e12 / ((2 * np.pi * F_D) ** 2 * L),
        Q_p=rep["Q_coil"], Q_d=2 * np.pi * F_D * L / coil.ac_resistance(F_D),
        eta=rep["eta"], cv=rep["cv"], len_lambda=rep["path_over_lambda"],
        map=dict(n=len(m["x"]), pixel=m["pixel"], x0=float(m["x"][0]),
                 v=_b64f(m["v"]), w=_b64u8(np.round(m["w"] * 255))),
        surf=dict(nphi=len(s["phi"]), nz=len(s["z"]), ncap=len(s["r"]),
                  side=_b64f(s["side"]), lo=_b64f(s["cap_lo"]), hi=_b64f(s["cap_hi"])),
        wires=dict(seg=_b64f(np.hstack([coil.starts, coil.ends])),
                   cur=_b64f(np.abs(coil.currents))),
    )
    if spec["type"] == "birdcage":
        # Low-pass birdcage: one capacitor per rung with I_k = I_max sin(phi_k); equating
        # stored energies gives w^2 = (N/2)/(L_eff C), so C_rung = (N/2) C_single.
        N = int(spec["rungs"])
        entry.update(rungs=N, C_rung_p_pF=N / 2 * entry["C_p_pF"], C_rung_d_pF=N / 2 * entry["C_d_pF"])
    return entry


def compute_entry_json(spec_json: str, cell_json: str, quality: str = "web") -> str:
    """JSON in, JSON out: the call the Pyodide worker makes."""
    return json.dumps(compute_entry(json.loads(spec_json), json.loads(cell_json), quality),
                      separators=(",", ":"))


def build_cell(diam: float, length: float = 50.0, quality: str = "full", log=print) -> dict:
    out = []
    for spec in presets(diam, length):
        e = compute_entry(spec, dict(diam=diam, length=length), quality)
        e["note"] = spec["note"]
        e["group"] = spec["group"]
        out.append(e)
        log(f"  {diam:g} mm  {e['name']:<28} L={e['L_nH']:7.1f} nH")
    return dict(diam=diam, radius=diam / 2, length=length, coils=out)


def lut(name: str = "viridis", n: int = 256) -> list[str]:
    import matplotlib
    return [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps[name](np.linspace(0, 1, n))]
