# nmr_coil_sim

Field simulations of cw-NMR coils for dynamically polarized solid targets: how much
of the NMR signal comes from the material inside the beam raster, what non-uniform
polarization does to deuteron polarization measurements, and whether a coil
geometry can be tuned. Started after the CLAS12 Run Group C NMR analysis, where
external coils sampled the unrastered shell of the 15 and 20 mm cells too
strongly (CLAS Note, Sec. 3.3).

Everything is plain NumPy. The same code runs offline and, through Pyodide, in the
browser front end.

## Web app

`web/` is a static site for GitHub Pages:
- the preset coils come precomputed in `web/data/default.json`;
- new coils you design are computed in the browser by `coilsim.py` and
  `explorer_data.py` running in Pyodide in a web worker.

You can drag the beam raster, rotate a 3D view of B_T² on the cell surface, set
core and shell polarizations to see the deuteron signal with its area and peak-fit
results, and compare inductance and tuning capacitance across coils.

**Two pages.** Both are served from the same site:

| Address | What it is |
|---|---|
| `https://jdmax.github.io/nmr_coil_sim/` | Full explorer, with the coil designer running in Pyodide |
| `https://jdmax.github.io/nmr_coil_sim/explorer.html` | Coil Raster Explorer as one self-contained file: preset coils only, no Python, works offline; you can also download it and open it locally |

**Publishing.** In the repository settings, under *Pages → Build and deployment*,
set *Source* to **GitHub Actions**. The workflow in `.github/workflows/pages.yml`
then:
1. runs the validation tests;
2. assembles the site with `python serve_web.py --build`;
3. deploys it on every push to `main`.

**Local preview.**

```
python serve_web.py          # builds _site/ and serves http://localhost:8000
```

Pyodide 0.27.2 loads from the jsDelivr CDN, so the designer needs a network
connection. The presets work without one.

## Quick start (Python)

```
pip install -r requirements.txt
python test_coilsim.py        # checks against closed-form results; prints ALL PASS
python compare_coils.py       # coil comparison on the 15 mm cell -> figures/
python fig8_reproduction.py   # CLAS Note Fig. 8 reproduction -> figures/
python deuteron_two_zone.py   # area vs. peak-fit deuteron polarization -> figures/
python build_explorer.py      # presets at full resolution -> web/data/default.json, web/explorer.html
```

```python
import coilsim as cs
cell = cs.Cell(radius=7.5, length=50)
coil = cs.saddle(radius=9.0, length=50, arc_deg=120)
cs.print_report(cs.sampling_report(coil, cell, raster_radius=6.0, raster_offset=(0, -1)))

m = cs.zavg_map(coil, cell)                  # B_T^2 averaged over z (Fig. 8 style)
cs.plot_zavg_map(m, 6.0, raster_offset=(0, -1))
cs.plot_cell_surface(coil, cell)             # 3D surface map with the wires
```

## Files

| File | What it is |
|---|---|
| `coilsim.py` | Library: Biot–Savart field of wire segments, partial-inductance L, skin-effect R and Q, coil builders (loop, rectangle, spiral, saddle, Alderman–Grant, birdcage), raster sampling metrics, field maps, plots |
| `explorer_data.py` | Coil specs → field maps and circuit numbers for the explorer pages; preset coils; runs in Pyodide |
| `deuteron_signals.py` | Two-zone deuteron signals and area vs. peak-fit polarization, with the DeuteronPeakFit lineshape |
| `test_coilsim.py` | Validation against analytic results (loop and wire fields, loop and square inductance, saddle and birdcage uniformity) |
| `compare_coils.py`, `fig8_reproduction.py`, `deuteron_two_zone.py` | Studies that write figures to `figures/` |
| `build_explorer.py`, `explorer_template.html` | Build the preset data (`web/data/default.json`) and the single-file explorer (`web/explorer.html`); `--from-json` rebuilds only the page |
| `serve_web.py`, `web/` | GitHub Pages front end and its build/preview script |

## Figures of merit

| Quantity | Meaning |
|---|---|
| Raster fraction f | CLAS Note Eq. 3: ∫raster B_T² dV / ∫cell B_T² dV |
| Uniform-coil ceiling | The raster's share of the cell area, (R_b/R_t)² for a centred raster |
| Worst 1 mm shift | Largest change in f when the raster moves 1 mm in any direction |
| η | Transverse filling factor ∫cell B_T² dV / (μ0 L); signal ∝ η·Q |
| L, C | Inductance and the series capacitance that tunes it. For birdcages: capacitance per rung of a low-pass birdcage, C_rung = (N/2)/(ω²L) |

## The RGC coil model

CLAS Note Fig. 8 shows a flat circular coil above the cell; the Fig. 6 photo shows
a 3-turn pancake spiral. `cs.spiral(7, 13, 3, center=(0, 16, 0), normal=(0, 1, 0))`
puts a spiral with r = 7–13 mm in a plane 16 mm above the axis. It reproduces the
Note's fractions: 0.558 centred and 0.436 at 1 mm away, against 0.56 and 0.43 in the
Note. These dimensions are a fit, not a measurement; replace them with the as-built
values.

Its modelled 257 nH needs 93 pF at 32.6 MHz, a little above the RGC cold board's
40–82 pF range. The leads to the board, which are not modelled, account for the
difference.

## Deuteron signals with non-uniform polarization

`deuteron_signals.compare(P_in, P_out, f)` gives the signal a coil with raster
fraction f sees: f·shape(P_in) + (1−f)·shape(P_out), using the absorption-only
Dulya lineshape from [DeuteronPeakFit](https://github.com/jdmax/DeuteronPeakFit),
copied in under its MIT licence.

- **Area method.** Area is exactly linear in P, so with a calibration constant taken
  on uniformly polarized material it returns the weighted mean.
- **Peak fit.** A single-r fit is pulled toward the zone with larger |P|, and the fit
  residual stays near 10⁻⁶ of the peak height, so fit quality cannot reveal the
  problem. The fast fit frees only G and r; freeing A, ωQ, ωL, η and ξ as well
  changes P by less than 10⁻⁴.
- **Using your own fit.** With lmfit installed,
  `peak_fit_polarization(..., use_deuteronpeakfit=True)` calls `deuteron_fit.fit()`
  directly.

## Model limits

- Quasi-static fields with imposed currents. Birdcages sit in their ideal mode, and
  Alderman–Grant strips are split into equal-current filaments.
- No eddy currents in the bath, horn or frame. These lower L.
- L excludes leads, so treat it as a lower bound.
- Coil Q uses room-temperature copper, for comparing coils only.
- Fields are valid while unbroken conductor runs stay below about λ/10; the
  `len/lam` column flags runs that don't.
