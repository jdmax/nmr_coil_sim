# %% [markdown]
# # Coil comparison for a 15 mm x 50 mm CLAS12 target cell
# Run as a script (`python compare_coils.py`) or cell-by-cell in VS Code
# (the `# %%` markers make an interactive notebook). Figures go to ./figures.
#
# ASSUMED GEOMETRY -- edit these to match the real drawings:
#   * cell: 15 mm diameter, 50 mm long (CLAS Note Sec. 2.2)
#   * "RGC-style external loop": a 40 x 15 mm flat loop, 12 mm off-axis on +x.
#     This is a placeholder, not the as-built coil. It gives ~54% centred /
#     ~41% at a 1 mm offset vs. the Note's ~60% / 40%.
#   * volume coils sit on the cell wall (r = 8.5 mm) or on the bath (r = 12 mm)

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import coilsim as cs

os.makedirs("figures", exist_ok=True)
F_P = 212.6e6     # proton Larmor at 5 T
F_D = 32.6e6      # deuteron Larmor at 5 T

cell = cs.Cell(radius=7.5, length=50.0)
RASTER = 6.0      # 12 mm raster (misaligned periods); try 7.0 for nominal 14 mm

coils = [
    cs.rect_loop(40, 15, center=(12, 0, 0), normal=(1, 0, 0),
                 name="RGC-style external loop"),
    cs.rect_loop(40, 10, center=(0, 0, 0), normal=(1, 0, 0),
                 name="Internal flat loop"),
    cs.saddle(8.5, 50, arc_deg=120, name="Saddle 120°, r=8.5"),
    cs.saddle(12.0, 50, arc_deg=120, name="Saddle 120°, r=12 (on bath)"),
    cs.alderman_grant(8.5, 50, strip_deg=60, name="Alderman–Grant, r=8.5"),
    cs.birdcage(8.5, 50, n_rungs=8, name="Birdcage 8-rung, r=8.5"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]

# %% Figures of merit at the proton frequency
reports = [cs.sampling_report(c, cell, RASTER, freq=F_P) for c in coils]
print(f"Cell r={cell.radius} mm, raster r={RASTER} mm, f={F_P/1e6:.1f} MHz")
cs.print_report(reports)
print("\nraster frac: CLAS Note Eq.3.  ideal: same for a perfectly uniform B_T.")
print("CV: spread of B_T^2 over the cell.  eta: filling factor (signal ~ eta*Q).")
print("Q_coil: copper at 300 K, coil only.  C pF: series tuning capacitance 1/(w^2 L)")
print("len/lam: unbroken-run check, keep < ~0.1 (not meaningful for a birdcage,")
print("         whose capacitors break every rung and ring segment)")

with open("figures/coil_report.csv", "w") as fh:
    keys = list(reports[0].keys())
    fh.write(",".join(keys) + "\n")
    for r in reports:
        fh.write(",".join(str(r[k]) for k in keys) + "\n")

# %% Mid-plane B_T^2 maps (normalised to the cell mean)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
for ax, c in zip(axes.flat, coils):
    im = cs.plot_midplane(c, cell, RASTER, ax=ax, extent=13.5)
axes.flat[0].legend(fontsize=7, loc="lower left", frameon=False)
fig.colorbar(im, ax=axes, shrink=0.6, label="B_T² / ⟨B_T²⟩ over cell cross-section")
fig.suptitle(f"Mid-plane sensitivity, 15 mm cell (solid), {2*RASTER:.0f} mm raster "
             "(dashed); dots = conductors")
fig.savefig("figures/midplane_maps.png", dpi=150)

# %% Raster-offset sensitivity (offset along x; + is toward the external loop)
offsets = np.linspace(-1.5, 1.5, 13)
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ideal = (RASTER / cell.radius) ** 2
ax.axhline(ideal, color="0.45", lw=1, ls="--")
ax.text(offsets[0], ideal + 0.006, "uniform-coil ceiling", color="0.35", fontsize=8)
for c, col in zip(coils, COLORS):
    f = cs.offset_scan(c, cell, RASTER, offsets, direction=(1, 0))
    ax.plot(offsets, f, color=col, lw=2, marker="o", ms=4, label=c.name)
ax.set_xlabel("raster offset along x (mm), + toward external loop")
ax.set_ylabel("fraction of NMR signal from raster")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
fig.savefig("figures/offset_scan.png", dpi=150)

# %% Axial uniformity along the beam
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
z = np.linspace(-cell.length / 2, cell.length / 2, 101)
for c, col in zip(coils, COLORS):
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    B = c.field(pts)
    bt = np.hypot(B[:, 0], B[:, 1])
    ax.plot(z, bt / bt[50], color=col, lw=2, label=c.name)
ax.set_xlabel("z along beam (mm), on axis")
ax.set_ylabel("B_T / B_T(centre)")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
fig.savefig("figures/axial_profile.png", dpi=150)

# %% Design curve: how far from the cell should a volume coil sit?
radii = np.array([8.5, 9.5, 11, 12.5, 14, 16])
families = [
    ("Saddle 120°", lambda r: cs.saddle(r, 50, 120)),
    ("Alderman–Grant 60°", lambda r: cs.alderman_grant(r, 50, 60, n_filaments=5)),
    ("Birdcage 8-rung", lambda r: cs.birdcage(r, 50, 8)),
    ("Birdcage 16-rung", lambda r: cs.birdcage(r, 50, 16)),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
print("\nRadius sweep (raster fraction / eta %):")
for (label, make), col in zip(families, COLORS[2:]):
    fr, et = [], []
    for r in radii:
        rep = cs.sampling_report(make(r), cell, RASTER, freq=None)
        fr.append(rep["raster_fraction"]); et.append(100 * rep["eta"])
    print(f"  {label:<20}", "  ".join(f"{r:>4.1f}mm {f:.3f}/{e:4.1f}"
                                     for r, f, e in zip(radii, fr, et)))
    a1.plot(radii, fr, color=col, lw=2, marker="o", ms=5, label=label)
    a2.plot(radii, et, color=col, lw=2, marker="o", ms=5, label=label)
a1.axhline(ideal, color="0.45", lw=1, ls="--")
a1.text(radii[0], ideal + 0.002, "uniform-coil ceiling", color="0.35", fontsize=8)
a1.set_ylabel("fraction of NMR signal from raster")
a2.set_ylabel("filling factor η (%)")
for a in (a1, a2):
    a.set_xlabel("coil radius (mm); cell radius 7.5 mm")
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25)
a2.legend(fontsize=8, frameon=False)
fig.suptitle("Uniformity vs. signal size as the coil moves out from the cell wall")
fig.savefig("figures/radius_sweep.png", dpi=150)

# %% Geometry gallery
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
for i, c in enumerate(coils):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    cs.plot_geometry(c, cell, ax=ax)
fig.savefig("figures/geometry.png", dpi=130)

# %% Deuteron (32.6 MHz): more turns are affordable at the lower frequency
print(f"\nDeuteron, f={F_D/1e6:.1f} MHz")
d_coils = [
    cs.rect_loop(40, 15, center=(12, 0, 0), normal=(1, 0, 0), turns=3,
                 name="RGC-style loop, 3 turns"),
    cs.saddle(8.5, 50, 120, turns=1, name="Saddle 1 turn"),
    cs.saddle(8.5, 50, 120, turns=3, name="Saddle 3 turns"),
    cs.birdcage(8.5, 50, n_rungs=8, name="Birdcage 8-rung"),
]
cs.print_report([cs.sampling_report(c, cell, RASTER, freq=F_D) for c in d_coils])

plt.show()
