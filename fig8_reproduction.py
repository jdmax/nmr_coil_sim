"""Reproduce CLAS Note Fig. 8 (RGC NMR coil sampling) with coilsim.

The RGC coil is modeled as a flat 3-turn circular spiral (r = 7-13 mm) lying in
a plane 16 mm above the cell axis, centered on the cell. Those dimensions were
FITTED so that the 12 mm raster collects 0.56 of the signal centered and 0.43
when displaced 1 mm away from the coil, the two values quoted in the Note.
Replace them with the as-built dimensions when you have them.

Run: python fig8_reproduction.py   -> figures/fig8_reproduction.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import coilsim as cs

os.makedirs("figures", exist_ok=True)
cell = cs.Cell(radius=7.5, length=50.0)
RASTER = 6.0
coil = cs.spiral(7.0, 13.0, 3, center=(0, 16.0, 0), normal=(0, 1, 0),
                 name="RGC coil model (3-turn spiral, 16 mm above axis)")

smap = cs.surface_map(coil, cell)
m = cs.zavg_map(coil, cell)

fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
ax3 = fig.add_subplot(1, 3, 1, projection="3d")
_, sm = cs.plot_cell_surface(coil, cell, smap, ax=ax3, vmax=3.0)
fig.colorbar(sm, ax=ax3, shrink=0.6, label="B_T² / volume mean")
ax3.set_title("B_T² on the cell surface", fontsize=10)

vmax = np.nanpercentile(np.where(m["w"] > 0, m["v"], np.nan), 99)
ax1 = fig.add_subplot(1, 3, 2)
im, f0 = cs.plot_zavg_map(m, RASTER, (0, 0), ax=ax1, vmax=vmax,
                          title="B_T² averaged across all z, raster centered")
ax2 = fig.add_subplot(1, 3, 3)
im, f1 = cs.plot_zavg_map(m, RASTER, (0, -1.0), ax=ax2, vmax=vmax,
                          title="Raster displaced 1 mm away from the coil")
fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="B_T² / cell mean")
fig.suptitle(f"CLAS Note Fig. 8 reproduction: raster fraction {f0:.2f} centered, "
             f"{f1:.2f} offset (Note: 0.56, 0.43)", fontsize=11)
fig.savefig("figures/fig8_reproduction.png", dpi=150)

L = coil.inductance()
print(f"centered {f0:.3f}, offset {f1:.3f}")
print(f"L = {L*1e9:.0f} nH -> C = {1e12/((2*np.pi*32.6e6)**2*L):.0f} pF at 32.6 MHz, "
      f"{1e12/((2*np.pi*212.6e6)**2*L):.1f} pF at 212.6 MHz")
