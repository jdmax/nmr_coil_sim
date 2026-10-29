"""Deuteron signal shape with non-uniform polarization: area vs. peak-fit methods.

Left: the signal the RGC coil sees when the rastered core and the unrastered shell
are polarized differently, its two parts, and the single-r peak fit.
Right: error of each method relative to the rastered material, for the RGC coil
(raster fraction 0.56) and a near-uniform volume coil (0.62), 12 mm raster in the
15 mm cell, as the rastered polarization varies with the shell fixed.

Run: python deuteron_two_zone.py  -> figures/deuteron_two_zone.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import deuteron_signals as ds

os.makedirs("figures", exist_ok=True)
freqs = ds.RGC_FREQS
P_IN, P_OUT, F_RGC, F_VOL = 0.30, 0.45, 0.558, 0.625

c = ds.compare(P_IN, P_OUT, F_RGC)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
scale = 1 / np.abs(c["signal"]).max()
a1.plot(freqs, c["signal"] * scale, color="#15202b", lw=2, label="signal seen by coil")
a1.plot(freqs, c["raster_part"] * scale, color="#b4531b", lw=1.6, ls="--",
        label=f"rastered core, P = {P_IN:.2f} (weight {F_RGC:.2f})")
a1.plot(freqs, c["shell_part"] * scale, color="#2a78d6", lw=1.6, ls="--",
        label=f"unrastered shell, P = {P_OUT:.2f} (weight {1-F_RGC:.2f})")
a1.plot(freqs, c["fit"] * scale, color="#1baf7a", lw=1.2, ls=":",
        label=f"single-r peak fit, P = {c['P_fit']:.3f}")
a1.set_xlim(32.40, 33.00)
a1.set_xlabel("frequency (MHz)"); a1.set_ylabel("absorption (normalised)")
a1.set_title(f"RGC coil: area gives P = {c['P_area']:.3f}, peak fit {c['P_fit']:.3f}, "
             f"raster is {P_IN:.2f}", fontsize=10)
a1.legend(fontsize=8, frameon=False, loc="upper left")
a1.spines[["top", "right"]].set_visible(False); a1.grid(alpha=0.25)

Pin = np.linspace(0.0, 0.7, 29)
for fr, col, lab in ((F_RGC, "#b4531b", "RGC coil"), (F_VOL, "#6250d6", "volume coil")):
    ea, ef = [], []
    for p in Pin:
        cc = ds.compare(p, P_OUT, fr)
        ea.append(cc["P_area"] - p); ef.append(cc["P_fit"] - p)
    a2.plot(Pin, ea, color=col, lw=2, label=f"{lab} (f = {fr:.2f}), area")
    a2.plot(Pin, ef, color=col, lw=2, ls="--", label=f"{lab} (f = {fr:.2f}), peak fit")
a2.axhline(0, color="0.5", lw=0.8); a2.axvline(P_OUT, color="0.5", lw=0.8, ls=":")
a2.text(P_OUT + 0.01, a2.get_ylim()[0] * 0.9, "shell P", color="0.4", fontsize=8)
a2.set_xlabel(f"rastered polarization (shell fixed at {P_OUT:.2f})")
a2.set_ylabel("measured − rastered polarization")
a2.set_title("Error relative to the material in the beam", fontsize=10)
a2.legend(fontsize=8, frameon=False)
a2.spines[["top", "right"]].set_visible(False); a2.grid(alpha=0.25)
fig.savefig("figures/deuteron_two_zone.png", dpi=150)
print(f"P_in {P_IN}, P_out {P_OUT}, f {F_RGC}: area {c['P_area']:.4f}, fit {c['P_fit']:.4f}, "
      f"rel. fit residual {c['resid_rms_rel']:.1e}")
