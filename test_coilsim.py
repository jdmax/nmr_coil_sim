"""Checks coilsim against closed-form results. Run: python test_coilsim.py"""
import numpy as np
import coilsim as cs

MU0 = cs.MU0
ok = True


def check(name, got, want, tol):
    global ok
    rel = abs(got - want) / abs(want)
    flag = "PASS" if rel < tol else "FAIL"
    ok &= rel < tol
    print(f"[{flag}] {name}: got {got:.6g}, expected {want:.6g} (rel err {rel:.2e})")


# 1. On-axis field of a circular loop: B = mu0 I R^2 / (2 (R^2+z^2)^1.5)
R = 10.0  # mm
loop = cs.circle_loop(R, n=360)
for z in (0.0, 5.0, 20.0):
    Bz = loop.field([[0, 0, z]])[0, 2]
    want = MU0 * (R * 1e-3) ** 2 / (2 * ((R * 1e-3) ** 2 + (z * 1e-3) ** 2) ** 1.5)
    check(f"loop on-axis Bz at z={z} mm", Bz, want, 1e-4)

# 2. Long straight wire: B = mu0 I / (2 pi d), and in-wire linear taper
wire = cs.polyline([[0, 0, -5000], [0, 0, 5000]], closed=False, wire_radius=0.5)
B = np.linalg.norm(wire.field([[3.0, 0, 0]])[0])
check("long wire at d=3 mm", B, MU0 / (2 * np.pi * 3e-3), 1e-5)
B = np.linalg.norm(wire.field([[0.25, 0, 0]])[0])
check("inside wire at d=a/2", B, MU0 * 0.25e-3 / (2 * np.pi * (0.5e-3) ** 2), 1e-4)

# 3. Inductance of a circular loop, surface current: L = mu0 R (ln(8R/a) - 2)
for R, a in ((10.0, 0.25), (5.0, 0.1)):
    loop = cs.circle_loop(R, n=180, wire_radius=a)
    want = MU0 * R * 1e-3 * (np.log(8 * R / a) - 2)
    check(f"loop inductance R={R} a={a}", loop.inductance(), want, 0.01)

# 4. Square loop (Grover, HF): L = (2 mu0 s/pi)(ln(s/a) - 0.774) approx
s, a = 20.0, 0.25
sq = cs.rect_loop(s, s, normal=(0, 0, 1), width_dir=(1, 0, 0), n_side=20, wire_radius=a)
want = 2 * MU0 * s * 1e-3 / np.pi * (np.log(s / a) - 0.774)
check("square loop inductance", sq.inductance(), want, 0.01)

# 5. Birdcage mode-1 field uniformity at center region and direction
bc = cs.birdcage(10.0, 50.0, n_rungs=16)
pts = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [-3, 3, 0]])
B = bc.field(pts)
print("birdcage B (uT/A) at test points:\n", np.round(B * 1e6, 3))
check("birdcage uniformity |B(3,0)|/|B(0,0)|", np.linalg.norm(B[1]),
      np.linalg.norm(B[0]), 0.01)
check("birdcage field along +x", B[0, 0], np.linalg.norm(B[0]), 1e-6)

# 6. Saddle field direction and Ginsberg-Melchner flatness near center
sd = cs.saddle(10.0, 40.0, 120.0)
B = sd.field([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
check("saddle field along +x", B[0, 0], np.linalg.norm(B[0]), 1e-6)
check("saddle 2nd-order flatness (x=1mm)", B[1, 0], B[0, 0], 1e-3)
check("saddle 2nd-order flatness (z=1mm)", B[3, 0], B[0, 0], 1e-3)

# 7. Long solenoid interior field (many loops superposed): B = mu0 n I
sol = None
Rs, Ls, Nt = 10.0, 200.0, 200
for k in range(Nt):
    l = cs.circle_loop(Rs, center=(0, 0, -Ls / 2 + (k + 0.5) * Ls / Nt), n=72,
                       wire_radius=0.3)
    sol = l if sol is None else sol + l
cell = cs.Cell(radius=7.0, length=40.0, nz=9, nr=12, nphi=24)
pts, w = cell.grid()
B = sol.field(pts)
check("solenoid interior B", np.mean(B[:, 2]), MU0 * Nt / (Ls * 1e-3), 0.01)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
