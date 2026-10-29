"""
coilsim.py -- quasi-static field, inductance and sampling toolkit for cw-NMR coils
on cylindrical polarized-target cells.

Dependencies: numpy, matplotlib (plots only).

Conventions
-----------
* Geometry is in millimetres everywhere. Fields come back in tesla per amp of
  drive current (T/A); inductance in henries; resistance in ohms.
* The static field B0 is along +z, which is also the cell (beam) axis, as in the
  CLAS12 solenoid. Only the transverse component B_T couples to the spins.
* A coil is a set of straight filament segments, each with its own current
  relative to a 1 A drive. That lets one object describe a series-wound loop,
  a saddle, a strip split into parallel filaments, or a birdcage with
  prescribed rung currents.
* Quasi-static (magnetostatic) approximation: valid while every unbroken
  conductor run is short compared with the wavelength (lambda/10 is a common
  rule of thumb; lambda = 1.41 m at 213 MHz). Use Coil.path_length() to check.
  Currents are imposed, not solved: a birdcage is assumed to sit in its ideal
  mode, and wide strips are approximated by evenly loaded filaments (real RF
  current crowds to the strip edges).

Quick start
-----------
    import coilsim as cs
    cell = cs.Cell(radius=7.5, length=50.0)
    coil = cs.saddle(radius=9.0, length=50.0, arc_deg=120)
    rep = cs.sampling_report(coil, cell, raster_radius=6.0)
    cs.print_report(rep)
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import numpy as np

MU0 = 4e-7 * np.pi
_MM = 1e-3  # mm -> m

__all__ = [
    "Coil", "Cell", "polyline", "circle_loop", "rect_loop", "spiral", "saddle",
    "alderman_grant", "birdcage", "sampling_report", "print_report",
    "offset_scan", "zavg_map", "map_fraction", "surface_map",
    "plot_midplane", "plot_geometry", "plot_axial_profile", "plot_zavg_map",
    "plot_cell_surface",
]


# ----------------------------------------------------------------------------
# Coil container
# ----------------------------------------------------------------------------
@dataclass
class Coil:
    starts: np.ndarray            # (N,3) segment start points, mm
    ends: np.ndarray              # (N,3) segment end points, mm
    currents: np.ndarray          # (N,) current per segment for 1 A drive
    wire_radius: float = 0.25     # mm, used for inductance, resistance, near-wire field
    name: str = "coil"

    def __post_init__(self):
        self.starts = np.asarray(self.starts, float).reshape(-1, 3)
        self.ends = np.asarray(self.ends, float).reshape(-1, 3)
        self.currents = np.broadcast_to(
            np.asarray(self.currents, float), (len(self.starts),)).copy()

    # -- composition --------------------------------------------------------
    def __add__(self, other: "Coil") -> "Coil":
        """Superpose two coils driven by the same current (series connection)."""
        return Coil(np.vstack([self.starts, other.starts]),
                    np.vstack([self.ends, other.ends]),
                    np.concatenate([self.currents, other.currents]),
                    wire_radius=self.wire_radius,
                    name=f"{self.name}+{other.name}")

    def scaled(self, k: float) -> "Coil":
        return Coil(self.starts, self.ends, self.currents * k,
                    self.wire_radius, self.name)

    def translated(self, d) -> "Coil":
        d = np.asarray(d, float)
        return Coil(self.starts + d, self.ends + d, self.currents,
                    self.wire_radius, self.name)

    def rotated(self, axis, angle_deg: float, origin=(0, 0, 0)) -> "Coil":
        """Rotate about `axis` through `origin` (right-hand rule)."""
        R = _rotation_matrix(axis, np.radians(angle_deg))
        o = np.asarray(origin, float)
        return Coil((self.starts - o) @ R.T + o, (self.ends - o) @ R.T + o,
                    self.currents, self.wire_radius, self.name)

    def renamed(self, name: str) -> "Coil":
        return Coil(self.starts, self.ends, self.currents, self.wire_radius, name)

    # -- geometry -------------------------------------------------------------
    @property
    def seg_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.ends - self.starts, axis=1)

    def path_length(self) -> float:
        """Current-weighted conductor length in mm (= wire length for a series coil;
        parallel filaments count once). Compare with the wavelength."""
        return float(np.sum(self.seg_lengths * np.abs(self.currents)))

    # -- field ----------------------------------------------------------------
    def field(self, points, max_elems: float = 2e6) -> np.ndarray:
        """Exact Biot-Savart field of finite straight segments at `points` (M,3) mm.
        Returns (M,3) in T/A. Inside a wire (distance < wire_radius) the field
        falls off linearly to zero, as for a uniform-current round wire."""
        p_all = np.atleast_2d(np.asarray(points, float))
        a, b, I = self.starts, self.ends, self.currents
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
        ab2 = np.einsum("ij,ij->i", b - a, b - a)
        out = np.empty_like(p_all)
        chunk = max(1, int(max_elems // max(1, len(a))))
        inv_r2w = 1.0 / self.wire_radius ** 2
        for s in range(0, len(p_all), chunk):
            p = p_all[s:s + chunk]
            px, py, pz = p[:, 0:1], p[:, 1:2], p[:, 2:3]
            r1x, r1y, r1z = px - ax, py - ay, pz - az
            r2x, r2y, r2z = px - bx, py - by, pz - bz
            n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
            n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
            cx = r1y * r2z - r1z * r2y
            cy = r1z * r2x - r1x * r2z
            cz = r1x * r2y - r1y * r2x
            n12 = n1 * n2
            denom = n12 * (n12 + r1x * r2x + r1y * r2y + r1z * r2z)
            # perpendicular distance^2 to the segment line, for the in-wire taper:
            # outside the wire the field is the usual 1/d law, inside it falls as d
            taper = np.minimum(1.0, (cx * cx + cy * cy + cz * cz) / ab2 * inv_r2w)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(denom > 1e-30, (I * taper) * (n1 + n2) / denom, 0.0)
            out[s:s + chunk, 0] = np.einsum("mn,mn->m", f, cx)
            out[s:s + chunk, 1] = np.einsum("mn,mn->m", f, cy)
            out[s:s + chunk, 2] = np.einsum("mn,mn->m", f, cz)
        return out * MU0 / (4 * np.pi) / _MM

    # -- circuit quantities ---------------------------------------------------
    def inductance(self, n_far: int = 4, n_near: int = 24, near: float = 3.0,
                   block: int = 64) -> float:
        """Effective inductance L = sum_ij I_i I_j M_ij (H) for a 1 A drive, from
        partial inductances: Neumann double integral between segments (Gauss-
        Legendre), analytic self term for a round wire with surface current
        (high-frequency limit). Pairs closer than `near` x segment length get
        `n_near` points. For coils with imposed distributed currents (birdcage,
        filament bundles) this is 2W/I^2, the energy-equivalent inductance."""
        a, b, I = self.starts, self.ends, self.currents
        rad = self.wire_radius
        N = len(a)
        lens = self.seg_lengths
        dirs = (b - a) / lens[:, None]
        mids = 0.5 * (a + b)

        def quad_pts(n):
            x, w = np.polynomial.legendre.leggauss(n)
            t = 0.5 * (x + 1)
            P = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]  # (N,n,3)
            W = 0.5 * w[None, :] * lens[:, None]                        # (N,n)
            return P, W

        P, W = quad_pts(n_far)
        DD = dirs @ dirs.T
        M = np.zeros((N, N))
        for s in range(0, N, block):
            Pi = P[s:s + block]                      # (bi,n,3)
            diff = Pi[:, :, None, None, :] - P[None, None, :, :, :]
            r = np.sqrt(np.einsum("abcdk,abcdk->abcd", diff, diff) + rad ** 2)
            K = np.einsum("ab,abcd,cd->ac", W[s:s + block], 1.0 / r, W)
            M[s:s + block] = K
        # refine close pairs
        Pn, Wn = quad_pts(n_near)
        dm = np.linalg.norm(mids[:, None, :] - mids[None, :, :], axis=-1)
        lmax = np.maximum(lens[:, None], lens[None, :])
        ii, jj = np.nonzero((dm < near * lmax) & ~np.eye(N, dtype=bool))
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        for s in range(0, len(ii), 256):
            i_, j_ = ii[s:s + 256], jj[s:s + 256]
            diff = Pn[i_][:, :, None, :] - Pn[j_][:, None, :, :]
            r = np.sqrt(np.einsum("pabk,pabk->pab", diff, diff) + rad ** 2)
            K = np.einsum("pa,pab,pb->p", Wn[i_], 1.0 / r, Wn[j_])
            M[i_, j_] = K
            M[j_, i_] = K
        M *= DD
        # analytic partial self-inductance of a straight tube of radius a
        l = lens
        np.fill_diagonal(M, 2 * (l * np.log((l + np.sqrt(l**2 + rad**2)) / rad)
                                 - np.sqrt(l**2 + rad**2) + rad))
        # (factor 2 because mu0/4pi prefactor is applied below; mu0/2pi = 2*mu0/4pi)
        return float(I @ M @ I) * MU0 / (4 * np.pi) * _MM

    def ac_resistance(self, freq: float, rho: float = 1.68e-8) -> float:
        """Skin-effect resistance (ohm) for a 1 A drive: R = sum_i R_i I_i^2.
        rho defaults to room-temperature copper. At 1-4 K the anomalous skin
        effect limits the improvement to a few x; treat cold numbers as optimistic."""
        delta = np.sqrt(rho / (np.pi * freq * MU0))          # m
        a = self.wire_radius * _MM
        area = np.where(delta < a, 2 * np.pi * a * delta, np.pi * a**2)
        R_i = rho * self.seg_lengths * _MM / area
        return float(np.sum(R_i * self.currents**2))

    def q_estimate(self, freq: float, rho: float = 1.68e-8) -> float:
        """Unloaded coil-only Q = wL/R. Ignores cable, capacitors, solder, radiation."""
        return 2 * np.pi * freq * self.inductance() / self.ac_resistance(freq, rho)


def _rotation_matrix(axis, theta):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


# ----------------------------------------------------------------------------
# Builders (all mm; cell axis along z)
# ----------------------------------------------------------------------------
def polyline(points, current: float = 1.0, closed: bool = True,
             wire_radius: float = 0.25, name: str = "polyline") -> Coil:
    """Coil from an ordered list of vertices; current flows in vertex order."""
    P = np.asarray(points, float)
    if closed and not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[:1]])
    return Coil(P[:-1], P[1:], current, wire_radius, name)


def _arc(radius, phi1, phi2, z, n):
    phi = np.linspace(phi1, phi2, n + 1)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi),
                            np.full_like(phi, z)])


def circle_loop(radius: float, center=(0, 0, 0), normal=(0, 0, 1), n: int = 72,
                turns: int = 1, pitch: float = 0.6, wire_radius: float = 0.25,
                name: str = "loop") -> Coil:
    """Circular loop (or stacked turns along the normal)."""
    coil = None
    for t in range(turns):
        P = _arc(radius, 0, 2 * np.pi, (t - (turns - 1) / 2) * pitch, n)
        c = polyline(P, 1.0, False, wire_radius, name)
        coil = c if coil is None else coil + c
    coil = _orient(coil, normal)
    return coil.translated(center).renamed(name)


def rect_loop(width: float, height: float, center=(0, 0, 0), normal=(1, 0, 0),
              width_dir=(0, 0, 1), turns: int = 1, pitch: float = 0.6,
              n_side: int = 10, wire_radius: float = 0.25,
              name: str = "rect loop") -> Coil:
    """Flat rectangular loop. `width` runs along `width_dir`, `height` along
    normal x width_dir. Multiple turns are stacked along the normal by `pitch`.
    Default: a loop in the y-z plane with its long side along the beam."""
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    u = np.asarray(width_dir, float); u -= u.dot(n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    coil = None
    for t in range(turns):
        off = (t - (turns - 1) / 2) * pitch * n
        pts = []
        for (c1, d1), (c2, d2) in zip(corners[:-1], corners[1:]):
            for s in np.linspace(0, 1, n_side, endpoint=False):
                cu = c1 + (c2 - c1) * s
                cv = d1 + (d2 - d1) * s
                pts.append(off + 0.5 * width * cu * u + 0.5 * height * cv * v)
        c = polyline(np.array(pts), 1.0, True, wire_radius, name)
        coil = c if coil is None else coil + c
    return coil.translated(center).renamed(name)


def _orient(coil: Coil, normal) -> Coil:
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    ax = np.cross(z, n)
    if np.linalg.norm(ax) < 1e-12:
        return coil if n[2] > 0 else coil.rotated((1, 0, 0), 180)
    return coil.rotated(ax, np.degrees(np.arccos(np.clip(z @ n, -1, 1))))


def spiral(r_in: float, r_out: float, turns: float, center=(0, 0, 0),
           normal=(0, 1, 0), n_per_turn: int = 72, return_lead: bool = True,
           lead_offset: float = 0.6, wire_radius: float = 0.25,
           name: str = "spiral") -> Coil:
    """Flat Archimedean spiral (a pancake coil) from r_in to r_out, lying in the
    plane through `center` with the given `normal`. With return_lead=True a
    straight radial lead, raised by `lead_offset` along the normal, closes the
    circuit from the inner end back past the outer edge (as in a real pancake
    coil). turns=1 with r_in == r_out is a single circular loop."""
    nseg = max(8, int(np.ceil(abs(turns) * n_per_turn)))
    th = np.linspace(0, 2 * np.pi * turns, nseg + 1)
    r = np.linspace(r_out, r_in, nseg + 1)
    P = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)])
    coil = polyline(P, 1.0, False, wire_radius, name)
    if return_lead:
        # raised radial lead out from the inner end, down, then back to the start
        end = P[-1]
        ang = np.arctan2(end[1], end[0])
        far = (r_out + 1.5) * np.array([np.cos(ang), np.sin(ang), 0])
        lead = np.array([end, end + [0, 0, lead_offset], far + [0, 0, lead_offset],
                         far, P[0]])
        coil = coil + polyline(lead, 1.0, False, wire_radius)
    coil = _orient(coil, normal)
    return coil.translated(center).renamed(name)


def saddle(radius: float, length: float, arc_deg: float = 120.0, turns: int = 1,
           turn_spacing: float = 0.8, phi0_deg: float = 0.0, n_axial: int = 8,
           n_arc: int = 24, center=(0, 0, 0), wire_radius: float = 0.25,
           name: str = "saddle") -> Coil:
    """Saddle pair on a cylinder of `radius` about z, windows facing +/-x so the
    field at the centre is along +x (rotate with phi0_deg). Ginsberg & Melchner
    optimum: arc 120 deg, length/diameter = 2. Additional turns are nested
    inside the first, each window narrowed by `turn_spacing` (mm, along the arc)
    on each side and shortened by the same amount at each end."""
    coil = None
    for t in range(turns):
        half = np.radians(arc_deg) / 2 - t * turn_spacing / radius
        zt = length / 2 - t * turn_spacing
        if half <= 0 or zt <= 0:
            raise ValueError("too many turns for the window size")
        for phic, sgn in ((0.0, +1), (np.pi, -1)):
            up = phic + sgn * half
            dn = phic - sgn * half
            zz = np.linspace(-zt, zt, n_axial + 1)
            leg_up = np.column_stack([radius * np.cos(up) * np.ones_like(zz),
                                      radius * np.sin(up) * np.ones_like(zz), zz])
            top = _arc(radius, up, dn, zt, n_arc)
            leg_dn = leg_up[::-1].copy()
            leg_dn[:, 0] = radius * np.cos(dn)
            leg_dn[:, 1] = radius * np.sin(dn)
            bot = _arc(radius, dn, up, -zt, n_arc)
            P = np.vstack([leg_up, top[1:], leg_dn[1:], bot[1:]])
            c = polyline(P, 1.0, False, wire_radius, name)
            coil = c if coil is None else coil + c
    return coil.rotated((0, 0, 1), phi0_deg).translated(center).renamed(name)


def alderman_grant(radius: float, length: float, strip_deg: float = 60.0,
                   n_filaments: int = 7, ring_width: float = 0.0,
                   phi0_deg: float = 0.0, n_axial: int = 8, n_arc: int = 36,
                   center=(0, 0, 0), wire_radius: float = 0.1,
                   name: str = "Alderman-Grant") -> Coil:
    """Alderman-Grant (slotted-tube) resonator approximated by filaments.
    Two axial strips centred at phi = +/-90 deg (field along +x), each split into
    `n_filaments` parallel filaments carrying equal shares of the current.
    End guard rings are modelled as filament arcs at z = +/-length/2, each
    strip's current splitting equally both ways round the ring. Real RF current
    peaks at strip edges; this is the uniform-share approximation. Note the
    real device is a mostly-closed metal sleeve: check microwave access."""
    phis = np.radians(90 + np.linspace(-strip_deg / 2, strip_deg / 2, n_filaments))
    zt = length / 2
    zz = np.linspace(-zt, zt, n_axial + 1)
    share = 1.0 / n_filaments
    parts = []
    for ph in phis:
        up = np.column_stack([radius * np.cos(ph) * np.ones_like(zz),
                              radius * np.sin(ph) * np.ones_like(zz), zz])
        dn_phi = ph + np.pi
        dn = np.column_stack([radius * np.cos(dn_phi) * np.ones_like(zz),
                              radius * np.sin(dn_phi) * np.ones_like(zz), zz[::-1]])
        parts.append(polyline(up, share, False, wire_radius))
        parts.append(polyline(dn, share, False, wire_radius))
        for direction in (+1, -1):          # split both ways round each ring
            parts.append(polyline(_arc(radius, ph, ph + direction * np.pi, zt, n_arc),
                                  share / 2, False, wire_radius))
            parts.append(polyline(_arc(radius, dn_phi, dn_phi + direction * np.pi,
                                       -zt, n_arc), share / 2, False, wire_radius))
    coil = parts[0]
    for p in parts[1:]:
        coil = coil + p
    return coil.rotated((0, 0, 1), phi0_deg).translated(center).renamed(name)


def birdcage(radius: float, length: float, n_rungs: int = 8, mode: int = 1,
             phi0_deg: float = 0.0, n_axial: int = 8, n_arc: int = 8,
             center=(0, 0, 0), wire_radius: float = 0.25,
             name: str = "birdcage") -> Coil:
    """Birdcage with the ideal mode-`mode` current pattern imposed:
    rung k at phi_k carries sin(mode*phi_k) (normalised to max |I| = 1), so
    mode 1 gives a uniform field along +x. End-ring currents follow from
    Kirchhoff's law at each node (zero-mean). Capacitor placement (low/high-pass)
    does not change the magnetostatic field of a given mode, only its frequency."""
    N = n_rungs
    phis = 2 * np.pi * np.arange(N) / N + np.radians(phi0_deg) + np.pi / N
    Ir = np.sin(mode * phis)
    Ir /= np.max(np.abs(Ir))
    zt = length / 2
    zz = np.linspace(-zt, zt, n_axial + 1)
    parts = []
    for ph, I in zip(phis, Ir):
        if abs(I) < 1e-12:
            continue
        rung = np.column_stack([radius * np.cos(ph) * np.ones_like(zz),
                                radius * np.sin(ph) * np.ones_like(zz), zz])
        parts.append(polyline(rung, I, False, wire_radius))
    # top ring: current arriving at node k from rung k leaves along ring segment k
    # (from phi_k to phi_{k+1}); J_k = J_{k-1} + I_k, choose zero mean.
    J = np.cumsum(Ir)
    J -= J.mean()
    for k in range(N):
        p1, p2 = phis[k], phis[k] + 2 * np.pi / N
        if abs(J[k]) < 1e-12:
            continue
        parts.append(polyline(_arc(radius, p1, p2, zt, n_arc), J[k], False, wire_radius))
        parts.append(polyline(_arc(radius, p1, p2, -zt, n_arc), -J[k], False, wire_radius))
    coil = parts[0]
    for p in parts[1:]:
        coil = coil + p
    return coil.translated(center).renamed(name)


# ----------------------------------------------------------------------------
# Target cell and sampling metrics
# ----------------------------------------------------------------------------
@dataclass
class Cell:
    """Cylindrical target volume about z. Midpoint polar grid with exact
    annulus weights, so area/volume fractions are exact for circles."""
    radius: float = 7.5
    length: float = 50.0
    center: tuple = (0.0, 0.0, 0.0)
    nr: int = 30
    nphi: int = 72
    nz: int = 25

    def grid(self):
        dr = self.radius / self.nr
        r = (np.arange(self.nr) + 0.5) * dr
        dphi = 2 * np.pi / self.nphi
        phi = (np.arange(self.nphi) + 0.5) * dphi
        dz = self.length / self.nz
        z = -self.length / 2 + (np.arange(self.nz) + 0.5) * dz
        R, PH, Z = np.meshgrid(r, phi, z, indexing="ij")
        pts = np.column_stack([(R * np.cos(PH)).ravel(), (R * np.sin(PH)).ravel(),
                               Z.ravel()]) + np.asarray(self.center, float)
        w = (R * dr * dphi * dz).ravel()
        return pts, w


def _bt2(B, b0_axis=(0, 0, 1)):
    b0 = np.asarray(b0_axis, float); b0 /= np.linalg.norm(b0)
    Bp = B @ b0
    return np.einsum("ij,ij->i", B, B) - Bp**2


def sampling_report(coil: Coil, cell: Cell, raster_radius: float,
                    raster_offset=(0.0, 0.0), freq: float | None = 212.6e6,
                    rho: float = 1.68e-8, b0_axis=(0, 0, 1),
                    _cache: dict | None = None) -> dict:
    """Figures of merit for one coil and one beam raster.

    raster_fraction : CLAS Note Eq. 3, int_raster B_T^2 dV / int_cell B_T^2 dV
    area_fraction   : the same ratio for a perfectly uniform coil (the ceiling)
    shell_weight    : 1 - raster_fraction; NMR error = shell_weight*(P_shell - P_beam)
    cv              : B_T^2 standard deviation / mean over the cell (0 = uniform)
    eta             : magnetic filling factor of the cell for the transverse field,
                      int_cell B_T^2 dV / (mu0 L)  (signal size ~ eta * Q)
    """
    pts, w = cell.grid()
    if _cache is not None and "B" in _cache:
        B = _cache["B"]
    else:
        B = coil.field(pts)
        if _cache is not None:
            _cache["B"] = B
    bt2 = _bt2(B, b0_axis)
    c = np.asarray(cell.center, float)
    dx, dy = raster_offset
    inside = ((pts[:, 0] - c[0] - dx) ** 2 + (pts[:, 1] - c[1] - dy) ** 2
              <= raster_radius ** 2)
    S = w * bt2
    frac = S[inside].sum() / S.sum()
    afrac = w[inside].sum() / w.sum()
    mean = S.sum() / w.sum()
    cv = np.sqrt(np.sum(w * (bt2 - mean) ** 2) / w.sum()) / mean
    rep = dict(name=coil.name, raster_fraction=frac, area_fraction=afrac,
               shell_weight=1 - frac, ideal_shell_weight=1 - afrac, cv=cv,
               mean_BT_uT_per_A=np.sqrt(mean) * 1e6,
               path_length_mm=coil.path_length())
    if _cache is not None and "L" in _cache:
        L = _cache["L"]
    else:
        L = coil.inductance()
        if _cache is not None:
            _cache["L"] = L
    rep["L_nH"] = L * 1e9
    rep["eta"] = S.sum() * _MM**3 / (MU0 * L)
    if freq:
        R = coil.ac_resistance(freq, rho)
        rep["R_ohm"] = R
        rep["Q_coil"] = 2 * np.pi * freq * L / R
        rep["freq_MHz"] = freq / 1e6
        rep["C_tune_pF"] = 1e12 / ((2 * np.pi * freq) ** 2 * L)
        rep["path_over_lambda"] = coil.path_length() * _MM / (2.998e8 / freq)
    return rep


def print_report(rep: dict | list):
    reps = rep if isinstance(rep, list) else [rep]
    hdr = (f"{'coil':<28}{'raster frac':>12}{'ideal':>8}{'CV(BT2)':>9}"
           f"{'<BT> uT/A':>11}{'L nH':>8}{'eta %':>8}{'Q_coil':>8}{'C pF':>7}"
           f"{'len/lam':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in reps:
        print(f"{r['name']:<28}{r['raster_fraction']:>12.3f}{r['area_fraction']:>8.3f}"
              f"{r['cv']:>9.2f}{r['mean_BT_uT_per_A']:>11.1f}{r['L_nH']:>8.1f}"
              f"{100*r['eta']:>8.1f}{r.get('Q_coil', np.nan):>8.0f}"
              f"{r.get('C_tune_pF', np.nan):>7.1f}"
              f"{r.get('path_over_lambda', np.nan):>9.3f}")


def offset_scan(coil: Coil, cell: Cell, raster_radius: float, offsets,
                direction=(1.0, 0.0)) -> np.ndarray:
    """Raster fraction vs raster displacement (mm) along `direction` in x-y."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    pts, w = cell.grid()
    bt2 = _bt2(coil.field(pts))
    c = np.asarray(cell.center, float)
    S = w * bt2
    out = []
    for o in offsets:
        cx, cy = c[0] + o * d[0], c[1] + o * d[1]
        inside = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 <= raster_radius ** 2
        out.append(S[inside].sum() / S.sum())
    return np.array(out)


def zavg_map(coil: Coil, cell: Cell, pixel: float = 0.15, nz: int = 21,
             b0_axis=(0, 0, 1)):
    """B_T^2 averaged over the cell length on a square pixel grid covering the
    cell cross-section (the quantity plotted in CLAS Note Fig. 8).
    Returns dict with x, y (pixel centres, mm, relative to cell centre),
    v (z-averaged B_T^2 normalised to its mean over the cell, 0 outside) and
    w (fraction of each pixel inside the cell wall, from 4x4 supersampling).
    The raster fraction for any circle is sum(v*w inside)/sum(v*w)."""
    R = cell.radius
    n = int(np.ceil(2 * R / pixel))
    edges = -n * pixel / 2 + pixel * np.arange(n + 1)
    xc = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(xc, xc)          # rows = y, cols = x
    # coverage by 4x4 supersampling
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    w = np.zeros_like(X)
    for sx in sub:
        for sy in sub:
            w += ((X + sx * pixel) ** 2 + (Y + sy * pixel) ** 2 <= R ** 2)
    w /= 16.0
    use = w > 0
    dz = cell.length / nz
    zs = -cell.length / 2 + (np.arange(nz) + 0.5) * dz
    c = np.asarray(cell.center, float)
    xy = np.column_stack([X[use], Y[use]])
    acc = np.zeros(len(xy))
    for z in zs:
        pts = np.column_stack([xy + c[:2], np.full(len(xy), c[2] + z)])
        acc += _bt2(coil.field(pts), b0_axis)
    acc /= nz
    v = np.zeros_like(X)
    v[use] = acc
    v /= np.sum(v * w) / np.sum(w)
    return dict(x=xc, y=xc, v=v, w=w, pixel=pixel, radius=R)


def map_fraction(m: dict, raster_radius: float, raster_offset=(0.0, 0.0)) -> float:
    """Raster fraction from a zavg_map result (pixel-centre test)."""
    X, Y = np.meshgrid(m["x"], m["y"])
    inside = (X - raster_offset[0]) ** 2 + (Y - raster_offset[1]) ** 2 <= raster_radius ** 2
    S = m["v"] * m["w"]
    return float(S[inside].sum() / S.sum())


def surface_map(coil: Coil, cell: Cell, nphi: int = 72, nz: int = 41, ncap: int = 8,
                b0_axis=(0, 0, 1), norm: float | None = None):
    """B_T^2 on the cell's outer surface: side wall (nz x nphi) and both end caps
    (ncap x nphi), normalised to the mean over the cell volume. For 3D maps like
    CLAS Note Fig. 8 (left)."""
    R, L = cell.radius, cell.length
    c = np.asarray(cell.center, float)
    phi = np.linspace(0, 2 * np.pi, nphi + 1)
    z = np.linspace(-L / 2, L / 2, nz)
    PH, Z = np.meshgrid(phi, z)
    side = np.column_stack([R * np.cos(PH).ravel(), R * np.sin(PH).ravel(), Z.ravel()]) + c
    rr = np.linspace(0, R, ncap + 1)
    RR, PHc = np.meshgrid(rr, phi, indexing="ij")
    caps = []
    for zc in (-L / 2, L / 2):
        caps.append(np.column_stack([(RR * np.cos(PHc)).ravel(), (RR * np.sin(PHc)).ravel(),
                                     np.full(RR.size, zc)]) + c)
    if norm is None:   # mean B_T^2 over the cell volume
        pts, w = cell.grid()
        norm = np.sum(w * _bt2(coil.field(pts), b0_axis)) / w.sum()
    vs = _bt2(coil.field(side), b0_axis).reshape(PH.shape) / norm
    vc = [_bt2(coil.field(p), b0_axis).reshape(RR.shape) / norm for p in caps]
    return dict(phi=phi, z=z, side=vs, r=rr, cap_lo=vc[0], cap_hi=vc[1],
                radius=R, length=L, center=c)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_zavg_map(m: dict, raster_radius: float, raster_offset=(0.0, 0.0), ax=None,
                  vmax: float | None = None, cmap: str = "viridis", title=None,
                  coil_side: str | None = None):
    """Fig. 8-style map of z-averaged B_T^2 with the raster circle and the
    resulting signal fraction printed inside, as in the CLAS Note."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4))
    v = np.where(m["w"] > 0, m["v"], np.nan)
    p = m["pixel"]
    ext = [m["x"][0] - p / 2, m["x"][-1] + p / 2, m["y"][0] - p / 2, m["y"][-1] + p / 2]
    im = ax.imshow(v, origin="lower", extent=ext, cmap=cmap, vmin=0,
                   vmax=vmax if vmax else np.nanpercentile(v, 99))
    th = np.linspace(0, 2 * np.pi, 300)
    R = m["radius"]
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.15", lw=1.2)
    ox, oy = raster_offset
    ax.plot(ox + raster_radius * np.cos(th), oy + raster_radius * np.sin(th),
            color="k", lw=1.2)
    f = map_fraction(m, raster_radius, raster_offset)
    txt = f"∫B_T²(raster) / ∫B_T²(total) = {f:.2f}\nraster radius = {raster_radius:.1f} mm"
    if ox or oy:
        txt += f"\noffset = ({ox:+.1f}, {oy:+.1f}) mm"
    ax.text(0, 0, txt, ha="center", va="center", fontsize=8, color="w")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    return im, f


def plot_cell_surface(coil: Coil, cell: Cell, smap: dict | None = None, ax=None,
                      vmax: float = 3.0, cmap: str = "viridis", elev: float = 22,
                      azim: float = -58, wire_color: str = "#d62728"):
    """3D view of B_T^2 on the cell surface with the coil drawn in red,
    in the style of CLAS Note Fig. 8 (left). Axes: z = beam, y = up."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    if smap is None:
        smap = surface_map(coil, cell)
    if ax is None:
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(projection="3d")
    norm = colors.Normalize(0, vmax)
    cmo = plt.get_cmap(cmap)
    R, c = smap["radius"], smap["center"]
    PH, Z = np.meshgrid(smap["phi"], smap["z"])
    # plot coordinates: (z, x, y) so the beam runs across and y stays up
    ax.plot_surface(Z + c[2], R * np.cos(PH) + c[0], R * np.sin(PH) + c[1],
                    facecolors=cmo(norm(smap["side"])), shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    RR, PHc = np.meshgrid(smap["r"], smap["phi"], indexing="ij")
    for key, zc in (("cap_lo", -smap["length"] / 2), ("cap_hi", smap["length"] / 2)):
        ax.plot_surface(np.full_like(RR, zc) + c[2], RR * np.cos(PHc) + c[0],
                        RR * np.sin(PHc) + c[1], facecolors=cmo(norm(smap[key])),
                        shade=False, rstride=1, cstride=1, linewidth=0)
    for a, b, I in zip(coil.starts, coil.ends, coil.currents):
        ax.plot([a[2], b[2]], [a[0], b[0]], [a[1], b[1]], color=wire_color,
                lw=0.6 + 1.8 * min(1, abs(I)))
    allp = np.vstack([coil.starts, coil.ends])
    lim = max(np.max(np.abs(allp)), smap["length"] / 2) * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("z (beam, mm)"); ax.set_ylabel("x (mm)"); ax.set_zlabel("y (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(coil.name, fontsize=10)
    sm = cm.ScalarMappable(norm=norm, cmap=cmo)
    return ax, sm
def plot_midplane(coil: Coil, cell: Cell, raster_radius: float | None = None,
                  raster_offset=(0, 0), z: float = 0.0, extent: float | None = None,
                  n: int = 161, ax=None, normalize: bool = True, vmax: float = 2.0,
                  show_wires: bool = True):
    """Map of B_T^2 in the plane z = const, normalised to its mean over the cell
    cross-section. Cell wall solid, raster dashed."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    c = np.asarray(cell.center, float)
    ext = extent or 1.6 * cell.radius
    xs = np.linspace(-ext, ext, n) + c[0]
    ys = np.linspace(-ext, ext, n) + c[1]
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, c[2] + z)])
    bt2 = _bt2(coil.field(pts)).reshape(X.shape)
    inside = (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= cell.radius ** 2
    img = bt2 / bt2[inside].mean() if normalize else bt2
    im = ax.pcolormesh(X, Y, np.where(inside, img, np.nan), shading="auto",
                       cmap="viridis", vmin=0, vmax=vmax if normalize else None)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c[0] + cell.radius * np.cos(th), c[1] + cell.radius * np.sin(th),
            color="0.2", lw=1.2)
    if raster_radius:
        ax.plot(c[0] + raster_offset[0] + raster_radius * np.cos(th),
                c[1] + raster_offset[1] + raster_radius * np.sin(th),
                color="w", lw=1.2, ls="--")
    if show_wires:
        # mark conductors crossing this plane
        a, b = coil.starts, coil.ends
        zc = c[2] + z + 1e-7   # nudge so vertices lying exactly on the plane count once
        cross = (a[:, 2] - zc) * (b[:, 2] - zc) < 0
        if cross.any():
            t = (zc - a[cross, 2]) / (b[cross, 2] - a[cross, 2])
            P = a[cross] + t[:, None] * (b[cross] - a[cross])
            sign = np.sign(coil.currents[cross] * (b[cross, 2] - a[cross, 2]))
            ax.scatter(P[sign > 0, 0], P[sign > 0, 1], s=18, c="tab:red",
                       marker="o", zorder=3, label="+z current")
            ax.scatter(P[sign < 0, 0], P[sign < 0, 1], s=18, c="tab:blue",
                       marker="o", zorder=3, label="-z current")
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(coil.name, fontsize=10)
    return im


def plot_axial_profile(coils, cell: Cell, ax=None, r: float = 0.0):
    """Relative B_T along z at radius r on the x axis, each normalised at z = 0."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    z = np.linspace(-cell.length / 2, cell.length / 2, 101)
    c = np.asarray(cell.center, float)
    for coil in coils:
        pts = np.column_stack([np.full_like(z, c[0] + r), np.full_like(z, c[1]), c[2] + z])
        bt = np.sqrt(_bt2(coil.field(pts)))
        ax.plot(z, bt / bt[len(z) // 2], label=coil.name)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("B_T / B_T(z=0)")
    ax.axhline(1, color="0.7", lw=0.8)
    ax.legend(fontsize=8)
    return ax


def plot_geometry(coil: Coil, cell: Cell | None = None, ax=None):
    """3D wire plot, line width ~ |current|; cell drawn as a translucent cylinder."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(projection="3d")
    for a, b, I in zip(coil.starts, coil.ends, coil.currents):
        ax.plot([a[2], b[2]], [a[0], b[0]], [a[1], b[1]], color="#b5541c",
                lw=0.4 + 2.0 * min(1, abs(I)))
    if cell is not None:
        th = np.linspace(0, 2 * np.pi, 40)
        zz = np.array([-cell.length / 2, cell.length / 2])
        T, Z = np.meshgrid(th, zz)
        c = np.asarray(cell.center, float)
        ax.plot_surface(Z + c[2], cell.radius * np.cos(T) + c[0],
                        cell.radius * np.sin(T) + c[1], alpha=0.15, color="0.5")
    ax.set_xlabel("z (beam)"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.set_title(coil.name, fontsize=10)
    try:
        ax.set_box_aspect((1, 1, 1))
        lim = np.max(np.abs(np.vstack([coil.starts, coil.ends])))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    except Exception:
        pass
    return ax
