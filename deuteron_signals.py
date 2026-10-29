"""
deuteron_signals.py -- two-zone deuteron NMR signals for non-uniform polarization.

Builds the cw-NMR absorption signal a coil sees when the material inside the beam
raster has polarization P_in and the unrastered shell has P_out, weighted by the
coil's sampling (the raster fraction f from coilsim), and compares the two ways
of turning it into a polarization:

  area method      P_area = CC * area. Area is exactly linear in P, so with a
                   calibration constant taken on uniformly polarized material
                   (TE or a clean golden event) P_area = f*P_in + (1-f)*P_out.
  peak-fit method  fit a single-r Dulya lineshape, P = (r^2-1)/(r^2+r+1).
                   Peak heights are nonlinear in P, so a two-zone signal does not
                   fit to the weighted mean.

Lineshape: absorption-only Dulya model as in jdmax/DeuteronPeakFit
(deuteron_fit._fit_func), copied here (MIT license) so this module runs without
lmfit. Where lmfit and deuteron_fit.py are importable, peak_fit_polarization(...,
use_deuteronpeakfit=True) calls the original fit() with all seven parameters free.

Default lineshape parameters are from an absorption-only fit to the RGC example
event in DeuteronPeakFit (frequencies in MHz).
"""
from __future__ import annotations
import numpy as np

# Fit to DeuteronPeakFit/example_data (RGC ND3, 5 T): absorption-only model
RGC_LINESHAPE = dict(A=0.0782, wQ=0.02722, wL=32.698, eta=0.0, xi=0.0)
RGC_FREQS = np.linspace(32.3, 33.1, 512)


# --- Dulya lineshape (from deuteron_fit.py, absorption part) ------------------
def _iplus(r, Q, R):
    r3QR = r ** (-3 * Q * R)
    NN = r * (r + r3QR) + 1
    return r * (r - r3QR) / NN


def _iminus(r, Q, R):
    r3QR = r ** (3 * Q * R)
    NN = r * (r + r3QR) + 1
    return (r * r3QR - 1) / NN


def _integrals(R, A, eps, Y2, etac2p):
    Y = np.sqrt(Y2)
    Yx2 = 2 * Y
    z2 = 1 - eps * R - etac2p
    q4 = z2 * z2 + A * A
    q2 = np.sqrt(q4)
    qq = np.sqrt(q2)
    cosa = z2 / q2
    cosa_2 = 1 / np.sqrt(2) * np.sqrt(1 + cosa)
    sina_2 = 1 / np.sqrt(2) * np.sqrt(1 - cosa)
    fTmp = Y2 + q2
    fVal = Yx2 * qq * cosa_2
    T = cosa_2 * (np.pi / 2 + np.arctan((Y2 - q2) / (Yx2 * qq * sina_2)))
    L = 0.5 * sina_2 * np.log((fTmp + fVal) / (fTmp - fVal))
    return (T + L) / (2 * qq)


def _f(R, A, eps, eta):
    if eta < 0.001:
        return _integrals(R, A, eps, 3, 0)
    FF = 0
    dphi = 1
    for i in (0, 1):
        ec2p = eta * np.cos(np.pi * dphi * i)
        Y2 = 3 - ec2p
        FF += 0.5 * np.sqrt(3) / np.sqrt(Y2) * _integrals(R, A, eps, Y2, 0)
    for N in [2 ** n for n in range(2, 6)]:
        dphi = 1 / N
        for i in range(N - 1, 0, -2):
            ec2p = eta * np.cos(np.pi * dphi * i)
            Y2 = 3 - ec2p
            FF += np.sqrt(3) / np.sqrt(Y2) * _integrals(R, A, eps, Y2, ec2p)
    return FF * dphi


def transition_shapes(freqs, A, wQ, wL, eta=0.0, xi=0.0):
    """r-independent pieces of the lineshape: R, F-, F+, and the asymmetry factor.
    lineshape(r) = G * fAsym * (I-(r) F- + I+(r) F+)."""
    w = np.asarray(freqs, float)
    R = (w - wL) / (3 * wQ)
    Fm = _f(R, A, -1, eta) / wQ
    Fp = _f(R, A, 1, eta) / wQ
    fAsym = 1 + 0.5 * xi * (1 + R)
    return dict(R=R, Fm=Fm, Fp=Fp, fAsym=fAsym, Q=wQ / wL)


def lineshape(r, ts, G=1.0):
    return G * ts["fAsym"] * (_iminus(r, ts["Q"], ts["R"]) * ts["Fm"]
                              + _iplus(r, ts["Q"], ts["R"]) * ts["Fp"])


# --- polarization <-> r ---------------------------------------------------------
def P_of_r(r):
    return (r * r - 1) / (r * r + r + 1)


def r_of_P(P):
    """Inverse of P(r): the positive root of (1-P) r^2 - P r - (1+P) = 0."""
    P = np.asarray(P, float)
    return (P + np.sqrt(4 - 3 * P * P)) / (2 * (1 - P))


def Pzz_of_r(r):
    return (r - 1) ** 2 / (r * r + r + 1)


# --- two-zone signal and the two methods ----------------------------------------
def two_zone_signal(P_in, P_out, f, ts):
    """Signal = f * shape(P_in) + (1-f) * shape(P_out), unit gain."""
    return f * lineshape(r_of_P(P_in), ts) + (1 - f) * lineshape(r_of_P(P_out), ts)


def area_polarization(signal, freqs, ts):
    """Area method with a calibration constant from a uniformly polarized signal
    of the same lineshape (TE or clean golden event)."""
    ref_P = 0.4
    cc = ref_P / np.trapezoid(lineshape(r_of_P(ref_P), ts), freqs)
    return cc * np.trapezoid(signal, freqs)


def _fit_G_r(signal, ts, P_lo=-0.97, P_hi=0.97):
    """Least-squares fit of G and r (other lineshape parameters fixed). G is linear,
    so chi2(r) = |y|^2 - (s.y)^2/(s.s); minimize over r by grid + golden section."""
    y = np.asarray(signal, float)

    def chi2(r):
        s = lineshape(r, ts)
        ss = s @ s
        return y @ y - (s @ y) ** 2 / ss if ss > 0 else np.inf

    Ps = np.linspace(P_lo, P_hi, 389)
    cs = np.array([chi2(r_of_P(p)) for p in Ps])
    k = int(np.argmin(cs))
    a, b = Ps[max(k - 1, 0)], Ps[min(k + 1, len(Ps) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = chi2(r_of_P(c)), chi2(r_of_P(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = chi2(r_of_P(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = chi2(r_of_P(d))
    P = 0.5 * (a + b)
    r = r_of_P(P)
    s = lineshape(r, ts)
    G = (s @ y) / (s @ s)
    return dict(P=float(P), r=float(r), G=float(G), fit=G * s)


def peak_fit_polarization(signal, freqs, ts=None, use_deuteronpeakfit=False,
                          start=None):
    """Peak-fit polarization. Default: fit G and r with the lineshape parameters
    fixed (in tests, freeing A, wQ, wL, eta, xi changes P by < 1e-4).
    use_deuteronpeakfit=True calls deuteron_fit.fit() (needs lmfit) with all
    seven parameters free, starting from RGC_LINESHAPE."""
    if not use_deuteronpeakfit:
        return _fit_G_r(signal, ts)
    import deuteron_fit as dfit   # from github.com/jdmax/DeuteronPeakFit
    p0 = dict(RGC_LINESHAPE)
    p0.update(start or {})
    res0 = _fit_G_r(signal, ts) if ts is not None else dict(r=1.5, G=1.0)
    init = dict(A=p0["A"], G=res0["G"], r=res0["r"], wQ=p0["wQ"], wL=p0["wL"],
                eta=p0["eta"], xi=p0["xi"])
    res = dfit.fit(freqs, signal, init)
    r = res.params["r"].value
    return dict(P=float(P_of_r(r)), r=float(r), G=res.params["G"].value, fit=res.best_fit)


def compare(P_in, P_out, f, freqs=RGC_FREQS, shape_params=RGC_LINESHAPE):
    """Everything for one scenario: signal, both polarizations and the fit."""
    ts = transition_shapes(freqs, **shape_params)
    sig = two_zone_signal(P_in, P_out, f, ts)
    pf = peak_fit_polarization(sig, freqs, ts)
    return dict(signal=sig, raster_part=f * lineshape(r_of_P(P_in), ts),
                shell_part=(1 - f) * lineshape(r_of_P(P_out), ts),
                P_area=float(area_polarization(sig, freqs, ts)), P_fit=pf["P"],
                fit=pf["fit"], resid_rms_rel=float(np.sqrt(np.mean((sig - pf["fit"]) ** 2))
                                                   / np.abs(sig).max()))
