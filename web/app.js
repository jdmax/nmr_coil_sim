// NMR Coil Raster Explorer -- GitHub Pages front end.
// Preset coils come precomputed in data/default.json; new coils are computed by the
// same Python code (coilsim.py + explorer_data.py) running in Pyodide in worker.js.
(async function main() {
"use strict";
let DATA;
try {
  DATA = await (await fetch("data/default.json")).json();
} catch (err) {
  document.querySelector("header p").textContent = "Could not load data/default.json: " + err;
  return;
}


/* ---------- decoding ---------- */
function b64ToBytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
const f32 = s => new Float32Array(b64ToBytes(s).buffer);
const u8 = s => b64ToBytes(s);

const LUT = DATA.lut.map(h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
const VMAX = 3;
function lutColor(v) { const i = Math.max(0, Math.min(255, Math.round(v / VMAX * 255))); return LUT[i]; }

function decodeEntry(c) {
  const m = c.map; m.v = f32(m.v); m.w = u8(m.w);
  const n = m.n, S = new Float32Array(n * n); let tot = 0;
  for (let i = 0; i < n * n; i++) { S[i] = m.v[i] * m.w[i] / 255; tot += S[i]; }
  m.S = S; m.tot = tot;
  m.coord = new Float32Array(n); for (let i = 0; i < n; i++) m.coord[i] = m.x0 + i * m.pixel;
  const s = c.surf; s.side = f32(s.side); s.lo = f32(s.lo); s.hi = f32(s.hi);
  c.seg = f32(c.wires.seg); c.cur = f32(c.wires.cur);
  return c;
}
for (const cell of DATA.cells) cell.coils.forEach(decodeEntry);

/* ---------- state ---------- */
const st = { cell: 0, coil: 0, rd: 12, ox: 0, oy: 0, yaw: -0.85, pitch: 0.42 };
const cell = () => DATA.cells[st.cell];
const coil = () => cell().coils[st.coil];

function fraction(m, r, ox, oy) {
  const n = m.n, p = m.pixel, x0 = m.x0, S = m.S, r2 = r * r;
  const j0 = Math.max(0, Math.floor((oy - r - x0) / p)), j1 = Math.min(n - 1, Math.ceil((oy + r - x0) / p));
  const i0 = Math.max(0, Math.floor((ox - r - x0) / p)), i1 = Math.min(n - 1, Math.ceil((ox + r - x0) / p));
  let s = 0;
  for (let j = j0; j <= j1; j++) {
    const dy = m.coord[j] - oy, dy2 = dy * dy; if (dy2 > r2) continue;
    const row = j * n;
    for (let i = i0; i <= i1; i++) { const dx = m.coord[i] - ox; if (dx * dx + dy2 <= r2) s += S[row + i]; }
  }
  return s / m.tot;
}
// largest |change| in fraction when the raster moves 1 mm in any of 16 directions
function worstShift(m, r, f0) {
  let worst = 0;
  for (let k = 0; k < 16; k++) {
    const a = 2 * Math.PI * k / 16;
    worst = Math.max(worst, Math.abs(fraction(m, r, st.ox + Math.cos(a), st.oy + Math.sin(a)) - f0));
  }
  return worst;
}
const fmt = (x, d = 3) => Number.isFinite(x) ? x.toFixed(d) : "–";

/* ---------- theme helpers ---------- */
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- controls ---------- */
const coilSel = document.getElementById("coilSel");
const rasterR = document.getElementById("rasterR");
const offX = document.getElementById("offX"), offY = document.getElementById("offY");

function buildControls() {
  const seg = document.getElementById("cellSeg");
  seg.innerHTML = "";
  DATA.cells.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = c.length === 50 ? `${c.diam} mm` : `${c.diam} × ${c.length} mm`; b.setAttribute("aria-pressed", i === st.cell);
    b.addEventListener("click", () => setCell(i));
    seg.appendChild(b);
  });
  coilSel.innerHTML = "";
  let og = null, last = null;
  cell().coils.forEach((c, i) => {
    if (c.group !== last) { og = document.createElement("optgroup"); og.label = c.group; coilSel.appendChild(og); last = c.group; }
    const o = document.createElement("option"); o.value = i; o.textContent = c.name; og.appendChild(o);
  });
  coilSel.value = st.coil;
  rasterR.max = cell().diam;
}
function setCell(i) {
  const key = coil().key;
  st.cell = i;
  st.coil = Math.max(0, cell().coils.findIndex(c => c.key === key));
  st.rd = Math.min(st.rd, cell().diam - 2);
  if (i === 1 && st.rd === 12) st.rd = 18;
  if (i === 0 && st.rd === 18) st.rd = 12;
  st.ox = 0; st.oy = 0;
  buildControls(); build3d(); updateAll();
}
coilSel.addEventListener("change", () => { st.coil = +coilSel.value; build3d(); updateAll(); });
rasterR.addEventListener("input", () => { st.rd = +rasterR.value; updateAll(); });
offX.addEventListener("change", () => { st.ox = +offX.value || 0; updateAll(); });
offY.addEventListener("change", () => { st.oy = +offY.value || 0; updateAll(); });
document.getElementById("btnCenter").addEventListener("click", () => { st.ox = 0; st.oy = 0; updateAll(); });
document.getElementById("btnFig8").addEventListener("click", () => { st.ox = 0; st.oy = -1; updateAll(); });

/* ---------- 2D map ---------- */
const mapCv = document.getElementById("map");
let mapImgCache = { key: null, canvas: null };
function mapImage(m, key) {
  if (mapImgCache.key === key) return mapImgCache.canvas;
  const n = m.n, oc = document.createElement("canvas"); oc.width = n; oc.height = n;
  const ctx = oc.getContext("2d"), img = ctx.createImageData(n, n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const src = j * n + i, dst = ((n - 1 - j) * n + i) * 4;   // flip: row 0 is y min
    if (m.w[src] === 0) { img.data[dst + 3] = 0; continue; }
    const c = lutColor(m.v[src]); img.data[dst] = c[0]; img.data[dst + 1] = c[1]; img.data[dst + 2] = c[2]; img.data[dst + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  mapImgCache = { key, canvas: oc };
  return oc;
}
function mapGeom() {
  const R = cell().radius, ext = R + 5.5;
  const W = mapCv.clientWidth, H = mapCv.clientHeight, s = Math.min(W, H) / (2 * ext);
  return { R, ext, W, H, s, cx: W / 2, cy: H / 2 };
}
function drawMap() {
  const dpr = window.devicePixelRatio || 1, g = mapGeom();
  mapCv.width = Math.round(g.W * dpr); mapCv.height = Math.round(g.H * dpr);
  const ctx = mapCv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, g.W, g.H);
  const c = coil(), m = c.map, R = g.R;
  const X = x => g.cx + x * g.s, Y = y => g.cy - y * g.s;
  // grid
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted"); ctx.font = `11px ${css("--f-mono")}`;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = -Math.floor(g.ext / 5) * 5; t <= g.ext; t += 5) {
    ctx.beginPath(); ctx.moveTo(X(t), Y(-g.ext)); ctx.lineTo(X(t), Y(g.ext)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(-g.ext), Y(t)); ctx.lineTo(X(g.ext), Y(t)); ctx.stroke();
    if (t !== 0) { ctx.fillText(t, X(t), Y(-g.ext) - 15); }
  }
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (let t = -Math.floor(g.ext / 5) * 5; t <= g.ext; t += 5) if (t !== 0) ctx.fillText(t, X(-g.ext) + 3, Y(t));
  // field map, clipped to the cell
  const img = mapImage(m, `${st.cell}:${c.key}`);
  const half = (m.n * m.pixel) / 2;
  ctx.save(); ctx.beginPath(); ctx.arc(X(0), Y(0), R * g.s, 0, 2 * Math.PI); ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(img, X(m.x0 - m.pixel / 2), Y(m.x0 - m.pixel / 2 + m.n * m.pixel), m.n * m.pixel * g.s, m.n * m.pixel * g.s);
  ctx.restore();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(X(0), Y(0), R * g.s, 0, 2 * Math.PI); ctx.stroke();
  // conductors crossing z = 0
  const seg = c.seg, cur = c.cur, nseg = cur.length; let anyIn = false, coilAbove = 0;
  for (let k = 0; k < nseg; k++) {
    const a = k * 6, za = seg[a + 2], zb = seg[a + 5];
    if ((za - 1e-4) * (zb - 1e-4) >= 0) continue;
    const t = (1e-4 - za) / (zb - za), x = seg[a] + t * (seg[a + 3] - seg[a]), y = seg[a + 1] + t * (seg[a + 4] - seg[a + 1]);
    if (Math.abs(x) < g.ext && Math.abs(y) < g.ext) {
      anyIn = true; ctx.fillStyle = css("--copper");
      ctx.beginPath(); ctx.arc(X(x), Y(y), 2.2 + 1.6 * Math.min(1, cur[k]), 0, 2 * Math.PI); ctx.fill();
    } else if (y > g.ext) coilAbove = Math.max(coilAbove, y);
  }
  if (coilAbove) {
    ctx.fillStyle = css("--copper"); ctx.font = `600 12px ${css("--f-body")}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(`▲ coil plane ${coilAbove.toFixed(0)} mm above axis`, X(0), Y(g.ext) + 6);
  }
  // raster
  const rr = st.rd / 2;
  ctx.lineWidth = 3.5; ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath(); ctx.arc(X(st.ox), Y(st.oy), rr * g.s, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 2; ctx.strokeStyle = "#ffffff";
  ctx.beginPath(); ctx.arc(X(st.ox), Y(st.oy), rr * g.s, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(X(st.ox), Y(st.oy), 3, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1; ctx.stroke();
  // axis labels
  ctx.fillStyle = css("--muted"); ctx.font = `11px ${css("--f-body")}`; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
  ctx.fillText("x (mm) →", g.W - 6, g.H - 4);
  ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText("↑ y (mm)", 6, 4);
}
// drag
let drag = null;
function mmFromEvent(e) { const r = mapCv.getBoundingClientRect(), g = mapGeom(); return { x: (e.clientX - r.left - g.cx) / g.s, y: -(e.clientY - r.top - g.cy) / g.s }; }
mapCv.addEventListener("pointerdown", e => {
  const p = mmFromEvent(e);
  drag = { dx: st.ox - p.x, dy: st.oy - p.y };
  if (Math.hypot(p.x - st.ox, p.y - st.oy) > st.rd / 2) { drag = { dx: 0, dy: 0 }; st.ox = p.x; st.oy = p.y; }
  mapCv.setPointerCapture(e.pointerId); mapCv.classList.add("dragging"); updateAll();
});
mapCv.addEventListener("pointermove", e => {
  if (!drag) return; const p = mmFromEvent(e), lim = cell().radius + 3;
  st.ox = Math.max(-lim, Math.min(lim, Math.round((p.x + drag.dx) * 20) / 20));
  st.oy = Math.max(-lim, Math.min(lim, Math.round((p.y + drag.dy) * 20) / 20));
  updateAll();
});
const endDrag = () => { drag = null; mapCv.classList.remove("dragging"); };
mapCv.addEventListener("pointerup", endDrag); mapCv.addEventListener("pointercancel", endDrag);
mapCv.addEventListener("keydown", e => {
  const d = e.shiftKey ? 1 : 0.1, k = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[e.key];
  if (!k) return; e.preventDefault(); st.ox = +(st.ox + k[0]).toFixed(2); st.oy = +(st.oy + k[1]).toFixed(2); updateAll();
});

/* ---------- 3D view ---------- */
const cv3 = document.getElementById("view3d");
let geo3 = null;
function build3d() {
  const c = coil(), C = cell(), s = c.surf, R = C.radius, L = C.length;
  const quads = [];
  const P = (r, ph, z) => [r * Math.cos(ph), r * Math.sin(ph), z];
  const nphi = s.nphi, nz = s.nz, ncap = s.ncap;
  const phiAt = i => 2 * Math.PI * i / (nphi - 1);
  for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nphi - 1; i++) {
    const z0 = -L / 2 + L * j / (nz - 1), z1 = -L / 2 + L * (j + 1) / (nz - 1);
    const v = (s.side[j * nphi + i] + s.side[j * nphi + i + 1] + s.side[(j + 1) * nphi + i] + s.side[(j + 1) * nphi + i + 1]) / 4;
    quads.push({ p: [P(R, phiAt(i), z0), P(R, phiAt(i + 1), z0), P(R, phiAt(i + 1), z1), P(R, phiAt(i), z1)], col: lutColor(v) });
  }
  for (const [arr, z] of [[s.lo, -L / 2], [s.hi, L / 2]]) {
    for (let a = 0; a < ncap - 1; a++) for (let i = 0; i < nphi - 1; i++) {
      const r0 = R * a / (ncap - 1), r1 = R * (a + 1) / (ncap - 1);
      const v = (arr[a * nphi + i] + arr[a * nphi + i + 1] + arr[(a + 1) * nphi + i] + arr[(a + 1) * nphi + i + 1]) / 4;
      quads.push({ p: [P(r0, phiAt(i), z), P(r0, phiAt(i + 1), z), P(r1, phiAt(i + 1), z), P(r1, phiAt(i), z)], col: lutColor(v) });
    }
  }
  const lines = []; const seg = c.seg, cur = c.cur;
  let rmax = Math.max(R, L / 2);
  for (let k = 0; k < cur.length; k++) {
    const a = [seg[k * 6], seg[k * 6 + 1], seg[k * 6 + 2]], b = [seg[k * 6 + 3], seg[k * 6 + 4], seg[k * 6 + 5]];
    // wires inside the cell are hidden by its surface; draw3d shows them dashed on top
    const inside = [a, b].every(p => Math.hypot(p[0], p[1]) < R && Math.abs(p[2]) <= L / 2);
    lines.push({ a, b, w: 1 + 1.8 * Math.min(1, cur[k]), inside });
    rmax = Math.max(rmax, Math.hypot(...a), Math.hypot(...b));
  }
  geo3 = { quads, lines, rmax };
}
function rot(p) {
  // yaw about y, then pitch about x
  const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
  const x1 = cy * p[0] + sy * p[2], z1 = -sy * p[0] + cy * p[2], y1 = p[1];
  const y2 = cp * y1 - sp * z1, z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}
function draw3d() {
  if (!geo3) return;
  const dpr = window.devicePixelRatio || 1, W = cv3.clientWidth, H = cv3.clientHeight;
  cv3.width = Math.round(W * dpr); cv3.height = Math.round(H * dpr);
  const ctx = cv3.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const sc = 0.46 * Math.min(W, H) / geo3.rmax, cx = W / 2, cy = H / 2;
  const prj = q => [cx + q[0] * sc, cy - q[1] * sc];
  const items = [];
  for (const q of geo3.quads) {
    const r = q.p.map(rot); const d = (r[0][2] + r[1][2] + r[2][2] + r[3][2]) / 4;
    items.push({ t: 0, d, pts: r.map(prj), col: q.col });
  }
  for (const l of geo3.lines) {
    const a = rot(l.a), b = rot(l.b);
    items.push({ t: 1, d: l.inside ? Infinity : (a[2] + b[2]) / 2 + 0.3, a: prj(a), b: prj(b), w: l.w, inside: l.inside });
  }
  // raster footprint on both end caps
  const C = cell(), rr = st.rd / 2;
  for (const z of [-C.length / 2, C.length / 2]) for (let i = 0; i < 48; i++) {
    const t0 = 2 * Math.PI * i / 48, t1 = 2 * Math.PI * (i + 1) / 48;
    const a = rot([st.ox + rr * Math.cos(t0), st.oy + rr * Math.sin(t0), z]), b = rot([st.ox + rr * Math.cos(t1), st.oy + rr * Math.sin(t1), z]);
    items.push({ t: 2, d: (a[2] + b[2]) / 2 + 0.2, a: prj(a), b: prj(b) });
  }
  items.sort((u, v) => u.d - v.d);   // far (negative z after rotation toward viewer = +z) first
  const copper = css("--copper");
  for (const it of items) {
    if (it.t === 0) {
      const c = `rgb(${it.col[0]},${it.col[1]},${it.col[2]})`;
      ctx.fillStyle = c; ctx.strokeStyle = c; ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.moveTo(...it.pts[0]); for (let k = 1; k < 4; k++) ctx.lineTo(...it.pts[k]); ctx.closePath(); ctx.fill(); ctx.stroke();
    } else if (it.t === 1) {
      ctx.strokeStyle = copper; ctx.lineWidth = it.w; ctx.lineCap = it.inside ? "butt" : "round";
      ctx.setLineDash(it.inside ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(...it.a); ctx.lineTo(...it.b); ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(...it.a); ctx.lineTo(...it.b); ctx.stroke();
    }
  }
  // axis triad
  const o = [44, H - 40], ink = css("--ink-2");
  ctx.font = `600 11px ${css("--f-body")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const [v, lab] of [[[1, 0, 0], "x"], [[0, 1, 0], "y"], [[0, 0, 1], "z beam"]]) {
    const r = rot(v); ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(o[0] + r[0] * 24, o[1] - r[1] * 24); ctx.stroke();
    ctx.fillStyle = ink; ctx.fillText(lab, o[0] + r[0] * 34, o[1] - r[1] * 34);
  }
}
let rotDrag = null;
cv3.addEventListener("pointerdown", e => { rotDrag = { x: e.clientX, y: e.clientY, yaw: st.yaw, pitch: st.pitch }; cv3.setPointerCapture(e.pointerId); });
cv3.addEventListener("pointermove", e => {
  if (!rotDrag) return;
  st.yaw = rotDrag.yaw + (e.clientX - rotDrag.x) * 0.01;
  st.pitch = Math.max(-1.5, Math.min(1.5, rotDrag.pitch + (e.clientY - rotDrag.y) * 0.01));
  requestAnimationFrame(draw3d);
});
cv3.addEventListener("pointerup", () => rotDrag = null); cv3.addEventListener("pointercancel", () => rotDrag = null);
cv3.addEventListener("keydown", e => {
  const k = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
  if (!k) return; e.preventDefault(); st.yaw += k[0]; st.pitch = Math.max(-1.5, Math.min(1.5, st.pitch + k[1])); draw3d();
});

/* ---------- tables ---------- */
function chipP(c) {
  const C = c.rungs ? c.C_rung_p_pF : c.C_p_pF;
  let h = C >= 6 ? `<span class="chip ok">${c.rungs ? "Tunable" : "Like RGC loop"}</span>` : C >= 3 ? `<span class="chip warn">Tight</span>`
    : `<span class="chip bad">Below strays</span>`;
  if (!c.rungs && c.len_lambda > 0.1) h += ` <span class="chip warn">Run &gt; λ/10</span>`;
  return h;
}
function chipD(c) {
  if (c.rungs) return `<span class="chip neutral">Per-rung capacitors</span>`;
  if (c.C_d_pF >= 40 && c.C_d_pF <= 82) return `<span class="chip ok">In RGC board range</span>`;
  if (c.C_d_pF > 82) return `<span class="chip neutral">Add fixed C</span>`;
  return `<span class="chip warn">Below board minimum</span>`;
}
function buildTables() {
  document.getElementById("fp").textContent = DATA.freqs.p.toFixed(1);
  document.getElementById("fd").textContent = DATA.freqs.d.toFixed(1);
  const tb = document.getElementById("tuneBody"); tb.innerHTML = "";
  cell().coils.forEach((c, i) => {
    const tr = document.createElement("tr"); tr.dataset.i = i;
    const rm = c.group === "Your coils" ? ` <button class="rm" type="button" title="Remove ${c.name}" aria-label="Remove ${c.name}">×</button>` : "";
    tr.innerHTML = `<td>${c.name}${rm}<span class="group">${c.group}</span></td>
      <td class="num">${c.L_nH.toFixed(0)}</td>
      <td class="num">${c.rungs ? c.C_rung_p_pF.toFixed(1) + `<span class="group">each of ${c.rungs} rungs</span>` : c.C_p_pF.toFixed(1)}</td><td>${chipP(c)}</td>
      <td class="num">${c.rungs ? c.C_rung_d_pF.toFixed(0) + `<span class="group">each of ${c.rungs} rungs</span>` : c.C_d_pF.toFixed(0)}</td><td>${chipD(c)}</td>
      <td class="num">${(100 * c.eta).toFixed(1)}</td>
      <td class="num">${c.rungs ? "–" : c.len_lambda.toFixed(2)}</td>`;
    tr.addEventListener("click", () => selectCoil(i));
    if (rm) tr.querySelector(".rm").addEventListener("click", e => { e.stopPropagation(); removeCoils([c]); });
    tb.appendChild(tr);
  });
  buildSignalRows();
  const cb = document.getElementById("cmpBody"); cb.innerHTML = "";
  cell().coils.forEach((c, i) => {
    const tr = document.createElement("tr"); tr.dataset.i = i;
    tr.innerHTML = `<td>${c.name}<span class="group">${c.group}</span></td>
      <td class="barcell"><div class="bar"><div class="fill"></div><div class="tick"></div></div></td>
      <td class="num f"></td><td class="num s"></td><td class="num pa"></td><td class="num pf"></td>`;
    tr.addEventListener("click", () => selectCoil(i));
    cb.appendChild(tr);
  });
}
function selectCoil(i) { st.coil = i; coilSel.value = i; build3d(); updateAll(); }

/* ---------- deuteron two-zone signal ---------- */
// Dulya absorption lineshape as in DeuteronPeakFit (eta = 0, xi = 0), parameters from an
// absorption-only fit to its RGC example event. F+ and F- do not depend on r, so they are
// computed once; a lineshape for any r is then I-(r) F- + I+(r) F+.
const DS = { A: 0.0782, wQ: 0.02722, wL: 32.698, f0: 32.30, f1: 33.10, n: 401 };
DS.freq = new Float64Array(DS.n); DS.R = new Float64Array(DS.n); DS.Fm = new Float64Array(DS.n); DS.Fp = new Float64Array(DS.n);
DS.df = (DS.f1 - DS.f0) / (DS.n - 1); DS.Q = DS.wQ / DS.wL;
function dulya(R, A, eps) {
  const Y2 = 3, Yx2 = 2 * Math.sqrt(3), z2 = 1 - eps * R, q2 = Math.sqrt(z2 * z2 + A * A), qq = Math.sqrt(q2);
  const cosa = z2 / q2, c2 = Math.SQRT1_2 * Math.sqrt(1 + cosa), s2 = Math.SQRT1_2 * Math.sqrt(1 - cosa);
  const fT = Y2 + q2, fV = Yx2 * qq * c2;
  const T = c2 * (Math.PI / 2 + Math.atan((Y2 - q2) / (Yx2 * qq * s2)));
  const L = 0.5 * s2 * Math.log((fT + fV) / (fT - fV));
  return (T + L) / (2 * qq);
}
for (let i = 0; i < DS.n; i++) {
  const w = DS.f0 + i * DS.df, R = (w - DS.wL) / (3 * DS.wQ);
  DS.freq[i] = w; DS.R[i] = R; DS.Fm[i] = dulya(R, DS.A, -1) / DS.wQ; DS.Fp[i] = dulya(R, DS.A, 1) / DS.wQ;
}
const P_of_r = r => (r * r - 1) / (r * r + r + 1);
const r_of_P = P => (P + Math.sqrt(4 - 3 * P * P)) / (2 * (1 - P));
function shapeR(r, out) {
  out = out || new Float64Array(DS.n);
  for (let i = 0; i < DS.n; i++) {
    const e = 3 * DS.Q * DS.R[i], rp = Math.pow(r, -e), rm = Math.pow(r, e);
    const ip = r * (r - rp) / (r * (r + rp) + 1), im = (r * rm - 1) / (r * (r + rm) + 1);
    out[i] = im * DS.Fm[i] + ip * DS.Fp[i];
  }
  return out;
}
const trap = y => { let s = 0; for (let i = 0; i < y.length; i++) s += y[i]; return (s - 0.5 * (y[0] + y[y.length - 1])) * DS.df; };
const CC = 0.4 / trap(shapeR(r_of_P(0.4)));            // calibration on uniformly polarized material
// grid of lineshapes for the fit's coarse scan
const GRID = []; for (let k = 0; k <= 388; k++) { const P = -0.97 + 1.94 * k / 388, s = shapeR(r_of_P(P)); let ss = 0; for (const v of s) ss += v * v; GRID.push({ P, s, ss }); }
function fitPeak(y) {
  let yy = 0; for (const v of y) yy += v * v;
  const chiS = (s, ss) => { let sy = 0; for (let i = 0; i < DS.n; i++) sy += s[i] * y[i]; return yy - sy * sy / ss; };
  const chiP = P => { const s = shapeR(r_of_P(P)); let ss = 0; for (const v of s) ss += v * v; return chiS(s, ss); };
  let kb = 0, cb = Infinity;
  GRID.forEach((g, k) => { const c = chiS(g.s, g.ss); if (c < cb) { cb = c; kb = k; } });
  let a = GRID[Math.max(0, kb - 1)].P, b = GRID[Math.min(GRID.length - 1, kb + 1)].P;
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a), d = a + gr * (b - a), fc = chiP(c), fd = chiP(d);
  for (let it = 0; it < 50; it++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = chiP(c); }
    else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = chiP(d); }
  }
  const P = 0.5 * (a + b), s = shapeR(r_of_P(P));
  let ss = 0, sy = 0; for (let i = 0; i < DS.n; i++) { ss += s[i] * s[i]; sy += s[i] * y[i]; }
  const G = sy / ss, fit = s.map(v => v * G);
  return { P, fit };
}
const dState = { pin: 0.30, pout: 0.45, key: "", Sin: null, Sout: null };
function deutParts() {
  const key = `${dState.pin}|${dState.pout}`;
  if (dState.key !== key) { dState.Sin = shapeR(r_of_P(dState.pin)); dState.Sout = shapeR(r_of_P(dState.pout)); dState.key = key; }
  return dState;
}
function deut(f, full) {
  const d = deutParts(), y = new Float64Array(DS.n);
  for (let i = 0; i < DS.n; i++) y[i] = f * d.Sin[i] + (1 - f) * d.Sout[i];
  const area = CC * trap(y), pf = fitPeak(y);
  if (!full) return { area, fit: pf.P };
  let rs = 0, mx = 0; for (let i = 0; i < DS.n; i++) { rs += (y[i] - pf.fit[i]) ** 2; mx = Math.max(mx, Math.abs(y[i])); }
  return { area, fit: pf.P, y, fitY: pf.fit, res: Math.sqrt(rs / DS.n) / mx };
}
const dsCv = document.getElementById("dsig");
function drawDeut(f, r) {
  const dpr = window.devicePixelRatio || 1, W = dsCv.clientWidth, H = dsCv.clientHeight;
  dsCv.width = Math.round(W * dpr); dsCv.height = Math.round(H * dpr);
  const ctx = dsCv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const d = deutParts(), x0 = 32.40, x1 = 33.00, padL = 44, padR = 22, padT = 12, padB = 30;
  let mx = 0, mn = 0; for (let i = 0; i < DS.n; i++) { mx = Math.max(mx, Math.abs(r.y[i])); }
  const norm = 1 / mx;
  const series = [
    { y: Array.from(d.Sin, v => v * f * norm), col: css("--copper"), w: 1.6, dash: [6, 4] },
    { y: Array.from(d.Sout, v => v * (1 - f) * norm), col: css("--shell"), w: 1.6, dash: [6, 4] },
    { y: Array.from(r.y, v => v * norm), col: css("--ink"), w: 2.2, dash: [] },
    { y: Array.from(r.fitY, v => v * norm), col: css("--fitc"), w: 1.6, dash: [2, 3] },
  ];
  for (const s of series) for (const v of s.y) { mn = Math.min(mn, v); }
  const ymin = Math.min(-0.05, mn * 1.05), ymax = 1.08;
  const X = w => padL + (w - x0) / (x1 - x0) * (W - padL - padR), Y = v => padT + (ymax - v) / (ymax - ymin) * (H - padT - padB);
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted"); ctx.font = `11px ${css("--f-mono")}`;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 32.4; t <= 33.0001; t += 0.1) { ctx.beginPath(); ctx.moveTo(X(t), Y(ymax)); ctx.lineTo(X(t), Y(ymin)); ctx.stroke(); ctx.fillText(t.toFixed(1), X(t), H - padB + 6); }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(padL, Y(t)); ctx.lineTo(W - padR, Y(t)); ctx.stroke(); ctx.fillText(t.toFixed(1), padL - 6, Y(t)); }
  ctx.strokeStyle = css("--muted"); ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
  ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--muted"); ctx.font = `11px ${css("--f-body")}`;
  ctx.fillText("frequency (MHz)", W - padR, H - 2);
  ctx.save(); ctx.beginPath(); ctx.rect(padL, padT, W - padL - padR, H - padT - padB); ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.col; ctx.lineWidth = s.w; ctx.setLineDash(s.dash); ctx.beginPath();
    for (let i = 0; i < DS.n; i++) { const px = X(DS.freq[i]), py = Y(s.y[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
    ctx.stroke();
  }
  ctx.restore(); ctx.setLineDash([]);
}
const pIn = document.getElementById("pIn"), pOut = document.getElementById("pOut");
pIn.addEventListener("input", () => { dState.pin = +pIn.value; updateAll(); });
pOut.addEventListener("input", () => { dState.pout = +pOut.value; updateAll(); });
const sgn = x => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
function updateDeut(f) {
  const r = deut(f, true);
  document.getElementById("pInVal").textContent = dState.pin.toFixed(2);
  document.getElementById("pOutVal").textContent = dState.pout.toFixed(2);
  document.getElementById("dF").textContent = fmt(f, 3);
  document.getElementById("dTrue").textContent = dState.pin.toFixed(3);
  document.getElementById("dArea").textContent = fmt(r.area, 3);
  document.getElementById("dAreaE").textContent = sgn(r.area - dState.pin);
  document.getElementById("dFit").textContent = fmt(r.fit, 3);
  document.getElementById("dFitE").textContent = sgn(r.fit - dState.pin);
  document.getElementById("dDiff").textContent = sgn(r.fit - r.area);
  document.getElementById("dRes").textContent = r.res < 1e-4 ? r.res.toExponential(1) : r.res.toFixed(4);
  drawDeut(f, r);
}

/* ---------- cw-NMR signal size ---------- */
// Series Q-meter circuit at resonance: the coil (L, R = wL/Q_coil) at the end of a lossy n·λ/2 cable, plus a
// damping resistor. To first order in the cable loss al the line presents Z_L + al (Z0 - Z_L^2/Z0), so it adds
// al (Z0 + (wL)^2/Z0) in series. Signal as a fraction of the Q-curve height: eta * Q_c * chi''.
const Z0 = 50, sig = { d: 0.15, p: 0.4, rd: 0 };
const NUC = ["d", "p"], REF = { d: "rgc_d", p: "rgc_p" };
function circuitQ(c, nuc) {
  const wL = 2 * Math.PI * DATA.freqs[nuc] * 1e6 * c.L_nH * 1e-9, al = sig[nuc] / 8.686;   // dB -> Np
  return wL / (wL / (nuc === "p" ? c.Q_p : c.Q_d) + al * (Z0 + wL * wL / Z0) + sig.rd);
}
const fx = x => !Number.isFinite(x) ? "–" : x >= 10 ? x.toFixed(0) : x >= 1 ? x.toFixed(1) : x.toFixed(2);
function buildSignalRows() {
  for (const n of NUC) document.getElementById(`sigF${n}`).textContent = DATA.freqs[n].toFixed(1);
  const sb = document.getElementById("sigBody"); sb.innerHTML = "";
  cell().coils.forEach((c, i) => {
    const tr = document.createElement("tr"); tr.dataset.i = i;
    tr.innerHTML = `<td>${c.name}<span class="group">${c.group}</span></td>` +
      NUC.map(n => `<td class="num q${n}"></td><td class="num t${n}"></td><td class="num r${n}"></td>`).join("");
    tr.addEventListener("click", () => selectCoil(i));
    sb.appendChild(tr);
  });
}
function updateSignals(r) {
  const coils = cell().coils, ref = {};
  for (const n of NUC) {
    const c = coils.find(e => e.key === REF[n]); if (!c) continue;
    const t = c.eta * circuitQ(c, n);
    ref[n] = { t, r: t * fraction(c.map, r, st.ox, st.oy) };
  }
  const rel = NUC.every(n => ref[n]);
  [...document.querySelectorAll("#sigBody tr")].forEach(tr => {
    const c = coils[+tr.dataset.i], fk = fraction(c.map, r, st.ox, st.oy);
    for (const n of NUC) {
      const q = circuitQ(c, n), t = c.eta * q;
      tr.querySelector(`.q${n}`).textContent = q.toFixed(1);
      tr.querySelector(`.t${n}`).textContent = rel ? fx(t / ref[n].t) + "×" : fx(t);
      tr.querySelector(`.r${n}`).textContent = rel ? fx(t * fk / ref[n].r) + "×" : fx(t * fk);
    }
    tr.classList.toggle("sel", +tr.dataset.i === st.coil);
  });
  document.getElementById("sigRef").textContent = rel
    ? "Signals relative to the RGC deuteron spiral and RGC proton loop on this cell, at this raster."
    : "No RGC coil on this cell size: signals are η·Q_c, per unit χ″, as a fraction of the Q-curve height.";
}
for (const [id, k] of [["sigLd", "d"], ["sigLp", "p"], ["sigRd", "rd"]]) {
  const el = document.getElementById(id); el.value = sig[k];
  el.addEventListener("input", () => {
    const v = +el.value; if (el.value === "" || !Number.isFinite(v) || v < 0) return;
    sig[k] = v; updateSignals(st.rd / 2);
  });
}

/* ---------- update ---------- */
function ceilingFrac() {
  // area of raster ∩ cell / area of cell, from the uniform weight map of any coil
  const m = cell().coils[0].map, n = m.n, r = st.rd / 2, r2 = r * r; let a = 0, t = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const w = m.w[j * n + i]; if (!w) continue; t += w;
    const dx = m.coord[i] - st.ox, dy = m.coord[j] - st.oy; if (dx * dx + dy * dy <= r2) a += w;
  }
  return a / t;
}
function updateAll() {
  const C = cell(), c = coil(), r = st.rd / 2;
  rasterR.value = st.rd; document.getElementById("rasterVal").textContent = `${st.rd.toFixed(1)} mm`;
  offX.value = st.ox.toFixed(2); offY.value = st.oy.toFixed(2);
  document.getElementById("coilNote").textContent = c.note;
  [...document.querySelectorAll("#cellSeg button")].forEach((b, i) => b.setAttribute("aria-pressed", i === st.cell));
  const f = fraction(c.map, r, st.ox, st.oy), ceil = ceilingFrac();
  const sens = worstShift(c.map, r, f);
  document.getElementById("fracBig").textContent = fmt(f, 3);
  document.getElementById("ceil").textContent = fmt(ceil, 3);
  document.getElementById("shell").textContent = fmt(1 - f, 3);
  document.getElementById("sens").textContent = "±" + fmt(sens, 3);
  const clips = Math.hypot(st.ox, st.oy) + r > C.radius + 1e-6;
  document.getElementById("clip").innerHTML = clips ? `<span class="chip bad">Clips the wall</span>` : `<span class="chip ok">Inside</span>`;
  [...document.querySelectorAll("#cmpBody tr")].forEach(tr => {
    const k = +tr.dataset.i, cc = C.coils[k];
    const fk = fraction(cc.map, r, st.ox, st.oy);
    const sk = worstShift(cc.map, r, fk);
    tr.querySelector(".fill").style.width = `${(100 * fk).toFixed(1)}%`;
    tr.querySelector(".tick").style.left = `calc(${(100 * ceil).toFixed(1)}% - 1px)`;
    tr.querySelector(".f").textContent = fmt(fk, 3);
    tr.querySelector(".s").textContent = "±" + fmt(sk, 3);
    const dk = deut(fk);
    tr.querySelector(".pa").textContent = fmt(dk.area, 3);
    tr.querySelector(".pf").textContent = fmt(dk.fit, 3);
    tr.classList.toggle("sel", k === st.coil);
  });
  [...document.querySelectorAll("#tuneBody tr")].forEach(tr => tr.classList.toggle("sel", +tr.dataset.i === st.coil));
  updateSignals(r);
  drawMap(); draw3d(); updateDeut(f);
}



/* ---------- coil designer (Pyodide worker) ---------- */
const STORE = "nmr_coil_sim.user.v1";
const dType = document.getElementById("dType"), dName = document.getElementById("dName");
const dCellD = document.getElementById("dCellD"), dCellL = document.getElementById("dCellL");
const dParams = document.getElementById("dParams");
const btnCompute = document.getElementById("btnCompute"), btnLoadSel = document.getElementById("btnLoadSel");
const btnRemove = document.getElementById("btnRemove"), btnRemoveAll = document.getElementById("btnRemoveAll"), computeMsg = document.getElementById("computeMsg");
const pyDot = document.getElementById("pyDot"), pyText = document.getElementById("pyText");
let pyReady = false, busy = false, jobId = 0;
const pending = new Map();

for (const [k, t] of Object.entries(DATA.types)) {
  const o = document.createElement("option"); o.value = k; o.textContent = t.label; dType.appendChild(o);
}
function renderParams(values) {
  const t = DATA.types[dType.value];
  dParams.innerHTML = "";
  for (const p of t.params) {
    const wrap = document.createElement("div"); wrap.className = "ctl";
    const id = `dp_${p.key}`;
    const v = values && values[p.key] !== undefined ? values[p.key] : p.default;
    if (p.unit === "bool") {
      wrap.innerHTML = `<label class="eyebrow" for="${id}">${p.label}</label>
        <select id="${id}"><option value="0">No</option><option value="1">Yes</option></select>`;
      wrap.querySelector("select").value = v ? "1" : "0";
    } else {
      const step = p.unit === "" ? 1 : 0.1;
      wrap.innerHTML = `<label class="eyebrow" for="${id}">${p.label}${p.unit ? ` (${p.unit})` : ""}</label>
        <input type="number" id="${id}" step="${step}" value="${v}">`;
    }
    dParams.appendChild(wrap);
  }
}
dType.addEventListener("change", () => { renderParams(); dName.value = DATA.types[dType.value].label; });
function readSpec() {
  const t = DATA.types[dType.value], spec = { type: dType.value, name: dName.value.trim() || t.label };
  for (const p of t.params) {
    const el = document.getElementById(`dp_${p.key}`); const v = +el.value;
    if (!Number.isFinite(v)) throw new Error(`${p.label} is not a number`);
    spec[p.key] = p.unit === "" ? Math.round(v) : v;
  }
  return spec;
}
function loadFromCoil(c) {
  const spec = c.spec || {};
  if (spec.type && DATA.types[spec.type]) { dType.value = spec.type; renderParams(spec); }
  dName.value = (c.name || "") + (c.group === "Your coils" ? "" : " (copy)");
  dCellD.value = cell().diam; dCellL.value = cell().length;
}
btnLoadSel.addEventListener("click", () => loadFromCoil(coil()));
function syncDesignerButtons() {
  btnRemove.hidden = !coil() || coil().group !== "Your coils";
  btnRemoveAll.hidden = !userCoils().length;
  btnCompute.disabled = !pyReady || busy;
}

// worker
let worker = null;
try {
  worker = new Worker("worker.js");
  worker.onmessage = e => {
    const msg = e.data;
    if (msg.type === "ready") {
      pyReady = true; pyDot.className = "dot ok"; pyText.textContent = `Python ready (Pyodide ${msg.version}, NumPy ${msg.numpy})`;
      syncDesignerButtons();
    } else if (msg.type === "progress") {
      pyText.textContent = msg.text;
    } else if (msg.type === "error" && msg.id === undefined) {
      pyDot.className = "dot err"; pyText.textContent = "Python failed to load: " + msg.error;
    } else if (pending.has(msg.id)) {
      const p = pending.get(msg.id); pending.delete(msg.id);
      msg.type === "result" ? p.resolve(msg) : p.reject(new Error(msg.error));
    }
  };
  worker.onerror = e => { pyDot.className = "dot err"; pyText.textContent = "Python worker error: " + (e.message || e); };
} catch (err) {
  pyDot.className = "dot err"; pyText.textContent = "Web workers are unavailable here: " + err;
}
function runPy(spec, cellSpec) {
  const id = ++jobId;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ type: "compute", id, spec, cell: cellSpec }); });
}

function cellIndexFor(diam, length) {
  let i = DATA.cells.findIndex(c => Math.abs(c.diam - diam) < 1e-6 && Math.abs(c.length - length) < 1e-6);
  if (i < 0) { DATA.cells.push({ diam, radius: diam / 2, length, coils: [] }); i = DATA.cells.length - 1; }
  return i;
}
function addEntry(entry, cellSpec, persist) {
  const ci = cellIndexFor(cellSpec.diam, cellSpec.length), list = DATA.cells[ci].coils;
  entry.group = "Your coils";
  const old = list.findIndex(c => c.key === entry.key);
  if (old >= 0) list.splice(old, 1);
  list.push(decodeEntry(entry));
  if (persist) saveUser();
  return { ci, k: list.length - 1 };
}
function saveUser() {
  const user = [];
  for (const c of DATA.cells) for (const e of c.coils) if (e.group === "Your coils" && e.raw) user.push({ cell: { diam: c.diam, length: c.length }, entry: e.raw });
  try { localStorage.setItem(STORE, JSON.stringify(user)); }
  catch (err) { computeMsg.textContent = "Couldn't save to this browser's storage; the coil stays until you reload."; }
}
function loadUser() {
  let user = [];
  try { user = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch (err) { user = []; }
  for (const u of user) {
    try { const e = addEntry(Object.assign(JSON.parse(u.entry), { raw: u.entry }), u.cell, false); } catch (err) { /* skip bad entry */ }
  }
}

btnCompute.addEventListener("click", async () => {
  let spec, cellSpec;
  try {
    spec = readSpec();
    cellSpec = { diam: +dCellD.value, length: +dCellL.value };
    if (!(cellSpec.diam > 0 && cellSpec.length > 0)) throw new Error("Set a cell diameter and length.");
  } catch (err) { computeMsg.textContent = err.message; return; }
  busy = true; syncDesignerButtons(); pyDot.className = "dot busy";
  const t0 = performance.now();
  computeMsg.textContent = `Computing ${spec.name}…`;
  try {
    const res = await runPy(spec, cellSpec);
    const entry = Object.assign(JSON.parse(res.json), { raw: res.json });
    const { ci, k } = addEntry(entry, cellSpec, true);
    const changed = ci !== st.cell;
    st.cell = ci; st.coil = k;
    if (changed) { st.ox = 0; st.oy = 0; st.rd = Math.min(st.rd, cellSpec.diam); }
    buildControls(); build3d(); updateAll();
    computeMsg.textContent = `Added ${entry.name} in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  } catch (err) {
    computeMsg.textContent = "Computation failed: " + err.message;
  } finally {
    busy = false; pyDot.className = pyReady ? "dot ok" : "dot err"; syncDesignerButtons();
  }
});
function userCoils() { return DATA.cells.flatMap(c => c.coils.filter(e => e.group === "Your coils")); }
// Drop the given user coils (and any cell left empty), keeping the current selection where it survives.
function removeCoils(gone) {
  if (!gone.length) return;
  const curCell = cell(), curCoil = coil();
  for (const c of DATA.cells) c.coils = c.coils.filter(e => !gone.includes(e));
  DATA.cells = DATA.cells.filter(c => c.coils.length);
  saveUser();
  st.cell = Math.max(0, DATA.cells.indexOf(curCell));
  st.coil = Math.max(0, cell().coils.indexOf(curCoil));
  if (cell() !== curCell) { st.ox = 0; st.oy = 0; st.rd = Math.min(st.rd, cell().diam); }
  buildControls(); build3d(); updateAll();
  computeMsg.textContent = gone.length === 1 ? `Removed ${gone[0].name}.` : `Removed ${gone.length} coils.`;
}
btnRemove.addEventListener("click", () => {
  const c = coil(); if (c && c.group === "Your coils") removeCoils([c]);
});
btnRemoveAll.addEventListener("click", () => {
  const all = userCoils();
  if (all.length && confirm(`Remove all ${all.length} of your coils from this browser?`)) removeCoils(all);
});

loadUser();
dType.value = "saddle"; renderParams(); dName.value = "My saddle";
dCellD.value = DATA.cells[0].diam; dCellL.value = DATA.cells[0].length;

/* ---------- boot ---------- */
const grad = "linear-gradient(90deg," + DATA.lut.filter((_, i) => i % 16 === 0 || i === 255).join(",") + ")";
document.getElementById("ramp1").style.background = grad; document.getElementById("ramp2").style.background = grad;
let lastCell = -1;
const _upd = updateAll;
updateAll = function () {
  const k = `${st.cell}:${cell().coils.map(c => c.key).join(",")}`;
  if (lastCell !== k) { buildTables(); lastCell = k; }
  _upd(); syncDesignerButtons();
};
buildControls(); build3d(); updateAll();
new ResizeObserver(() => { drawMap(); draw3d(); }).observe(document.querySelector(".stage"));
new ResizeObserver(() => updateDeut(fraction(coil().map, st.rd / 2, st.ox, st.oy))).observe(dsCv);
const rerender = () => { drawMap(); draw3d(); updateDeut(fraction(coil().map, st.rd / 2, st.ox, st.oy)); };
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

})();
