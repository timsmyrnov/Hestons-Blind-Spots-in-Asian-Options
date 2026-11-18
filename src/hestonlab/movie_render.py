"""Presentation-only WebGL renderer. No pricing, fitting, or neural dependencies."""

from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd

THREE_VERSION = "0.160.0"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  @font-face { font-family: "CM"; font-weight: 400;
    src: url(data:font/ttf;base64,__CMR__) format("truetype"); }
  @font-face { font-family: "CM"; font-weight: 700;
    src: url(data:font/ttf;base64,__CMB__) format("truetype"); }
  @font-face { font-family: "CMI"; font-weight: 400;
    src: url(data:font/ttf;base64,__CMI__) format("truetype"); }
  :root { --ink: #1f2328; --muted: #5f6672; }
  html, body { margin: 0; height: 100%; overflow: hidden;
    background: radial-gradient(120% 95% at 50% 40%, #ffffff 0%, #f7f7fa 58%, #eceef3 100%);
    font-family: "CM", "Latin Modern Roman", "CMU Serif", serif; }
  canvas { display: block; }
  .hud { position: fixed; left: 28px; top: 22px; color: var(--ink); pointer-events: none; }
  .date { font-size: 23px; font-weight: 700; }
  .avg { margin-top: 3px; font-size: 15px; color: var(--muted); }
  .warning { margin-top: 4px; font-size: 14px; color: #926321; }
  .view-note { margin-top: 4px; font-size: 14px; color: #604b83; }
  .line-note { position: fixed; left: 28px; bottom: 49px; font-size: 13px; color: var(--muted); }
  .legend { position: fixed; left: 28px; bottom: 22px; display: flex; gap: 18px;
    font-size: 14px; color: var(--muted); pointer-events: none; }
  .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
    margin-right: 6px; vertical-align: -1px; }
  .num { font-size: 12.5px; color: var(--ink); padding: 0 4px; border-radius: 4px;
    background: rgba(255, 255, 255, 0.82); opacity: 0; white-space: nowrap; }
  .num.day { font-size: 14px; font-weight: 700; }
  .bend { font-size: 13px; font-weight: 700; color: #17634d; padding: 4px 7px;
    background: rgba(237, 249, 241, 0.95); border: 1px solid #64a58a;
    border-radius: 5px; white-space: nowrap; opacity: 0; }
  .tick { font-size: 11.5px; color: var(--muted); white-space: nowrap; }
  .axis { font-size: 13.5px; color: var(--muted); white-space: nowrap; }
  .manifold, .math { white-space: nowrap; color: var(--ink); }
  .manifold { font-size: 14px; } .math { font-size: 20px; }
  .manifold sub, .math sub { font-size: 70%; }
  .replay { position: fixed; right: 20px; bottom: 16px; width: 30px; height: 30px;
    border-radius: 50%; border: 1px solid rgba(31, 35, 40, 0.12);
    background: rgba(255, 255, 255, 0.75); color: var(--muted); font-size: 15px;
    cursor: pointer; opacity: 0; transition: opacity 0.5s; }
  .replay.show { opacity: 1; }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__V__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__V__/examples/jsm/" } }
</script>
</head>
<body>
<div class="hud"><div class="date"></div><div class="avg"></div><div class="view-note"></div><div class="warning"></div></div>
<div class="legend">
  <span><i style="background:#4AAEEA"></i>Market manifold</span>
  <span><i style="background:#8B5BD0"></i>Heston manifold</span>
</div>
<div class="line-note"></div>
<button class="replay" aria-label="Replay">&#8635;</button>
<script type="module">
import * as THREE from "three";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const D = __DATA__;
document.querySelector(".line-note").textContent = D.localLineNote || "";
document.querySelector(".view-note").textContent = D.viewNote || "";
const sheetNames = D.sheetNames || ["Market manifold", "Heston manifold"];
const sheetSymbols = D.sheetSymbols || ["M_m", "M_h"];
document.querySelectorAll(".legend span").forEach((el, i) => el.lastChild.textContent = sheetNames[i]);
const days = D.days, nDays = days.length;
const FLOW = D.flow || D.seconds, FINALE = D.finale || 0;  // days play, then the finale
const query = new URLSearchParams(location.search);
const frozen = query.has("t") ? Math.min(Math.max(parseFloat(query.get("t")), 0), 1) : null;
const exportTo = query.get("export");  // video recording: frames go to a local receiver
const deg = Math.PI / 180;
const clamp = t => Math.min(Math.max(t, 0), 1);
const ease = t => 0.5 - 0.5 * Math.cos(Math.PI * clamp(t));
const smooth = t => { t = clamp(t); return t * t * (3 - 2 * t); };
const fmt = x => x.toFixed(3);

// ---- one frame for all days: floor scale shared, vertical scale stretched to be readable
const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
const grow = p => { for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); } };
for (const d of days) {
  for (const key of ["heston", "market"]) for (const row of d[key]) for (const p of row) if (p) grow(p);
  for (const l of d.lines) { grow(l.s); grow(l.e); }
}
const mid = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
const Sf = 10 / Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-12);
const Su = 6.5 / Math.max(hi[2] - lo[2], 1e-12);  // heights: about 2/3 of the floor width
const V = p => new THREE.Vector3((p[0] - mid[0]) * Sf, (p[2] - mid[2]) * Su, -(p[1] - mid[1]) * Sf);
const lerp = (a, b, w) => [0, 1, 2].map(k => a[k] + (b[k] - a[k]) * w);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true,
                                           preserveDrawingBuffer: frozen !== null || !!exportTo });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: "fixed", inset: "0", pointerEvents: "none" });
document.body.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, innerWidth / innerHeight, 0.1, 500);
scene.add(new THREE.HemisphereLight(0xf6f3ff, 0xdbe3ec, 1.2));
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(6, 12, 8);
scene.add(key);
const fill = new THREE.DirectionalLight(0xe0e8ff, 0.6);
fill.position.set(-9, 3, -6);
scene.add(fill);

function ramp(stops, t) {
  const c = stops.map(h => new THREE.Color(h));
  const x = clamp(t) * (c.length - 1), i = Math.min(Math.floor(x), c.length - 2);
  return c[i].clone().lerp(c[i + 1], x - i);
}

const texts = [];  // every label, so video frames can draw them too
const MATH_KINDS = new Set(["manifold", "math"]);
function parseMath(text) {  // "T_pM_h" -> [["T", 0], ["p", 1], ["M", 0], ["h", 1]]
  const out = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "_" && i + 1 < text.length) { out.push([text[i + 1], 1]); i++; }
    else out.push([text[i], 0]);
  }
  return out;
}
function label(kind, text, position, center = [0.5, 0.5]) {
  const el = document.createElement("div");
  el.className = kind;
  el.textContent = text;
  const obj = new CSS2DObject(el);
  obj.position.copy(position);
  obj.center.set(center[0], center[1]);
  scene.add(obj);
  const item = { kind, el, obj, center };
  if (MATH_KINDS.has(kind)) {
    item.segs = parseMath(text);
    el.innerHTML = item.segs.map(([t, sub]) => {
      const f = /[0-9+]/.test(t) ? "CM" : "CMI";
      return sub ? `<sub style="font-family:${f}">${t}</sub>` : `<span style="font-family:${f}">${t}</span>`;
    }).join("");
  }
  texts.push(item);
  return item;
}

// ---- sheets: one mesh each, morphed between days
function sheet(keyName, stops, opacity, order) {
  const n = days[0][keyName].length, m = days[0][keyName][0].length, id = [];
  let count = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++)
    id.push(days.every(d => d[keyName][i][j]) ? count++ : -1);
  const frames = days.map(d => {
    const a = new Float32Array(count * 3);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const k = id[i * m + j];
      if (k >= 0) { const v = V(d[keyName][i][j]); a[3 * k] = v.x; a[3 * k + 1] = v.y; a[3 * k + 2] = v.z; }
    }
    return a;
  });
  // Visibility per point and day: 0 where that day's model is not valid (faded out).
  const fades = days.some(d => d[keyName + "Alpha"]) ? days.map(d => {
    const a = new Float32Array(count).fill(1), al = d[keyName + "Alpha"];
    if (al) for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const k = id[i * m + j];
      if (k >= 0) a[k] = al[i][j];
    }
    return a;
  }) : null;
  let ylo = Infinity, yhi = -Infinity;
  for (let k = 1; k < frames[0].length; k += 3) { ylo = Math.min(ylo, frames[0][k]); yhi = Math.max(yhi, frames[0][k]); }
  const col = new Float32Array(count * 4), edge = new Float32Array(count * 4);
  const ink = new THREE.Color(stops[2]);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const k = id[i * m + j];
    if (k < 0) continue;
    const h = yhi > ylo ? (frames[0][3 * k + 1] - ylo) / (yhi - ylo) : 0.5;
    // A restrained diagonal gradient, anchored to the mesh rather than animated.
    const tone = 0.10 + 0.80 * (0.60 * j / (m - 1) + 0.25 * i / (n - 1) + 0.15 * h);
    const c = ramp(stops, tone);
    col.set([c.r, c.g, c.b, fades ? fades[0][k] : 1], 4 * k);
    edge.set([ink.r, ink.g, ink.b, fades ? fades[0][k] : 1], 4 * k);
  }
  const at = (i, j) => id[i * m + j], index = [], seg = [];
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < m - 1; j++) {
    const a = at(i, j), b = at(i, j + 1), c = at(i + 1, j), d = at(i + 1, j + 1);
    if (a >= 0 && b >= 0 && c >= 0 && d >= 0) index.push(a, c, b, b, c, d);
  }
  const step = Math.max(1, Math.round(n / 11));
  for (let i = 0; i < n; i += step) for (let j = 0; j < m - 1; j++)
    if (at(i, j) >= 0 && at(i, j + 1) >= 0) seg.push(at(i, j), at(i, j + 1));
  for (let j = 0; j < m; j += step) for (let i = 0; i < n - 1; i++)
    if (at(i, j) >= 0 && at(i + 1, j) >= 0) seg.push(at(i, j), at(i + 1, j));
  const g = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(frames[0].slice(), 3);
  g.setAttribute("position", position);
  const color = new THREE.BufferAttribute(col, 4);
  g.setAttribute("color", color);
  g.setIndex(index);
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshPhysicalMaterial({
    vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity,
    roughness: 0.36, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.32,
    depthWrite: opacity > 0.9 && !fades, polygonOffset: true, polygonOffsetFactor: 1,
    polygonOffsetUnits: 1 }));
  mesh.renderOrder = order;
  scene.add(mesh);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", position);
  const edgeColor = new THREE.BufferAttribute(edge, 4);
  lg.setAttribute("color", edgeColor);
  lg.setIndex(seg);
  const wire = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.15, depthWrite: false }));
  wire.renderOrder = order + 0.5;
  scene.add(wire);
  return { mesh, frames, position, geometry: g, fades, color, edgeColor, id, n, m };
}
function morph(s, k, w) {
  const a = s.frames[k], b = s.frames[Math.min(k + 1, nDays - 1)], p = s.position.array;
  for (let i = 0; i < p.length; i++) p[i] = a[i] + (b[i] - a[i]) * w;
  if (gain !== 1 || shift !== 0) for (let i = 1; i < p.length; i += 3) p[i] = gain * p[i] + shift;
  s.position.needsUpdate = true;
  s.geometry.computeVertexNormals();
  if (!s.fades) return;
  const fa = s.fades[k], fb = s.fades[Math.min(k + 1, nDays - 1)];
  for (let i = 0; i < fa.length; i++) {
    const v = fa[i] + (fb[i] - fa[i]) * w;
    s.color.array[4 * i + 3] = v;
    s.edgeColor.array[4 * i + 3] = v;
  }
  s.color.needsUpdate = s.edgeColor.needsUpdate = true;
}
const MARKET = ["#D3ECFB", "#4AAEEA", "#1F6FB0"];
const HESTON = ["#E2D4F5", "#8B5BD0", "#45207F"];
const hestonOnTop = D.topSheet === "heston";
const heston = sheet("heston", HESTON, hestonOnTop ? 0.64 : 0.96, hestonOnTop ? 3 : 0);
const market = sheet("market", MARKET, hestonOnTop ? 0.96 : 0.74, hestonOnTop ? 0 : 3);

// ---- the gap axis can zoom in once a stormy stretch has passed (D.rescale)
let gain = 1, shift = 0;  // drawn height y -> gain * y + shift
const RS = D.rescale ? (() => {
  let ylo = Infinity, yhi = -Infinity;
  const take = y => { ylo = Math.min(ylo, y); yhi = Math.max(yhi, y); };
  for (const s of [heston, market]) for (let k = D.rescale.from; k < nDays; k++) {
    const f = s.frames[k], fa = s.fades ? s.fades[k] : null;
    for (let q = 0; q < f.length / 3; q++) if (!fa || fa[q] > 0.05) take(f[3 * q + 1]);
  }
  for (let k = D.rescale.from; k < nDays; k++)
    for (const l of days[k].lines) if (l.v > 0.05) { take(V(l.s).y); take(V(l.e).y); }
  const full = [(lo[2] - mid[2]) * Su, (hi[2] - mid[2]) * Su];  // the box keeps its height
  const g = (full[1] - full[0]) / Math.max(yhi - ylo, 1e-9);
  return { g, off: full[0] - g * ylo };
})() : null;
function setScale(seconds) {
  if (D.verticalRanges) {
    const [k, w] = when(seconds / D.seconds);
    const a = D.verticalRanges[k], b = D.verticalRanges[Math.min(k + 1, nDays - 1)];
    const low = a[0] + (b[0] - a[0]) * w, high = a[1] + (b[1] - a[1]) * w;
    gain = (hi[2] - lo[2]) / (high - low);
    shift = (lo[2] - mid[2]) * Su - gain * (low - mid[2]) * Su;
    return;
  }
  if (!RS) return;
  const x = smooth((seconds - D.rescale.at) / D.rescale.over);
  gain = 1 + (RS.g - 1) * x;
  shift = RS.off * x;
}
const Y = v => { v.y = gain * v.y + shift; return v; };

// ---- distance lines: thin tubes, grey first, then colored by distance with numbers
const GREY = new THREE.Color("#8a8f98");
const DIST = ["#2E9E5B", "#9BC53D", "#E8B931", "#E07B39", "#C8413A"];
// Colour = the gap itself (USD/bbl), one fixed log scale for the whole movie: green at
// D.gapScale[0] or less, red at D.gapScale[1] or more. Equal gaps, equal colours.
const [G0, G1] = D.gapScale || [0.02, 0.2];
const gapShade = d => clamp(Math.log(Math.max(d, 1e-9) / G0) / Math.log(G1 / G0));
// Every vector uses this same fixed value scale, including local and corner vectors.
const up = new THREE.Vector3(0, 1, 0);
const lines = days[0].lines.map((l0, i) => {
  const r = l0.day ? 0.036 : 0.010;
  const mat = new THREE.MeshBasicMaterial({ color: GREY.clone(), transparent: true,
                                            opacity: 0.55, depthWrite: false, toneMapped: false });
  const tube = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 14, 1, true), mat);
  tube.renderOrder = 4;  // after the translucent upper sheet, so it never tints a line
  scene.add(tube);
  return { tube, mat, text: label(l0.day ? "num day" : "num", "", new THREE.Vector3(), [-0.12, 0.5]) };
});
const dots = [["#4AAEEA", 0], ["#8B5BD0", 1]].map(([hex]) => {
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.075, 28, 18),
    new THREE.MeshStandardMaterial({ color: hex, roughness: 0.3, emissive: hex, emissiveIntensity: 0.15 }));
  dot.renderOrder = 4;
  scene.add(dot);
  return dot;
});

// ---- M_h and M_m: tiny labels riding on a corner of each sheet
function vertexAt(s, i, j) {  // nearest grid point that exists, searching inward
  for (let r = 0; r < Math.max(s.n, s.m); r++)
    for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) {
      const a = Math.min(Math.max(i + di, 0), s.n - 1), b = Math.min(Math.max(j + dj, 0), s.m - 1);
      if (s.id[a * s.m + b] >= 0) return s.id[a * s.m + b];
    }
  return -1;
}
function seen(s, idx, k, w) {
  if (!s.fades) return 1;
  const a = s.fades[k][idx], b = s.fades[Math.min(k + 1, nDays - 1)][idx];
  return a + (b - a) * w;
}
const at = (s, idx) => new THREE.Vector3(s.position.array[3 * idx], s.position.array[3 * idx + 1],
                                         s.position.array[3 * idx + 2]);
const tags = [[heston, sheetSymbols[1], "#45207F", 0, 1], [market, sheetSymbols[0], "#1F6FB0", 0, 1]].map(([s, text, hex, ci, cj]) => {
  const idx = vertexAt(s, ci * (s.n - 1), cj * (s.m - 1));
  const tag = label("manifold", text, new THREE.Vector3(), [0.5, 0.5]);
  tag.color = hex;
  tag.el.style.color = hex;
  tag.place = () => {
    if (idx < 0) { tag.el.style.opacity = "0"; return; }
    const corner = at(s, idx), out = corner.clone().setY(0).normalize().multiplyScalar(0.45);
    tag.obj.position.copy(corner).add(out);
    const [k, w] = current;
    tag.el.style.opacity = clamp(seen(s, idx, k, w)).toFixed(3);
  };
  return tag;
});
let current = [0, 0];

// Two local tangent patches on the displayed Heston slice. Geometry and bend
// are interpolated before camera scaling; zoom cannot change the reported bend.
const tangentPatches = (D.tangentPlanes ? ["A", "B"] : []).map((name, index) => {
  const fill = new THREE.MeshBasicMaterial({ color: index ? 0x8acaa3 : 0x66b99e,
    side: THREE.DoubleSide, transparent: true, opacity: 0, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, toneMapped: false });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(18), 3));
  const mesh = new THREE.Mesh(geometry, fill);
  mesh.renderOrder = 5;
  mesh.frustumCulled = false;
  const ink = new THREE.LineBasicMaterial({ color: 0x267c60, transparent: true,
    opacity: 0, depthWrite: false, toneMapped: false });
  const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
    Array.from({length: 4}, () => new THREE.Vector3())), ink);
  const cross = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(
    Array.from({length: 4}, () => new THREE.Vector3())), ink);
  const leader = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(), new THREE.Vector3()]), ink);
  for (const item of [rim, cross, leader]) { item.renderOrder = 6; item.frustumCulled = false; }
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0x17634d, transparent: true, opacity: 0, depthWrite: false }));
  dot.renderOrder = 7;
  const text = label("bend", "", new THREE.Vector3());
  scene.add(mesh, rim, cross, leader, dot);
  function positions(object, points) {
    const attr = object.geometry.attributes.position;
    points.forEach((p, i) => attr.setXYZ(i, p.x, p.y, p.z));
    attr.needsUpdate = true;
  }
  return { show(progress, k, w) {
    const alpha = smooth((progress - D.tangentReveal) / 0.06);
    for (const item of [mesh, rim, cross, leader, dot]) item.visible = alpha > 0;
    text.el.style.opacity = alpha.toFixed(3);
    if (!alpha) return;
    const a = days[k].tangents[index], b = days[Math.min(k + 1, nDays - 1)].tangents[index];
    const p = lerp(a.p, b.p, w), u = lerp(a.u, b.u, w), v = lerp(a.v, b.v, w);
    const point = (x, y) => Y(V(p.map((q, axis) => q + x * u[axis] + y * v[axis])));
    const corners = [[-1,-1], [1,-1], [1,1], [-1,1]].map(([x,y]) => point(x,y));
    positions(mesh, [corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]]);
    positions(rim, corners);
    positions(cross, [point(-1,0), point(1,0), point(0,-1), point(0,1)]);
    const center = point(0,0);
    dot.position.copy(center);
    // Callouts sit above the patches, away from the gap labels between sheets.
    const screenUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    const screenRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    text.obj.position.copy(center).addScaledVector(screenUp, 2.6)
      .addScaledVector(screenRight, index ? 0.65 : -0.65);
    positions(leader, [center, text.obj.position]);
    const bend = Math.max(...a.departures.map((d, i) => Math.abs(d + (b.departures[i] - d) * w))) * 1000;
    text.el.textContent = `Tangent ${name}, bend ${bend < 0.005 ? "under 0.01" : bend.toFixed(2)} mUSD/bbl`;
    fill.opacity = 0.55 * alpha;
    ink.opacity = 0.8 * alpha;
    dot.material.opacity = alpha;
  }};
});

// ---- the finale: p on M_h, its tangent plane T_pM_h and two tangent vectors
const finale = FINALE > 0 ? (() => {
  const s = heston, last = s.frames[nDays - 1];
  const i = Math.round(0.28 * (s.n - 1)), j = Math.round(0.72 * (s.m - 1));
  const ok = (a, b) => a >= 0 && b >= 0 && a < s.n && b < s.m && s.id[a * s.m + b] >= 0;
  let pi = i, pj = j;  // nearest grid point with all four neighbours on the sheet
  search: for (let r = 0; r < s.n; r++)
    for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++)
      if (ok(i + di, j + dj) && ok(i + di + 1, j + dj) && ok(i + di - 1, j + dj) &&
          ok(i + di, j + dj + 1) && ok(i + di, j + dj - 1)) { pi = i + di; pj = j + dj; break search; }
  const P = (a, b) => {
    const q = s.id[a * s.m + b], y = last[3 * q + 1];
    return new THREE.Vector3(last[3 * q], RS ? RS.g * y + RS.off : y, last[3 * q + 2]);
  };
  const p = P(pi, pj);
  const t1 = P(pi, pj + 1).sub(P(pi, pj - 1)).normalize();   // along slice direction 1
  const t2 = P(pi - 1, pj).sub(P(pi + 1, pj)).normalize();   // along slice direction 2 (toward the viewer)
  const normal = t1.clone().cross(t2).normalize();
  const a = 1.5, lift = normal.clone().multiplyScalar(0.012 * (normal.y >= 0 ? 1 : -1));
  const corner = (x, y) => p.clone().addScaledVector(t1, x * a).addScaledVector(t2, y * a).add(lift);
  const plane = new THREE.BufferGeometry().setFromPoints([corner(-1, -1), corner(1, -1), corner(1, 1),
                                                         corner(-1, -1), corner(1, 1), corner(-1, 1)]);
  const planeMat = new THREE.MeshBasicMaterial({ color: 0xf2f0f7, transparent: true, opacity: 0,
                                                 side: THREE.DoubleSide, depthWrite: false, toneMapped: false });
  const planeMesh = new THREE.Mesh(plane, planeMat);
  planeMesh.renderOrder = 6;
  const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
    [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]),
    new THREE.LineBasicMaterial({ color: 0x5f6672, transparent: true, opacity: 0, depthWrite: false }));
  rim.renderOrder = 7;
  scene.add(planeMesh, rim);
  const ink = new THREE.MeshBasicMaterial({ color: 0x1f2328, transparent: true, opacity: 0,
                                            depthWrite: false, toneMapped: false });
  function arrow(dir, length) {
    const g = new THREE.Group(), head = 0.22;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, length - head, 12), ink);
    shaft.position.y = (length - head) / 2;
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.075, head, 18), ink);
    tip.position.y = length - head / 2;
    g.add(shaft, tip);
    g.position.copy(p).add(lift);
    g.quaternion.setFromUnitVectors(up, dir);
    g.children.forEach(c => { c.renderOrder = 8; });
    scene.add(g);
    return g;
  }
  arrow(t1, 1.15);
  arrow(t2, 1.15);
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 24, 16), ink);
  dot.position.copy(p).add(lift);
  dot.renderOrder = 9;
  scene.add(dot);
  const names = [
    label("math", "p", p.clone().addScaledVector(t1, -0.22).addScaledVector(t2, -0.22).add(lift), [0.5, 0.5]),
    label("math", "v_1", p.clone().addScaledVector(t1, 1.42).add(lift), [0.5, 0.5]),
    label("math", "v_2", p.clone().addScaledVector(t2, 1.42).add(lift), [0.5, 0.5]),
    label("math", "T_pM_h", corner(1.25, -1.15), [0.5, 0.5]),
  ];
  names.forEach(n => { n.el.style.opacity = "0"; });
  return { p, show(u) {
    const shape = smooth((u - 0.3) / 0.3), words = smooth((u - 0.5) / 0.25);
    planeMat.opacity = 0.62 * shape;
    rim.material.opacity = 0.8 * shape;
    ink.opacity = shape;
    dot.visible = shape > 0.001;
    names.forEach(n => { n.el.style.opacity = words.toFixed(3); });
  } };
})() : null;

// ---- axes: three edges of the box, ticks in real units
const box = { x: [(lo[0] - mid[0]) * Sf, (hi[0] - mid[0]) * Sf], y: [(lo[2] - mid[2]) * Su, (hi[2] - mid[2]) * Su],
              z: [-(hi[1] - mid[1]) * Sf, -(lo[1] - mid[1]) * Sf] };
for (const a of ["x", "y", "z"]) { const pad = 0.05 * (box[a][1] - box[a][0]); box[a] = [box[a][0] - pad, box[a][1] + pad]; }
const axisMaterial = new THREE.LineBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.9 });
function segment(a, b) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  scene.add(new THREE.Line(g, axisMaterial));
}
function niceTicks(a, b) {
  const raw = (b - a) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw);
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-12; t += step) out.push(Math.abs(t) < step * 1e-6 ? 0 : t);
  let digits = 0;  // enough decimals for the step itself (0.25 needs two)
  while (digits < 6 && Math.abs(step * 10 ** digits - Math.round(step * 10 ** digits)) > 1e-6) digits++;
  return { out, digits };
}
const toScene = { x: t => (t - mid[0]) * Sf, y: t => (t - mid[2]) * Su, z: t => -(t - mid[1]) * Sf };
const dataRange = { x: [box.x[0] / Sf + mid[0], box.x[1] / Sf + mid[0]],
                    y: [box.y[0] / Su + mid[2], box.y[1] / Su + mid[2]],
                    z: [-box.z[1] / Sf + mid[1], -box.z[0] / Sf + mid[1]] };
const X0 = box.x[0], X1 = box.x[1], Y0 = box.y[0], Y1 = box.y[1], Z0 = box.z[0], Z1 = box.z[1];
const tickLen = 0.18, axisUnits = D.axisUnits || [D.units, D.units, D.units];
segment(new THREE.Vector3(X0, Y0, Z1), new THREE.Vector3(X1, Y0, Z1));  // slice direction 1, front
segment(new THREE.Vector3(X0, Y0, Z0), new THREE.Vector3(X0, Y0, Z1));  // slice direction 2, left
segment(new THREE.Vector3(X0, Y0, Z0), new THREE.Vector3(X0, Y1, Z0));  // gap, back-left
const T1 = niceTicks(...dataRange.x), T2 = niceTicks(...dataRange.z);
for (const t of T1.out) { const x = toScene.x(t); if (x < X0 || x > X1) continue;
  segment(new THREE.Vector3(x, Y0, Z1), new THREE.Vector3(x, Y0, Z1 + tickLen));
  label("tick", t.toFixed(T1.digits), new THREE.Vector3(x, Y0, Z1 + 3.2 * tickLen)); }
for (const t of T2.out) { const z = toScene.z(t); if (z < Z0 || z > Z1) continue;
  segment(new THREE.Vector3(X0, Y0, z), new THREE.Vector3(X0 - tickLen, Y0, z));
  label("tick", t.toFixed(T2.digits), new THREE.Vector3(X0 - 3.6 * tickLen, Y0, z)); }
const gapTicks = Array.from({ length: 9 }, () => {  // relabelled every frame (the gap axis may zoom)
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(), new THREE.Vector3()]), axisMaterial);
  scene.add(line);
  return { line, text: label("tick", "", new THREE.Vector3()) };
});
function placeGapTicks() {
  const toData = y => (y - shift) / gain / Su + mid[2];
  const T = niceTicks(toData(Y0), toData(Y1));
  gapTicks.forEach((tick, i) => {
    const t = T.out[i], y = t === undefined ? NaN : gain * (t - mid[2]) * Su + shift;
    const show = Number.isFinite(y) && y >= Y0 - 1e-9 && y <= Y1 + 1e-9;
    tick.line.visible = show;
    tick.text.el.style.opacity = show ? "1" : "0";
    if (!show) return;
    const at = tick.line.geometry.attributes.position;
    at.setXYZ(0, X0, y, Z0); at.setXYZ(1, X0 - tickLen, y, Z0); at.needsUpdate = true;
    tick.text.obj.position.set(X0 - 3.6 * tickLen, y, Z0);
    tick.text.el.textContent = t.toFixed(T.digits);
  });
}
label("axis", D.axes[0] + ` (${axisUnits[0]})`, new THREE.Vector3((X0 + X1) / 2, Y0, Z1 + 11 * tickLen));
label("axis", D.axes[1] + ` (${axisUnits[1]})`, new THREE.Vector3(X0 - 13 * tickLen, Y0, (Z0 + Z1) / 2));
label("axis", D.axes[2] + ` (${axisUnits[2]})`, new THREE.Vector3(X0 - 1.5 * tickLen, Y1 + 2.5 * tickLen, Z0));

let bottom = Infinity, radius = 0;
for (const s of [heston, market]) for (const f of s.frames) for (let i = 0; i < f.length; i += 3) {
  bottom = Math.min(bottom, f[i + 1]); radius = Math.max(radius, Math.hypot(f[i], f[i + 1], f[i + 2])); }
for (const x of [X0, X1]) for (const y of [Y0, Y1]) for (const z of [Z0, Z1]) radius = Math.max(radius, Math.hypot(x, y, z));
const shade = document.createElement("canvas");
shade.width = shade.height = 256;
const ctx = shade.getContext("2d");
const glow = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
glow.addColorStop(0, "rgba(40, 45, 75, 0.14)");
glow.addColorStop(1, "rgba(40, 45, 75, 0)");
ctx.fillStyle = glow;
ctx.fillRect(0, 0, 256, 256);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(18, 18), new THREE.MeshBasicMaterial({
  map: new THREE.CanvasTexture(shade), transparent: true, depthWrite: false }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = Math.min(bottom, Y0) - 0.6;
scene.add(floor);

let distance = 20;
function fit() {
  const v = camera.fov * deg, h = 2 * Math.atan(Math.tan(v / 2) * camera.aspect);
  distance = 0.99 * radius / Math.sin(Math.min(v, h) / 2);
}
fit();

// ---- timeline: hold on each day, glide to the next
const dateEl = document.querySelector(".date"), avgEl = document.querySelector(".avg");
const warningEl = document.querySelector(".warning");
let errorReveal = 0;
function when(progress) {
  const t = Math.min(clamp(progress) * D.seconds, FLOW), seg = D.hold + D.move;
  const k = Math.min(Math.floor(t / seg), nDays - 1), local = t - k * seg;
  const x = (local - D.hold) / D.move;  // no hold: a steady glide, so long movies flow
  return [k, k < nDays - 1 && local > D.hold ? (D.hold > 0 ? ease(x) : clamp(x)) : 0];
}
function frame(progress) {
  const t = clamp(progress) * D.seconds;
  setScale(t);
  const e = ease(Math.min(t / FLOW, 1));
  const zoom = FINALE > 0 ? ease((t - FLOW) / FINALE) : 0;  // the finale: zoom onto p
  const az = (D.fixedCamera ? -48 : -58 + 96 * e + 16 * zoom) * deg;
  const el = (D.fixedCamera ? 26 : D.steadySurfaces ? 32 : 32 - 14 * e + 16 * zoom) * deg;
  const [k, w] = when(progress), A = days[k], B = days[Math.min(k + 1, nDays - 1)];
  current = [k, w];
  morph(heston, k, w);
  morph(market, k, w);
  const target = finale ? finale.p.clone().multiplyScalar(zoom) : new THREE.Vector3();
  const dist = distance * (1 - 0.7 * zoom);
  camera.position.set(target.x + dist * Math.cos(el) * Math.sin(az), target.y + dist * Math.sin(el),
                      target.z + dist * Math.cos(el) * Math.cos(az));
  camera.lookAt(target);
  tags.forEach(tag => tag.place());
  tangentPatches.forEach(patch => patch.show(progress, k, w));
  placeGapTicks();
  if (finale) finale.show((t - FLOW) / FINALE);
  const b = smooth((progress - D.reveal) / 0.06);
  errorReveal = b;
  document.querySelector(".line-note").style.opacity = b.toFixed(3);
  let sum = 0, seen = 0;
  lines.forEach((L, i) => {
    const a = Y(V(lerp(A.lines[i].s, B.lines[i].s, w))), z = Y(V(lerp(A.lines[i].e, B.lines[i].e, w)));
    const dir = z.clone().sub(a), len = Math.max(dir.length(), 1e-6);
    L.tube.position.copy(a).addScaledVector(dir, 0.5);
    L.tube.quaternion.setFromUnitVectors(up, dir.clone().divideScalar(len));
    L.tube.scale.set(1, len, 1);
    L.text.obj.position.copy(a).addScaledVector(dir, 0.55);
    const d = A.lines[i].d + (B.lines[i].d - A.lines[i].d) * w;
    const vis = A.lines[i].v + (B.lines[i].v - A.lines[i].v) * w;  // 0 where a model is not valid
    const central = A.lines[i].day;
    if (central) { sum += d * vis; seen += vis; }
    L.mat.color.copy(GREY).lerp(ramp(DIST, gapShade(Number(fmt(d)))), b);
    L.mat.opacity = (central ? 0.55 + 0.42 * b : 0.55 + 0.35 * b) * vis * (1 - 0.8 * zoom);
    L.tube.visible = vis > 0.001;
    L.text.el.textContent = (central ? "" : "≈") + fmt(d);
    L.text.el.style.opacity = (b * vis * (1 - zoom)).toFixed(3);
    if (i === 0) {
      dots[0].position.copy(a); dots[1].position.copy(z);
      dots[0].visible = vis > 0.001; dots[1].visible = vis > 0.001;
    }
  });
  dateEl.textContent = (w < 0.5 ? A : B).date;
  avgEl.textContent = progress < D.reveal ? D.note || ""
    : (seen > 1e-6 ? `weekly gap ${fmt(sum / seen)} ${D.units}` : "weekly gap unavailable")
      + (D.note ? `, ${D.note}` : "");
  const warnings = [...new Set([A.warning, B.warning].filter(Boolean).flatMap(x => x.split(" · ")))]
    .filter(x => !x.includes("Numerical checks"));
  warningEl.textContent = D.showWarnings === false ? "" : D.compactHud ? [
    warnings.some(x => x.includes("beyond observed support")) ? "outside market support" : ""
  ].filter(Boolean).join(" · ") : warnings.join(" · ");
  market.mesh.material.opacity = hestonOnTop ? 0.96 :
    (D.steadySurfaces ? 0.64 : (0.74 - 0.2 * b) * (1 - 0.45 * zoom));
  renderer.render(scene, camera);
  if (!exportTo) labels.render(scene, camera);
  return b;
}

// ---- video frames: WebGL image plus every label, drawn onto one canvas
const FONT = '"CM", serif';
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
const STYLE = { "num": [400, 12.5, "#1f2328"], "num day": [700, 14, "#1f2328"],
                "bend": [700, 13, "#17634d"],
                "tick": [400, 11.5, "#5f6672"], "axis": [400, 13.5, "#5f6672"],
                "manifold": [400, 14, "#1f2328"], "math": [400, 20, "#1f2328"] };
function mathFont(text, sub, size) {
  return `400 ${(sub ? 0.7 : 1) * size}px ${/[0-9+]/.test(text) ? '"CM"' : '"CMI"'}, serif`;
}
function drawMath(c, segs, x, y, size, color) {
  c.fillStyle = color;
  for (const [text, sub] of segs) {
    c.font = mathFont(text, sub, size);
    c.fillText(text, x, y + (sub ? 0.3 * size : 0));
    x += c.measureText(text).width;
  }
}
function mathWidth(c, segs, size) {
  let w = 0;
  for (const [text, sub] of segs) { c.font = mathFont(text, sub, size); w += c.measureText(text).width; }
  return w;
}
function compose(out, c) {
  const W = out.width, H = out.height, k = H / 900;
  const bg = c.createRadialGradient(W / 2, H * 0.4, 0, W / 2, H * 0.4, Math.hypot(W, H) * 0.62);
  bg.addColorStop(0, "#ffffff"); bg.addColorStop(0.58, "#f7f7fa"); bg.addColorStop(1, "#eceef3");
  c.globalAlpha = 1; c.fillStyle = bg; c.fillRect(0, 0, W, H);
  c.drawImage(renderer.domElement, 0, 0, W, H);
  c.textBaseline = "middle";
  const v = new THREE.Vector3();
  for (const t of texts) {
    const alpha = t.el.style.opacity === "" ? 1 : parseFloat(t.el.style.opacity);
    if (alpha < 0.01) continue;
    t.obj.getWorldPosition(v).project(camera);
    if (v.z > 1) continue;
    const [weight, size, ink] = STYLE[t.kind], color = t.color || ink;
    if (t.segs) {
      const mw = mathWidth(c, t.segs, size * k);
      c.globalAlpha = alpha;
      drawMath(c, t.segs, (v.x + 1) / 2 * W - t.center[0] * mw, (1 - v.y) / 2 * H, size * k, color);
      continue;
    }
    c.font = `${weight} ${size * k}px ${FONT}`;
    const tw = c.measureText(t.el.textContent).width, bend = t.kind === "bend";
    const pill = t.kind.startsWith("num") || bend;
    const padding = bend ? 7 : 4;
    const w = tw + (pill ? 2 * padding * k : 0), h = (size + (bend ? 8 : 4)) * k;
    const x = (v.x + 1) / 2 * W - t.center[0] * w, y = (1 - v.y) / 2 * H;
    c.globalAlpha = alpha;
    if (pill) {
      c.fillStyle = bend ? "rgba(237, 249, 241, 0.95)" : "rgba(255, 255, 255, 0.82)";
      roundRect(c, x, y - h / 2, w, h, 4 * k); c.fill();
      if (bend) { c.strokeStyle = "#64a58a"; c.lineWidth = k; c.stroke(); }
    }
    c.fillStyle = color;
    c.fillText(t.el.textContent, x + (pill ? padding * k : 0), y + 0.5 * k);
  }
  c.globalAlpha = 1; c.textBaseline = "top";
  c.fillStyle = "#1f2328"; c.font = `700 ${23 * k}px ${FONT}`;
  c.fillText(dateEl.textContent, 28 * k, 24 * k);
  c.fillStyle = "#5f6672"; c.font = `400 ${15 * k}px ${FONT}`;
  c.fillText(avgEl.textContent, 28 * k, 54 * k);
  c.fillStyle = "#926321"; c.font = `400 ${14 * k}px ${FONT}`;
  c.fillText(warningEl.textContent, 28 * k, (D.viewNote ? 98 : 76) * k);
  c.fillStyle = "#604b83";
  c.fillText(D.viewNote || "", 28 * k, 76 * k);
  c.fillStyle = "#5f6672"; c.font = `400 ${13 * k}px ${FONT}`;
  c.globalAlpha = errorReveal;
  c.fillText(D.localLineNote || "", 28 * k, H - 64 * k);
  c.globalAlpha = 1;
  c.textBaseline = "middle"; c.font = `400 ${14 * k}px ${FONT}`;
  let x = 28 * k;
  for (const [hex, name] of [["#4AAEEA", sheetNames[0]], ["#8B5BD0", sheetNames[1]]]) {
    c.fillStyle = hex; roundRect(c, x, H - 38 * k, 11 * k, 11 * k, 3 * k); c.fill();
    c.fillStyle = "#5f6672"; c.fillText(name, x + 17 * k, H - 32 * k);
    x += 17 * k + c.measureText(name).width + 18 * k;
  }
  x += 14 * k;  // gap colour scale: the line colours read as numbers
  c.globalAlpha = errorReveal;
  c.fillStyle = "#5f6672"; c.fillText("gap", x, H - 32 * k);
  x += c.measureText("gap").width + 8 * k;
  const barW = 120 * k, bar = c.createLinearGradient(x, 0, x + barW, 0);
  DIST.forEach((hex, i) => bar.addColorStop(i / (DIST.length - 1), hex));
  c.fillStyle = bar; roundRect(c, x, H - 37 * k, barW, 9 * k, 3 * k); c.fill();
  c.font = `400 ${11.5 * k}px ${FONT}`; c.fillStyle = "#5f6672"; c.textAlign = "center";
  [G0, Math.sqrt(G0 * G1), G1].forEach((g, i) => c.fillText(g.toFixed(g < 0.1 ? 3 : 2), x + i * barW / 2, H - 18 * k));
  c.textAlign = "left";
  c.globalAlpha = 1;
}
async function record() {
  await document.fonts.load(`400 16px "CM"`);
  await document.fonts.load(`700 16px "CM"`);
  const fps = parseFloat(query.get("fps") || "60"), count = Math.round(D.seconds * fps);
  const W = parseInt(query.get("w") || innerWidth), H = parseInt(query.get("h") || innerHeight);
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);  // exact video size, independent of the browser window
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  fit();
  const out = document.createElement("canvas");
  out.width = W; out.height = H;
  const c = out.getContext("2d");
  const first = parseInt(query.get("from") || "0");
  const last = Math.min(parseInt(query.get("to") || String(count)), count);
  for (let i = first; i < last; i++) {
    frame(count > 1 ? i / (count - 1) : 1);
    compose(out, c);
    const png = await new Promise(resolve => out.toBlob(resolve, "image/png"));
    await fetch(`${exportTo}/frame/${i}`, { method: "POST", body: png });
  }
  await fetch(`${exportTo}/done`, { method: "POST", body: String(last - first) });
}

const replay = document.querySelector(".replay");
let start = performance.now();
function tick(now) {
  const progress = frozen ?? clamp((now - start) / 1000 / D.seconds);
  frame(progress);
  replay.classList.toggle("show", frozen === null && progress >= 1);
  requestAnimationFrame(tick);
}
if (exportTo) record(); else requestAnimationFrame(tick);
addEventListener("click", () => { start = performance.now(); });
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
  fit();
});
</script>
</body>
</html>
"""


def _font(name):
    import matplotlib

    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
    return base64.b64encode(path.read_bytes()).decode()


def _edge_alpha(valid):
    """Visibility on one day's grid: 0 where the model cannot reach the floor point, 1
    from 4 grid cells inside, rising smoothly between. Measuring the distance rounds
    off the grid's stair steps, so the visible edge is a smooth curve."""
    from scipy.ndimage import distance_transform_edt

    if valid.all():
        return np.ones(valid.shape)
    x = np.clip((distance_transform_edt(valid) - 1.0) / 3.0, 0.0, 1.0)
    return np.where(valid, x * x * (3 - 2 * x), 0.0)


def _fill_days(days):
    """Sheets and lines for every day with no gaps, and how visible each part is.

    Where a model cannot reach a floor point the sheet fades out (see _edge_alpha);
    those points keep their floor position and borrow the height of the nearest point,
    so every point can morph. A sheet a day marks hidden (`<key>_hidden`) keeps the
    nearest shown day's shape and fades out entirely. A point never visible is dropped
    (NaN). A line missing on a day borrows the nearest day's line and is hidden there.
    Returns sheets[key] (days, n, n, 3), alphas[key] (None when all visible) and lines
    per day (with a `visible` flag).
    """
    from scipy.ndimage import distance_transform_edt

    sheets, alphas = {}, {}
    for key in ("heston", "market"):
        hidden = [bool(d.get(key + "_hidden")) for d in days]
        if all(hidden):
            raise ValueError(f"The {key} sheet is hidden on every day")
        points = np.array([d[key] for d in days], dtype=float)
        shown = [k for k in range(len(days)) if not hidden[k]]
        for k in range(len(days)):
            if hidden[k]:
                points[k] = points[min(shown, key=lambda g: (abs(g - k), g > k))]
        found = np.isfinite(points).all(axis=-1)
        counts = found.sum(axis=0)
        floor = (
            np.where(found[..., None], points, 0.0)[..., :2].sum(axis=0)
            / np.maximum(counts, 1)[..., None]
        )
        filled, alpha = points.copy(), np.zeros(found.shape)
        for k in range(len(days)):
            if not found[k].any():
                raise ValueError(f"The {key} sheet is empty on some day")
            near = distance_transform_edt(~found[k], return_distances=False, return_indices=True)
            height = points[k][near[0], near[1], 2]
            filled[k] = np.where(
                found[k][..., None], points[k], np.concatenate((floor, height[..., None]), axis=-1)
            )
            given = days[k].get(key + "_alpha")
            alpha[k] = (
                0.0
                if hidden[k]
                else (
                    np.where(found[k], given, 0.0) if given is not None else _edge_alpha(found[k])
                )
            )
        ever = (alpha > 1e-3).any(axis=0) & (counts > 0)
        filled[:, ~ever] = np.nan
        sheets[key] = filled
        alphas[key] = None if (alpha[:, ever] >= 1 - 1e-9).all() else alpha
    n_lines = len(days[0]["lines"])
    if any(len(d["lines"]) != n_lines for d in days):
        raise ValueError("Every day needs the same number of lines")

    def ok(line):
        return bool(
            np.isfinite(line["start"]).all()
            and np.isfinite(line["end"]).all()
            and np.isfinite(line["distance"])
        )

    lines = [[] for _ in days]
    for i in range(n_lines):
        good = [k for k, d in enumerate(days) if ok(d["lines"][i])]
        if not good:
            continue  # never valid: drop this line from the whole movie
        for k in range(len(days)):
            j = k if k in good else min(good, key=lambda g: (abs(g - k), g > k))
            seen = float(days[k]["lines"][i].get("visible", 1.0)) if k in good else 0.0
            lines[k].append({**days[j]["lines"][i], "visible": seen})
    return sheets, alphas, lines


def flow_frames(dates, slices, every=4, sigma=4.0):
    """Keyframes for a long, fast movie: one every `every` trading days, each a Gaussian
    average (sd `sigma` trading days) of the nearby days' sheets, fades, lines and
    numbers. Played without holds, the movie flows instead of flickering day to day.
    Returns (keyframes, labels) for write_html; labels are month and year.
    """
    sheets, alphas, lines = _fill_days(slices)
    days = np.arange(len(slices))
    centers = np.arange(0, len(slices), every, dtype=float)
    if centers[-1] != days[-1]:
        centers = np.append(centers, days[-1])
    frames, labels = [], []
    for c in centers:
        w = np.exp(-0.5 * ((days - c) / sigma) ** 2)
        w /= w.sum()
        frame = {"slice": slices[0]["slice"]}
        for key in ("heston", "market"):
            if alphas[key] is None:
                frame[key] = np.einsum("d,dijk->ijk", w, sheets[key])
                frame[key + "_alpha"] = np.ones(sheets[key].shape[1:3])
                continue
            # Positions count only where shown, so stand-in heights never leak in.
            shown = np.einsum("d,dij->dij", w, alphas[key])
            total = shown.sum(axis=0)
            plain = np.einsum("d,dijk->ijk", w, sheets[key])
            mixed = (
                np.einsum("dij,dijk->ijk", shown, sheets[key]) / np.maximum(total, 1e-12)[..., None]
            )
            frame[key] = np.where((total > 1e-6)[..., None], mixed, plain)
            frame[key + "_alpha"] = total
        frame["lines"] = []
        for i in range(len(lines[0])):
            seen = np.array([lines[d][i]["visible"] for d in days], dtype=float)
            weight = w * seen
            total = weight.sum()
            use = weight / total if total > 1e-9 else w

            def mean(field):
                return np.einsum("d,dk->k", use, np.array([lines[d][i][field] for d in days]))

            frame["lines"].append(
                {
                    "start": mean("start"),
                    "end": mean("end"),
                    "day": lines[0][i]["day"],
                    "distance": float(use @ np.array([lines[d][i]["distance"] for d in days])),
                    "visible": float(np.clip(w @ seen, 0, 1)),
                }
            )
        frames.append(frame)
        labels.append(f"{pd.Timestamp(dates[int(round(c))]):%b %Y}")
    return frames, labels


def gap_scale(distances):
    """Fixed colour scale for the distance lines: from the 5th to the 95th percentile of
    the movie's gaps (so a single extreme day cannot wash out the rest), widened to
    clean numbers. Colours are logarithmic in between."""
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if not d.size:
        return [0.02, 0.2]
    low, high = np.percentile(d, [5, 95])
    steps = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8])

    def clean(x, up):
        mag = 10.0 ** np.floor(np.log10(x))
        candidates = np.concatenate([steps * mag / 10, steps * mag, steps * mag * 10])
        return float(candidates[candidates >= x].min() if up else candidates[candidates <= x].max())

    low, high = clean(low, False), clean(high, True)
    return [round(low, 6), round(high if high > low else 2 * low, 6)]


def tangent_patches(surface):
    """Central-difference tangents and signed height departures on two local patches.

    This measures the displayed 2D slice, not intrinsic full-manifold curvature.
    Keeping signed departures lets playback evaluate the maximum after temporal
    interpolation rather than incorrectly interpolating precomputed maxima.
    """
    surface = np.asarray(surface, dtype=float)
    n, m, _ = surface.shape
    if min(n, m) < 5 or not np.isfinite(surface).all():
        raise ValueError("Tangent patches need a finite grid with at least five points per side")
    radius = max(1, min(4, (min(n, m) - 1) // 8))
    patches = []
    for fi, fj in ((0.30, 0.25), (0.70, 0.75)):
        i = int(np.clip(round(fi * (n - 1)), radius, n - radius - 1))
        j = int(np.clip(round(fj * (m - 1)), radius, m - radius - 1))
        p = surface[i, j]
        u = (surface[i, j + 1] - surface[i, j - 1]) / 2
        v = (surface[i + 1, j] - surface[i - 1, j]) / 2
        offsets = np.arange(-radius, radius + 1)
        dj, di = np.meshgrid(offsets, offsets)
        plane = p + dj[..., None] * u + di[..., None] * v
        local = surface[i - radius : i + radius + 1, j - radius : j + radius + 1]
        np.testing.assert_allclose(local[..., :2], plane[..., :2], atol=1e-10)
        patches.append({
            "p": p.tolist(), "u": (radius * u).tolist(), "v": (radius * v).tolist(),
            "departures": (local[..., 2] - plane[..., 2]).ravel().tolist(),
        })
    return patches


def write_html(
    days,
    path,
    dates,
    hold=2.2,
    move=1.0,
    reveal=0.25,
    units="USD/bbl",
    labels=None,
    note="",
    finale=0.0,
    rescale=None,
    fixed_camera=False,
    axis_units=None,
    local_line_note="",
    steady_surfaces=False,
    vertical_ranges=None,
    compact_hud=False,
    axes=None,
    view_note="",
    sheet_names=None,
    sheet_symbols=None,
    top_sheet="market",
    show_warnings=True,
    tangent_planes=False,
):
    """Self-contained WebGL movie page: holds on each day, then glides to the next.

    `days` are day_slice() / day_sheets() outputs, each in its own day frame.
    `rescale` = {"at": s, "over": s, "from": i}: from second `at`, over `over` seconds,
    the gap axis zooms in to fit only days i onward (e.g. once a stormy year is over). Opens in a browser (click to replay);
    `?t=0.6` freezes it at 60%. Returns the path; the duration is in the page data.
    """
    if not days or len(days) != len(dates):
        raise ValueError("One projected slice per date required")

    def point(p):
        return [round(float(v), 6) for v in p]

    def grid(array):
        return [[point(p) if np.isfinite(p).all() else None for p in row] for row in array]

    sheets, alphas, lines = _fill_days(days)
    payload = []
    for k, date in enumerate(dates):
        when = pd.Timestamp(date)
        day = {
            "date": labels[k] if labels else f"{when:%b} {when.day}, {when.year}",
            "warning": days[k].get("warning", ""),
            "lines": [
                {
                    "s": point(line["start"]),
                    "e": point(line["end"]),
                    "d": float(line["distance"]),
                    "day": bool(line["day"]),
                    "v": float(line["visible"]),
                }
                for line in lines[k]
            ],
        }
        for key in ("heston", "market"):
            day[key] = grid(sheets[key][k])
            if alphas[key] is not None:
                day[key + "Alpha"] = np.round(alphas[key][k], 3).tolist()
        if tangent_planes:
            day["tangents"] = tangent_patches(sheets["heston"][k])
        payload.append(day)
    flow = movie_seconds(len(days), hold, move)
    shown = [line["distance"] for day in lines for line in day if line["visible"] >= 0.5]
    data = {
        "days": payload,
        "hold": hold,
        "move": move,
        "units": units,
        "note": note,
        "reveal": reveal * flow / (flow + finale),
        "flow": flow,
        "finale": finale,
        "rescale": rescale,
        "fixedCamera": fixed_camera,
        "axisUnits": axis_units,
        "localLineNote": local_line_note,
        "steadySurfaces": steady_surfaces,
        "verticalRanges": vertical_ranges,
        "compactHud": compact_hud,
        "viewNote": view_note,
        "sheetNames": sheet_names,
        "sheetSymbols": sheet_symbols,
        "topSheet": top_sheet,
        "showWarnings": show_warnings,
        "tangentPlanes": tangent_planes,
        "tangentReveal": 0.5,
        "gapScale": gap_scale(shown),
        "seconds": flow + finale,
        "axes": axes or ["slice direction 1", "slice direction 2", "gap"],
    }
    first, last = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()
    html = (
        HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__V__", THREE_VERSION)
        .replace("__CMR__", _font("cmr10.ttf"))
        .replace("__CMB__", _font("cmb10.ttf"))
        .replace("__CMI__", _font("cmmi10.ttf"))
        .replace("__TITLE__", f"{first} to {last}, slice {days[0]['slice']}")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def movie_seconds(n_days, hold=2.2, move=1.0):
    return n_days * hold + (n_days - 1) * move


CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_chrome():
    import os
    import shutil

    for candidate in (os.environ.get("CHROME_PATH"), *CHROME_PATHS):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    raise FileNotFoundError("Chrome/Chromium not found; set CHROME_PATH")


def record_video(
    html_path, mp4_path, seconds, fps=60, size=(1920, 1080), chunk=120, stall=45, retries=3
):
    """Render every frame of a movie page in headless Chrome, then encode H.264 MP4.

    Frames are deterministic (frame i shows time i / fps), so playback is perfectly
    smooth regardless of how fast this machine renders. Long headless sessions can
    stall, so frames are rendered in chunks, each in a fresh browser; a chunk that
    makes no progress for `stall` seconds is retried. Needs internet for three.js.
    """
    import http.server
    import shutil
    import subprocess
    import tempfile
    import threading
    import imageio_ffmpeg

    frames = Path(tempfile.mkdtemp(prefix="manifold_frames_"))
    profile = Path(tempfile.mkdtemp(prefix="manifold_chrome_"))
    done = threading.Event()

    class Receiver(http.server.BaseHTTPRequestHandler):
        def _reply(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_OPTIONS(self):
            self._reply()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/frame/"):
                (frames / f"frame_{int(self.path.rsplit('/', 1)[1]):05d}.png").write_bytes(body)
            elif self.path == "/done":
                done.set()
            self._reply()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = (
        Path(html_path).resolve().as_uri()
        + f"?export=http://127.0.0.1:{server.server_address[1]}&fps={fps}"
        + f"&w={size[0]}&h={size[1]}"
    )
    command = [
        find_chrome(),
        "--headless=new",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={size[0]},{size[1]}",
        "--disable-web-security",
        f"--user-data-dir={profile}",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
    ]
    count = int(round(seconds * fps))

    def written(first, last):
        return sum((frames / f"frame_{i:05d}.png").is_file() for i in range(first, last))

    def attempt(first, last):
        done.clear()
        browser = subprocess.Popen(
            [*command, f"{url}&from={first}&to={last}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            progress, idle = written(first, last), 0
            while not done.wait(1):
                now = written(first, last)
                idle = 0 if now > progress else idle + 1
                progress = now
                if browser.poll() is not None or idle > stall:
                    return False
            return written(first, last) == last - first
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=20)
            except subprocess.TimeoutExpired:
                browser.kill()

    try:
        for first in range(0, count, chunk):
            last = min(first + chunk, count)
            if not any(attempt(first, last) for _ in range(retries)):
                raise RuntimeError(f"Frames {first}-{last} of {html_path} failed {retries} times")
    finally:
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    try:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frames / "frame_%05d.png"),
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frames, ignore_errors=True)
    return Path(mp4_path)
