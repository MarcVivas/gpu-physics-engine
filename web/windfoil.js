// Closed-form antialiased filling of quadratic Bezier outlines, after "Windfoil" (DesLauriers, arXiv 2610.02468).
//
// A path is a list of closed contours; a contour is a list of quadratic segments [x0, y0, x1, y1, x2, y2]
// (a straight edge has its control point halfway). The coverage of a pixel is the box-filtered winding number
//
//     F = 1 / (sx sy)  of the integral of w(x, y) over the pixel's footprint            (nonzero: C = min(|F|, 1);
//                                                                                        even-odd: r = |F| mod 2, C = min(r, 2 - r))
//
// With w(x, y) = the signed number of curve crossings to the right of (x, y), the integral over a footprint is a sum over curve
// pieces that are monotone in x and in y:   the integral of G(x(t)) y'(t) dt  with G(x) = clamp(x - x0, 0, sx)
// (a crossing right of the footprint counts its whole width, one left of it nothing). Each piece splits at the t where it enters and
// leaves the footprint's columns: the parts outside contribute sx * (change in y) or nothing and the part inside
// the closed form  integral of (x(t) + hx) y'(t) dt  =  a quartic in t.  No sampling, so there is no noise and no aliasing.
//
// This file: the CPU side (splitting into monotone pieces, row bands for the shader), a CPU evaluation of the same formula
// (the reference the WGSL is tested against) and the WebGPU pipeline (class Windfoil).

/** One monotone piece = 6 floats + its y range; `pieces` is a Float32Array of stride 8: p0, p1, p2, ymin, ymax. */
export const PIECE_STRIDE = 8;

function lerp(a, b, t) { return a + (b - a) * t; }
function splitQuad(q, t) {
  const [x0, y0, x1, y1, x2, y2] = q;
  const ax = lerp(x0, x1, t), ay = lerp(y0, y1, t), bx = lerp(x1, x2, t), by = lerp(y1, y2, t);
  const mx = lerp(ax, bx, t), my = lerp(ay, by, t);
  return [[x0, y0, ax, ay, mx, my], [mx, my, bx, by, x2, y2]];
}

/** Splits a quadratic at its x and y extrema: up to three pieces, each monotone in x and in y. */
export function monotonePieces(q) {
  const cuts = [];
  for (const k of [0, 1]) {
    const a = q[k], b = q[2 + k], c = q[4 + k], den = a - 2 * b + c;
    if (Math.abs(den) > 1e-12) { const t = (a - b) / den; if (t > 1e-6 && t < 1 - 1e-6) cuts.push(t); }
  }
  cuts.sort((a, b) => a - b);
  const out = []; let rest = q, done = 0;
  for (const t of cuts) {
    if (t - done < 1e-6) continue;
    const [l, r] = splitQuad(rest, (t - done) / (1 - done));
    out.push(l); rest = r; done = t;
  }
  out.push(rest);
  return out;
}

/** Pieces of a whole path as a Float32Array (horizontal pieces, which add nothing to the winding, are dropped). */
export function buildPieces(path) {
  const list = [];
  for (const contour of path) for (const q of contour) for (const p of monotonePieces(q)) {
    const lo = Math.min(p[1], p[3], p[5]), hi = Math.max(p[1], p[3], p[5]);
    if (hi - lo > 1e-9) list.push([...p, lo, hi]);
  }
  const out = new Float32Array(PIECE_STRIDE * list.length);
  list.forEach((p, i) => out.set(p, PIECE_STRIDE * i));
  return out;
}

/** The t in [0, 1] where the monotone quadratic A t^2 + B t + C reaches v, clamped (before the start: the end it begins at). */
function tCross(A, B, C, v) {
  const q0 = C, q1 = A + B + C, inc = q0 <= q1;
  if (v <= Math.min(q0, q1)) return inc ? 0 : 1;
  if (v >= Math.max(q0, q1)) return inc ? 1 : 0;
  const c = C - v;
  let t;
  if (Math.abs(A) < 1e-7 * Math.max(Math.abs(B), 1e-12)) t = -c / B;
  else {
    const sq = Math.sqrt(Math.max(B * B - 4 * A * c, 0)), qq = -0.5 * (B + (B >= 0 ? sq : -sq));
    const t1 = qq / A, t2 = qq !== 0 ? c / qq : t1;
    t = t1 >= -1e-4 && t1 <= 1 + 1e-4 ? t1 : t2;
  }
  return Math.min(1, Math.max(0, t));
}

/** The integral of w over the box [cx-hx, cx+hx] x [cy-hy, cy+hy], summed over `pieces[from..from+count)` (indices of pieces, or all). */
export function boxWinding(pieces, cx, cy, hx, hy, list) {
  const sx = 2 * hx, sy = 2 * hy;
  let total = 0;
  const n = list ? list.length : pieces.length / PIECE_STRIDE;
  for (let k = 0; k < n; k++) {
    const o = PIECE_STRIDE * (list ? list[k] : k);
    if (pieces[o + 6] >= cy + hy || pieces[o + 7] <= cy - hy) continue;
    const x0 = pieces[o] - cx, y0 = pieces[o + 1] - cy, x1 = pieces[o + 2] - cx, y1 = pieces[o + 3] - cy, x2 = pieces[o + 4] - cx, y2 = pieces[o + 5] - cy;
    const ax2 = x0 - 2 * x1 + x2, ax1 = 2 * (x1 - x0), ax0 = x0;
    const ay2 = y0 - 2 * y1 + y2, ay1 = 2 * (y1 - y0), ay0 = y0;
    const ty0 = tCross(ay2, ay1, ay0, -hy), ty1 = tCross(ay2, ay1, ay0, hy);
    const ta = Math.min(ty0, ty1), tb = Math.max(ty0, ty1);
    if (tb <= ta) continue;
    const inc = x0 <= x2;
    const tl = tCross(ax2, ax1, ax0, -hx), tr = tCross(ax2, ax1, ax0, hx);
    const sm = Math.min(Math.max(Math.min(tl, tr), ta), tb), lg = Math.min(Math.max(Math.max(tl, tr), ta), tb);
    const yAt = (t) => (ay2 * t + ay1) * t + ay0;
    // before the footprint's columns: nothing (x left of it) or its whole width (x right of it); inside: the closed form; then the other side
    total += (inc ? 0 : sx) * (yAt(sm) - yAt(ta));
    const kx = ax0 + hx;
    const P = (t) => t * (kx * ay1 + t * ((ax1 * ay1 + 2 * kx * ay2) / 2 + t * ((ax2 * ay1 + 2 * ax1 * ay2) / 3 + t * (ax2 * ay2 / 2))));
    total += P(lg) - P(sm);
    total += (inc ? sx : 0) * (yAt(tb) - yAt(lg));
  }
  return total / (sx * sy);
}

export function coverage(F, rule = "nonzero") {
  const a = Math.abs(F);
  if (rule === "evenodd") { const r = a - 2 * Math.floor(a / 2); return Math.min(r, 2 - r); }
  return Math.min(a, 1);
}

// ------------------------------------------------------------------------------------------- helpers to make paths
/** A circle as `n` quadratic arcs (counter-clockwise unless `cw`): a few thousandths off round for n = 8. */
export function circleContour(cx, cy, r, n = 8, cw = false) {
  const out = [], step = 2 * Math.PI / n, rc = r / Math.cos(step / 2);
  for (let i = 0; i < n; i++) {
    const a0 = i * step, a1 = (i + 1) * step, am = (a0 + a1) / 2;
    out.push([cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + rc * Math.cos(am), cy + rc * Math.sin(am), cx + r * Math.cos(a1), cy + r * Math.sin(a1)]);
  }
  return cw ? reverseContour(out) : out;
}
export function reverseContour(c) { return c.map((q) => [q[4], q[5], q[2], q[3], q[0], q[1]]).reverse(); }
/** A polygon (straight edges) from [x, y, x, y, ...]. */
export function polyContour(pts) {
  const out = [], n = pts.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, x0 = pts[2 * i], y0 = pts[2 * i + 1], x1 = pts[2 * j], y1 = pts[2 * j + 1];
    out.push([x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1]);
  }
  return out;
}
/** A closed contour through the curve points of a smooth outline: the on-curve points `pts` (x, y, ...) joined by quadratics
 *  whose controls are the chord midpoints' neighbours (a quadratic B-spline through the midpoints): a smooth closed blob. */
export function blobContour(pts) {
  const n = pts.length / 2, out = [];
  const mid = (i) => { const a = i % n, b = (i + 1) % n; return [(pts[2 * a] + pts[2 * b]) / 2, (pts[2 * a + 1] + pts[2 * b + 1]) / 2]; };
  for (let i = 0; i < n; i++) {
    const m0 = mid(i + n - 1), m1 = mid(i), c = (i) % n;
    out.push([m0[0], m0[1], pts[2 * c], pts[2 * c + 1], m1[0], m1[1]]);
  }
  return out;
}

/** The exact winding number at a point by flattening each quadratic into `n` straight steps (the supersampling reference). */
export function windingAt(path, px, py, n = 64) {
  let w = 0;
  for (const contour of path) for (const q of contour) {
    let ax = q[0], ay = q[1];
    for (let i = 1; i <= n; i++) {
      const t = i / n, u = 1 - t;
      const bx = u * u * q[0] + 2 * u * t * q[2] + t * t * q[4], by = u * u * q[1] + 2 * u * t * q[3] + t * t * q[5];
      if ((ay <= py) !== (by <= py)) {                       // the step crosses the horizontal line through the point
        const xc = ax + (py - ay) / (by - ay) * (bx - ax);
        if (xc > px) w += by > ay ? 1 : -1;
      }
      ax = bx; ay = by;
    }
  }
  return w;
}

// ------------------------------------------------------------------------------------------------- WebGPU
const WGSL = /* wgsl */ `
struct Params {
  view: vec4<f32>,      // camera x, camera y (world point at the screen centre), pixels per world unit, padding in pixels
  res: vec4<f32>,       // target width, height, band y0, band height
  bbox: vec4<f32>,      // path bounds: x0, y0, x1, y1
  color: vec4<f32>,
  misc: vec4<u32>,      // rule (0 nonzero, 1 even-odd), number of bands
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pieces: array<f32>;      // stride 8
@group(0) @binding(2) var<storage, read> bands: array<u32>;       // start, count per band
@group(0) @binding(3) var<storage, read> items: array<u32>;

struct VOut { @builtin(position) pos: vec4<f32> };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  let pad = P.view.w / P.view.z;
  let lo = P.bbox.xy - vec2(pad);
  let hi = P.bbox.zw + vec2(pad);
  let w = mix(lo, hi, c[vi]);
  let ndc = (w - P.view.xy) * P.view.z * 2.0 / P.res.xy;
  var o: VOut;
  o.pos = vec4(ndc, 0.0, 1.0);
  return o;
}

fn tcross(A: f32, B: f32, C: f32, v: f32) -> f32 {
  let q0 = C;
  let q1 = A + B + C;
  let inc = q0 <= q1;
  if (v <= min(q0, q1)) { return select(1.0, 0.0, inc); }
  if (v >= max(q0, q1)) { return select(0.0, 1.0, inc); }
  let c = C - v;
  var t = 0.0;
  if (abs(A) < 1e-7 * max(abs(B), 1e-12)) {
    t = -c / B;
  } else {
    let sq = sqrt(max(B * B - 4.0 * A * c, 0.0));
    let qq = -0.5 * (B + select(-sq, sq, B >= 0.0));
    let t1 = qq / A;
    var t2 = t1;
    if (qq != 0.0) { t2 = c / qq; }
    t = select(t2, t1, t1 >= -1e-4 && t1 <= 1.0001);
  }
  return clamp(t, 0.0, 1.0);
}

@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let hx = 0.5 / P.view.z;
  let hy = hx;
  let sx = 2.0 * hx;
  // the pixel centre in the world, from the framebuffer position (exact; an interpolated varying is not)
  let cx = P.view.x + (in.pos.x - 0.5 * P.res.x) / P.view.z;
  let cy = P.view.y - (in.pos.y - 0.5 * P.res.y) / P.view.z;
  var band = i32(floor((cy - P.res.z) / P.res.w));
  band = clamp(band, 0, i32(P.misc.y) - 1);
  let start = bands[2 * band];
  let count = bands[2 * band + 1];
  var total = 0.0;
  for (var k = 0u; k < count; k++) {
    let o = items[start + k] * 8u;
    if (pieces[o + 6u] >= cy + hy || pieces[o + 7u] <= cy - hy) { continue; }
    let x0 = pieces[o] - cx; let y0 = pieces[o + 1u] - cy;
    let x1 = pieces[o + 2u] - cx; let y1 = pieces[o + 3u] - cy;
    let x2 = pieces[o + 4u] - cx; let y2 = pieces[o + 5u] - cy;
    let ax2 = x0 - 2.0 * x1 + x2; let ax1 = 2.0 * (x1 - x0); let ax0 = x0;
    let ay2 = y0 - 2.0 * y1 + y2; let ay1 = 2.0 * (y1 - y0); let ay0 = y0;
    let ty0 = tcross(ay2, ay1, ay0, -hy);
    let ty1 = tcross(ay2, ay1, ay0, hy);
    let ta = min(ty0, ty1);
    let tb = max(ty0, ty1);
    if (tb <= ta) { continue; }
    let inc = x0 <= x2;
    let tl = tcross(ax2, ax1, ax0, -hx);
    let tr = tcross(ax2, ax1, ax0, hx);
    let sm = clamp(min(tl, tr), ta, tb);
    let lg = clamp(max(tl, tr), ta, tb);
    let kx = ax0 + hx;
    let a4 = ax2 * ay2 / 2.0;
    let a3 = (ax2 * ay1 + 2.0 * ax1 * ay2) / 3.0;
    let a2 = (ax1 * ay1 + 2.0 * kx * ay2) / 2.0;
    let a1 = kx * ay1;
    let ysm = (ay2 * sm + ay1) * sm + ay0;
    let ylg = (ay2 * lg + ay1) * lg + ay0;
    let yta = (ay2 * ta + ay1) * ta + ay0;
    let ytb = (ay2 * tb + ay1) * tb + ay0;
    let psm = sm * (a1 + sm * (a2 + sm * (a3 + sm * a4)));
    let plg = lg * (a1 + lg * (a2 + lg * (a3 + lg * a4)));
    total += select(sx, 0.0, inc) * (ysm - yta) + (plg - psm) + select(0.0, sx, inc) * (ytb - ylg);
  }
  let f = abs(total / (sx * 2.0 * hy));
  var cov = min(f, 1.0);
  if (P.misc.x == 1u) {
    let r = f - 2.0 * floor(f / 2.0);
    cov = min(r, 2.0 - r);
  }
  let a = P.color.a * cov;
  return vec4(P.color.rgb * a, a);
}
`;

/** One filled path on the GPU. `const wf = Windfoil.create(device, "rgba8unorm"); wf.setPath(path); wf.draw(pass, {...})`. */
export class Windfoil {
  static create(device, format) { return new Windfoil(device, format); }
  constructor(device, format) {
    this.device = device; this.format = format;
    const module = device.createShaderModule({ code: WGSL, label: "windfoil" });
    this.pipeline = device.createRenderPipeline({
      layout: "auto", label: "windfoil",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format, blend: {
        color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } } }] },
      primitive: { topology: "triangle-list" },
    });
    this.params = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "windfoil params" });
    this.pieces = null; this.bind = null; this.count = 0;
  }
  /** Uploads a path. `pad` is how far (in world units) a pixel's footprint can reach over a row band's edge: the size of a pixel at the
   *  smallest zoom you will draw at (a piece within `pad` of a band belongs to it); `bands` defaults to about sqrt(#pieces). */
  setPath(path, { pad = 0.05, bands } = {}) {
    const pieces = buildPieces(path), n = pieces.length / PIECE_STRIDE;
    this.count = n; this.pieceData = pieces;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const o = PIECE_STRIDE * i;
      for (const k of [0, 2, 4]) { x0 = Math.min(x0, pieces[o + k]); x1 = Math.max(x1, pieces[o + k]); }
      y0 = Math.min(y0, pieces[o + 6]); y1 = Math.max(y1, pieces[o + 7]);
    }
    if (!n) { x0 = y0 = x1 = y1 = 0; }
    const nb = Math.max(1, Math.min(256, bands ?? Math.ceil(Math.sqrt(n)))), bh = Math.max((y1 - y0) / nb, 1e-6);
    const info = new Uint32Array(2 * nb), list = [];
    for (let b = 0; b < nb; b++) {
      const lo = y0 + b * bh - pad, hi = y0 + (b + 1) * bh + pad;
      info[2 * b] = list.length;
      for (let i = 0; i < n; i++) if (pieces[PIECE_STRIDE * i + 7] > lo && pieces[PIECE_STRIDE * i + 6] < hi) list.push(i);
      info[2 * b + 1] = list.length - info[2 * b];
    }
    this.bounds = [x0, y0, x1, y1]; this.nb = nb; this.bh = bh; this.y0 = y0;
    const mk = (data, label) => {
      const buf = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label });
      this.device.queue.writeBuffer(buf, 0, data); return buf;
    };
    const items = Uint32Array.from(list);
    this.bufs = [mk(pieces.length ? pieces : new Float32Array(8), "pieces"), mk(info, "bands"), mk(items.length ? items : new Uint32Array(4), "items")];
    this.bind = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } },
      ...this.bufs.map((buffer, i) => ({ binding: i + 1, resource: { buffer } })) ] });
    this.itemCount = list.length;
    return this;
  }
  /** Draws into `pass`: camera = the world point at the target's centre, scale = pixels per world unit, res = target size in pixels.
   *  Alpha is premultiplied; `rule` is "nonzero" or "evenodd". Y is up in the world. */
  draw(pass, { camera = [0, 0], scale = 1, res, color = [0, 0, 0, 1], rule = "nonzero" }) {
    if (!this.count) return;
    const u = new ArrayBuffer(80), f = new Float32Array(u), w = new Uint32Array(u);
    f.set([camera[0], camera[1], scale, 1.5, res[0], res[1], this.y0, this.bh, ...this.bounds, ...color], 0);
    w[16] = rule === "evenodd" ? 1 : 0; w[17] = this.nb;
    this.device.queue.writeBuffer(this.params, 0, u);
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.bind); pass.draw(6);
  }
}
