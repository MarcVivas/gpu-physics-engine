// gpuphys_webgpu.js -- the engine's circle pipeline on WebGPU: the same kernels as gpuphys.c and the Python host,
// as WGSL (converted from SPIR-V by tools/spv2wgsl.py), dispatched from JavaScript. Nothing is read back to
// draw: the positions live in a GPU buffer the renderer binds directly (see renderer.js).
//
//   const device = await (await navigator.gpu.requestAdapter()).requestDevice({ requiredLimits: GpuWorld.limits(adapter) });
//   const w = await GpuWorld.create(device, { kernels: "gpu/spirv", capacity: 20000, world: [64, 36], gravity: [0, -18] });
//   w.setTileGrid(64, 36, cells);                  // Uint32Array, row-major, y up, 1 = solid
//   const crate = w.addBox(10, 20, 1, 1);          // bodies are clusters of circles joined by distance joints
//   w.brushSand(30, 25, 1.5, 200);                 // free circles (tag 1)
//   w.step(1 / 60);                                // one submit; await w.readPositions() only for tests
//
// A step is: integrate, grid + radix sort + collision cells, `iterations` x (colour-batched collision solver,
// joints, tiles), tile friction. Per-body material, filters and sensors (Box2D's fixture settings) switch the
// step to the prowl_material kernels, exactly as the Python host does. Joints between bodies, contact events and
// polygons are CPU-side in gpuphys.c: use the WebAssembly build (runtime/wasm) when a game needs them.

export const TAG = { BALL: 0, SAND: 1, CRATE: 2, SHARD: 3, ROPE: 4 };
const RIGID = 0, ROPE_J = 1, STRUT = 2, FIXED = 0xffffffff;
const NUM_BLOCKS = 45, RADIX_WG = 256, MAX_CELLS = 4, PREFIX_LIMIT = 256 * 256;
const FRICTION_UNIT = 0.375, ALL = 0xffffffff;
const ceilDiv = (a, b) => Math.floor((a + b - 1) / b);
const U = typeof GPUBufferUsage !== "undefined" ? GPUBufferUsage : {};

function pack(fmt, ...vals) {                       // "f"/"u"/"i" per value: the kernels' parameter blocks
  const b = new ArrayBuffer(4 * fmt.length), dv = new DataView(b);
  for (let i = 0; i < fmt.length; i++) {
    if (fmt[i] === "f") dv.setFloat32(4 * i, vals[i], true);
    else if (fmt[i] === "i") dv.setInt32(4 * i, vals[i], true);
    else dv.setUint32(4 * i, vals[i] >>> 0, true);
  }
  return b;
}

class Kernels {
  constructor(device, base, manifest) {
    this.device = device; this.base = base; this.manifest = manifest;
    this.entries = {}; this.modules = {}; this.pipes = {};
    for (const [name, rec] of Object.entries(manifest)) {
      if (name === "_meta") continue;
      for (const e of rec.entry_points) this.entries[e.name] = { module: name, info: e };
    }
  }
  static async load(device, base) {
    const manifest = await (await fetch(base + "/manifest.json")).json();
    if (manifest._meta?.wgsl?.params !== "uniform") throw new Error(base + " has no WGSL (build.py --wgsl)");
    return new Kernels(device, base, manifest);
  }
  has(entry) { return entry in this.entries; }
  async module(name) {
    if (!this.modules[name]) {
      this.modules[name] = (async () => {
        const code = await (await fetch(`${this.base}/${this.manifest[name].wgsl}`)).text();
        const m = this.device.createShaderModule({ code, label: name });
        const msgs = (await m.getCompilationInfo()).messages.filter((x) => x.type === "error");
        if (msgs.length) throw new Error(`${name}.wgsl: ${msgs.map((x) => x.message).join("; ")}`);
        return m;
      })();
    }
    return this.modules[name];
  }
  /** The compute pipeline of an entry point, with an explicit bind group layout built from the manifest. */
  async pipeline(entry) {
    if (!this.pipes[entry]) {
      this.pipes[entry] = (async () => {
        const { module, info } = this.entries[entry];
        const mod = await this.module(module);
        const entries = info.buffers.map((b) => ({ binding: b.binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }));
        if (info.push?.length) entries.push({ binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } });
        const bgl = this.device.createBindGroupLayout({ entries, label: entry });
        const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        const pipe = await this.device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: entry }, label: entry });
        return { pipe, bgl, info, local: info.local_size[0] };
      })();
      this.pipes[entry].then((r) => { this.pipes[entry].resolved = r; });
    }
    return this.pipes[entry];
  }
}

export class GpuWorld {
  /** Limits to ask requestDevice for: the filtered solver binds 10 storage buffers. */
  static limits(adapter) {
    return { maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
             maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
  }
  static async create(device, opts = {}) {
    const k = await Kernels.load(device, opts.kernels ?? "gpu/spirv");
    const w = new GpuWorld(device, k, opts);
    await w.prepare();
    return w;
  }

  constructor(device, kernels, o) {
    this.device = device; this.k = kernels;
    this.capacity = o.capacity ?? 4096;
    this.world = [o.world?.[0] ?? 64, o.world?.[1] ?? 36];
    this.gravity = [o.gravity?.[0] ?? 0, o.gravity?.[1] ?? -18];
    this.damping = o.damping ?? 0.999;
    this.sandDamping = o.sandDamping ?? 0.985;
    this.maxRadius = o.maxRadius ?? 0.5;
    this.iterations = o.iterations ?? 2;
    this.friction = o.friction ?? 0.15;               // kernel units: 0.375 x Unity's friction
    this.bounceLoss = o.bounceLoss ?? 0.6;
    this.bounceThreshold = 1.0;
    this.n = 0; this.steps = 0;
    const cap = this.capacity, pairs = MAX_CELLS * cap;
    const buf = (bytes, usage = 0, label = "") => device.createBuffer({ size: Math.max(16, bytes), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | usage, label });
    this.buf = buf;
    this.positions = buf(8 * cap, U.VERTEX, "positions"); this.previous = buf(8 * cap, 0, "previous");
    this.radius = buf(4 * cap, 0, "radius"); this.tags = buf(4 * cap, 0, "tags"); this.groups = buf(4 * cap, 0, "groups");
    this.cellIds = buf(4 * pairs); this.objectIds = buf(4 * pairs);
    this.sortKeysB = buf(4 * pairs); this.sortPayloadB = buf(4 * pairs);
    const wgs = Math.max(1, ceilDiv(ceilDiv(pairs, NUM_BLOCKS), RADIX_WG));
    this.histogram = buf(4 * RADIX_WG * wgs);
    this.chunksCap = ceilDiv(pairs, 4);
    this.chunkCounts = buf(4 * Math.max(256, this.chunksCap));
    this.collisionCells = buf(4 * Math.max(4, pairs));
    this.indirect = buf(16, U.INDIRECT, "indirect");
    this.cellUniform = buf(16);
    this.sums = [];
    for (let m = this.chunksCap; ; m = ceilDiv(m, 256)) {
      this.sums.push(buf(4 * Math.max(256, ceilDiv(m, 256))));
      if (m < PREFIX_LIMIT) break;
    }
    this.copyScratch = buf(16, 0, "copy scratch");
    this.tiles = buf(16); this.tilesDims = [0, 0];
    this.maxJoints = o.maxJoints ?? 4 * cap;
    this.jointI = buf(16 * this.maxJoints); this.jointF = buf(16 * this.maxJoints);
    // material path
    this.useMaterial = false;
    this.materialBuf = buf(20 * cap, 0, "material"); this.filterBuf = buf(16 * cap, 0, "filter"); this.omegaBuf = buf(4 * cap, 0, "omega");
    // CPU mirrors of what the host needs to know about each circle
    this.owner = []; this.tagMirror = []; this.radMirror = [];
    this.m = { custom: [], mat: [], dens: [], drag: [], flt: [], sensor: [] };
    this.bodies = new Map(); this.nextBody = 1;
    this.joints = new Map(); this.nextJoint = 1; this.jointBatches = []; this.jointsDirty = false;
    this.matDirty = true; this.matDt = null;
    this.paramPool = []; this.groupCache = new Map(); this.ids = new WeakMap(); this.idn = 0; this.slot = 0;
    this.bounceTile = null;
    // Bezier colliders (kernels/prowl_curves.rs): 16 floats each
    this.curveCap = 0; this.curveData = new Float32Array(0); this.nCurves = 0; this.curveBuf = buf(16, 0, "curves");
  }

  async prepare() {
    const need = ["integrate_tagged", "build_cell_ids_array", "build_histogram", "scatter_keys", "count_objects_for_each_chunk",
      "prefix_sum_of_each_block", "prefix_sum_of_the_block_sums", "add_block_prefix_sums_to_the_buffer", "build_collision_cells_array",
      "solve_collisions_grouped", "solve_joints", "tile_collide", "tile_friction"];
    if (this.hasMaterialKernels()) need.push("integrate_material", "solve_collisions_filtered", "tile_material");
    if (this.hasCurveKernels()) need.push("curve_collide", "curve_material");
    await Promise.all(need.map((e) => this.k.pipeline(e)));
  }
  hasMaterialKernels() { return ["integrate_material", "solve_collisions_filtered", "tile_material"].every((e) => this.k.has(e)); }
  hasCurveKernels() { return this.hasMaterialKernels() && ["curve_collide", "curve_material"].every((e) => this.k.has(e)); }
  get cellSize() { return this.maxRadius * 2.2; }
  write(buffer, offset, data) { this.device.queue.writeBuffer(buffer, offset, data); }

  // ------------------------------------------------------------------ circles
  addCircle(x, y, radius = 0.5, vx = 0, vy = 0, dt = 1 / 60, tag = 0, group = 0) {
    if (this.n >= this.capacity) throw new RangeError(`GPU world is full (${this.capacity} circles)`);
    const i = this.n++;
    this.write(this.positions, 8 * i, new Float32Array([x, y]));
    this.write(this.previous, 8 * i, new Float32Array([x - vx * dt, y - vy * dt]));
    this.write(this.radius, 4 * i, new Float32Array([radius]));
    this.write(this.tags, 4 * i, new Uint32Array([tag]));
    this.write(this.groups, 4 * i, new Uint32Array([group]));
    this.owner.push(group); this.tagMirror.push(tag); this.radMirror.push(radius);
    this.m.custom.push(false); this.m.mat.push([this.friction, 0, 1]); this.m.dens.push(1); this.m.drag.push(0);
    this.m.flt.push([1, ALL, 0]); this.m.sensor.push(false);
    this.maxRadius = Math.max(this.maxRadius, radius);
    this.matDirty = true;
    return i;
  }
  _newBody(kind, circles, extra = {}) {
    const id = this.nextBody++;
    const groups = new Uint32Array(circles.length).fill(id);
    for (const c of circles) { this.owner[c] = id; this.write(this.groups, 4 * c, groups.subarray(0, 1)); }
    this.bodies.set(id, { id, kind, circles, joints: [], ...extra });
    return id;
  }
  addBall(x, y, radius = 0.35, vx = 0, vy = 0, dt = 1 / 60) {
    return this._newBody("ball", [this.addCircle(x, y, radius, vx, vy, dt, TAG.BALL)]);
  }
  /** A crate: a grid of circles (at least 2 x 2) held by stiff distance joints. */
  addBox(x, y, w = 1, h = 1, angle = 0, vx = 0, vy = 0, dt = 1 / 60, stiffness = 1, kind = "box", ttl = null) {
    const r0 = Math.min(w, h) / 4;
    const rows = Math.max(2, Math.round(h / (2 * r0))), cols = Math.max(2, Math.round(w / (2 * r0)));
    const r = Math.min(w / (2 * cols), h / (2 * rows));
    if (this.n + rows * cols > this.capacity) throw new RangeError("GPU world is full");
    const ca = Math.cos(angle), sa = Math.sin(angle), circles = [], pos = [];
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const lx = -w / 2 + r + i * (w - 2 * r) / (cols - 1), ly = -h / 2 + r + j * (h - 2 * r) / (rows - 1);
      const px = x + lx * ca - ly * sa, py = y + lx * sa + ly * ca;
      pos.push([px, py]);
      circles.push(this.addCircle(px, py, r, vx, vy, dt, kind === "shard" ? TAG.SHARD : TAG.CRATE));
    }
    const id = this._newBody(kind, circles, { w, h, cols, rows, ttl });
    const body = this.bodies.get(id);
    for (let a = 0; a < circles.length; a++) for (let b = a + 1; b < circles.length; b++) {
      if (Math.max(Math.abs(a % cols - b % cols), Math.abs(Math.floor(a / cols) - Math.floor(b / cols))) <= 2)
        body.joints.push(this._addJoint(circles[a], circles[b], RIGID, Math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]), stiffness));
    }
    return id;
  }
  addRope(ax, ay, bx, by, segments = 10, radius = 0.07, pinA = true, pinB = false, stiffness = 1, slack = 1) {
    const n = segments + 1;
    if (this.n + n > this.capacity) throw new RangeError("GPU world is full");
    const circles = [];
    for (let k = 0; k < n; k++) circles.push(this.addCircle(ax + (bx - ax) * k / segments, ay + (by - ay) * k / segments, radius, 0, 0, 1 / 60, TAG.ROPE));
    const id = this._newBody("rope", circles), body = this.bodies.get(id);
    const seg = Math.hypot(bx - ax, by - ay) / segments * slack;
    for (let k = 0; k < segments; k++) body.joints.push(this._addJoint(circles[k], circles[k + 1], RIGID, seg, stiffness));
    if (pinA) body.joints.push(this.pin(circles[0], ax, ay));
    if (pinB) body.joints.push(this.pin(circles[n - 1], bx, by));
    return id;
  }
  addSand(x, y, radius = 0.08, vx = 0, vy = 0) { return this.n < this.capacity ? this.addCircle(x, y, radius, vx, vy, 1 / 60, TAG.SAND) : null; }
  brushSand(x, y, brush = 0.5, count = 8, radius = 0.08, rnd = Math.random) {
    let made = 0;
    for (let i = 0; i < count; i++) {
      const a = rnd() * Math.PI * 2, d = brush * Math.sqrt(rnd());
      if (this.addSand(x + Math.cos(a) * d, y + Math.sin(a) * d, radius) !== null) made++;
    }
    return made;
  }
  removeBody(id) {
    const body = this.bodies.get(id);
    this.bodies.delete(id);
    for (const j of body.joints) this.joints.delete(j);
    for (const c of [...body.circles].sort((a, b) => b - a)) {
      for (const [jid, j] of this.joints) if (j[0] === c || j[1] === c) this.joints.delete(jid);
      body.circles.splice(body.circles.indexOf(c), 1);
      this._swapRemove(c);
    }
    this.jointsDirty = true;
  }
  _swapRemove(i) {
    const last = this.n - 1;
    if (i !== last) {
      const enc = this.device.createCommandEncoder();
      // WebGPU cannot copy inside one buffer: go through a scratch buffer (copies run in order)
      for (const [b, w] of [[this.positions, 8], [this.previous, 8], [this.radius, 4], [this.tags, 4], [this.groups, 4]]) {
        enc.copyBufferToBuffer(b, w * last, this.copyScratch, 0, w);
        enc.copyBufferToBuffer(this.copyScratch, 0, b, w * i, w);
      }
      this.device.queue.submit([enc.finish()]);
      for (const arr of [this.owner, this.tagMirror, this.radMirror, ...Object.values(this.m)]) arr[i] = arr[last];
      for (const j of this.joints.values()) { if (j[0] === last) j[0] = i; if (j[1] === last) j[1] = i; }
      const body = this.bodies.get(this.owner[i]);
      if (body) body.circles[body.circles.indexOf(last)] = i;
      this.jointsDirty = true;
    }
    for (const arr of [this.owner, this.tagMirror, this.radMirror, ...Object.values(this.m)]) arr.pop();
    this.n--; this.matDirty = true;
  }

  // ------------------------------------------------------------------- joints
  _addJoint(a, b, flags, rest, stiffness, ax = 0, ay = 0) {
    if (this.joints.size >= this.maxJoints) throw new RangeError("too many joints");
    const id = this.nextJoint++;
    this.joints.set(id, [a, b, flags, rest, stiffness, ax, ay]);
    this.jointsDirty = true;
    return id;
  }
  addCircleJoint(a, b, rest, kind = "rigid", stiffness = 1) {
    return this._addJoint(a, b, { rigid: RIGID, rope: ROPE_J, strut: STRUT }[kind], rest, stiffness);
  }
  pin(a, x, y, rest = 0, kind = "rigid", stiffness = 1) {
    return this._addJoint(a, FIXED, { rigid: RIGID, rope: ROPE_J, strut: STRUT }[kind], rest, stiffness, x, y);
  }
  removeJoint(id) { this.joints.delete(id); this.jointsDirty = true; }
  _uploadJoints() {
    const used = new Map(), colour = new Map();
    for (const [id, j] of this.joints) {
      const ends = j[1] === FIXED ? [j[0]] : [j[0], j[1]];
      const taken = new Set(); for (const e of ends) for (const c of used.get(e) ?? []) taken.add(c);
      let c = 0; while (taken.has(c)) c++;
      colour.set(id, c);
      for (const e of ends) { if (!used.has(e)) used.set(e, new Set()); used.get(e).add(c); }
    }
    const order = [...this.joints.keys()].sort((a, b) => colour.get(a) - colour.get(b));
    const ji = new Uint32Array(4 * order.length), jf = new Float32Array(4 * order.length), batches = [];
    order.forEach((id, k) => {
      const [a, b, flags, rest, stiff, ax, ay] = this.joints.get(id);
      ji.set([a, b, flags, 0], 4 * k); jf.set([rest, stiff, ax, ay], 4 * k);
      if (!batches.length || batches[batches.length - 1][2] !== colour.get(id)) batches.push([k, 0, colour.get(id)]);
      batches[batches.length - 1][1]++;
    });
    this.jointBatches = batches.map(([f, c]) => [f, c]);
    if (order.length) { this.write(this.jointI, 0, ji); this.write(this.jointF, 0, jf); }
    this.jointsDirty = false;
  }

  // ---------------------------------------------------------------- level
  setTileGrid(width, height, cells) {
    const data = cells instanceof Uint32Array ? cells : Uint32Array.from(cells);
    if (data.byteLength > this.tiles.size) { this.tiles = this.buf(data.byteLength, 0, "tiles"); this.groupCache.clear(); }
    this.write(this.tiles, 0, data);
    this.tilesDims = [width | 0, height | 0];
  }

  // ------------------------------------------------- Bezier colliders
  /** A static quadratic Bezier collider p0 -> p1 (control) -> p2 in world units. side 0: a thin wall solid on both faces;
   *  +1 / -1: one edge of a solid contour whose inside is on the left / right of the direction p0 -> p2.
   *  friction is Unity's, bounciness 0..1. Returns the curve's index, or -1 when the kernels are missing. */
  addCurve(x0, y0, x1, y1, x2, y2, friction = 0.4, bounciness = 0, side = 0) {
    if (!this.hasCurveKernels()) return -1;
    if (this.nCurves === this.curveCap) {
      this.curveCap = Math.max(16, this.curveCap * 2);
      const grown = new Float32Array(16 * this.curveCap); grown.set(this.curveData);
      this.curveData = grown;
      this.curveBuf = this.buf(4 * grown.length, 0, "curves"); this.groupCache.clear();
    }
    this.curveData.set([x0, y0, x1, y1, x2, y2, Math.min(x0, x1, x2), Math.min(y0, y1, y2), Math.max(x0, x1, x2), Math.max(y0, y1, y2),
      FRICTION_UNIT * Math.max(0, friction), Math.min(1, Math.max(0, bounciness)), side > 0 ? 1 : side < 0 ? -1 : 0, 0, 0, 0], 16 * this.nCurves);
    this.write(this.curveBuf, 0, this.curveData);
    this.useMaterial = true; this.matDirty = true;
    return this.nCurves++;
  }
  clearCurves() { this.nCurves = 0; }
  get curveCount() { return this.nCurves; }

  // ----------------------------------------------- Box2D's per-body settings
  _circlesOf(body) { return this.bodies.get(body).circles; }
  setMaterial(body, { friction = 0.4, bounciness = 0, density = 1, gravityScale = 1, drag = 0 } = {}) {
    if (!this.hasMaterialKernels()) return false;
    this.useMaterial = true;
    for (const c of this._circlesOf(body)) {
      this.m.custom[c] = true;
      this.m.mat[c] = [FRICTION_UNIT * Math.max(0, friction), Math.min(1, Math.max(0, bounciness)), gravityScale];
      this.m.dens[c] = density > 0 ? density : 1; this.m.drag[c] = Math.max(0, drag);
    }
    this.matDirty = true; return true;
  }
  setFilter(body, { category = 1, mask = ALL, group = 0 } = {}) {
    if (!this.hasMaterialKernels()) return false;
    this.useMaterial = true;
    for (const c of this._circlesOf(body)) this.m.flt[c] = [category >>> 0, mask >>> 0, group | 0];
    this.matDirty = true; return true;
  }
  setSensor(body, on = true) {
    if (!this.hasMaterialKernels()) return false;
    this.useMaterial = true;
    for (const c of this._circlesOf(body)) this.m.sensor[c] = !!on;
    this.matDirty = true; return true;
  }
  setBounceThreshold(speed) { this.bounceThreshold = Math.max(0, speed); }
  setWorldMaterial({ friction, bounciness } = {}) {
    if (friction !== undefined) this.friction = FRICTION_UNIT * Math.max(0, friction);
    if (bounciness !== undefined) this.bounceLoss = 1 - Math.min(1, Math.max(0, bounciness));
    this.matDirty = true;
  }
  _uploadMaterial(dt) {
    const n = this.n, mat = new Float32Array(5 * n), flt = new Uint32Array(4 * n);
    for (let i = 0; i < n; i++) {
      let [f, e, gs] = this.m.mat[i];
      if (!this.m.custom[i]) f = this.friction;
      const sand = this.tagMirror[i] === 1 ? this.sandDamping / this.damping : 1;
      mat.set([f, e, gs, sand / (1 + this.m.drag[i] * dt), 1 / (this.radMirror[i] * this.m.dens[i])], 5 * i);
      let [cat, mask, grp] = this.m.flt[i];
      if (this.m.sensor[i]) cat = mask = grp = 0;
      flt.set([cat, mask, grp >>> 0, this.owner[i]], 4 * i);
    }
    this.write(this.materialBuf, 0, mat); this.write(this.filterBuf, 0, flt);
  }

  // --------------------------------------------------------------- dispatch
  _params(data) {
    let p = this.paramPool[this.slot];
    if (!p) { p = this.paramPool[this.slot] = this.device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST, label: "params" + this.slot }); }
    if (data) this.write(p, 0, data);
    this.slot++;
    return p;
  }
  _bind(entry, pl, params, byName) {
    const ids = pl.info.buffers.map((b) => this._id(byName[b.name], entry, b.name)).join(",");
    const key = `${entry}#${params ? this._id(params) : "-"}#${ids}`;
    let g = this.groupCache.get(key);
    if (!g) {
      const entries = pl.info.buffers.map((b) => ({ binding: b.binding, resource: { buffer: byName[b.name] } }));
      if (params) entries.push({ binding: 30, resource: { buffer: params } });
      g = this.device.createBindGroup({ layout: pl.bgl, entries, label: entry });
      this.groupCache.set(key, g);
    }
    return g;
  }
  _id(buffer, entry = "", name = "") {
    if (!buffer) throw new Error(`${entry}: no buffer for parameter ${name}`);
    let id = this.ids.get(buffer);
    if (id === undefined) { id = ++this.idn; this.ids.set(buffer, id); }
    return id;
  }
  _pl(entry) {                                  // pipelines are prepared before the first step
    const p = this.k.pipes[entry];
    if (!p || !p.resolved) throw new Error("pipeline not prepared: " + entry);
    return p.resolved;
  }
  _run(pass, entry, items, params, byName) {
    const pl = this._pl(entry);
    const pbuf = pl.info.push?.length ? this._params(params) : null;
    pass.setPipeline(pl.pipe);
    pass.setBindGroup(0, this._bind(entry, pl, pbuf, byName));
    pass.dispatchWorkgroups(ceilDiv(items, pl.local));
  }

  _radixSort(pass, count) {
    if (count <= 1) return;
    const threads = ceilDiv(count, NUM_BLOCKS), wgs = ceilDiv(threads, RADIX_WG);
    let a = [this.cellIds, this.objectIds], z = [this.sortKeysB, this.sortPayloadB];
    for (let p = 0; p < 4; p++) {
      const params = pack("uuuu", count, p * 8, wgs, NUM_BLOCKS);
      this._run(pass, "build_histogram", threads, params, { keys_a: a[0], histogram: this.histogram });
      this._run(pass, "scatter_keys", threads, params, { keys_a: a[0], histogram: this.histogram, payload_a: a[1], keys_b: z[0], payload_b: z[1] });
      [a, z] = [z, a];
    }
  }
  _prefixSum(pass, data, count, level = 0) {
    if (count <= 0) return;
    const sums = this.sums[level], p = pack("u", count);
    this._run(pass, "prefix_sum_of_each_block", count, p, { data, block_sums: sums });
    if (count >= PREFIX_LIMIT) this._prefixSum(pass, sums, ceilDiv(count, 256), level + 1);
    else this._run(pass, "prefix_sum_of_the_block_sums", 256, null, { block_sums: sums });
    this._run(pass, "add_block_prefix_sums_to_the_buffer", count, p, { data, block_sums: sums });
  }
  _buildCells(pass, n) {
    const pairs = MAX_CELLS * n, chunks = ceilDiv(pairs, 4);
    this.write(this.cellUniform, 0, new Uint32Array([chunks, pairs]));
    this._run(pass, "build_cell_ids_array", n, pack("fu", this.cellSize, n),
      { positions: this.positions, cell_ids: this.cellIds, object_ids: this.objectIds, radius: this.radius });
    this._radixSort(pass, pairs);
    this._run(pass, "count_objects_for_each_chunk", chunks, null,
      { chunk_obj_count: this.chunkCounts, uniform_data: this.cellUniform, cell_ids: this.cellIds });
    this._prefixSum(pass, this.chunkCounts, chunks);
    this._run(pass, "build_collision_cells_array", chunks, null,
      { chunk_obj_count: this.chunkCounts, collision_cells: this.collisionCells, indirect_args: this.indirect,
        uniform_data: this.cellUniform, cell_ids: this.cellIds });
  }
  _solve(pass, entry, extra, dt) {
    const pl = this._pl(entry);
    const byName = { chunk_obj_count: this.chunkCounts, collision_cells: this.collisionCells, cell_ids: this.cellIds,
      object_ids: this.objectIds, positions: this.positions, radius: this.radius, uniform_data: this.cellUniform, ...extra };
    for (let color = 1; color <= 4; color++) {
      const params = entry === "solve_collisions_filtered" ? pack("uf", color, this.bounceThreshold * dt) : pack("u", color);
      const pbuf = this._params(params);
      pass.setPipeline(pl.pipe);
      pass.setBindGroup(0, this._bind(entry, pl, pbuf, byName));
      pass.dispatchWorkgroupsIndirect(this.indirect, 0);
    }
  }

  /** Advances the world one tick: one command submission. */
  step(dt = 1 / 60) {
    this._expire(dt);
    const n = this.n;
    if (!n) { this.steps++; return; }
    if (this.jointsDirty) this._uploadJoints();
    const mat = this.useMaterial;
    if (mat && (this.matDirty || this.matDt !== dt)) { this._uploadMaterial(dt); this.matDirty = false; this.matDt = dt; }
    this.slot = 0;
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    const [tw, th] = this.tilesDims, haveTiles = tw && th;
    const cs = { positions: this.positions, previous_positions: this.previous, radius: this.radius };
    if (mat) this._run(pass, "integrate_material", n, pack("ffffffu", dt, this.gravity[0], this.gravity[1], this.damping, this.world[0], this.world[1], n),
      { ...cs, material: this.materialBuf });
    else this._run(pass, "integrate_tagged", n, pack("fffffffu", dt, this.gravity[0], this.gravity[1], this.damping, this.sandDamping, this.world[0], this.world[1], n),
      { ...cs, tags: this.tags });
    this._buildCells(pass, n);
    for (let it = 0; it < Math.max(1, this.iterations); it++) {
      if (mat) this._solve(pass, "solve_collisions_filtered", { filter: this.filterBuf, material: this.materialBuf, previous_positions: this.previous }, dt);
      else this._solve(pass, "solve_collisions_grouped", { groups: this.groups }, dt);
      for (const [first, count] of this.jointBatches)
        this._run(pass, "solve_joints", count, pack("uu", first, count), { positions: this.positions, radius: this.radius, joint_i: this.jointI, joint_f: this.jointF });
      if (haveTiles) this._run(pass, "tile_collide", n, pack("uuu", tw, th, n), { positions: this.positions, radius: this.radius, tiles: this.tiles });
      if (this.nCurves) this._run(pass, "curve_collide", n, pack("uu", n, this.nCurves), { ...cs, material: this.materialBuf, curves: this.curveBuf });
    }
    if (haveTiles) {
      if (mat) this._run(pass, "tile_material", n, pack("uuufff", tw, th, n, this.friction, 1 - this.bounceLoss, this.bounceThreshold * dt),
        { ...cs, tiles: this.tiles, material: this.materialBuf, omega: this.omegaBuf });
      else if (this.friction > 0 || this.bounceLoss > 0) this._run(pass, "tile_friction", n, pack("uuuff", tw, th, n, this.friction, this.bounceLoss),
        { ...cs, tiles: this.tiles });
    }
    if (this.nCurves) this._run(pass, "curve_material", n, pack("uuf", n, this.nCurves, this.bounceThreshold * dt),
      { ...cs, material: this.materialBuf, omega: this.omegaBuf, curves: this.curveBuf });
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.steps++;
  }
  _expire(dt) {
    for (const b of [...this.bodies.values()]) if (b.ttl != null && (b.ttl -= dt) <= 0) this.removeBody(b.id);
  }

  // ---------------------------------------------------------------- readback (tests, tools)
  async readBuffer(buffer, bytes, offset = 0) {
    const stage = this.device.createBuffer({ size: Math.max(4, bytes), usage: U.COPY_DST | U.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset, stage, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await stage.mapAsync(GPUMapMode.READ);
    const out = stage.getMappedRange().slice(0, bytes);
    stage.destroy();
    return out;
  }
  async readPositions() { return new Float32Array(await this.readBuffer(this.positions, 8 * this.n)); }
  async readPrevious() { return new Float32Array(await this.readBuffer(this.previous, 8 * this.n)); }
  /** Pushes every circle within `radius` of (x, y) away (a read and a write of the previous positions); returns how many. */
  async blast(x, y, radius = 3.5, force = 60, dt = 1 / 60) {
    if (!this.n) return 0;
    const pos = await this.readPositions(), prv = await this.readPrevious();
    let pushed = 0;
    for (let i = 0; i < this.n; i++) {
      let dx = pos[2 * i] - x, dy = pos[2 * i + 1] - y, d = Math.hypot(dx, dy);
      if (d >= radius) continue;
      if (d < 1e-4) { dx = 0; dy = 1; d = 1; }
      const k = force * (1 - d / radius) * dt / d;
      prv[2 * i] -= dx * k; prv[2 * i + 1] -= dy * k; pushed++;
    }
    if (pushed) this.write(this.previous, 0, prv);
    return pushed;
  }
  async bodyPosition(id) {
    const pos = await this.readPositions(), cs = this.bodies.get(id).circles;
    let x = 0, y = 0; for (const c of cs) { x += pos[2 * c]; y += pos[2 * c + 1]; }
    return { x: x / cs.length, y: y / cs.length };
  }
}
