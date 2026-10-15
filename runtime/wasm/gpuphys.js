// gpuphys.js -- the gpu-physics-engine C runtime (gpuphys.c) compiled to WebAssembly, with a JavaScript API.
//
//   import { loadGpuPhys } from "./gpuphys.js";
//   const gp = await loadGpuPhys("gpuphys.wasm");        // a URL (browser, Node 18+) or the bytes
//   const w = gp.createWorld({ capacity: 4096, width: 64, height: 36, gravity: [0, -18] });
//   const ball = w.addBall(10, 20, 0.35);
//   w.setTiles(64, 36, floorTiles);
//   w.step(1 / 60);
//   const { x, y, angle } = w.bodyPose(ball);
//
// Everything the C header offers is here (joints, events, polygons, material, filters, ray casts); names are
// the header's without the `gp_` prefix, in camelCase. The simulation runs on the CPU, in WebAssembly: it is the
// engine for browsers without WebGPU, for tests, and for the features that are CPU-side (joints, contact events).
// Positions are world units, y up, as in Prowl2D.

export const Tag = { BALL: 0, SAND: 1, CRATE: 2, SHARD: 3, ROPE: 4 };
export const JointKind = { RIGID: 0, ROPE: 1, STRUT: 2 };               // addJoint (circle level)
export const Joint = { DISTANCE: 1, REVOLUTE: 2, WELD: 3, PRISMATIC: 4, WHEEL: 5, MOUSE: 6 };
export const Event = { BEGIN: 1, END: 2, TRIGGER_ENTER: 3, TRIGGER_EXIT: 4 };
export const Ray = { TILES: 1, CIRCLES: 2 };

const JOINT_DEF_SIZE = 80, EVENT_SIZE = 36, HIT_SIZE = 32;

export async function loadGpuPhys(source) {
  let bytes = source;
  if (typeof source === "string" || source instanceof URL) {
    if (typeof process !== "undefined" && process.versions?.node && !/^https?:/.test(String(source))) {
      const fs = await import("node:fs");
      bytes = fs.readFileSync(source instanceof URL ? source : source);
    } else {
      bytes = await (await fetch(source)).arrayBuffer();
    }
  }
  const f = Math.fround;
  const env = {
    sinf: (x) => f(Math.sin(x)), cosf: (x) => f(Math.cos(x)), atan2f: (y, x) => f(Math.atan2(y, x)),
    powf: (x, y) => f(Math.pow(x, y)), atanf: (x) => f(Math.atan(x)), acosf: (x) => f(Math.acos(x)),
    asinf: (x) => f(Math.asin(x)), tanf: (x) => f(Math.tan(x)), expf: (x) => f(Math.exp(x)),
    fminf: (a, b) => (a < b ? a : b), fmaxf: (a, b) => (a > b ? a : b),
  };
  const { instance } = await WebAssembly.instantiate(bytes, { env });
  return new GpuPhys(instance.exports);
}

class GpuPhys {
  constructor(x) {
    this.x = x;
    this.scratch = x.malloc(8192);
  }
  get u8() { return new Uint8Array(this.x.memory.buffer); }
  get dv() { return new DataView(this.x.memory.buffer); }
  alloc(n) { return this.x.malloc(n); }
  free(p) { this.x.free(p); }
  /** Runs fn(pointer) with `bytes` of scratch memory the call may fill; returns what fn returns. */
  scratchCall(bytes, fn) {
    const p = bytes <= 8192 ? this.scratch : this.x.malloc(bytes);
    try { return fn(p); } finally { if (p !== this.scratch) this.x.free(p); }
  }
  createWorld(o = {}) {
    const width = o.width ?? 64, height = o.height ?? 36;
    const capacity = o.capacity ?? 4096;
    const gravity = o.gravity ?? [0, -18];
    const ptr = this.x.gp_create(capacity, o.maxJoints ?? capacity * 4, o.originX ?? 0, o.originY ?? 0, width, height,
      o.cell ?? 1, gravity[0], gravity[1]);
    if (!ptr) throw new Error("gp_create failed (out of memory?)");
    return new World(this, ptr);
  }
}

class World {
  constructor(gp, ptr) { this.gp = gp; this.x = gp.x; this.ptr = ptr; }
  destroy() { if (this.ptr) { this.x.gp_destroy(this.ptr); this.ptr = 0; } }
  step(dt = 1 / 60) { this.x.gp_step(this.ptr, dt); }
  setTiles(width, height, cells) {
    const n = width * height, p = this.gp.alloc(4 * n);
    new Uint32Array(this.x.memory.buffer, p, n).set(cells);
    this.x.gp_set_tiles(this.ptr, width, height, p);
    this.gp.free(p);                                    // the engine copies them
  }
  setMaterial(friction, bounciness) { this.x.gp_set_material(this.ptr, friction, bounciness); }
  setJointIterations(n) { this.x.gp_set_joint_iterations(this.ptr, n); }
  setBounceThreshold(speed) { this.x.gp_set_bounce_threshold(this.ptr, speed); }

  // ---- bodies
  addBall(x, y, radius = 0.35, vx = 0, vy = 0, dt = 1 / 60) { return this.x.gp_add_ball(this.ptr, x, y, radius, vx, vy, dt); }
  addBox(x, y, w = 1, h = 1, angle = 0, vx = 0, vy = 0, dt = 1 / 60, stiffness = 1) {
    return this.x.gp_add_box(this.ptr, x, y, w, h, angle, vx, vy, dt, stiffness);
  }
  addRope(ax, ay, bx, by, segments = 12, radius = 0.08, pinA = true, pinB = false) {
    return this.x.gp_add_rope(this.ptr, ax, ay, bx, by, segments, radius, pinA ? 1 : 0, pinB ? 1 : 0);
  }
  /** verts: [x0, y0, x1, y1, ...] relative to (x, y), convex. */
  addPolygon(x, y, verts, { angle = 0, radius = 0, vx = 0, vy = 0, dt = 1 / 60, stiffness = 1 } = {}) {
    const n = verts.length / 2, p = this.gp.alloc(4 * verts.length);
    new Float32Array(this.x.memory.buffer, p, verts.length).set(verts);
    const id = this.x.gp_add_polygon(this.ptr, x, y, angle, p, n, radius, vx, vy, dt, stiffness);
    this.gp.free(p);
    return id;
  }
  /** A static quadratic Bezier collider p0 -> p1 (control) -> p2. side 0: a thin wall; +1 / -1: one edge of a solid contour
   *  whose inside is on the left / right of the direction p0 -> p2. friction is Unity's, bounciness 0..1.
   *  Returns the curve's index (-1 if the build lacks the curve kernels). */
  addCurve(x0, y0, x1, y1, x2, y2, friction = 0.4, bounciness = 0, side = 0) {
    return this.x.gp_add_curve(this.ptr, x0, y0, x1, y1, x2, y2, friction, bounciness, side);
  }
  clearCurves() { this.x.gp_clear_curves(this.ptr); }
  get curveCount() { return this.x.gp_curve_count(this.ptr); }
  /** A free circle (no body), tag sand: returns its circle index. */
  addSand(x, y, radius = 0.08, vx = 0, vy = 0, dt = 1 / 60) { return this.x.gp_add_sand(this.ptr, x, y, radius, vx, vy, dt); }
  /** Pours `count` grains in a disc of radius `brush` around (x, y); returns how many fitted. */
  brushSand(x, y, brush = 0.5, count = 8, radius = 0.08, rnd = Math.random) {
    let made = 0;
    for (let i = 0; i < count; i++) {
      const a = rnd() * Math.PI * 2, d = brush * Math.sqrt(rnd());
      if (this.addSand(x + Math.cos(a) * d, y + Math.sin(a) * d, radius) >= 0) made++;
    }
    return made;
  }
  /** Pushes every circle within `radius` of (x, y) away: `force` is the speed given at the centre (units/s). */
  blast(x, y, radius = 3.5, force = 60) { return this.x.gp_blast(this.ptr, x, y, radius, force); }
  addCircleJoint(a, b, kind, rest, stiffness = 1, fx = 0, fy = 0) { return this.x.gp_add_joint(this.ptr, a, b, kind, rest, stiffness, fx, fy); }
  /** The circle index of the k-th circle of a body (a rope's points, a crate's grid). */
  bodyCircle(b, k) { return this.x.gp_body_circle(this.ptr, b, k); }
  get bodyCount() { return this.x.gp_body_count(this.ptr); }
  bodyCircles(b) { return this.x.gp_body_circles(this.ptr, b); }
  bodyPose(b) {
    return this.gp.scratchCall(16, (p) => {
      this.x.gp_body_pose(this.ptr, b, p, p + 4, p + 8);
      const dv = this.gp.dv;
      return { x: dv.getFloat32(p, true), y: dv.getFloat32(p + 4, true), angle: dv.getFloat32(p + 8, true) };
    });
  }
  bodyOrigin(b) {
    return this.gp.scratchCall(8, (p) => {
      this.x.gp_body_origin(this.ptr, b, p, p + 4);
      const dv = this.gp.dv;
      return { x: dv.getFloat32(p, true), y: dv.getFloat32(p + 4, true) };
    });
  }
  bodyVelocity(b, dt = 1 / 60) {
    return this.gp.scratchCall(8, (p) => {
      this.x.gp_body_velocity(this.ptr, b, dt, p, p + 4);
      const dv = this.gp.dv;
      return { x: dv.getFloat32(p, true), y: dv.getFloat32(p + 4, true) };
    });
  }
  setBodyVelocity(b, vx, vy, dt = 1 / 60) { this.x.gp_body_set_velocity(this.ptr, b, dt, vx, vy); }
  addBodyVelocity(b, vx, vy, dt = 1 / 60) { this.x.gp_body_add_velocity(this.ptr, b, dt, vx, vy); }
  setBodyPosition(b, x, y) { this.x.gp_body_set_position(this.ptr, b, x, y); }
  setBodyActive(b, on) { this.x.gp_body_set_active(this.ptr, b, on ? 1 : 0); }
  setBodyKinematic(b, on) { this.x.gp_body_set_kinematic(this.ptr, b, on ? 1 : 0); }
  moveBody(b, x, y) { this.x.gp_body_move(this.ptr, b, x, y); }
  lockRotation(b, on) { this.x.gp_body_lock_rotation(this.ptr, b, on ? 1 : 0); }
  angularVelocity(b, dt = 1 / 60) { return this.x.gp_body_angular_velocity(this.ptr, b, dt); }
  setAngularVelocity(b, omega, dt = 1 / 60) { this.x.gp_body_set_angular_velocity(this.ptr, b, dt, omega); }
  addAngularVelocity(b, domega, dt = 1 / 60) { this.x.gp_body_add_angular_velocity(this.ptr, b, dt, domega); }
  setBodyAngle(b, a) { this.x.gp_body_set_angle(this.ptr, b, a); }

  // ---- Box2D's per-body settings
  setBodyMaterial(b, { friction = 0.4, bounciness = 0, density = 1, gravityScale = 1, drag = 0 } = {}) {
    return !!this.x.gp_body_set_material(this.ptr, b, friction, bounciness, density, gravityScale, drag);
  }
  setBodyFilter(b, { category = 1, mask = 0xffffffff, group = 0 } = {}) {
    return !!this.x.gp_body_set_filter(this.ptr, b, category >>> 0, mask >>> 0, group | 0);
  }
  setBodySensor(b, on = true) { return !!this.x.gp_body_set_sensor(this.ptr, b, on ? 1 : 0); }

  // ---- joints between bodies
  /** def: { type, a, b (-1 = the world), x, y (anchor) | ax, ay, bx, by, axisX, axisY, enableLimit, lower, upper,
   *         enableMotor, motorSpeed, maxMotor, frequency, dampingRatio, length, maxForce, collideConnected } */
  addJoint(def) {
    return this.gp.scratchCall(JOINT_DEF_SIZE, (p) => {
      this.x.gp_joint_def(p, def.type, def.a, def.b ?? -1, def.x ?? def.ax ?? 0, def.y ?? def.ay ?? 0);
      const dv = this.gp.dv;
      const setF = (off, v) => { if (v !== undefined) dv.setFloat32(p + off, v, true); };
      const setI = (off, v) => { if (v !== undefined) dv.setInt32(p + off, v | 0, true); };
      setF(12, def.ax); setF(16, def.ay); setF(20, def.bx); setF(24, def.by);
      setF(28, def.axisX); setF(32, def.axisY);
      setI(36, def.enableLimit === undefined ? undefined : +def.enableLimit); setF(40, def.lower); setF(44, def.upper);
      setI(48, def.enableMotor === undefined ? undefined : +def.enableMotor); setF(52, def.motorSpeed); setF(56, def.maxMotor);
      setF(60, def.frequency); setF(64, def.dampingRatio); setF(68, def.length); setF(72, def.maxForce);
      setI(76, def.collideConnected === undefined ? undefined : +def.collideConnected);
      return this.x.gp_add_body_joint(this.ptr, p);
    });
  }
  setJointTarget(j, x, y) { this.x.gp_joint_set_target(this.ptr, j, x, y); }
  setJointMotor(j, enable, speed, maxMotor) { this.x.gp_joint_set_motor(this.ptr, j, enable ? 1 : 0, speed, maxMotor); }
  setJointLimits(j, enable, lower, upper) { this.x.gp_joint_set_limits(this.ptr, j, enable ? 1 : 0, lower, upper); }
  destroyJoint(j) { this.x.gp_joint_destroy(this.ptr, j); }
  jointAngle(j) { return this.x.gp_joint_angle(this.ptr, j); }
  jointTranslation(j) { return this.x.gp_joint_translation(this.ptr, j); }

  // ---- contact events
  enableEvents(on = true) { this.x.gp_enable_events(this.ptr, on ? 1 : 0); }
  /** Queued events, oldest first: { kind, a, b, tile, x, y, nx, ny, speed }. */
  pollEvents(max = 64) {
    return this.gp.scratchCall(EVENT_SIZE * max, (p) => {
      const n = this.x.gp_poll_events(this.ptr, p, max), dv = this.gp.dv, out = [];
      for (let i = 0; i < n; i++) {
        const o = p + i * EVENT_SIZE;
        out.push({ kind: dv.getInt32(o, true), a: dv.getInt32(o + 4, true), b: dv.getInt32(o + 8, true), tile: dv.getInt32(o + 12, true),
          x: dv.getFloat32(o + 16, true), y: dv.getFloat32(o + 20, true), nx: dv.getFloat32(o + 24, true),
          ny: dv.getFloat32(o + 28, true), speed: dv.getFloat32(o + 32, true) });
      }
      return out;
    });
  }

  // ---- circles and queries
  get circleCount() { return this.x.gp_circle_count(this.ptr); }
  circle(i) { return { x: this.x.gp_circle_x(this.ptr, i), y: this.x.gp_circle_y(this.ptr, i), r: this.x.gp_circle_radius(this.ptr, i), body: this.x.gp_circle_body(this.ptr, i) }; }
  /** [x, y, radius] for every circle, in one Float32Array: what a renderer draws. */
  readCircles(into) {
    const n = this.circleCount;
    const out = into && into.length >= 3 * n ? into : new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      out[3 * i] = this.x.gp_circle_x(this.ptr, i); out[3 * i + 1] = this.x.gp_circle_y(this.ptr, i);
      out[3 * i + 2] = this.x.gp_circle_radius(this.ptr, i);
    }
    return out;
  }
  raycast(ox, oy, dx, dy, distance = 1e9, { mask = Ray.TILES | Ray.CIRCLES, ignoreBody = -1 } = {}) {
    return this.gp.scratchCall(HIT_SIZE, (p) => {
      if (!this.x.gp_raycast(this.ptr, ox, oy, dx, dy, distance, mask, ignoreBody, p)) return null;
      const dv = this.gp.dv;
      return { x: dv.getFloat32(p, true), y: dv.getFloat32(p + 4, true), nx: dv.getFloat32(p + 8, true), ny: dv.getFloat32(p + 12, true),
        distance: dv.getFloat32(p + 16, true), kind: dv.getInt32(p + 20, true), body: dv.getInt32(p + 24, true), tile: dv.getInt32(p + 28, true) };
    });
  }
  overlapCircle(x, y, radius) {
    return this.gp.scratchCall(4, (p) => (this.x.gp_overlap_circle(this.ptr, x, y, radius, p) ? this.gp.dv.getInt32(p, true) : null));
  }
  tileAt(x, y) {
    return this.gp.scratchCall(4, (p) => (this.x.gp_tile_at(this.ptr, x, y, p) ? this.gp.dv.getInt32(p, true) : null));
  }
  bodyPairs(max = 256) {
    return this.gp.scratchCall(8 * max, (p) => {
      const n = this.x.gp_body_pairs(this.ptr, p, p + 4 * max, max), dv = this.gp.dv, out = [];
      for (let i = 0; i < n; i++) out.push([dv.getInt32(p + 4 * i, true), dv.getInt32(p + 4 * max + 4 * i, true)]);
      return out;
    });
  }
}
