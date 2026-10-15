// renderer.js -- draws a GpuWorld's circles and tiles straight from its GPU buffers (no readback).
//
//   const r = new Renderer(device, canvas, world);
//   r.draw();                       // after world.step(): clears and draws tiles then circles
//
// World units, y up; the whole world box is fitted into the canvas.
const WGSL = /* wgsl */`
struct View { scale: vec2f, offset: vec2f, world: vec2f, tiles: vec2u };
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> rad: array<f32>;
@group(0) @binding(3) var<storage, read> tag: array<u32>;
@group(0) @binding(4) var<storage, read> tiles: array<u32>;

struct VOut { @builtin(position) p: vec4f, @location(0) uv: vec2f, @location(1) color: vec3f };
const quad = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));

fn tagColor(t: u32) -> vec3f {
  switch (t) {
    case 1u: { return vec3f(0.86, 0.74, 0.42); }      // sand
    case 2u: { return vec3f(0.62, 0.45, 0.28); }      // crate
    case 3u: { return vec3f(0.85, 0.6, 0.35); }       // shard
    case 4u: { return vec3f(0.62, 0.5, 0.3); }        // rope
    default: { return vec3f(0.35, 0.65, 0.95); }      // ball
  }
}

@vertex fn vs_circle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let q = quad[vi];
  let c = pos[ii] + q * rad[ii];
  var o: VOut;
  o.p = vec4f(c * view.scale + view.offset, 0.0, 1.0);
  o.uv = q;
  o.color = tagColor(tag[ii]);
  return o;
}
@fragment fn fs_circle(in: VOut) -> @location(0) vec4f {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let shade = 0.78 + 0.22 * (1.0 - d) - 0.12 * in.uv.y;
  return vec4f(in.color * shade, 1.0);
}

@vertex fn vs_tile(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let tx = ii % view.tiles.x;
  let ty = ii / view.tiles.x;
  let q = (quad[vi] + vec2f(1.0)) * 0.5;
  let c = vec2f(f32(tx), f32(ty)) + q;
  o.p = vec4f(c * view.scale + view.offset, 0.0, 1.0);
  if (tiles[ii] == 0u) { o.p = vec4f(2.0, 2.0, 0.0, 1.0); }
  o.uv = q;
  o.color = vec3f(0.30, 0.34, 0.40);
  return o;
}
@fragment fn fs_tile(in: VOut) -> @location(0) vec4f {
  let e = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
  return vec4f(in.color * (0.85 + 0.15 * smoothstep(0.0, 0.08, e)), 1.0);
}
`;

export class Renderer {
  constructor(device, canvas, world, { clear = [0.07, 0.08, 0.1, 1] } = {}) {
    this.device = device; this.canvas = canvas; this.world = world; this.clear = clear;
    this.ctx = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    const module = device.createShaderModule({ code: WGSL });
    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ...[1, 2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } })) ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const mk = (vs, fs) => device.createRenderPipeline({ layout, vertex: { module, entryPoint: vs },
      fragment: { module, entryPoint: fs, targets: [{ format: this.format }] }, primitive: { topology: "triangle-list" } });
    this.circles = mk("vs_circle", "fs_circle"); this.tilesPipe = mk("vs_tile", "fs_tile");
    this.bgl = bgl;
    this.view = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.group = null; this.groupFor = null;
  }
  /** Converts a canvas pixel to world units. */
  toWorld(px, py) {
    const [W, H] = this.world.world, r = this.canvas.getBoundingClientRect();
    const s = this.scale, fx = ((px - r.left) / r.width) * this.canvas.width, fy = ((py - r.top) / r.height) * this.canvas.height;
    return [(fx - this.ox) / s, H - (fy - this.oy) / s];
  }
  draw(target = null) {
    const w = this.world, [W, H] = w.world, cw = target ? target.width : this.canvas.width, ch = target ? target.height : this.canvas.height;
    this.scale = Math.min(cw / W, ch / H);
    this.ox = (cw - W * this.scale) / 2; this.oy = (ch - H * this.scale) / 2;
    const sx = (2 * this.scale) / cw, sy = (2 * this.scale) / ch;
    const data = new ArrayBuffer(48), f = new Float32Array(data), u = new Uint32Array(data);
    f.set([sx, sy, -1 + (2 * this.ox) / cw, -1 + (2 * this.oy) / ch, W, H]); u.set([w.tilesDims[0] || 1, w.tilesDims[1] || 1], 6);
    this.device.queue.writeBuffer(this.view, 0, data);
    if (this.groupFor !== w.tiles) {
      this.group = this.device.createBindGroup({ layout: this.bgl, entries: [
        { binding: 0, resource: { buffer: this.view } }, { binding: 1, resource: { buffer: w.positions } },
        { binding: 2, resource: { buffer: w.radius } }, { binding: 3, resource: { buffer: w.tags } },
        { binding: 4, resource: { buffer: w.tiles } }] });
      this.groupFor = w.tiles;
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: (target ?? this.ctx.getCurrentTexture()).createView(),
      clearValue: { r: this.clear[0], g: this.clear[1], b: this.clear[2], a: this.clear[3] }, loadOp: "clear", storeOp: "store" }] });
    pass.setBindGroup(0, this.group);
    if (w.tilesDims[0]) { pass.setPipeline(this.tilesPipe); pass.draw(6, w.tilesDims[0] * w.tilesDims[1]); }
    if (w.n) { pass.setPipeline(this.circles); pass.draw(6, w.n); }
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  /** Renders one frame to an offscreen texture and returns it as ImageData (tests, screenshots, thumbnails). */
  async snapshot(width = this.canvas.width, height = this.canvas.height) {
    const tex = this.device.createTexture({ size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.draw(tex);
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const out = this.device.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: out, bytesPerRow: bpr }, [width, height]);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await out.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(out.getMappedRange()), img = new ImageData(width, height), bgra = this.format.startsWith("bgra");
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = y * bpr + x * 4, o = (y * width + x) * 4;
      img.data[o] = src[i + (bgra ? 2 : 0)]; img.data[o + 1] = src[i + 1]; img.data[o + 2] = src[i + (bgra ? 0 : 2)]; img.data[o + 3] = 255;
    }
    out.unmap(); tex.destroy(); out.destroy();
    return img;
  }
}
