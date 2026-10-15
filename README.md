# GPU Physics Engine 🚀

A high-performance 2D physics engine built with Rust and wgpu, leveraging GPU compute shaders for massive parallel processing of physics simulations.

[![Feature Demo Video Thumbnail](images/image.png)](https://www.youtube.com/watch?v=TxYFeai6B9Q)

[Demo on YouTube](https://www.youtube.com/watch?v=TxYFeai6B9Q)


## ✨ Features

- **GPU-Accelerated**: All physics computations run on the GPU for maximum performance
- **Spatial Grid Partitioning**: Efficient broad-phase collision detection using GPU-based spatial grids. **Learn more**: [NVIDIA GPU Gems - Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
- **Verlet Integration**: Stable numerical integration for smooth particle motion
- **Real-time Interaction**: Interactive particle spawning and mouse-based attraction forces
- **Scalable**: Handle millions of particles with high framerates
- **Crust / SPIR-V port**: every compute shader also builds from Rust-subset kernels to SPIR-V without rustc ([details](#-crust--spir-v-kernels-no-rustc-no-wgsl))

## 🎮 Controls

| Input | Action |
|-------|--------|
| `W` or `↑` | Move camera up |
| `A` or `←` | Move camera left |
| `S` or `↓` | Move camera down |
| `D` or `→` | Move camera right |
| `P` | Spawn 100 particles at mouse position |
| `Left Click` | Attract particles to mouse |
| `Mouse Wheel` | Zoom in/out |

## 🚀 Quick Start
### Running the Engine
```bash
cargo run --release
```
### Tests
```
cargo test
```

### Benchmark
The benchmark shows the performance for each of the compute shaders at the end of the execution. It creates `benchmark.json` file that can be visualized at `edge://tracing/` or `chrome://tracing/`. 
```
cargo run --release --features benchmark
```


## 🔧 Implementation Details

### GPU Spatial Grid Partitioning
The engine uses a sophisticated spatial partitioning system that divides the simulation space into a uniform grid. Each particle is assigned to grid cells, dramatically reducing the number of collision checks from O(n²) to approximately O(n*k) in most cases.

**Learn more**: [NVIDIA GPU Gems - Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

### GPU Collision Response
All collision detection and response calculations are performed in parallel on the GPU using compute shaders, allowing for real-time simulation of millions of interacting particles.

### Verlet Integration
The engine employs Verlet integration for numerical stability and energy conservation, ensuring smooth and realistic particle motion over time.

## Morton encoding
Every 4 seconds, the particles are sorted using morton codes to improve cache locality. 

### Current Limitations
- **2D Only**: Currently supports 2D simulations
- **Circle Shapes**: Only circular particles are supported at this time

## 🦀 Crust / SPIR-V kernels (no rustc, no WGSL)

Every compute shader also exists as a Rust-subset kernel in [`kernels/`](kernels), compiled straight to
SPIR-V by [Crust](https://github.com/brentharts/crust) without rustc or cargo. The WGSL files stay the
reference; the ports are checked against them.

```bash
python3 build.py --crust ../crust --status        # every WGSL shader and its port status
python3 build.py --crust ../crust                 # compile kernels/*.rs to build/spirv/*.spv
python3 build.py --crust ../crust --check         # ...and run each one against a Python model of the WGSL
python3 build.py --crust ../crust --push-buffer --out DIR   # scalar parameters in a storage buffer (binding 30)
```

`--check` runs the compiled SPIR-V in Crust's reference interpreter (no GPU needed) and compares every
output byte with a plain-Python model of the WGSL, including f32 rounding. `--push-buffer` builds need no
push-constant device feature, which is what wgpu / WebGPU hosts use. Crust (`$CRUST_DIR` or `--crust`) is
the only requirement.

| Kernel file | Entry points | Origin |
|---|---|---|
| `home_cell_ids.rs`, `grid.rs` | `create_home_cell_ids`, `build_cell_ids_array` | ports of the grid shaders |
| `particle_integration.rs`, `rearrange.rs` | `verlet_integration`, `rearrange` | ports |
| `radix_sort.rs`, `prefix_sum.rs` | `build_histogram`, `scatter_keys`, `prefix_sum_of_each_block`, ... | ports (workgroup memory, barriers, atomics) |
| `collision_cell_builder.rs`, `collision_solver.rs` | `count_objects_for_each_chunk`, `build_collision_cells_array`, `solve_collisions` | ports |
| `prowl_world.rs` | `integrate_world`, `tile_collide` | added for [Prowl2D](https://github.com/crustos/Prowl2D): gravity and world box; circles against a grid of solid tiles |
| `prowl_bodies.rs` | `solve_joints`, `solve_collisions_grouped`, `integrate_tagged`, `tile_friction`, `raycast`, `sensor_count` | added for Prowl2D: see below |
| `prowl_material.rs` | `integrate_material`, `solve_collisions_filtered`, `tile_material` | Box2D-style per-circle settings: friction, bounciness, gravity scale, damping, inverse mass (0 = static/kinematic), and category/mask/group collision filtering; see below |

### Beyond particles: bodies, joints, queries

`prowl_bodies.rs` adds what a game needs on top of the circle solver, all still built from circles:

- **Joints**: distance constraints between two circles or a circle and a fixed point (rigid, rope, strut,
  spring), solved in colour batches so no circle appears twice in a dispatch. Ropes are chains of them.
- **Groups**: circles of one body (the corner circles of a crate) ignore each other in
  `solve_collisions_grouped`, so a **crate** is a grid of circles held by stiff joints.
- **Tile friction**: slows circles touching a solid tile and absorbs part of the bounce (sand piles, crates stop sliding).
- **Ray casts**: a batch of rays against the tile grid (exact grid walk) and all circles.
- **Sensors**: counts of circles inside boxes or circles, filtered by a per-circle tag.

Prowl2D drives all of this from Python through wgpu-py: see its `GPU.md`.

### Material, mass and filtering (Box2D's fixture settings)

`prowl_material.rs` reads three per-circle arrays: `material` (friction, bounciness, gravity scale, damping), `filter` (category bits,
mask bits, group index, as in Box2D) and `inv_mass` (0 makes a circle immovable). Pairs combine as in Box2D: friction is the geometric mean,
bounciness the larger. A shared positive group always collides, a shared negative group never does; otherwise each mask must contain the
other's category. Bounce and friction act on the Verlet velocity (`position - previous`), so they are approximate on stacks but exact for a
single contact (`selftest.py` has the known answers: equal circles swap velocities at bounciness 1, stay together at 0). Bounce against a
tile leaves at `bounciness` times the approach speed (slower than a threshold, 1 unit/s by default, just stops).

In the C runtime: `gp_body_set_material(w, body, friction, bounciness, density, gravity_scale, drag)`, `gp_body_set_filter(w, body, category, mask, group)`
and `gp_set_bounce_threshold`. The first call switches the step to these kernels (a world that never calls them steps exactly as before);
`python3 runtime/check.py` builds the runtime with gcc and runs `runtime/test_gpuphys.c` (bounce heights, gravity scale, drag, filters). A host
that links the translated kernel files itself must add `gpk_prowl_material.c` (`runtime/gpk.py` writes it); without it the calls return 0.
The Prowl2D Python driver does not use them yet.

### Contact events (C runtime)

`gp_enable_events(w, 1)` makes every `gp_step` queue `GP_EVENT_BEGIN` / `END` (body against body, or against the tiles: `b = -1` and a tile index), with
the contact point, the normal from `a` toward `b` and the closing speed, and `GP_EVENT_TRIGGER_ENTER` / `EXIT` for bodies made sensors with
`gp_body_set_sensor` (a sensor collides with nothing; the category/mask filter says what it sees). `gp_poll_events` drains the queue. A contact
begins within 0.01 tile units of touching and ends beyond 0.05, so a resting contact does not flicker. No kernel is involved: it is computed on the
CPU from the step's collision cells. These are what a Unity-style `OnCollisionEnter2D` / `OnTriggerEnter2D` needs.

### Joints between bodies (C runtime)

`gp_add_body_joint` makes Box2D's **revolute** (angle limits, motor), **weld**, **prismatic** (translation limits, motor), **wheel** (a slider with a
spring and damping, plus a spin motor), **distance** (rod, rope with `enable_limit`, or a spring with `frequency`) and **mouse** joints
(`gp_joint_def` fills the defaults; `gp_joint_set_target / set_motor / set_limits`, `gp_joint_angle / translation` read and drive them). They are
solved on the CPU in position form, a few passes a step (`gp_set_joint_iterations`), on top of the circle solver: a **box** is one rigid body
(mass and inertia from its circles), a ball or one circle of a rope is a node with its own spin
(see below), so a revolute anchor may sit off a ball's centre; angle limits and motors need a box or polygon. Joined bodies do not collide with each other (a chain of joints is one collision group) unless
`collide_connected`. `runtime/test_gpuphys.c` checks each type against its physics: a hinge keeps its pin within 0.002, a limit holds a plank up, a
motor turns at its speed, a slider stops at its limit, a 1 Hz suspension sags to 0.23 (theory 0.25), a rope stops a fall at its length.

### Rotation and polygons (C runtime)

A **ball has a spin**: `gp_body_set_angle`, `gp_body_[add_|set_]angular_velocity` drive it, and on a tile it rolls toward the no-slip spin
instead of just sliding. **`gp_add_polygon`** makes a convex polygon: a fill of circles kept rigid by shape matching, with the rigid
velocity projected back so it keeps spinning and tumbling (`gp_body_origin` gives back the point it was made at). Tile friction acts on the
slip speed `v.t - r*omega` through a per-circle `omega` binding of `tile_material`, so a rolling ball keeps rolling; this needs the material
path (any `gp_body_set_material / filter / sensor` call), the legacy path still slows rolling balls. Polygons are rounded by their circle
radius, and a coarse fill (a hexagon of 8 circles) rests within about 0.05 rad of a flat edge.

### Smooth surfaces: quadratic Bezier colliders (`kernels/prowl_curves.rs`)

Static curves that circles roll on, with no stair-stepping of a tile grid. A curve is a quadratic Bezier `p0 -> p1 (control) -> p2`
(a straight segment is one with its control point halfway; longer outlines are several curves end to end).

    gp_add_curve(w, x0,y0, x1,y1, x2,y2, friction, bounciness, side)     // C;  world.addCurve(...) in wasm and WebGPU

* `side = 0`: a **thin wall**, solid on both faces (a ramp, a bowl, a bridge). A circle stays on the face it was on in the previous step.
* `side = +1 / -1`: one edge of a **solid contour** whose inside is on the left / right of the direction `p0 -> p2` (a hill, a cave floor):
  circles are pushed out of it, and a contour's curves share their end points.
* `curve_collide` runs in the solver iterations after the tiles (nearest point on the curve by 9 samples and Newton steps, then a push to one radius);
  `curve_material` runs after them and applies the contact's friction (geometric mean with the circle's), bounciness (the larger) and the slip, like `tile_material`.
* A circle that moves more than its radius in one step can still tunnel, and a circle whose centre is deeper inside a solid contour than its radius is not pulled out.
  Sensor bodies are blocked by curves. Adding a curve switches the world to the material kernels. Curves are static; Bezier-outlined moving bodies are not implemented.
* Available in the C runtime (`gpuphys.c`), the wasm build and the WebGPU host; the Python host has no curve calls yet.

#### Drawing the curves: `web/windfoil.js`

A closed-form antialiased fill for quadratic outlines after [Windfoil](https://arxiv.org/abs/2610.02468) (DesLauriers): a pixel's coverage is the
box-filtered winding number over its footprint, summed over curve pieces that are monotone in x and y, with the part of each piece inside the footprint's
columns integrated exactly (a quartic in t). No sampling, so no noise. `Windfoil.create(device, format).setPath(path).draw(pass, {camera, scale, res, color, rule})`
draws a path (a list of closed contours of `[x0,y0,x1,y1,x2,y2]` segments; rule `nonzero` or `evenodd`); pieces are binned in row bands, so a pixel only visits the
pieces near its row. `node web/windfoil_test.mjs` checks the formula against supersampling on the CPU, and `web/check.py` renders shapes in headless Chromium and compares the pixels.
Limits: the footprint is the axis-aligned pixel (no rotated or anisotropic filtering), and `pad` of `setPath` must be at least one pixel at the smallest zoom drawn.

### In the browser: WebAssembly and WebGPU

Two ways to run the engine in a page; the first has every feature, the second has the speed.

**WebAssembly (`runtime/wasm`).** `python3 runtime/wasm/build.py --crust ../crust` compiles `gpuphys.c` and Crust's C translation of the kernels to
`gpuphys.wasm` with plain clang (`--target=wasm32`, wasm-ld, no emscripten, no wasi-sdk; `libc.c` is the 150 lines of libc it needs, and `sinf`/`cosf`
are imported from JavaScript's `Math`). `gpuphys.js` wraps the whole C header: bodies, polygons, material and filters, the joints, contact events, ray
casts, sand and blasts. It runs on the CPU, so it suits a few hundred bodies and a few thousand grains. `python3 runtime/wasm/check.py --crust ../crust`
builds it, builds the same scenario natively and runs `tests/test.mjs` in Node: both builds agree to three decimals (a tumbling hexagon differs by 0.02 rad, from
JavaScript's `sin`/`cos`).

**WebGPU (`web/`).** WebGPU takes WGSL, so `python3 build.py --wgsl` (which implies `--push-buffer`) also writes `NAME.wgsl` for each kernel file, converted from
the SPIR-V by [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) (`cargo install naga-cli --locked`; `tools/spv2wgsl.py`) and validated again after the
two changes WebGPU needs: the scalar parameters become a `var<uniform>`, and `solve_collisions_filtered` fits the 10 storage buffers most adapters allow
(its inverse masses live in the material array). `web/gpuphys_webgpu.js` is the engine's pipeline in JavaScript: integrate, grid, radix sort, collision
cells, the coloured solver, joints, tiles, and the material path, in one command submission per step; `web/renderer.js` draws the circles and tiles from those
buffers with no readback; `web/demo.html` is a small playground. Bodies are the circle clusters of the other hosts (balls, crates, ropes, sand); joints
between bodies, contact events and polygons are CPU-side and stay in the WebAssembly build. `python3 web/check.py --crust ../crust` runs `web/test.html` in
headless Chromium (WebGPU on SwiftShader, no GPU needed): a ball and a crate come to rest, sand settles, a rope hangs, rubber bounces, masks and sensors
pass through, a body can be removed, 4000 grains step. Needs Playwright for Node.

Two things the browser run taught: a short-circuit `||` around a buffer load in `tile_collide` made SwiftShader compile it for 74 seconds (now 15, no driver
has complained), and Chromium's headless SwiftShader loses the WebGPU device as soon as a *canvas* is presented, so the tests read back offscreen textures
(`Renderer.snapshot()`); a real browser is not affected.

### Current limits of the kernels

- Circles only; boxes, polygons, ropes and joints are circles plus constraints, rounded at the corners.
- Positions are interleaved `f32` pairs, y up; helper functions in the kernel language cannot take buffers.
- The wgpu path of the Python host has been verified only on the reference interpreter so far; the WebGPU host runs on SwiftShader (software Vulkan) in the tests, not yet on a hardware GPU.

## 📊 Performance

**Test Configuration:**
- **GPU**: AMD Radeon RX 6800 XT (with severe thermal throttling)
- **OS**: Windows 11
- **Build**: Release mode (`--release`)
- **Gravity**: disabled

The engine demonstrates excellent scaling characteristics, maintaining smooth framerates even with millions of active particles thanks to GPU parallelization.
![Performance Graph](./images/performance_graph.png)

| Particles | Frame Time |
|-----------|------------|
| 100,000   | 1.03 ms    |
| 256,000   | 1.28 ms    |
| 512,000   | 1.88 ms    |
| 756,000   | 2.53 ms    |
| 1,000,000 | 3.66 ms    |
| 2,000,000 | 8.417 ms   |
| 3,000,000 | 15.62 ms   |
| 4,000,000 | 22.70 ms   |

## 🔮 Future Enhancements

- Compute shaders optimization
- Rendering optimization
- Improve physics


---

*Built with ❤️ using Rust and wgpu*

## Crust `unity_pack --gpu-physics`

`unity_glue.py` plus `runtime/` (`gpuphys.c/.h`, `gpk.py`) let Crust's `unity_pack` run a packed
Unity project's Rigidbody2D / Collider2D physics on this engine's circle solver instead of Box2D:

```
python3 tools/unity_pack.py MyGame --gpu-physics path/to/gpu-physics-engine -o out
```

`gpk.py` translates the kernels to C with Crust; `gpuphys.c` runs them once per invocation on the CPU
(the sort and prefix sum are plain C) behind a world-units API (`gpuphys.h`); `unity_glue.py` generates
`physics_gpu.c`, which maps the Unity tables onto it. Supported and refused scenes are listed in
Crust's `UNITY_PACK.md` ("GPU physics").
