// Kernels the Prowl2D integration adds to the engine (no WGSL original):
//
//   integrate_world   Verlet step with gravity, damping and the world box.
//                     The WGSL integrator has no gravity; a platformer needs it.
//   tile_collide      Pushes circles out of a grid of solid tiles (the level's
//                     ground and walls), circle against axis-aligned square.
//
// Positions are interleaved f32 pairs (x, y), world units, y up -- the same
// layout as the rest of the engine. `tiles` is one u32 per tile, row-major,
// index = ty * tiles_w + tx with tile (tx, ty) covering [tx, tx+1) x [ty, ty+1);
// non-zero means solid. Tiles outside the grid are empty.

#[kernel(local_size = 64)]
fn integrate_world(
    #[binding(0)] positions: &mut [f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    delta_time: f32,
    gravity_x: f32,
    gravity_y: f32,
    damping: f32,
    world_width: f32,
    world_height: f32,
    num_particles: u32,
) {
    let index: u32 = gpu_global_id(0);
    if index >= num_particles {
        return;
    }
    let i: usize = (index * 2) as usize;
    let cx: f32 = positions[i];
    let cy: f32 = positions[i + 1];
    let vx: f32 = (cx - previous_positions[i]) * damping;
    let vy: f32 = (cy - previous_positions[i + 1]) * damping;

    let dt2: f32 = delta_time * delta_time;
    let mut nx: f32 = cx + vx + gravity_x * dt2;
    let mut ny: f32 = cy + vy + gravity_y * dt2;

    previous_positions[i] = cx;
    previous_positions[i + 1] = cy;

    let r: f32 = radius[index as usize];
    nx = nx.clamp(r, world_width - r);
    ny = ny.clamp(r, world_height - r);
    positions[i] = nx;
    positions[i + 1] = ny;
}

/// Index of tile (tx, ty) in the grid, or -1 outside it. (A helper cannot take
/// the buffer itself, so each caller does `tiles[idx]`.)
fn tile_index(tiles_w: i32, tiles_h: i32, tx: i32, ty: i32) -> i32 {
    if tx < 0 || ty < 0 || tx >= tiles_w || ty >= tiles_h {
        return -1;
    }
    ty * tiles_w + tx
}

#[kernel(local_size = 64)]
fn tile_collide(
    #[binding(0)] positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(5)] tiles: &[u32],
    tiles_w: u32,
    tiles_h: u32,
    num_particles: u32,
) {
    let index: u32 = gpu_global_id(0);
    if index >= num_particles {
        return;
    }
    let i: usize = (index * 2) as usize;
    let w: i32 = tiles_w as i32;
    let h: i32 = tiles_h as i32;
    let r: f32 = radius[index as usize];
    let mut x: f32 = positions[i];
    let mut y: f32 = positions[i + 1];

    // Every tile the circle's bounding box touches (at most 3 x 3 for r <= 1).
    let x0: i32 = (x - r).floor() as i32;
    let x1: i32 = (x + r).floor() as i32;
    let y0: i32 = (y - r).floor() as i32;
    let y1: i32 = (y + r).floor() as i32;

    for ty in y0..=y1 {
        for tx in x0..=x1 {
            let here: i32 = tile_index(w, h, tx, ty);
            if here >= 0 && tiles[here as usize] != 0 {
                let min_x: f32 = tx as f32;
                let min_y: f32 = ty as f32;
                let qx: f32 = x.clamp(min_x, min_x + 1.0);
                let qy: f32 = y.clamp(min_y, min_y + 1.0);
                let dx: f32 = x - qx;
                let dy: f32 = y - qy;
                let d2: f32 = dx * dx + dy * dy;
                if d2 > 0.0 {
                    if d2 < r * r {
                        let d: f32 = d2.sqrt();
                        let push: f32 = (r - d) / d;
                        x += dx * push;
                        y += dy * push;
                    }
                } else {
                    // The centre is inside the tile: leave through the nearest
                    // face that does not lead into another solid tile.
                    // distance to each face; a face that leads into another solid tile is made unreachable
                    // (plain nested ifs: a `||` around a load makes some drivers' shader compilers take
                    // forever, SwiftShader's among them)
                    let mut left: f32 = x - min_x;
                    let mut right: f32 = min_x + 1.0 - x;
                    let mut down: f32 = y - min_y;
                    let mut up: f32 = min_y + 1.0 - y;
                    let il: i32 = tile_index(w, h, tx - 1, ty);
                    let ir: i32 = tile_index(w, h, tx + 1, ty);
                    let id: i32 = tile_index(w, h, tx, ty - 1);
                    let iu: i32 = tile_index(w, h, tx, ty + 1);
                    if il >= 0 {
                        if tiles[il as usize] != 0 {
                            left = 1.0e9;
                        }
                    }
                    if ir >= 0 {
                        if tiles[ir as usize] != 0 {
                            right = 1.0e9;
                        }
                    }
                    if id >= 0 {
                        if tiles[id as usize] != 0 {
                            down = 1.0e9;
                        }
                    }
                    if iu >= 0 {
                        if tiles[iu as usize] != 0 {
                            up = 1.0e9;
                        }
                    }
                    let mut mx: f32 = 0.0;
                    let mut my: f32 = 0.0;
                    let mut best: f32 = 1.0e9;
                    if left < best {
                        best = left;
                        mx = -(left + r);
                        my = 0.0;
                    }
                    if right < best {
                        best = right;
                        mx = right + r;
                        my = 0.0;
                    }
                    if down < best {
                        best = down;
                        mx = 0.0;
                        my = -(down + r);
                    }
                    if up < best {
                        best = up;
                        mx = 0.0;
                        my = up + r;
                    }
                    x += mx;
                    y += my;
                }
            }
        }
    }
    positions[i] = x;
    positions[i + 1] = y;
}
