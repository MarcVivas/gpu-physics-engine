// Kernels for the Prowl2D integration beyond the circle solver (no WGSL original):
//
//   solve_joints            distance constraints between two circles, or a circle and a
//                           fixed point (ropes, rods, hinges, the stiff links of crates)
//   solve_collisions_grouped  collision_solver with a group id per circle: circles of the
//                           same non-zero group (the corners of one crate) ignore each other
//   integrate_tagged        integrate_world with a separate damping for tag 1 (sand)
//   tile_friction           slows circles that touch a solid tile (sand piles, crates stop sliding)
//   raycast                 a batch of rays against the solid tiles and the circles
//   sensor_count            how many circles overlap each sensor region (atomic counts)
//
// Layouts: positions are interleaved f32 pairs, y up. Per-circle tags (u32) say what
// a circle is: 0 ball, 1 sand, 2 crate corner, 3 shard corner, 4.. free for scripts.
//
// joints are two arrays of stride 4:
//   joint_i: [a, b, flags, 0]   b = 0xFFFFFFFF: the other end is a fixed point
//   joint_f: [rest, stiffness, ax, ay]   (ax, ay: the fixed point)
//   flags bit 0: rope (acts only when stretched), bit 1: strut (only when squeezed)
// The host colours the joints so no circle appears twice in one batch, and
// dispatches one batch (first, count) per colour.

const STIFFNESS: f32 = 0.6;
const MIN_DISTANCE: f32 = 0.0001;

#[kernel(local_size = 64)]
fn solve_joints(
    #[binding(0)] positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(6)] joint_i: &[u32],
    #[binding(7)] joint_f: &[f32],
    first: u32,
    count: u32,
) {
    let t: u32 = gpu_global_id(0);
    if t >= count {
        return;
    }
    let j: usize = ((first + t) * 4) as usize;
    let a: u32 = joint_i[j];
    let b: u32 = joint_i[j + 1];
    let flags: u32 = joint_i[j + 2];
    let rest: f32 = joint_f[j];
    let k: f32 = joint_f[j + 1];

    let ax: f32 = positions[(a * 2) as usize];
    let ay: f32 = positions[(a * 2 + 1) as usize];
    let inv_a: f32 = 1.0 / radius[a as usize];
    let mut bx: f32 = joint_f[j + 2];
    let mut by: f32 = joint_f[j + 3];
    let mut inv_b: f32 = 0.0;
    if b != 0xFFFFFFFF {
        bx = positions[(b * 2) as usize];
        by = positions[(b * 2 + 1) as usize];
        inv_b = 1.0 / radius[b as usize];
    }
    let dx: f32 = bx - ax;
    let dy: f32 = by - ay;
    let d: f32 = (dx * dx + dy * dy).sqrt();
    if d < MIN_DISTANCE {
        return;
    }
    let diff: f32 = d - rest;
    if (flags & 1) != 0 && diff < 0.0 {
        return;
    }
    if (flags & 2) != 0 && diff > 0.0 {
        return;
    }
    let corr: f32 = diff * k / (inv_a + inv_b);
    let ux: f32 = dx / d;
    let uy: f32 = dy / d;
    positions[(a * 2) as usize] = ax + ux * corr * inv_a;
    positions[(a * 2 + 1) as usize] = ay + uy * corr * inv_a;
    if b != 0xFFFFFFFF {
        positions[(b * 2) as usize] = bx - ux * corr * inv_b;
        positions[(b * 2 + 1) as usize] = by - uy * corr * inv_b;
    }
}

/// Compacts bits from every other position to the lower 16 bits.
fn unsplit_by_bits(n: u32) -> u32 {
    let mut x: u32 = n & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    x
}

fn get_cell_color(cell_hash: u32) -> u32 {
    let gx: u32 = unsplit_by_bits(cell_hash);
    let gy: u32 = unsplit_by_bits(cell_hash >> 1);
    1 + (gx % 2) + (gy % 2) * 2
}

#[kernel(local_size = 64)]
fn solve_collisions_grouped(
    #[binding(0)] chunk_obj_count: &[u32],
    #[binding(1)] collision_cells: &[u32],
    #[binding(2)] cell_ids: &[u32],
    #[binding(3)] object_ids: &[u32],
    #[binding(4)] positions: &mut [f32],
    #[binding(5)] radius: &[f32],
    #[binding(6)] uniform_data: &[u32],
    #[binding(7)] groups: &[u32],
    current_cell_color: u32,
) {
    let tid: u32 = gpu_global_id(0);
    let num_counting_chunks: u32 = uniform_data[0];
    let total_cell_ids: u32 = uniform_data[1];
    let num_collision_cells: u32 = chunk_obj_count[(num_counting_chunks - 1) as usize];
    if tid >= num_collision_cells {
        return;
    }
    let start: u32 = collision_cells[tid as usize];
    let cell_hash: u32 = cell_ids[start as usize];
    if get_cell_color(cell_hash) != current_cell_color {
        return;
    }
    let mut i: u32 = start;
    while i < total_cell_ids {
        if cell_ids[i as usize] != cell_hash {
            break;
        }
        let a: usize = object_ids[i as usize] as usize;
        let mut j: u32 = i + 1;
        while j < total_cell_ids {
            if cell_ids[j as usize] != cell_hash {
                break;
            }
            let b: usize = object_ids[j as usize] as usize;
            let same: bool = groups[a] != 0 && groups[a] == groups[b];
            if !same {
                let ax: f32 = positions[a * 2];
                let ay: f32 = positions[a * 2 + 1];
                let bx: f32 = positions[b * 2];
                let by: f32 = positions[b * 2 + 1];
                let ra: f32 = radius[a];
                let rb: f32 = radius[b];
                let vx: f32 = ax - bx;
                let vy: f32 = ay - by;
                let distance: f32 = (vx * vx + vy * vy).sqrt();
                let radius_sum: f32 = ra + rb;
                if radius_sum * radius_sum > distance * distance && distance > MIN_DISTANCE {
                    let penetration: f32 = radius_sum - distance;
                    let dx: f32 = vx / distance;
                    let dy: f32 = vy / distance;
                    let cx: f32 = dx * penetration * STIFFNESS;
                    let cy: f32 = dy * penetration * STIFFNESS;
                    let inv_a: f32 = 1.0 / ra;
                    let inv_b: f32 = 1.0 / rb;
                    let fa: f32 = inv_a / (inv_a + inv_b);
                    let fb: f32 = inv_b / (inv_a + inv_b);
                    positions[a * 2] = ax + cx * fa;
                    positions[a * 2 + 1] = ay + cy * fa;
                    positions[b * 2] = bx - cx * fb;
                    positions[b * 2 + 1] = by - cy * fb;
                }
            }
            j += 1;
        }
        i += 1;
    }
}

#[kernel(local_size = 64)]
fn integrate_tagged(
    #[binding(0)] positions: &mut [f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(3)] tags: &[u32],
    delta_time: f32,
    gravity_x: f32,
    gravity_y: f32,
    damping: f32,
    sand_damping: f32,
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
    let mut damp: f32 = damping;
    if tags[index as usize] == 1 {
        damp = sand_damping;
    }
    let vx: f32 = (cx - previous_positions[i]) * damp;
    let vy: f32 = (cy - previous_positions[i + 1]) * damp;
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

/// Index of tile (tx, ty) in the grid, or -1 outside it.
fn tile_index(tiles_w: i32, tiles_h: i32, tx: i32, ty: i32) -> i32 {
    if tx < 0 || ty < 0 || tx >= tiles_w || ty >= tiles_h {
        return -1;
    }
    ty * tiles_w + tx
}

/// A circle resting on (or within 0.02 of) a solid tile loses `mu` of the part of
/// its velocity that runs along the tile's surface, and `absorb` of the speed it is
/// leaving the tile with (the push-out otherwise shows up as a bounce) (the first contact found counts,
/// so a flat floor of several tiles does not apply it twice).
#[kernel(local_size = 64)]
fn tile_friction(
    #[binding(0)] positions: &[f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(5)] tiles: &[u32],
    tiles_w: u32,
    tiles_h: u32,
    num_particles: u32,
    mu: f32,
    absorb: f32,
) {
    let index: u32 = gpu_global_id(0);
    if index >= num_particles {
        return;
    }
    let i: usize = (index * 2) as usize;
    let w: i32 = tiles_w as i32;
    let h: i32 = tiles_h as i32;
    let r: f32 = radius[index as usize] + 0.02;
    let x: f32 = positions[i];
    let y: f32 = positions[i + 1];
    let px: f32 = previous_positions[i];
    let py: f32 = previous_positions[i + 1];
    let vx: f32 = x - px;
    let vy: f32 = y - py;
    let mut rx: f32 = 0.0;
    let mut ry: f32 = 0.0;
    let mut touched: bool = false;
    let x0: i32 = (x - r).floor() as i32;
    let x1: i32 = (x + r).floor() as i32;
    let y0: i32 = (y - r).floor() as i32;
    let y1: i32 = (y + r).floor() as i32;
    for ty in y0..=y1 {
        for tx in x0..=x1 {
            let here: i32 = tile_index(w, h, tx, ty);
            if here >= 0 && tiles[here as usize] != 0 {
                let qx: f32 = x.clamp(tx as f32, tx as f32 + 1.0);
                let qy: f32 = y.clamp(ty as f32, ty as f32 + 1.0);
                let dx: f32 = x - qx;
                let dy: f32 = y - qy;
                let d2: f32 = dx * dx + dy * dy;
                if d2 > 0.0 && d2 < r * r && !touched {
                    touched = true;
                    let d: f32 = d2.sqrt();
                    let nx: f32 = dx / d;
                    let ny: f32 = dy / d;
                    let vn: f32 = vx * nx + vy * ny;
                    rx += (vx - vn * nx) * mu;
                    ry += (vy - vn * ny) * mu;
                    if vn > 0.0 {
                        rx += vn * nx * absorb;
                        ry += vn * ny * absorb;
                    }
                }
            }
        }
    }
    previous_positions[i] = px + rx;
    previous_positions[i + 1] = py + ry;
}

// ---------------------------------------------------------------- ray casts
// rays: [origin x, origin y, direction x, direction y (unit), max distance] x 8 floats
// (3 spare). hits: [t, normal x, normal y, kind] x 4 floats per ray; t < 0: no hit;
// kind 1 tile, 2 circle. hit_ids: the circle's index, or the tile's index.
// mask: bit 0 tiles, bit 1 circles. Circles whose group equals ignore_group
// (when non-zero) are skipped, so a crate's own ray does not hit itself.
#[kernel(local_size = 64)]
fn raycast(
    #[binding(0)] positions: &[f32],
    #[binding(2)] radius: &[f32],
    #[binding(3)] groups: &[u32],
    #[binding(5)] tiles: &[u32],
    #[binding(8)] rays: &[f32],
    #[binding(9)] hits: &mut [f32],
    #[binding(10)] hit_ids: &mut [u32],
    tiles_w: u32,
    tiles_h: u32,
    num_circles: u32,
    num_rays: u32,
    mask: u32,
    ignore_group: u32,
) {
    let ri: u32 = gpu_global_id(0);
    if ri >= num_rays {
        return;
    }
    let q: usize = (ri * 8) as usize;
    let ox: f32 = rays[q];
    let oy: f32 = rays[q + 1];
    let dx: f32 = rays[q + 2];
    let dy: f32 = rays[q + 3];
    let max_t: f32 = rays[q + 4];
    let mut best: f32 = max_t;
    let mut kind: f32 = 0.0;
    let mut hid: u32 = 0;
    let mut hnx: f32 = 0.0;
    let mut hny: f32 = 0.0;

    if (mask & 1) != 0 && tiles_w > 0 {
        let w: i32 = tiles_w as i32;
        let h: i32 = tiles_h as i32;
        let mut tx: i32 = ox.floor() as i32;
        let mut ty: i32 = oy.floor() as i32;
        let mut step_x: i32 = 1;
        let mut step_y: i32 = 1;
        let mut tmax_x: f32 = 1.0e30;
        let mut tmax_y: f32 = 1.0e30;
        let mut tdx: f32 = 1.0e30;
        let mut tdy: f32 = 1.0e30;
        if dx > 0.0 {
            tdx = 1.0 / dx;
            tmax_x = (tx as f32 + 1.0 - ox) / dx;
        }
        if dx < 0.0 {
            step_x = -1;
            tdx = -1.0 / dx;
            tmax_x = (ox - tx as f32) / -dx;
        }
        if dy > 0.0 {
            tdy = 1.0 / dy;
            tmax_y = (ty as f32 + 1.0 - oy) / dy;
        }
        if dy < 0.0 {
            step_y = -1;
            tdy = -1.0 / dy;
            tmax_y = (oy - ty as f32) / -dy;
        }
        let mut cur: f32 = 0.0;
        let mut axis: u32 = 0;
        let mut steps: i32 = 0;
        let limit: i32 = w + h + 4;
        while steps < limit && cur <= best {
            let here: i32 = tile_index(w, h, tx, ty);
            if here >= 0 && tiles[here as usize] != 0 {
                best = cur;
                kind = 1.0;
                hid = here as u32;
                if axis == 1 {
                    hnx = -(step_x as f32);
                    hny = 0.0;
                }
                if axis == 2 {
                    hnx = 0.0;
                    hny = -(step_y as f32);
                }
                break;
            }
            if tmax_x < tmax_y {
                cur = tmax_x;
                tmax_x += tdx;
                tx += step_x;
                axis = 1;
            } else {
                cur = tmax_y;
                tmax_y += tdy;
                ty += step_y;
                axis = 2;
            }
            steps += 1;
        }
    }

    if (mask & 2) != 0 {
        for c in 0..num_circles {
            let skip: bool = ignore_group != 0 && groups[c as usize] == ignore_group;
            if !skip {
                let cx: f32 = positions[(c * 2) as usize];
                let cy: f32 = positions[(c * 2 + 1) as usize];
                let r: f32 = radius[c as usize];
                let ex: f32 = ox - cx;
                let ey: f32 = oy - cy;
                let b: f32 = ex * dx + ey * dy;
                let cc: f32 = ex * ex + ey * ey - r * r;
                let disc: f32 = b * b - cc;
                if disc >= 0.0 {
                    let s: f32 = disc.sqrt();
                    let mut t: f32 = -b - s;
                    if cc < 0.0 {
                        t = 0.0;
                    }
                    if t >= 0.0 && t < best {
                        best = t;
                        kind = 2.0;
                        hid = c;
                        let px: f32 = ox + dx * t - cx;
                        let py: f32 = oy + dy * t - cy;
                        hnx = px / r;
                        hny = py / r;
                    }
                }
            }
        }
    }

    let o: usize = (ri * 4) as usize;
    if kind > 0.0 {
        hits[o] = best;
    } else {
        hits[o] = -1.0;
    }
    hits[o + 1] = hnx;
    hits[o + 2] = hny;
    hits[o + 3] = kind;
    hit_ids[ri as usize] = hid;
}

// ---------------------------------------------------------------- sensors
// sensors: [x, y, half width, half height] x 4 floats (a box), or, when the half
// height is negative, a circle of radius = half width. sensor_mask: which circle
// tags (bit i = tag i; 0 = all) the sensor counts. sensor_counts[s] gets the number
// of overlapping circles; the host zeroes it first.
#[kernel(local_size = 64)]
fn sensor_count(
    #[binding(0)] positions: &[f32],
    #[binding(2)] radius: &[f32],
    #[binding(4)] tags: &[u32],
    #[binding(11)] sensors: &[f32],
    #[binding(12)] sensor_mask: &[u32],
    #[binding(13)] sensor_counts: &mut [u32],
    num_circles: u32,
    num_sensors: u32,
) {
    let c: u32 = gpu_global_id(0);
    if c >= num_circles {
        return;
    }
    let x: f32 = positions[(c * 2) as usize];
    let y: f32 = positions[(c * 2 + 1) as usize];
    let r: f32 = radius[c as usize];
    let tag: u32 = tags[c as usize];
    for s in 0..num_sensors {
        let m: u32 = sensor_mask[s as usize];
        let wanted: bool = m == 0 || ((m >> (tag & 31)) & 1) != 0;
        if wanted {
            let q: usize = (s * 4) as usize;
            let sx: f32 = sensors[q];
            let sy: f32 = sensors[q + 1];
            let hw: f32 = sensors[q + 2];
            let hh: f32 = sensors[q + 3];
            let mut hit: bool = false;
            if hh < 0.0 {
                let dx: f32 = x - sx;
                let dy: f32 = y - sy;
                let rr: f32 = hw + r;
                hit = dx * dx + dy * dy < rr * rr;
            } else {
                let qx: f32 = x.clamp(sx - hw, sx + hw);
                let qy: f32 = y.clamp(sy - hh, sy + hh);
                let dx: f32 = x - qx;
                let dy: f32 = y - qy;
                hit = dx * dx + dy * dy < r * r;
            }
            if hit {
                gpu_atomic_add(&sensor_counts[s as usize], 1);
            }
        }
    }
}
