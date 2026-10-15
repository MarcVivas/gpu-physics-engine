// Per-circle material, mass and collision filtering (Box2D's fixture/body settings), for the Prowl2D
// integration. No WGSL original.
//
//   integrate_material         Verlet step with a gravity scale and a damping per circle, and circles
//                              of zero inverse mass (static / kinematic) left where they are
//   solve_collisions_filtered  collision_solver with Box2D's filter rule, mass-weighted response,
//                              restitution and friction between circles
//   tile_material              tile_friction with the circle's own friction and bounciness
//
// Per-circle arrays (all indexed by circle):
//   material  f32, stride 5: [friction, bounciness, gravity_scale, damping, inv_mass]
//             friction >= 0 (Coulomb, 0.4 is Unity's default), bounciness 0..1,
//             gravity_scale multiplies gravity (0 floats, -1 rises), damping is a per-step
//             multiplier on the Verlet velocity (1 = none), on top of the world's damping.
//   filter    u32, stride 4: [category, mask, group (an i32), body]  -- Box2D's categoryBits, maskBits
//             and groupIndex: two circles with the same non-zero group always collide when it is
//             positive and never when it is negative; otherwise they collide when each one's
//             mask has the other's category bit. `body` is the engine's own rule, kept apart from
//             the user's filter: circles with the same non-zero body (the circles of one crate)
//             never collide with each other.
//   inv_mass (material[4]) is 1 / mass, 0 for an immovable circle (the plain engine uses 1 / radius).
//             It lives in `material` to keep each kernel within WebGPU's 10 storage buffers.
//
// A circle's velocity is `positions - previous_positions` (Verlet), so a collision changes a
// velocity by moving `previous_positions`. Combining two circles' materials follows Box2D:
// friction is the geometric mean, bounciness the larger one.

const STIFFNESS: f32 = 0.6;
const MIN_DISTANCE: f32 = 0.0001;

#[kernel(local_size = 64)]
fn integrate_material(
    #[binding(0)] positions: &mut [f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(11)] material: &[f32],
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
    let m: usize = (index * 5) as usize;
    if material[m + 4] <= 0.0 {
        // static or kinematic: it stays where the host put it, with no velocity of its own
        previous_positions[i] = cx;
        previous_positions[i + 1] = cy;
        return;
    }
    let gscale: f32 = material[m + 2];
    let damp: f32 = damping * material[m + 3];
    let vx: f32 = (cx - previous_positions[i]) * damp;
    let vy: f32 = (cy - previous_positions[i + 1]) * damp;
    let dt2: f32 = delta_time * delta_time;
    let mut nx: f32 = cx + vx + gravity_x * gscale * dt2;
    let mut ny: f32 = cy + vy + gravity_y * gscale * dt2;
    previous_positions[i] = cx;
    previous_positions[i + 1] = cy;
    let r: f32 = radius[index as usize];
    nx = nx.clamp(r, world_width - r);
    ny = ny.clamp(r, world_height - r);
    positions[i] = nx;
    positions[i + 1] = ny;
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

/// Box2D's collision filter: a shared non-zero group decides (positive collides, negative does
/// not); otherwise each side's mask must contain the other's category.
fn can_collide(ca: u32, ma: u32, ga: i32, cb: u32, mb: u32, gb: i32) -> bool {
    if ga != 0 && ga == gb {
        return ga > 0;
    }
    (ma & cb) != 0 && (mb & ca) != 0
}

/// One invocation per collision cell, as solve_collisions. `bounce_threshold` is the slowest
/// approach (world units per step) that still bounces: slower contacts are resting contacts.
#[kernel(local_size = 64)]
fn solve_collisions_filtered(
    #[binding(0)] chunk_obj_count: &[u32],
    #[binding(1)] collision_cells: &[u32],
    #[binding(2)] cell_ids: &[u32],
    #[binding(3)] object_ids: &[u32],
    #[binding(4)] positions: &mut [f32],
    #[binding(5)] radius: &[f32],
    #[binding(6)] uniform_data: &[u32],
    #[binding(7)] filter: &[u32],
    #[binding(8)] material: &[f32],
    #[binding(10)] previous_positions: &mut [f32],
    current_cell_color: u32,
    bounce_threshold: f32,
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
            let fa: usize = a * 4;
            let fb: usize = b * 4;
            let ma: usize = a * 5;
            let mb: usize = b * 5;
            let same_body: bool = filter[fa + 3] != 0 && filter[fa + 3] == filter[fb + 3];
            let ok: bool = !same_body && can_collide(
                filter[fa],
                filter[fa + 1],
                filter[fa + 2] as i32,
                filter[fb],
                filter[fb + 1],
                filter[fb + 2] as i32,
            );
            let inv_a: f32 = material[ma + 4];
            let inv_b: f32 = material[mb + 4];
            let wsum: f32 = inv_a + inv_b;
            if ok && wsum > 0.0 {
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
                    let nx: f32 = vx / distance;
                    let ny: f32 = vy / distance;
                    let corr: f32 = (radius_sum - distance) * STIFFNESS;
                    let share_a: f32 = inv_a / wsum;
                    let share_b: f32 = inv_b / wsum;

                    // the approach, before the push-out: velocity is position - previous
                    let pax: f32 = previous_positions[a * 2];
                    let pay: f32 = previous_positions[a * 2 + 1];
                    let pbx: f32 = previous_positions[b * 2];
                    let pby: f32 = previous_positions[b * 2 + 1];
                    let rvx: f32 = (ax - pax) - (bx - pbx);
                    let rvy: f32 = (ay - pay) - (by - pby);
                    let vn: f32 = rvx * nx + rvy * ny;
                    let vt: f32 = rvx * (0.0 - ny) + rvy * nx;

                    positions[a * 2] = ax + nx * corr * share_a;
                    positions[a * 2 + 1] = ay + ny * corr * share_a;
                    positions[b * 2] = bx - nx * corr * share_b;
                    positions[b * 2 + 1] = by - ny * corr * share_b;

                    // Box2D's combination: friction is the geometric mean, bounciness the larger
                    let mu: f32 = (material[ma] * material[mb]).sqrt();
                    let mut e: f32 = material[ma + 1];
                    if material[mb + 1] > e {
                        e = material[mb + 1];
                    }

                    // normal: the push-out already separates them at `corr` per step; the
                    // contact should leave at e * approach (0 for a resting or soft contact)
                    let mut want: f32 = vn;
                    let mut load: f32 = corr;
                    if vn <= 0.0 {
                        want = 0.0;
                        if 0.0 - vn > bounce_threshold {
                            want = e * (0.0 - vn);
                        }
                        load = corr + (0.0 - vn);
                    }
                    let dvn: f32 = want - (vn + corr);

                    // friction: the sliding speed shrinks by at most mu times the normal load
                    let limit: f32 = mu * load;
                    let dvt: f32 = (0.0 - vt).clamp(0.0 - limit, limit);

                    // a velocity change of dv on a is prev_a -= dv * share_a (b gets the opposite)
                    let cxn: f32 = dvn * nx + dvt * (0.0 - ny);
                    let cyn: f32 = dvn * ny + dvt * nx;
                    previous_positions[a * 2] = pax - cxn * share_a;
                    previous_positions[a * 2 + 1] = pay - cyn * share_a;
                    previous_positions[b * 2] = pbx + cxn * share_b;
                    previous_positions[b * 2 + 1] = pby + cyn * share_b;
                }
            }
            j += 1;
        }
        i += 1;
    }
}

/// Index of tile (tx, ty) in the grid, or -1 outside it.
fn tile_index(tiles_w: i32, tiles_h: i32, tx: i32, ty: i32) -> i32 {
    if tx < 0 || ty < 0 || tx >= tiles_w || ty >= tiles_h {
        return -1;
    }
    ty * tiles_w + tx
}

/// tile_friction with each circle's own material: against a tile the friction is
/// sqrt(circle friction * tile_friction) and the bounciness e = max(circle bounciness,
/// tile_bounciness). A circle moving into the tile faster than `bounce_threshold` (world
/// units per step) leaves it at e times that speed, a slower one stops; one that is only
/// being pushed out keeps e of the push-out speed. (The first contact found counts.)
/// `omega` is each circle's spin in radians per step (0 for one that does not turn): friction
/// acts on the slip at the contact, the tangential speed of the circle's surface against the
/// tile, so a rolling ball is not slowed.
#[kernel(local_size = 64)]
fn tile_material(
    #[binding(0)] positions: &[f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(5)] tiles: &[u32],
    #[binding(11)] material: &[f32],
    #[binding(14)] omega: &[f32],
    tiles_w: u32,
    tiles_h: u32,
    num_particles: u32,
    tile_friction: f32,
    tile_bounciness: f32,
    bounce_threshold: f32,
) {
    let index: u32 = gpu_global_id(0);
    if index >= num_particles {
        return;
    }
    let m: usize = (index * 5) as usize;
    if material[m + 4] <= 0.0 {
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
    let mu: f32 = (material[m] * tile_friction).sqrt();
    let mut e: f32 = material[m + 1];
    if tile_bounciness > e {
        e = tile_bounciness;
    }
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
                    let tx: f32 = 0.0 - ny;
                    let ty: f32 = nx;
                    let slip: f32 = vx * tx + vy * ty - radius[index as usize] * omega[index as usize];
                    rx += slip * tx * mu;
                    ry += slip * ty * mu;
                    // the normal speed after the contact, as a change dvn of vn
                    let mut dvn: f32 = 0.0;
                    if vn > 0.0 {
                        dvn = 0.0 - vn * (1.0 - e);
                    } else {
                        let mut leave: f32 = 0.0;
                        if 0.0 - vn > bounce_threshold {
                            leave = e * (0.0 - vn);
                        }
                        dvn = leave - vn;
                    }
                    rx += 0.0 - dvn * nx;
                    ry += 0.0 - dvn * ny;
                }
            }
        }
    }
    previous_positions[i] = px + rx;
    previous_positions[i + 1] = py + ry;
}
