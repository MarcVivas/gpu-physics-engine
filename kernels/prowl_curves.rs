// Smooth static colliders: quadratic Bezier curves.
//
//   curve_collide    pushes circles out of every curve (runs inside the solver iterations)
//   curve_material   friction and bounciness of a curve contact, after the iterations
//                    (the same velocity edit as tile_material, with the curve's own numbers)
//
// curves  f32, stride 16 per curve:
//   [p0x, p0y, p1x, p1y, p2x, p2y, minx, miny, maxx, maxy, friction, bounciness, side, 0, 0, 0]
//   friction is in kernel units (0.375 x Unity's), min/max is the curve's bounding box.
//   side  0: a thin wall that is solid on both faces. A circle stays on the face it was on
//            (decided from its previous position), so it cannot tunnel through unless it moves
//            more than its radius in a step.
//         +1 / -1: one edge of a solid contour whose inside is on the left (+1) or right (-1) of
//            the curve's direction p0 -> p2. A circle whose centre is inside is pushed out.
//   Curves are tested one after the other; a contour is several curves with the same side.
//
// Nearest point: 9 samples, then Newton steps on d/dt |B(t) - P|^2 / 2, clamped to [0, 1].

const MIN_DISTANCE: f32 = 0.000001;

fn bez(a: f32, b: f32, c: f32, t: f32) -> f32 {
    let u: f32 = 1.0 - t;
    u * u * a + 2.0 * u * t * b + t * t * c
}

fn dbez(a: f32, b: f32, c: f32, t: f32) -> f32 {
    2.0 * ((1.0 - t) * (b - a) + t * (c - b))
}

/// The parameter t of the point of the curve nearest to (px, py).
fn nearest_t(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    let mut best: f32 = 0.0;
    let mut best_d: f32 = 1.0e30;
    for i in 0..9 {
        let t: f32 = (i as f32) / 8.0;
        let dx: f32 = bez(ax, bx, cx, t) - px;
        let dy: f32 = bez(ay, by, cy, t) - py;
        let d: f32 = dx * dx + dy * dy;
        if d < best_d {
            best_d = d;
            best = t;
        }
    }
    let sx: f32 = ax - 2.0 * bx + cx;
    let sy: f32 = ay - 2.0 * by + cy;
    let mut t: f32 = best;
    for k in 0..5 {
        let dx: f32 = bez(ax, bx, cx, t) - px;
        let dy: f32 = bez(ay, by, cy, t) - py;
        let vx: f32 = dbez(ax, bx, cx, t);
        let vy: f32 = dbez(ay, by, cy, t);
        let g: f32 = dx * vx + dy * vy;
        let h: f32 = vx * vx + vy * vy + 2.0 * (dx * sx + dy * sy);
        if h > 0.000001 {
            t = (t - g / h).clamp(0.0, 1.0);
        }
    }
    t
}

#[kernel(local_size = 64)]
fn curve_collide(
    #[binding(0)] positions: &mut [f32],
    #[binding(1)] previous_positions: &[f32],
    #[binding(2)] radius: &[f32],
    #[binding(11)] material: &[f32],
    #[binding(15)] curves: &[f32],
    num_particles: u32,
    num_curves: u32,
) {
    let index: u32 = gpu_global_id(0);
    if index >= num_particles {
        return;
    }
    if material[(index * 5 + 4) as usize] <= 0.0 {
        return;
    }
    let i: usize = (index * 2) as usize;
    let r: f32 = radius[index as usize];
    let mut x: f32 = positions[i];
    let mut y: f32 = positions[i + 1];
    let px: f32 = previous_positions[i];
    let py: f32 = previous_positions[i + 1];
    for c in 0..num_curves {
        let k: usize = (c * 16) as usize;
        if x >= curves[k + 6] - r && x <= curves[k + 8] + r && y >= curves[k + 7] - r && y <= curves[k + 9] + r {
            let ax: f32 = curves[k];
            let ay: f32 = curves[k + 1];
            let bx: f32 = curves[k + 2];
            let by: f32 = curves[k + 3];
            let cx: f32 = curves[k + 4];
            let cy: f32 = curves[k + 5];
            let t: f32 = nearest_t(x, y, ax, ay, bx, by, cx, cy);
            let qx: f32 = bez(ax, bx, cx, t);
            let qy: f32 = bez(ay, by, cy, t);
            let dx: f32 = x - qx;
            let dy: f32 = y - qy;
            let d: f32 = (dx * dx + dy * dy).sqrt();
            if d < r {
                // the curve's tangent and its left normal
                let mut tx: f32 = dbez(ax, bx, cx, t);
                let mut ty: f32 = dbez(ay, by, cy, t);
                let tl: f32 = (tx * tx + ty * ty).sqrt();
                if tl < MIN_DISTANCE {
                    tx = cx - ax;
                    ty = cy - ay;
                }
                let tl2: f32 = (tx * tx + ty * ty).sqrt().max(MIN_DISTANCE);
                let lx: f32 = 0.0 - ty / tl2;
                let ly: f32 = tx / tl2;
                let side: f32 = curves[k + 12];
                // the unit vector pointing out of the solid on the circle's side
                let mut ox: f32 = lx;
                let mut oy: f32 = ly;
                if side > 0.5 {
                    ox = 0.0 - lx;
                    oy = 0.0 - ly;
                } else if side >= -0.5 {
                    if (px - qx) * lx + (py - qy) * ly < 0.0 {
                        ox = 0.0 - lx;
                        oy = 0.0 - ly;
                    }
                }
                let sd: f32 = dx * ox + dy * oy;
                let at_end: bool = t <= 0.0001 || t >= 0.9999;
                if (sd >= 0.0 || (at_end && side > -0.5 && side < 0.5)) && d > MIN_DISTANCE {
                    // outside (or round the free end of a thin wall): straight away from the nearest point
                    x = qx + dx / d * r;
                    y = qy + dy / d * r;
                } else {
                    x = qx + ox * r;
                    y = qy + oy * r;
                }
            }
        }
    }
    positions[i] = x;
    positions[i + 1] = y;
}

#[kernel(local_size = 64)]
fn curve_material(
    #[binding(0)] positions: &[f32],
    #[binding(1)] previous_positions: &mut [f32],
    #[binding(2)] radius: &[f32],
    #[binding(11)] material: &[f32],
    #[binding(14)] omega: &[f32],
    #[binding(15)] curves: &[f32],
    num_particles: u32,
    num_curves: u32,
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
    let rr: f32 = radius[index as usize];
    let r: f32 = rr + 0.02;
    let x: f32 = positions[i];
    let y: f32 = positions[i + 1];
    let px: f32 = previous_positions[i];
    let py: f32 = previous_positions[i + 1];
    let vx: f32 = x - px;
    let vy: f32 = y - py;
    let mut rx: f32 = 0.0;
    let mut ry: f32 = 0.0;
    let mut touched: bool = false;
    for c in 0..num_curves {
        let k: usize = (c * 16) as usize;
        if !touched && x >= curves[k + 6] - r && x <= curves[k + 8] + r && y >= curves[k + 7] - r && y <= curves[k + 9] + r {
            let ax: f32 = curves[k];
            let ay: f32 = curves[k + 1];
            let bx: f32 = curves[k + 2];
            let by: f32 = curves[k + 3];
            let cx: f32 = curves[k + 4];
            let cy: f32 = curves[k + 5];
            let t: f32 = nearest_t(x, y, ax, ay, bx, by, cx, cy);
            let qx: f32 = bez(ax, bx, cx, t);
            let qy: f32 = bez(ay, by, cy, t);
            let dx: f32 = x - qx;
            let dy: f32 = y - qy;
            let d: f32 = (dx * dx + dy * dy).sqrt();
            // a solid contour only touches from outside
            let mut tx: f32 = dbez(ax, bx, cx, t);
            let mut ty: f32 = dbez(ay, by, cy, t);
            let tl: f32 = (tx * tx + ty * ty).sqrt().max(MIN_DISTANCE);
            tx = tx / tl;
            ty = ty / tl;
            let side: f32 = curves[k + 12];
            let outside: f32 = (dx * (0.0 - ty) + dy * tx) * (0.0 - side);
            if d > MIN_DISTANCE && d < r && (side > -0.5 && side < 0.5 || outside >= 0.0) {
                touched = true;
                let mu: f32 = (material[m] * curves[k + 10]).sqrt();
                let mut e: f32 = material[m + 1];
                if curves[k + 11] > e {
                    e = curves[k + 11];
                }
                let nx: f32 = dx / d;
                let ny: f32 = dy / d;
                let vn: f32 = vx * nx + vy * ny;
                let sx: f32 = 0.0 - ny;
                let sy: f32 = nx;
                let slip: f32 = vx * sx + vy * sy - rr * omega[index as usize];
                rx += slip * sx * mu;
                ry += slip * sy * mu;
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
    previous_positions[i] = px + rx;
    previous_positions[i + 1] = py + ry;
}
