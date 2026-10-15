// Port of src/particles/rearrange.wgsl: gathers the particle arrays into the
// order given by particle_ids (the sorted order). Same bindings 0..=6.

#[kernel(local_size = 64)]
fn rearrange(
    positions_read: &[f32],
    radius_read: &[f32],
    previous_positions_read: &[f32],
    particle_ids: &[u32],
    positions_write: &mut [f32],
    radius_write: &mut [f32],
    previous_positions_write: &mut [f32],
    num_particles: u32,
) {
    let obj_id: u32 = gpu_global_id(0);
    if obj_id >= num_particles {
        return;
    }

    let reading_idx: u32 = particle_ids[obj_id as usize];
    let r: usize = (reading_idx * 2) as usize;
    let w: usize = (obj_id * 2) as usize;

    positions_write[w] = positions_read[r];
    positions_write[w + 1] = positions_read[r + 1];
    radius_write[obj_id as usize] = radius_read[reading_idx as usize];
    previous_positions_write[w] = previous_positions_read[r];
    previous_positions_write[w + 1] = previous_positions_read[r + 1];
}
