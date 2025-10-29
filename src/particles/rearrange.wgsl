
override WORKGROUP_SIZE = 64u;

struct PushConstantsData{
    num_particles: u32, 
}

@group(0) @binding(0) var<storage, read> positions_read: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> radius_read: array<f32>;
@group(0) @binding(2) var<storage, read> previous_positions_read: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> particle_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> positions_write: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> radius_write: array<f32>;
@group(0) @binding(6) var<storage, read_write> previous_positions_write: array<vec2<f32>>;

var<push_constant> push_constant_data: PushConstantsData;

@compute @workgroup_size(WORKGROUP_SIZE)
fn rearrange(@builtin(global_invocation_id) global_id: vec3<u32>){

    let obj_id = global_id.x;

    if obj_id >= push_constant_data.num_particles {
        return;
    }
    
    let reading_idx = particle_ids[obj_id];
    let position = positions_read[reading_idx];
    let radius = radius_read[reading_idx]; 
    let prev_position = previous_positions_read[reading_idx]; 
    
    positions_write[obj_id] = position; 
    radius_write[obj_id] = radius; 
    previous_positions_write[obj_id] = prev_position; 
}
