
override WORKGROUP_SIZE = 64u;

struct PushConstantsData{
    num_particles: u32, 
    cell_size: f32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> home_cell_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> particle_ids: array<u32>;

var<push_constant> push_constant_data: PushConstantsData;

@compute @workgroup_size(WORKGROUP_SIZE)
fn create_home_cell_ids(@builtin(global_invocation_id) global_id: vec3<u32>){

    let obj_id = global_id.x;

    if obj_id >= push_constant_data.num_particles {
        return;
    }

    let pos = positions[obj_id];

    // Convert to grid coordinates.
    let home_cell_coord = vec2<i32>(floor(pos / push_constant_data.cell_size));
    let home_cell_hash = morton_encode(home_cell_coord);
    
    // Store home cell id
    home_cell_ids[obj_id] = home_cell_hash;
    // Store particle index
    particle_ids[obj_id] = obj_id;
}

/// Spreads the lower 16 bits of an integer to every other bit.
/// Example (2-bit): n = 3 (binary 11) becomes 5 (binary 0101).
fn split_by_bits(n: u32) -> u32 {
    var x = n & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

/// Encodes 2D coordinates (16-bit max) into a 1D Morton index.
/// Example: (x=3, y=3) -> (binary 11, 11) -> interleaved 1111 -> 15.
fn morton_encode(v: vec2<i32>) -> u32 {
    return split_by_bits(u32(v.x)) | (split_by_bits(u32(v.y)) << 1);
}
