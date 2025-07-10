
const WORKGROUP_SIZE = 64u;

struct UniformData {
    num_particles: u32,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> uniform: UniformData;
@group(0) @binding(2) var<storage, read_write> cell_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> object_ids: array<u32>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn build_grid(@builtin(global_invocation_id) global_id: vec3<u32>){

    let thread_id = global_id.x;

    if thread_id >= uniform.num_particles {
        return;
    }

    positions[thread_id] = vec2<f32>(f32(thread_id), f32(thread_id));

}
