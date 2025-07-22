const WORKGROUP_SIZE = 64u;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> num_elems: u32;
@group(0) @binding(2) var<storage, read_write> intermediate_data: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum_1(@builtin(global_invocation_id) global_id: vec3<u32>){
    let tid = global_id.x;

    if tid >= num_elems {
        return;
    }

    let value = data[tid];
    



}
