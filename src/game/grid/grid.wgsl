
const WORKGROUP_SIZE = 64u; 

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_grid(@builtin(global_invocation_id) global_id: vec3<u32>){

}