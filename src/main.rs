
use game_engine::run; 
fn main() {
    #[cfg(target_arch = "wasm32")]
    run_web().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    run().unwrap();

}
