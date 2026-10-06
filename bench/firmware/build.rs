//! Turns scenario S2 into Rust constants (`$OUT_DIR/s2.rs`), so both firmware images embed
//! identical data: the model in f32 and the first `STEPS` measurements.

use std::env;
use std::fs;
use std::path::PathBuf;

const STEPS: usize = 1000;

fn matrix(json: &serde_json::Value, key: &str) -> String {
    let rows: Vec<String> = json[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let values: Vec<String> = row
                .as_array()
                .unwrap()
                .iter()
                .map(|v| format!("{:?}f32", v.as_f64().unwrap() as f32))
                .collect();
            format!("[{}]", values.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../tests/vectors/S2");
    let model_path = dir.join("model.json");
    let zs_path = dir.join("measurements.bin");
    println!("cargo:rerun-if-changed={}", model_path.display());
    println!("cargo:rerun-if-changed={}", zs_path.display());
    println!("cargo:rerun-if-changed=memory.x");

    // Lets cortex-m-rt's link.x find memory.x, and links with it. Unlike config rustflags,
    // this can't be overridden by a RUSTFLAGS environment variable.
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg-bins=-Tlink.x");

    let model: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(model_path).unwrap()).unwrap();
    let bytes = fs::read(zs_path).unwrap();
    let values: Vec<f32> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .take(STEPS * 2)
        .map(|b| f64::from_le_bytes(*b) as f32)
        .collect();
    let zs: Vec<String> = values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[x, y]| format!("[{x:?}f32, {y:?}f32]"))
        .collect();
    let x0: Vec<String> = model["x0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| format!("{:?}f32", v.as_f64().unwrap() as f32))
        .collect();

    let source = format!(
        "pub const STEPS: usize = {STEPS};\n\
         pub const F: [[f32; 4]; 4] = {};\n\
         pub const H: [[f32; 4]; 2] = {};\n\
         pub const Q: [[f32; 4]; 4] = {};\n\
         pub const R: [[f32; 2]; 2] = {};\n\
         pub const X0: [f32; 4] = [{}];\n\
         pub const P0: [[f32; 4]; 4] = {};\n\
         pub static ZS: [[f32; 2]; STEPS] = [{}];\n",
        matrix(&model, "f"),
        matrix(&model, "h"),
        matrix(&model, "q"),
        matrix(&model, "r"),
        x0.join(", "),
        matrix(&model, "p0"),
        zs.join(", "),
    );
    fs::write(out.join("s2.rs"), source).unwrap();
}
