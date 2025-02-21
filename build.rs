use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .file_descriptor_set_path(out_dir.join("vector_service.bin"))
        .compile(
            &["proto/vector_service.proto"],
            &["proto/"],
        )?;
    Ok(())
}
