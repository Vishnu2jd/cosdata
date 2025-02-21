pub mod collections;
pub mod vectors;
pub mod server;
pub mod error;
pub mod metadata;

pub mod proto {
    tonic::include_proto!("vector_service");

    pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/vector_service.bin"
    ));
}
