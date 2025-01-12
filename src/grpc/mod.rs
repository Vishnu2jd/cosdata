pub mod collections;
pub mod server;
pub mod error;

pub mod proto {
    tonic::include_proto!("vector_service");
}
