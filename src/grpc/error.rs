use tonic::Status;
use crate::models::common::WaCustomError;

impl From<WaCustomError> for Status {
    fn from(error: WaCustomError) -> Self {
        match error {
            WaCustomError::InvalidParams => Status::invalid_argument(error.to_string()),
            WaCustomError::NotFound(msg) => Status::not_found(msg),
            WaCustomError::FsError(msg) => Status::internal(format!("Filesystem error: {}", msg)),
            WaCustomError::DatabaseError(msg) => Status::internal(format!("Database error: {}", msg)),
            WaCustomError::SerializationError(msg) => Status::internal(format!("Serialization error: {}", msg)),
            _ => Status::internal(error.to_string()),
        }
    }
}