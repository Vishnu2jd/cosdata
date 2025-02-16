use crate::{
    distance::{
        cosine::{CosineDistance, CosineSimilarity},
        dotproduct::DotProductDistance,
        euclidean::EuclideanDistance,
        hamming::HammingDistance,
    },
    models::{
        buffered_io::{BufIoError, BufferManager},
        types::{FileOffset, MetricResult},
    },
};
use std::io;

use super::SimpleSerialize;

// @SERIALIZED_SIZE: 1 byte for the tag + 4 bytes for the value = 5 bytes
impl SimpleSerialize for MetricResult {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let (variant, value) = match self {
            Self::CosineSimilarity(value) => (0, value.0),
            Self::CosineDistance(value) => (1, value.0),
            Self::EuclideanDistance(value) => (2, value.0),
            Self::HammingDistance(value) => (3, value.0),
            Self::DotProductDistance(value) => (4, value.0),
        };
        let start = bufman.cursor_position(cursor)? as u32;
        bufman.update_u8_with_cursor(cursor, variant)?;
        bufman.update_f32_with_cursor(cursor, value)?;
        Ok(start)
    }

    fn deserialize(
        bufman: &BufferManager,
        FileOffset(offset): FileOffset,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let variant = bufman.read_u8_with_cursor(cursor)?;
        let value = bufman.read_f32_with_cursor(cursor)?;
        let metric = match variant {
            0 => Self::CosineSimilarity(CosineSimilarity(value)),
            1 => Self::CosineDistance(CosineDistance(value)),
            2 => Self::EuclideanDistance(EuclideanDistance(value)),
            3 => Self::HammingDistance(HammingDistance(value)),
            4 => Self::DotProductDistance(DotProductDistance(value)),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid MetricResult variant: {}", variant),
                )
                .into());
            }
        };
        bufman.close_cursor(cursor)?;
        Ok(metric)
    }
}
