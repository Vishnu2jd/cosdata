use super::CustomSerialize;
use crate::models::chunked_list::LazyItem;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItem<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = match self {
            LazyItem::Ready(item, existing_offset) => {
                let read_guard = existing_offset.read().unwrap();
                if let Some(existing_offset) = *read_guard {
                    writer.seek(SeekFrom::Start(existing_offset as u64))?;
                    item.serialize(writer)?;
                    existing_offset
                } else {
                    drop(read_guard);
                    let offset = item.serialize(writer)?;
                    *existing_offset.write().unwrap() = Some(offset);
                    offset
                }
            }
            LazyItem::LazyLoad(file_offset) => *file_offset,
            LazyItem::Null => u32::MAX,
        };

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let item = T::deserialize(reader, offset)?;
        Ok(LazyItem::Ready(
            Arc::new(item),
            Arc::new(RwLock::new(Some(offset))),
        ))
    }
}