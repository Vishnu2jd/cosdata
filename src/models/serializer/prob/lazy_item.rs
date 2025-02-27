use std::{collections::HashSet, sync::atomic::Ordering};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::FileIndex,
    prob_lazy_load::lazy_item::{ProbLazyItemState, ReadyState},
    prob_node::SharedNode,
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

impl ProbSerialize for SharedNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
        is_level_0: bool,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        match lazy_item.unsafe_get_state() {
            ProbLazyItemState::Pending(file_index) => Ok(file_index.get_offset().unwrap().0),
            ProbLazyItemState::Ready(ReadyState {
                data,
                file_offset,
                persist_flag,
                version_id,
                is_serialized,
                ..
            }) => {
                let bufman = if is_level_0 {
                    level_0_bufmans.get(*version_id)?
                } else {
                    bufmans.get(*version_id)?
                };

                if is_serialized.load(Ordering::Acquire) {
                    if !persist_flag.swap(false, Ordering::Acquire) {
                        return Ok(file_offset.0);
                    }

                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        bufman.open_cursor()?
                    };

                    data.update_serialized(
                        bufmans,
                        level_0_bufmans,
                        *version_id,
                        *file_offset,
                        cursor,
                        is_level_0,
                    )?;

                    if version_id != &version {
                        bufman.close_cursor(cursor)?;
                    }
                } else {
                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        bufman.open_cursor()?
                    };

                    bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;

                    is_serialized.store(true, Ordering::Release);
                    persist_flag.store(false, Ordering::Release);

                    data.serialize(bufmans, level_0_bufmans, *version_id, cursor, is_level_0)?;

                    if version_id != &version {
                        bufman.close_cursor(cursor)?;
                    }
                }

                Ok(file_offset.0)
            }
        }
    }

    fn deserialize(
        _bufmans: &BufferManagerFactory<Hash>,
        _level_0_bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm, is_level_0)
    }
}
