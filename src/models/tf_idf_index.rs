use std::{
    fmt::Debug,
    fs::OpenOptions,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering},
        Arc, RwLock,
    },
};

use crate::indexes::hnsw::offset_counter::IndexFileId;

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    common::TSHashTable,
    lazy_item::ProbLazyItem,
    serializer::tf_idf::{TFIDFIndexSerialize, TF_IDF_INDEX_DATA_CHUNK_SIZE},
    types::FileOffset,
    utils::calculate_path,
    versioning::VersionNumber,
};

// Term quotient (upper 16 bits of the hash)
pub type TermQuotient = u16;
// Outer map from term quotients to TermInfo
pub type QuotientMap = TSHashTable<TermQuotient, Arc<TermInfo>>;

pub struct VersionedVec<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub list: Vec<T>,
    pub next: Option<Box<VersionedVec<T>>>,
}

unsafe impl<T: Send> Send for VersionedVec<T> {}
unsafe impl<T: Sync> Sync for VersionedVec<T> {}

impl<T> VersionedVec<T> {
    pub fn new(version: VersionNumber) -> VersionedVec<T> {
        Self {
            serialized_at: RwLock::new(None),
            version,
            list: Vec::new(),
            next: None,
        }
    }

    pub fn push(&mut self, version: VersionNumber, value: T) {
        if self.version == version {
            return self.list.push(value);
        }

        if let Some(next) = &mut self.next {
            next.push(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.push(version, value);
            self.next = Some(new_next);
        }
    }

    pub fn iter(&self) -> VersionedVecIter<'_, T> {
        let iter = self.list.iter();
        VersionedVecIter {
            current_iter: iter,
            next: self.next.as_deref(),
        }
    }

    pub fn len(&self) -> usize {
        let current_len = self.list.len();

        match &self.next {
            Some(next_node) => current_len + next_node.len(),
            None => current_len,
        }
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl VersionedVec<(u32, f32)> {
    pub fn push_sorted(&mut self, version: VersionNumber, value: (u32, f32)) {
        if self.version == version {
            let mut i = self.list.len();
            while i > 0 && self.list[i - 1].0 > value.0 {
                i -= 1;
            }
            self.list.insert(i, value);
            return;
        }

        if let Some(next) = &mut self.next {
            next.push_sorted(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.push_sorted(version, value);
            self.next = Some(new_next);
        }
    }
}

impl<T: PartialEq> PartialEq for VersionedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version && self.list == other.list && self.next == other.next
    }
}

impl<T: Debug> Debug for VersionedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedList")
            .field("version", &self.version)
            .field("list", &self.list)
            .field("next", &self.next)
            .finish()
    }
}

// Iterator over &T
pub struct VersionedVecIter<'a, T> {
    current_iter: std::slice::Iter<'a, T>,
    next: Option<&'a VersionedVec<T>>,
}

impl<'a, T> Iterator for VersionedVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.current_iter.next() {
            return Some(item);
        }

        if let Some(next_node) = self.next {
            self.current_iter = next_node.list.iter();
            self.next = next_node.next.as_deref();
            self.current_iter.next()
        } else {
            None
        }
    }
}

pub struct TermInfo {
    pub documents: RwLock<VersionedVec<(u32, f32)>>,
    pub sequence_idx: u16,
}

impl TermInfo {
    #[allow(unused)]
    pub fn new(sequence_idx: u16, version: VersionNumber) -> Self {
        Self {
            documents: RwLock::new(VersionedVec::new(version)),
            sequence_idx,
        }
    }
}

#[cfg(test)]
impl PartialEq for TermInfo {
    fn eq(&self, other: &Self) -> bool {
        *self.documents.read().unwrap() == *other.documents.read().unwrap()
            && self.sequence_idx == other.sequence_idx
    }
}

#[cfg(test)]
impl std::fmt::Debug for TermInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermInfo")
            .field("documents", &self.documents)
            .field("sequence_idx", &self.sequence_idx)
            .finish()
    }
}

pub struct TFIDFIndexNodeData {
    // Map from term quotients to TermInfo
    pub map: QuotientMap,
    pub map_len: AtomicU16,
    pub num_entries_serialized: RwLock<u16>,
}

#[cfg(test)]
impl PartialEq for TFIDFIndexNodeData {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.map_len.load(Ordering::Relaxed) == other.map_len.load(Ordering::Relaxed)
            && *self.num_entries_serialized.read().unwrap()
                == *other.num_entries_serialized.read().unwrap()
    }
}

#[cfg(test)]
impl std::fmt::Debug for TFIDFIndexNodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TFIDFIndexNodeData")
            .field("map", &self.map)
            .field("map_len", &self.map_len.load(Ordering::Relaxed))
            .field(
                "num_entries_serialized",
                &*self.num_entries_serialized.read().unwrap(),
            )
            .finish()
    }
}

impl Default for TFIDFIndexNodeData {
    fn default() -> Self {
        Self {
            map: QuotientMap::new(16),
            map_len: AtomicU16::new(0),
            num_entries_serialized: RwLock::new(0),
        }
    }
}

impl TFIDFIndexNodeData {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct TFIDFIndexNode {
    pub is_serialized: AtomicBool,
    pub is_dirty: AtomicBool,
    pub file_offset: FileOffset,
    pub dim_index: u32,
    pub data: *mut ProbLazyItem<TFIDFIndexNodeData>,
    pub children: AtomicArray<TFIDFIndexNode, 16>,
}

#[cfg(test)]
impl PartialEq for TFIDFIndexNode {
    fn eq(&self, other: &Self) -> bool {
        self.file_offset == other.file_offset
            && self.dim_index == other.dim_index
            && self.children == other.children
            && unsafe { *self.data == *other.data }
    }
}

#[cfg(test)]
impl std::fmt::Debug for TFIDFIndexNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("TFIDFIndexNode")
            .field("file_offset", &self.file_offset)
            .field("dim_index", &self.dim_index)
            .field("children", &self.children)
            .finish()
    }
}

pub struct TFIDFIndexRoot {
    pub root: TFIDFIndexNode,
    pub cache: TFIDFIndexCache,
    // total number of documents in the index
    pub total_documents_count: AtomicU32,
    pub data_file_parts: u8,
}

#[cfg(test)]
impl PartialEq for TFIDFIndexRoot {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.total_documents_count.load(Ordering::Relaxed)
                == other.total_documents_count.load(Ordering::Relaxed)
            && self.data_file_parts == other.data_file_parts
    }
}

#[cfg(test)]
impl std::fmt::Debug for TFIDFIndexRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TFIDFIndexRoot")
            .field("root", &self.root)
            .field(
                "total_documents_count",
                &self.total_documents_count.load(Ordering::Relaxed),
            )
            .field("data_file_parts", &self.data_file_parts)
            .finish()
    }
}

unsafe impl Send for TFIDFIndexNode {}
unsafe impl Sync for TFIDFIndexNode {}
unsafe impl Send for TFIDFIndexRoot {}
unsafe impl Sync for TFIDFIndexRoot {}

impl TFIDFIndexNode {
    pub fn new(dim_index: u32, file_offset: FileOffset) -> Self {
        let data = ProbLazyItem::new(
            TFIDFIndexNodeData::new(),
            IndexFileId::invalid(),
            FileOffset(file_offset.0 + 4),
        );

        Self {
            is_serialized: AtomicBool::new(false),
            is_dirty: AtomicBool::new(true),
            file_offset,
            dim_index,
            data,
            children: AtomicArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_or_create_node(&self, path: &[u8], mut offset_fn: impl FnMut() -> u32) -> &Self {
        let mut current_node = self;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + (1u32 << (child_index * 2));
            if let Some(child) = current_node.children.get(child_index as usize) {
                let res = unsafe { &*child };
                current_node = res;
                continue;
            }
            let (new_child, _is_newly_created) = current_node
                .children
                .get_or_insert(child_index as usize, || {
                    Box::into_raw(Box::new(Self::new(new_dim_index, FileOffset(offset_fn()))))
                });
            let res = unsafe { &*new_child };
            current_node = res;
        }

        current_node
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &self,
        quotient: TermQuotient,
        value: f32,
        document_id: u32,
        cache: &TFIDFIndexCache,
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        // Get node data
        let data = unsafe { &*self.data }.try_get_data(cache, self.dim_index)?;
        // Get or create inner map for this quotient
        data.map.modify_or_insert(
            quotient,
            |term| {
                term.documents
                    .write()
                    .unwrap()
                    .push_sorted(version, (document_id, value));
            },
            || {
                // Create new inner map if quotient not found
                let mut documents = VersionedVec::new(version);
                let sequence_idx = data.map_len.fetch_add(1, Ordering::Relaxed);
                documents.push(version, (document_id, value));
                Arc::new(TermInfo {
                    documents: RwLock::new(documents),
                    sequence_idx,
                })
            },
        );

        // Mark node as dirty
        self.is_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// See [`crate::models::serializer::tf_idf::node`] for how its calculated
    pub fn get_serialized_size() -> u32 {
        TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 74
    }
}

impl TFIDFIndexRoot {
    pub fn new(root_path: PathBuf, data_file_parts: u8) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = TFIDFIndexNode::get_serialized_size();
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(node_size + 4);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.clone().into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = TFIDFIndexCache::new(dim_bufman, data_bufmans, offset_counter, data_file_parts);

        Ok(TFIDFIndexRoot {
            root: TFIDFIndexNode::new(0, FileOffset(4)),
            cache,
            total_documents_count: AtomicU32::new(0),
            data_file_parts,
        })
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&TFIDFIndexNode> {
        let mut current_node = &self.root;
        let path = calculate_path(dim_index, 0);
        for child_index in path {
            let child = current_node.children.get(child_index as usize)?;
            let node_res = unsafe { &*child };
            current_node = node_res;
        }

        assert_eq!(current_node.dim_index, dim_index);

        Some(current_node)
    }

    // Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(
        &self,
        hash_dim: u32,
        value: f32,
        document_id: u32,
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        // Split the hash dimension
        let storage_dim = hash_dim & (u16::MAX as u32);
        let quotient = (hash_dim >> 16) as TermQuotient;

        let path = calculate_path(storage_dim, 0);
        let node = self.root.find_or_create_node(&path, || {
            self.cache
                .offset_counter
                .fetch_add(TFIDFIndexNode::get_serialized_size(), Ordering::Relaxed)
        });
        debug_assert_eq!(node.dim_index, storage_dim);
        // value will be quantized while being inserted into the Node.
        node.insert(quotient, value, document_id, &self.cache, version)
    }

    pub fn serialize(&self) -> Result<(), BufIoError> {
        let cursor = self.cache.dim_bufman.open_cursor()?;
        self.cache
            .dim_bufman
            .update_u32_with_cursor(cursor, self.total_documents_count.load(Ordering::Relaxed))?;
        self.root.serialize(
            &self.cache.dim_bufman,
            &self.cache.data_bufmans,
            &self.cache.offset_counter,
            0,
            self.data_file_parts,
            cursor,
        )?;
        self.cache.dim_bufman.close_cursor(cursor)?;
        Ok(())
    }

    pub fn deserialize(root_path: PathBuf, data_file_parts: u8) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = TFIDFIndexNode::get_serialized_size();
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(dim_bufman.file_size() as u32);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.clone().into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = TFIDFIndexCache::new(dim_bufman, data_bufmans, offset_counter, data_file_parts);
        let root = TFIDFIndexNode::deserialize(
            &cache.dim_bufman,
            &cache.data_bufmans,
            FileOffset(4),
            0,
            data_file_parts,
            &cache,
        )?;
        let cursor = cache.dim_bufman.open_cursor()?;
        let total_documents_count = AtomicU32::new(cache.dim_bufman.read_u32_with_cursor(cursor)?);
        cache.dim_bufman.close_cursor(cursor)?;

        Ok(Self {
            root,
            cache,
            total_documents_count,
            data_file_parts,
        })
    }
}
