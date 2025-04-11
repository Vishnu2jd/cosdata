pub(crate) mod data;
pub(crate) mod transaction;
use lmdb::Transaction;

use std::{
    path::PathBuf,
    ptr,
    sync::{atomic::AtomicPtr, RwLock},
};

use transaction::InvertedIndexIDFTransaction;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    inverted_index_idf::InvertedIndexIDFRoot,
    types::{MetaDb, VectorId},
    versioning::{Hash, VersionControl},
};
use crate::macros::key;

pub struct InvertedIndexIDF {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub max_vectors: Option<i32>,
    pub root: InvertedIndexIDFRoot,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: AtomicPtr<InvertedIndexIDFTransaction>,
    pub vcs: VersionControl,
    pub vec_raw_manager: BufferManagerFactory<Hash>,
}

unsafe impl Send for InvertedIndexIDF {}
unsafe impl Sync for InvertedIndexIDF {}

impl InvertedIndexIDF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        auto_create_index: bool,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
        vec_raw_manager: BufferManagerFactory<Hash>,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexIDFRoot::new(root_path, data_file_parts)?;

        Ok(Self {
            name,
            auto_create_index,
            description,
            max_vectors,
            root,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            vec_raw_manager,
        })
    }

    pub fn insert(
        &self,
        hash_dim: u32,
        value: f32,
        document_id: u32,
        version: Hash,
    ) -> Result<(), BufIoError> {
        self.root.insert(hash_dim, value, document_id, version)
    }

    pub fn set_current_version(&self, version: Hash) {
        *self.current_version.write().unwrap() = version;
    }

    pub fn contains_vector_id(&self, vector_id_u32: u32) -> bool {
        let env = self.lmdb.env.clone();
        let db = *self.lmdb.db;
        let txn = match env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => {
                log::error!("LMDB RO txn failed for IDF contains_vector_id check: {}", e);
                return false;
            }
        };

        let vector_id_obj = VectorId(vector_id_u32 as u64);
        let embedding_key = key!(e: &vector_id_obj);

        let found = match txn.get(db, &embedding_key) {
            Ok(_) => true,
            Err(lmdb::Error::NotFound) => false,
            Err(e) => {
                log::error!("LMDB error during IDF contains_vector_id get for {}: {}", vector_id_u32, e);
                false
            }
        };

        txn.abort();
        found
    }
}
