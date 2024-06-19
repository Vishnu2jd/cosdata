use super::common::{tuple_to_string, WaCustomError};
use super::types::{
    HNSWLevel, NeighbourRef, Node, NodeFileRef, NodeProp, NodeRef, VectorId, VectorQt, VectorStore,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_cbor;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Seek, SeekFrom, Write};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

// Assuming the fixed size for neighbors and quant_vec
const MAX_NEIGHBORS: usize = 10; // Adjust as needed
const MAX_QUANT_VEC: usize = 10; // Adjust as needed

// persist structures

type NodePersistRef = (u32, u32); // (file_number, offset)

#[derive(Clone, Serialize, Deserialize)]
pub struct NeighbourPersist {
    pub node: NodePersistRef,
    pub cosine_similarity: f32,
}
#[derive(Serialize, Deserialize)]
pub struct NodePersist {
    // prop is not serialized in this context
    // #[serde(skip_serializing)]
    // pub prop: Arc<NodeProp>,
    pub prop_location: NodePersistRef,
    pub hnsw_level: HNSWLevel,
    pub neighbors: Vec<NeighbourPersist>,
    pub parent: Option<NodePersistRef>,
    pub child: Option<NodePersistRef>,
}

impl NodePersist {
    pub fn new(
        hnsw_level: HNSWLevel,
        location: NodePersistRef,
        neighbors: Vec<NeighbourPersist>,
        parent: Option<NodePersistRef>,
        child: Option<NodePersistRef>,
    ) -> NodePersist {
        NodePersist {
            // prop,
            hnsw_level,
            neighbors,
            parent,
            child,
            prop_location: location,
        }
    }
}

pub fn persist_node_prop_update_loc(
    prop_file: Arc<File>,
    wal_file: Arc<File>,
    node: NodeRef,
    prop: Arc<NodeProp>,
    hnsw_level: HNSWLevel,
) -> Result<(), WaCustomError> {
    let prop_location = write_prop_to_file(&prop, &prop_file);

    persist_node_update_loc(prop_location, wal_file, node, hnsw_level)
}

pub fn persist_node_update_loc(
    prop_location: NodeFileRef,
    wal_file: Arc<File>,
    node: NodeRef,
    hnsw_level: HNSWLevel,
) -> Result<(), WaCustomError> {
    // Lock the Mutex to access the neighbors
    println!(" while Node {} having nie {:?}", node.prop.id, node.neighbors);
    let neighbors_lock = node
        .neighbors
        .read()
        .map_err(|_| WaCustomError::MutexPoisoned("convert_node_to_node_persist".to_owned()))?;

    // Convert neighbors from NodeRef to NodePersistRef
    let neighbors: Result<Vec<NeighbourPersist>, _> = neighbors_lock
        .iter()
        .map(|neighbor| match neighbor {
            NeighbourRef::Ready {
                node: nodex,
                cosine_similarity,
            } => match nodex.get_location() {
                Some(loca) => Ok(NeighbourPersist {
                    node: loca,
                    cosine_similarity: *cosine_similarity,
                }),
                None => Err(WaCustomError::InvalidLocationNeighborEncountered(
                    "neighbours loop".to_owned(),
                    nodex.prop.id.clone(),
                )),
            },
            NeighbourRef::Pending(x) => Err(WaCustomError::PendingNeighborEncountered(
                tuple_to_string(*x),
            )),
        })
        .collect();

    // Handle the Result of the neighbors conversion
    let neighbors = neighbors?;

    // Convert parent and child
    let parent = node
        .parent
        .read()
        .unwrap()
        .as_ref()
        .map(|parent_node| parent_node.get_location().unwrap());
    let child = node
        .child
        .read()
        .unwrap()
        .as_ref()
        .map(|child_node| child_node.get_location().unwrap());

    let mut nprst = NodePersist {
        hnsw_level,
        neighbors,
        parent,
        child,
        prop_location,
    };

    let file_loc = write_node_to_file(&mut nprst, &wal_file);
    node.set_prop_location(prop_location);
    node.set_location(file_loc);
    return Ok(());
}

pub fn map_node_persist_ref_to_node(
    vec_store: VectorStore,
    node_ref: NodePersistRef,
    cosine_similarity: f32,
    vec_level: HNSWLevel,
    vec_id: VectorId,
) -> NeighbourRef {
    // logic to map NodePersistRef to Node
    //
    match vec_store.cache.get(&(vec_level, vec_id)) {
        Some(nodex) => {
            return NeighbourRef::Ready {
                node: nodex.value().clone(),
                cosine_similarity,
            }
        }
        None => return NeighbourRef::Pending(node_ref),
    };
}

pub fn load_node_from_node_persist(
    vec_store: VectorStore,
    node_persist: NodePersist,
    persist_loc: NodeFileRef,
    prop: Arc<NodeProp>,
) -> NodeRef {
    // Convert neighbors from NodePersistRef to NeighbourRef
    let neighbors_result: Vec<NeighbourRef> = node_persist
        .neighbors
        .iter()
        .map(|nref| {
            map_node_persist_ref_to_node(
                vec_store.clone(),
                nref.node,
                nref.cosine_similarity,
                node_persist.hnsw_level,
                prop.id.clone(),
            )
        })
        .collect();

    // Wrap neighbors in Arc<Mutex<Vec<NeighbourRef>>>
    let neighbors = Arc::new(RwLock::new(neighbors_result));

    // Convert parent and child
    let parent = if let Some(parent_ref) = node_persist.parent {
        vec_store
            .cache
            .get(&(node_persist.hnsw_level, prop.id.clone()))
            .map(|node| node.value().clone())
    } else {
        None
    };
    let parent = Arc::new(RwLock::new(parent));

    let child = if let Some(child_ref) = node_persist.child {
        vec_store
            .cache
            .get(&(node_persist.hnsw_level, prop.id.clone()))
            .map(|node| node.value().clone())
    } else {
        None
    };
    let child = Arc::new(RwLock::new(child));

    // Create and return NodeRef
    Arc::new(Node {
        prop,
        location: Arc::new(RwLock::new(Some(persist_loc))),
        prop_location: Arc::new(RwLock::new(Some(node_persist.prop_location))),
        neighbors,
        parent,
        child,
    })
}

pub fn write_prop_to_file(prop: &NodeProp, mut file: &File) -> (u32, u32) {
    let mut prop_bytes = Vec::new();
    //let result = encode(&prop);
    let result = serde_cbor::to_vec(&prop).unwrap();

    prop_bytes.extend_from_slice(result.as_ref());

    file.write_all(&prop_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - prop_bytes.len() as u64;
    (offset as u32, prop_bytes.len() as u32)
}

// fn write_to_end_of_file(file_path: &str, data: &[u8]) -> std::io::Result<(u64, usize)> {
//     let mut file = OpenOptions::new().append(true).open(file_path)?;
//     let offset = file.seek(SeekFrom::End(0))?;
//     file.write_all(data)?;
//     Ok((offset, data.len()))
// }

pub fn write_node_to_file(node: &mut NodePersist, mut file: &File) -> (u32, u32) {
    let mut node_bytes = Vec::new();
    let result = serde_cbor::to_vec(&node);
    if let Err(err) = result {
        panic!("Failed to CBOR encode NodePersist: {}", err);
    }
    node_bytes.extend_from_slice(result.unwrap().as_ref());

    file.write_all(&node_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - node_bytes.len() as u64;
    (offset as u32, node_bytes.len() as u32)
}

pub fn write_node_to_file_at_offset(
    node: &mut NodePersist,
    filename: &str,
    offset: u64,
) -> (u32, u32) {
    let mut node_bytes = Vec::new();
    let result = serde_cbor::to_vec(&node);
    if let Err(err) = result {
        panic!("Failed to CBOR encode NodePersist: {}", err);
    }
    node_bytes.extend_from_slice(result.unwrap().as_ref());

    let mut file = OpenOptions::new()
        .write(true)
        .open(filename)
        .expect("Failed to open file for writing");

    // Seek to the specified offset before writing
    file.seek(SeekFrom::Start(offset as u64))
        .expect("Failed to seek in file");

    file.write_all(&node_bytes)
        .expect("Failed to write to file");
    let written_bytes = node_bytes.len() as u32;
    (offset as u32, written_bytes)
}