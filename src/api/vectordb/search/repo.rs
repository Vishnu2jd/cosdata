use std::sync::Arc;
use std::collections::HashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    app_context::AppContext,
    api_service::{
        ann_vector_query,
        batch_ann_vector_query,
    },
    models::{
        common::WaCustomError,
        types::{MetricResult, VectorId, SparseVector},
        sparse_ann_query::{SparseAnnQueryBasic, SparseAnnResult},
    },
    indexes::{
        inverted::types::SparsePair,
        inverted::InvertedIndex,
        inverted_idf::InvertedIndexIDF,
    },
    distance::dotproduct::DotProductDistance,
    vector_store::get_sparse_embedding_by_id,
    config_loader::Config,
};
use super::dtos;
use super::error::SearchError;
use crate::metadata::query_filtering::Filter;


#[allow(dead_code)]
pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::DenseSearchRequestDto,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let hnsw_index = ctx.ain_env.collections_map.get_hnsw_index(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("Dense index not found for collection '{}'", collection_id)))?;

     let metadata_filter: Option<Filter> = request.filter;

    ann_vector_query(
        ctx,
        hnsw_index.clone(),
        request.query_vector,
        metadata_filter,
        request.top_k,
    ).await
}

#[allow(dead_code)]
pub(crate) async fn batch_dense_search(
     ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchDenseSearchRequestDto,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
     let hnsw_index = ctx.ain_env.collections_map.get_hnsw_index(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("Dense index not found for collection '{}'", collection_id)))?;

     let metadata_filter: Option<Filter> = request.filter;

    batch_ann_vector_query(
        ctx,
        hnsw_index.clone(),
        request.query_vectors,
        metadata_filter,
        request.top_k,
    ).await
}


#[allow(dead_code)]
pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::SparseSearchRequestDto,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
     if let Some(inverted_index) = ctx.ain_env.collections_map.get_inverted_index(collection_id) {
         // Filter field removed from DTO, no check needed
         let threshold = request.early_terminate_threshold.unwrap_or(ctx.config.search.early_terminate_threshold);
         // Call synchronous helper logic
         sparse_ann_vector_query_logic(&ctx.config, inverted_index.clone(), &request.query_terms, request.top_k, threshold)
     }
     else if let Some(idf_index) = ctx.ain_env.collections_map.get_idf_inverted_index(collection_id) {
        log::debug!("Using IDF index for sparse search in collection '{}'", collection_id);
        let query_sparse_vector = SparseVector { vector_id: u32::MAX, entries: request.query_terms.iter().map(|p| (p.0, p.1)).collect() };
        // Call synchronous search_bm25
        SparseAnnQueryBasic::new(query_sparse_vector)
            .search_bm25(&idf_index.root, request.top_k)
            .map(|idf_results| {
                idf_results.into_iter().map(|res| {
                    (VectorId(res.document_id as u64), MetricResult::DotProductDistance(DotProductDistance(res.score)))
                }).collect()
            })
            .map_err(|e| WaCustomError::BufIo(Arc::new(e)))
     }
     else {
        Err(WaCustomError::NotFound(format!("No sparse index (regular or IDF) found for collection '{}'", collection_id)))
     }
}


fn batch_sparse_idf_search_logic(
    idf_index: Arc<InvertedIndexIDF>,
    queries: &[Vec<SparsePair>],
    top_k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
     queries
        .par_iter()
        .map(|query_terms| {
            let query_sparse_vector = SparseVector { vector_id: u32::MAX, entries: query_terms.iter().map(|p| (p.0, p.1)).collect() };
            SparseAnnQueryBasic::new(query_sparse_vector)
                .search_bm25(&idf_index.root, top_k)
                .map(|idf_results| {
                    idf_results.into_iter().map(|res| {
                        (VectorId(res.document_id as u64), MetricResult::DotProductDistance(DotProductDistance(res.score)))
                    }).collect()
                })
                .map_err(|e| WaCustomError::BufIo(Arc::new(e)))
        })
        .collect()
}


pub(crate) async fn batch_sparse_search(
     ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchSparseSearchRequestDto,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
     if let Some(inverted_index) = ctx.ain_env.collections_map.get_inverted_index(collection_id) {
          let threshold = request.early_terminate_threshold.unwrap_or(ctx.config.search.early_terminate_threshold);
          batch_sparse_ann_vector_query_logic(&ctx.config, inverted_index.clone(), &request.query_terms_list, request.top_k, threshold)
     }
      else if let Some(idf_index) = ctx.ain_env.collections_map.get_idf_inverted_index(collection_id) {
        log::debug!("Using IDF index for batch sparse search in collection '{}'", collection_id);
        batch_sparse_idf_search_logic(idf_index.clone(), &request.query_terms_list, request.top_k)
     }
      else {
        Err(WaCustomError::NotFound(format!("No sparse index (regular or IDF) found for collection '{}'", collection_id)))
     }
}


fn sparse_ann_vector_query_logic(
    config: &Config,
    inverted_index: Arc<InvertedIndex>,
    query: &[SparsePair],
    top_k: Option<usize>,
    early_terminate_threshold: f32,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let sparse_vec = SparseVector { vector_id: u32::MAX, entries: query.iter().map(|pair| (pair.0, pair.1)).collect() };


    let intermediate_results = SparseAnnQueryBasic::new(sparse_vec).sequential_search(
        &inverted_index.root,
        inverted_index.root.root.quantization_bits,
        *inverted_index.values_upper_bound.read().unwrap(),
        early_terminate_threshold,
        if config.rerank_sparse_with_raw_values { config.sparse_raw_values_reranking_factor } else { 1 },
        top_k,
    )?;

    if config.rerank_sparse_with_raw_values {
        finalize_sparse_ann_results(inverted_index, intermediate_results, query, top_k)
    } else {
        Ok(intermediate_results.into_iter().map(|result| {
            (VectorId(result.vector_id as u64), MetricResult::DotProductDistance(DotProductDistance(result.similarity as f32)))
        }).collect())
    }
}

// Synchronous batch helper
fn batch_sparse_ann_vector_query_logic(
    config: &Config,
    inverted_index: Arc<InvertedIndex>,
    queries: &[Vec<SparsePair>],
    top_k: Option<usize>,
    early_terminate_threshold: f32,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .par_iter()
        .map(|query| {
            sparse_ann_vector_query_logic(config, inverted_index.clone(), query, top_k, early_terminate_threshold)
        })
        .collect()
}

// Synchronous finalization helper
fn finalize_sparse_ann_results(
    inverted_index: Arc<InvertedIndex>,
    intermediate_results: Vec<SparseAnnResult>,
    query: &[SparsePair],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));
    for result in intermediate_results {
        let id = VectorId(result.vector_id as u64);
        match get_sparse_embedding_by_id(&inverted_index.lmdb, &inverted_index.vec_raw_manager, &id) {
            Ok(embedding) => {
                let map = embedding.into_map();
                let mut dp = 0.0;
                for pair in query { if let Some(val) = map.get(&pair.0) { dp += val * pair.1; } }
                results.push((id, MetricResult::DotProductDistance(DotProductDistance(dp))));
            }
            Err(WaCustomError::NotFound(_)) => { log::warn!("Sparse embedding ID {} not found during finalization.", id); }
            Err(e) => { log::error!("Error fetching sparse embedding ID {} during finalization: {}", id, e); /* Decide whether to return Err(e) */ }
        }
    }
    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k_val) = k { results.truncate(k_val); }
    Ok(results)
}


pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::HybridSearchRequestDto,
) -> Result<Vec<(VectorId, f32)>, SearchError> {

    let hnsw_index = ctx.ain_env.collections_map.get_hnsw_index(collection_id)
        .ok_or_else(|| SearchError::IndexNotFound("Dense index required for hybrid search.".to_string()))?;

    // Perform Search on *Available* Sparse Index (Synchronous Call)
    let sparse_results: Vec<(VectorId, MetricResult)> =
        if let Some(inverted_index) = ctx.ain_env.collections_map.get_inverted_index(collection_id) {
            let sparse_k = request.top_k * 3;
            let threshold = ctx.config.search.early_terminate_threshold;
            // Call synchronous helper
            sparse_ann_vector_query_logic(&ctx.config, inverted_index.clone(), &request.query_terms, Some(sparse_k), threshold)
                .map_err(|e| SearchError::SearchFailed(format!("Hybrid: Sparse component (regular) failed: {}", e)))?
        } else if let Some(idf_index) = ctx.ain_env.collections_map.get_idf_inverted_index(collection_id) {
             log::debug!("Using IDF index for hybrid sparse component in collection '{}'", collection_id);
             let sparse_k = request.top_k * 3;
             let query_sparse_vector = SparseVector { vector_id: u32::MAX, entries: request.query_terms.iter().map(|p| (p.0, p.1)).collect() };
             // Call synchronous search_bm25
             SparseAnnQueryBasic::new(query_sparse_vector)
                .search_bm25(&idf_index.root, Some(sparse_k))
                .map(|idf_results| { idf_results.into_iter().map(|res| { (VectorId(res.document_id as u64), MetricResult::DotProductDistance(DotProductDistance(res.score))) }).collect() })
                .map_err(|e| SearchError::SearchFailed(format!("Hybrid: Sparse component (IDF) failed: {}", e)))?
        } else {
            return Err(SearchError::IndexNotFound("Sparse index (regular or IDF) required for hybrid search.".to_string()));
        };

    let dense_k = request.top_k * 3;
    let dense_results = ann_vector_query(
        ctx.clone(),
        hnsw_index.clone(),
        request.query_vector,
        None, // Pass None for filter
        Some(dense_k),
    ).await
    .map_err(|e| SearchError::SearchFailed(format!("Hybrid: Dense component failed: {}", e)))?;

    let mut final_scores: HashMap<VectorId, f32> = HashMap::new();
    let constant_k = request.fusion_constant_k;
    if constant_k < 0.0 { log::warn!("RRF fusion_constant_k ({}) is non-positive.", constant_k); }
    for (rank, (vector_id, _score)) in dense_results.iter().enumerate() {
        let score = 1.0 / (rank as f32 + constant_k + f32::EPSILON);
        final_scores.insert(vector_id.clone(), score);
    }
    for (rank, (vector_id, _score)) in sparse_results.iter().enumerate() {
         let score = 1.0 / (rank as f32 + constant_k + f32::EPSILON);
        *final_scores.entry(vector_id.clone()).or_insert(0.0) += score;
    }

    let mut final_results: Vec<(VectorId, f32)> = final_scores.into_iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(request.top_k);

    Ok(final_results)
}
