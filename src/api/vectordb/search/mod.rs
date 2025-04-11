use actix_web::{web, Scope};
use controller::{batch_dense_search, dense_search, sparse_search, batch_sparse_search, hybrid_search};

mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
mod repo;
mod service;

pub(crate) fn search_module() -> Scope {
    web::scope("/collections/{collection_id}/search")
        .route("/dense", web::post().to(dense_search))
        .route("/batch-dense", web::post().to(batch_dense_search))
        .route("/sparse", web::post().to(sparse_search))
        .route("/batch-sparse", web::post().to(batch_sparse_search))
        .route("/hybrid", web::post().to(hybrid_search))
}
