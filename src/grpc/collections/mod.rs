use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::app_context::AppContext;
use crate::models::collection::{Collection, DenseVectorOptions, SparseVectorOptions, CollectionConfig};
use crate::models::common::WaCustomError;

use super::proto::collections_service_server::CollectionsService;
use super::proto::{
    Collection as ProtoCollection,
    CreateCollectionRequest, CreateCollectionResponse,
    GetCollectionsRequest, GetCollectionsResponse,
    GetCollectionRequest, DeleteCollectionRequest,
};

pub struct CollectionsServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl CollectionsService for CollectionsServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        // Create new collection
        let collection = Collection::new(
            req.name.clone(),
            req.description.clone(),
            DenseVectorOptions {
                dimension: req.dense_vector.map(|d| d.dimension as usize).unwrap_or(0),
                enabled: req.dense_vector.is_some(),
                auto_create_index: true,
            },
            SparseVectorOptions {
                enabled: req.sparse_vector.is_some(),
                auto_create_index: true,
            },
            req.metadata_schema,
            CollectionConfig {
                max_vectors: None,
                replication_factor: None,
            },
        ).map_err(Status::from)?;

        // Store collection using public method
        self.context.ain_env.collections_map
            .insert_collection(Arc::new(collection.clone()))
            .map_err(Status::from)?;

        Ok(Response::new(CreateCollectionResponse {
            id: collection.name.clone(),
            name: collection.name,
            description: collection.description,
        }))
    }

    async fn get_collections(
        &self,
        _request: Request<GetCollectionsRequest>,
    ) -> Result<Response<GetCollectionsResponse>, Status> {
        let collections: Vec<ProtoCollection> = self.context.ain_env.collections_map
            .iter_collections()
            .map(|entry| ProtoCollection {
                name: entry.key().clone(),
                description: entry.value().description.clone(),
            })
            .collect();

        Ok(Response::new(GetCollectionsResponse { collections }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<ProtoCollection>, Status> {
        let collection = self.context.ain_env.collections_map
            .get_collection(&request.into_inner().id)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        Ok(Response::new(ProtoCollection {
            name: collection.name.clone(),
            description: collection.description.clone(),
        }))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<()>, Status> {
        let collection_id = request.into_inner().id;
        self.context.ain_env.collections_map
            .remove_collection(&collection_id)
            .map_err(|e| match e {
                WaCustomError::NotFound(_) => Status::not_found("Collection not found"),
                _ => Status::internal(format!("Failed to delete collection: {}", e))
            })?;

        Ok(Response::new(()))
    }
}
