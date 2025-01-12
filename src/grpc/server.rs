use std::sync::Arc;
use tonic::transport::Server;
use log::{info, error};

use crate::app_context::AppContext;
use super::collections::CollectionsServiceImpl;
use super::proto::collections_service_server::CollectionsServiceServer;

pub async fn start_grpc_server(context: Arc<AppContext>) {
    let addr = "[::1]:50051".parse().unwrap();

    let collections_service = CollectionsServiceImpl {
        context: context.clone(),
    };

    info!("gRPC server listening on {}", addr);

    match Server::builder()
        .add_service(CollectionsServiceServer::new(collections_service))
        .serve(addr)
        .await
    {
        Ok(_) => info!("gRPC server shutdown gracefully"),
        Err(e) => error!("gRPC server error: {}", e),
    }
}
