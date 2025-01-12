mod api_service;
mod app_context;
pub mod macros;
mod models;
mod vector_store;
mod web_server;
pub(crate) mod api;
pub mod config_loader;
pub mod cosql;
pub mod distance;
pub mod indexes;
pub mod quantization;
pub mod storage;
pub mod grpc;

use std::sync::Arc;
use tokio::spawn;
use web_server::run_actix_server;
use crate::{
    models::common::*,
    app_context::AppContext,
    config_loader::Config,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = config_loader::load_config();

    let context = Arc::new(AppContext::new(config)?);

    let grpc_context = context.clone();
    spawn(async move {
        grpc::server::start_grpc_server(grpc_context).await;
    });

    if let Err(e) = run_actix_server() {
        eprintln!("HTTP server error: {}", e);
    }

    Ok(())
}
