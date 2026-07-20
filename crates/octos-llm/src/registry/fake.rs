use std::sync::Arc;

use eyre::Result;

use crate::fake::FakeProvider;
use crate::provider::LlmProvider;

use super::{CreateParams, ProviderEntry};

pub const ENTRY: ProviderEntry = ProviderEntry {
    name: "fake",
    aliases: &["splash", "mock", "counter"],
    default_model: Some("splash-counter"),
    api_key_env: None,
    key_env_aliases: &[],
    default_base_url: None,
    requires_api_key: false,
    requires_base_url: false,
    requires_model: false,
    detect_patterns: &["splash-counter"],
    create,
};

fn create(_p: CreateParams) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(FakeProvider::new(_p.model)))
}
