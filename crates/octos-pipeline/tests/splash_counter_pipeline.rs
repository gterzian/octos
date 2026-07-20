//! Splash Counter Pipeline integration tests.
//!
//! Tests that the splash_counter.dot pipeline graph can be parsed and
//! validated, and that it produces valid Splash DSL output when executed
//! with the fake provider.


use octos_llm::fake::FakeProvider;
use octos_llm::{ChatConfig, LlmProvider};

/// The Splash counter pipeline DOT definition (imported from assets).
const SPLASH_COUNTER_PIPELINE: &str = include_str!(
    "../../octos-agent/src/assets/pipelines/splash_counter.dot"
);

#[test]
fn should_parse_splash_counter_pipeline() {
    let graph = octos_pipeline::parse_dot(SPLASH_COUNTER_PIPELINE)
        .expect("failed to parse splash_counter.dot");
    assert_eq!(graph.id, "splash_counter");
    assert_eq!(graph.nodes.len(), 2, "expected 2 nodes: generate_app, deploy_app");
    assert_eq!(graph.edges.len(), 1, "expected 1 edge: generate_app -> deploy_app");

    // Check node existence and attributes
    let generate = graph.nodes.get("generate_app")
        .expect("missing generate_app node");
    assert_eq!(generate.label.as_deref(), Some("Generate App"));
    assert!(generate.prompt.is_some());

    let deploy = graph.nodes.get("deploy_app")
        .expect("missing deploy_app node");
    assert_eq!(deploy.label.as_deref(), Some("Deploy App"));
    assert!(deploy.prompt.is_some());
}

#[test]
fn should_validate_splash_counter_pipeline() {
    let graph = octos_pipeline::parse_dot(SPLASH_COUNTER_PIPELINE)
        .expect("failed to parse splash_counter.dot");

    // The 'splash-counter' model is a fake provider model, not in the
    // production model catalog, so validation will produce "known model" errors.
    // This is expected — the pipeline works with the fake provider.
    let diagnostics = octos_pipeline::diagnostics(&graph);
    let model_errors: Vec<_> = diagnostics.iter()
        .filter(|d| d.severity == octos_pipeline::Severity::Error)
        .collect();

    // All errors should be about unknown model (which is expected for fake provider)
    for err in &model_errors {
        assert!(
            err.message.contains("unknown pipeline model"),
            "unexpected error: {:?}", err
        );
    }

    assert!(!model_errors.is_empty(), "expected model validation errors for fake provider");
    assert!(graph.detect_cycles().is_ok(), "pipeline should be acyclic");
}

#[test]
fn should_detect_no_cycles_in_splash_counter() {
    let graph = octos_pipeline::parse_dot(SPLASH_COUNTER_PIPELINE)
        .expect("failed to parse splash_counter.dot");
    assert!(graph.detect_cycles().is_ok());
}

/// Verify the fake provider returns a valid Splash DSL counter app.
#[tokio::test]
async fn fake_provider_returns_splash_counter_app() {
    let provider = FakeProvider::new(None);
    let response = provider
        .chat(&[], &[], &ChatConfig::default())
        .await
        .expect("fake provider should return response");

    let content = response.content.expect("response should have content");

    // Must contain valid Splash DSL structures
    assert!(content.contains("App {"), "should start with App block");
    assert!(content.contains("Window {"), "should have a Window");
    assert!(content.contains("Body {"), "should have a Body");
    assert!(content.contains("VStack {"), "should have VStack layout");
    assert!(content.contains("HStack {"), "should have HStack layout");

    // Must have counter functionality
    assert!(content.contains("increment"), "should have increment action");
    assert!(content.contains("decrement"), "should have decrement action");
    assert!(content.contains("reset"), "should have reset action");
    assert!(content.contains("count_display"), "should have count display widget");

    // Must have agent communication
    assert!(content.contains("__pi_response.set_text"), "should support pi response");
}

/// Verify the fake provider works with streaming.
#[tokio::test]
async fn fake_provider_streams_splash_counter_app() {
    use futures::StreamExt;

    let provider = FakeProvider::new(Some("splash-counter".into()));
    let mut stream = provider
        .chat_stream(&[], &[], &ChatConfig::default())
        .await
        .expect("fake provider should return stream");

    let mut full_text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            octos_llm::StreamEvent::TextDelta(delta) => {
                full_text.push_str(&delta);
            }
            octos_llm::StreamEvent::Done(reason) => {
                assert_eq!(reason, octos_llm::StopReason::EndTurn);
            }
            octos_llm::StreamEvent::Usage(usage) => {
                assert_eq!(usage.input_tokens, 0);
                assert_eq!(usage.output_tokens, 0);
            }
            _ => {}
        }
    }

    assert!(full_text.contains("Window"), "streamed text should contain Window");
    assert!(full_text.contains("increment"), "streamed text should contain increment");
}
