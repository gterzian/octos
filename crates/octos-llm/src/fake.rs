//! Fake LLM provider — returns a pre-baked Splash DSL counter app.
//!
//! This provider ignores all input and returns a valid Splash DSL body
//! for a simple counter application. Designed for testing the Makepad
//! channel adapter and pipeline deployment flows without needing a real
//! LLM API key.

use async_trait::async_trait;
use eyre::Result;
use octos_core::{Message, MessageRole, ToolCall};

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, ChatStream, StopReason, StreamEvent, TokenUsage, ToolSpec};

/// The Splash DSL body for a minimal counter application.
///
/// Uses Makepad widget DSL syntax: flat orphan widgets with on_click handlers,
/// let declarations, and ui.name.set_text() for widget references.
/// The Makepad host wraps this in SPLASH_PREFIX/SPLASH_SUFFIX and renders it.
const SPLASH_COUNTER_APP: &str = r#"let count = 0
display := Label{text:"0" draw_text.text_style.font_size:24}
ButtonFlat{text:"-" on_click:||{count -= 1; ui.display.set_text("" + count)}}
ButtonFlat{text:"+" on_click:||{count += 1; ui.display.set_text("" + count)}}
ButtonFlat{text:"Reset" on_click:||{count = 0; ui.display.set_text("0")}}
btn_hello := ButtonFlat{text:"Send to Pi" on_click:||{__pi_response.set_text("Hello from octos! Count is " + count)}}"#;

/// A fake provider that always returns a Splash DSL counter app.
///
/// Ignores messages, tools, and config. Useful for integration testing
/// the Makepad native UI channel without incurring LLM API costs.
pub struct FakeProvider {
    model_id: String,
}

impl FakeProvider {
    pub fn new(model_id: Option<String>) -> Self {
        Self {
            model_id: model_id.unwrap_or_else(|| "splash-counter".into()),
        }
    }

    /// Build a response based on conversation state.
    /// Only calls launch_app on the first turn; subsequent turns just end.
    fn build_response(&self, messages: &[Message]) -> Result<ChatResponse> {
        // Check if there are tool results in the conversation — if so,
        // the launch_app tool was already executed, so just end the turn.
        let has_tool_results = messages.iter().any(|m| matches!(m.role, MessageRole::Tool));

        if has_tool_results {
            Ok(ChatResponse {
                content: Some(
                    "The counter app is now running in the Makepad host window! \
                     You can interact with it using the buttons."
                        .to_string(),
                ),
                reasoning_content: None,
                tool_calls: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    reasoning_tokens: 0,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                provider_index: None,
            })
        } else {
            Ok(ChatResponse {
                content: Some("I'll launch the counter app on the Makepad host!".to_string()),
                reasoning_content: None,
                tool_calls: vec![ToolCall {
                    id: "fake-tc-1".to_string(),
                    name: "launch_app".to_string(),
                    arguments: serde_json::json!({
                        "splash_body": SPLASH_COUNTER_APP,
                        "app_id": "counter-1"
                    }),
                    metadata: None,
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    reasoning_tokens: 0,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                provider_index: None,
            })
        }
    }
}

#[async_trait]
impl LlmProvider for FakeProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _config: &ChatConfig,
    ) -> Result<ChatResponse> {
        self.build_response(messages)
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _config: &ChatConfig,
    ) -> Result<ChatStream> {
        let response = self.build_response(messages)?;
        let mut events: Vec<StreamEvent> = Vec::new();
        if let Some(text) = response.content.clone() {
            events.push(StreamEvent::TextDelta(text));
        }
        for (i, tc) in response.tool_calls.iter().enumerate() {
            events.push(StreamEvent::ToolCallDelta {
                index: i,
                id: Some(tc.id.clone()),
                name: Some(tc.name.clone()),
                arguments_delta: tc.arguments.to_string(),
            });
        }
        events.push(StreamEvent::Usage(response.usage));
        events.push(StreamEvent::Done(response.stop_reason));
        Ok(Box::pin(futures::stream::iter(events)))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider_name(&self) -> &str {
        "fake"
    }

    fn context_window(&self) -> u32 {
        4096
    }

    fn max_output_tokens(&self) -> u32 {
        2048
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn should_call_launch_app_on_first_turn() {
        let provider = FakeProvider::new(None);
        // Empty messages = first turn
        let response = provider
            .chat(&[], &[], &ChatConfig::default())
            .await
            .unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "launch_app");
    }

    #[tokio::test]
    async fn should_end_turn_after_tool_executed() {
        let provider = FakeProvider::new(None);
        // Simulate messages with a tool result using the available constructors
        let messages = vec![
            Message::assistant("I'll launch the counter app"),
            Message::tool_with_thread(
                "App launched successfully",
                "tc-1",
                octos_core::ThreadId::new("test"),
            ),
        ];
        let response = provider
            .chat(&messages, &[], &ChatConfig::default())
            .await
            .unwrap();
        assert_eq!(response.tool_calls.len(), 0, "should not call tool again");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn should_include_splash_in_tool_call_arguments() {
        let provider = FakeProvider::new(None);
        let response = provider
            .chat(&[], &[], &ChatConfig::default())
            .await
            .unwrap();
        let tc = &response.tool_calls[0];
        let splash = tc.arguments["splash_body"].as_str().unwrap();
        assert!(splash.contains("let count"));
        assert!(splash.contains("ButtonFlat{"));
        assert!(splash.contains("__pi_response.set_text"));
    }

    #[tokio::test]
    async fn should_stream_launch_app_then_end() {
        let provider = FakeProvider::new(Some("test-model".into()));
        let mut stream = provider
            .chat_stream(&[], &[], &ChatConfig::default())
            .await
            .unwrap();

        let mut seen_tool = false;
        let mut seen_done = false;
        while let Some(event) = stream.next().await {
            match &event {
                StreamEvent::ToolCallDelta { name, .. } => {
                    if name.as_deref() == Some("launch_app") {
                        seen_tool = true;
                    }
                }
                StreamEvent::Done(_) => seen_done = true,
                _ => {}
            }
        }
        assert!(seen_tool, "should have tool call in stream");
        assert!(seen_done, "should have done event");
    }

    #[test]
    fn should_expose_model_metadata() {
        let provider = FakeProvider::new(Some("splash-v1".into()));
        assert_eq!(provider.model_id(), "splash-v1");
        assert_eq!(provider.provider_name(), "fake");
        assert_eq!(provider.context_window(), 4096);
        assert_eq!(provider.max_output_tokens(), 2048);
    }

    #[test]
    fn should_use_default_model_id_when_none_given() {
        let provider = FakeProvider::new(None);
        assert_eq!(provider.model_id(), "splash-counter");
    }
}
