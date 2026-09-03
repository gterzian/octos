//! `scenario`: a deterministic, key-less LLM provider for offline testing.
//!
//! No network, no credentials, no randomness. Useful wherever a real model
//! would otherwise be required to exercise the plumbing: Robrix's manual
//! end-to-end "make a counter app" run, octos-cli acceptance tests against a
//! live agent loop, CI that must not depend on a provider being reachable.
//! Configure it exactly like any other provider (`provider: "scenario"` in
//! config.json, or `--provider scenario`), and it plays BOTH roles a
//! generation needs:
//!
//! - **Host-tool role** (offered tools include `launch_splash_app`): the
//!   first `chat()` returns a `ToolUse` for that tool, passing the user's
//!   request verbatim as its `description`; once the tool result comes back
//!   the next `chat()` ends the turn with a summary. This is what makes the
//!   model decide to call a Robrix host tool — the whole point of the ACP
//!   `session/new` `mcpServers` plumbing.
//! - **Writer role** (no host tool offered): ends the turn immediately with a
//!   canned, valid Splash app in a ```splash fence — deterministic source the
//!   generation pipeline can validate and package offline.
//!
//! Choose the scenario with `model`: `scenario` (the default) auto-detects
//! the role from the offered tools. The distinction is exact in Robrix's
//! flow: the chat-session agent is the only one ever offered
//! `launch_splash_app`, while the pipeline's code-writing agent sees only
//! built-in file tools.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use eyre::Result;
use octos_core::{Message, MessageRole, ToolCall};

use crate::config::ChatConfig;
use crate::provider::LlmProvider;
use crate::types::{ChatResponse, StopReason, TokenUsage, ToolSpec};

use super::{CreateParams, ProviderEntry};

/// The host tool whose presence marks the host-tool (chat-session) role:
/// the tool Robrix's own MCP server exposes for building mini-apps.
const HOST_LAUNCH_TOOL: &str = "launch_splash_app";

/// The canned Splash app the writer role emits. Deterministic so the whole
/// create→validate→package chain runs offline; mirrors the app `fake_acp`
/// (Robrix's offline ACP stand-in) emits for the same purpose.
const CANNED_APP: &str = "// name: Counter\n// icon: 🔢\n// tint: #4466ee\n\
let count = 0\n\
fn show(){ ui.count.set_text(\"\" + count) }\n\
View{\n\
    width: Fill height: Fill flow: Down spacing: 12 padding: 16\n\
    align: Align{x: 0.5 y: 0.5}\n\
    glass.H1{text: \"Counter\"}\n\
    count := Label{text: \"0\"}\n\
    glass.GlassButton{text: \"+1\" width: 90 height: 44 on_click: || { count += 1 show() }}\n\
}";

pub const ENTRY: ProviderEntry = ProviderEntry {
    name: "scenario",
    aliases: &[],
    default_model: Some("scenario"),
    api_key_env: None,
    key_env_aliases: &[],
    default_base_url: None,
    requires_api_key: false,
    requires_base_url: false,
    requires_model: false,
    // Never auto-detected from a model name — a user must opt in explicitly.
    detect_patterns: &[],
    create,
};

fn create(p: CreateParams) -> Result<std::sync::Arc<dyn LlmProvider>> {
    let model = p.model.unwrap_or_else(|| "scenario".to_string());
    Ok(std::sync::Arc::new(ScenarioProvider {
        model,
        calls: AtomicUsize::new(0),
    }))
}

struct ScenarioProvider {
    /// The scenario id (the `--model` value), echoed by `model_id()`.
    model: String,
    /// `chat()` call index for this session: 0 is the first turn's first call.
    calls: AtomicUsize,
}

/// The user's own latest words, for use as a tool `description`/reply echo.
/// Looks for the request markers Robrix's prompt builders emit, falling back
/// to the last User message's full text (a /ai session prompt IS the user's
/// words).
fn latest_user_request(messages: &[Message]) -> String {
    for message in messages.iter().rev() {
        if message.role == MessageRole::User {
            for marker in ["User's change request:", "User request:"] {
                if let Some(at) = message.content.rfind(marker) {
                    return message.content[at + marker.len()..].trim().to_string();
                }
            }
            return message.content.trim().to_string();
        }
    }
    String::new()
}

/// The text of the last Tool result, if one is present (used to echo what the
/// host tool returned in the turn's final summary).
fn latest_tool_output(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Tool)
        .map(|m| m.content.trim().to_string())
}

fn tool_use_response(name: &str, arguments: serde_json::Value) -> ChatResponse {
    ChatResponse {
        content: None,
        reasoning_content: None,
        tool_calls: vec![ToolCall {
            id: "call-scenario-1".to_string(),
            name: name.to_string(),
            arguments,
            metadata: None,
        }],
        stop_reason: StopReason::ToolUse,
        usage: TokenUsage::default(),
        provider_index: None,
    }
}

fn end_turn_response(text: String) -> ChatResponse {
    ChatResponse {
        content: Some(text),
        reasoning_content: None,
        tool_calls: vec![],
        stop_reason: StopReason::EndTurn,
        usage: TokenUsage::default(),
        provider_index: None,
    }
}

#[async_trait]
impl LlmProvider for ScenarioProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        _config: &ChatConfig,
    ) -> Result<ChatResponse> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let offered = tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();

        // Host-tool role: the first call of the first turn decides to build
        // the mini-app the user asked for; everything after the tool ran ends
        // the turn with a summary echoing the tool result.
        if offered.contains(&HOST_LAUNCH_TOOL) {
            if call == 0 {
                let description = latest_user_request(messages);
                return Ok(tool_use_response(
                    HOST_LAUNCH_TOOL,
                    serde_json::json!({ "description": description }),
                ));
            }
            let echo = latest_tool_output(messages).unwrap_or_default();
            return Ok(end_turn_response(format!(
                "The app was built and launched.\n\nTool result: {echo}"
            )));
        }

        // Writer role (or any agent with no host tools): end with the canned
        // app in a fenced block, deterministic source the pipeline can
        // validate and package offline.
        Ok(end_turn_response(format!("```splash\n{CANNED_APP}\n```")))
    }

    fn context_window(&self) -> u32 {
        128_000
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "scenario"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use octos_core::MessageRole;

    fn user_message(text: &str) -> Message {
        Message {
            role: MessageRole::User,
            content: text.to_string(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            client_message_id: None,
            thread_id: None,
            timestamp: chrono::Utc::now(),
        }
    }

    fn tool_result_message(text: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: text.to_string(),
            media: vec![],
            tool_calls: None,
            tool_call_id: Some("call-scenario-1".into()),
            reasoning_content: None,
            client_message_id: None,
            thread_id: None,
            timestamp: chrono::Utc::now(),
        }
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    fn provider() -> ScenarioProvider {
        ScenarioProvider {
            model: "scenario".into(),
            calls: AtomicUsize::new(0),
        }
    }

    #[tokio::test]
    async fn host_tool_role_calls_launch_splash_app_with_the_user_request() {
        let p = provider();
        let tools = vec![spec(HOST_LAUNCH_TOOL), spec("send_room_message")];
        let messages = vec![user_message("make a counter app")];
        let resp = p.chat(&messages, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let call = &resp.tool_calls[0];
        assert_eq!(call.name, HOST_LAUNCH_TOOL);
        assert_eq!(call.arguments["description"], "make a counter app");

        // After the tool ran, the same session ends the turn with a summary.
        let messages = vec![
            user_message("make a counter app"),
            tool_result_message("{\"status\":\"installed_and_running\"}"),
        ];
        let resp = p.chat(&messages, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        let text = resp.content.unwrap_or_default();
        assert!(text.contains("installed_and_running"), "summary echoes the tool result: {text}");
    }

    #[tokio::test]
    async fn writer_role_emits_a_fenced_splash_app() {
        let p = provider();
        let tools = vec![spec("write_file"), spec("read_file")];
        let messages = vec![user_message("build a counter app")];
        let resp = p.chat(&messages, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        let text = resp.content.unwrap_or_default();
        assert!(text.contains("```splash"), "writer emits a fenced splash block");
        assert!(text.contains("name: Counter"));
    }
}
