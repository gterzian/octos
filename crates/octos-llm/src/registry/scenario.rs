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
//! - **Host-tool role** (offered tools include `launch_splash_app`, i.e. a
//!   Robrix AI-room chat session): deterministic per-session sequence.
//!   Prompt 0 answers `"pong"` through the `send_message` tool (an
//!   obviously identifiable reply that exercises the whole tool-call path,
//!   whatever the user wrote); prompt 1 calls `launch_splash_app` with the
//!   user's request as its `description`; every prompt after that answers
//!   `"pong"` again. The `chat()` that follows a tool result ends the turn
//!   — with a brief `"pong"` echo after `send_message` (the message already
//!   went to the room; Robrix drops the echo), with a summary echoing the
//!   tool output after `launch_splash_app`. This is what makes the model
//!   decide to call a Robrix host tool — the whole point of the ACP
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

/// The other Robrix chat-session tool: posts a plain text message to the
/// session's room. The "pong" answers go through this tool so the whole
/// tool-call path (agent → MCP relay → Robrix host → room) is exercised
/// deterministically on every pong turn.
const SEND_MESSAGE_TOOL: &str = "send_message";

/// The canned Splash app the writer role emits. Deterministic so the whole
/// create→validate→package chain runs offline; deliberately the SIMPLEST
/// thing that renders — a text box saying "this is a mini-app".
///
/// Two constraints shaped it. First, no `glass.*` widgets: the host's
/// isolate prelude may not expose them, and nothing in a canned offline demo
/// needs them. Second, no `Fill` heights: the host mounts a mini-app's
/// Splash at `height: Fit` inside a scroll view (see Robrix's
/// `MiniAppHost`), so every level of a canned app must size itself
/// naturally (`Fit` / fixed / content-derived), exactly as the built-in demo
/// apps (room_peek, room_pins, …) do — a `Fill` root collapses to zero and
/// renders blank.
const CANNED_APP: &str = "// name: Mini App\n// icon: 💬\n// tint: #4466ee\n\
View{\n\
    width: Fill, height: Fit\n\
    flow: Down\n\
    padding: 20\n\
    align: Align{x: 0.5}\n\
    RoundedView{\n\
        width: Fit, height: Fit\n\
        padding: 22\n\
        show_bg: true\n\
        draw_bg +: {\n\
            color: #xEEF5FF\n\
            border_radius: 10.0\n\
        }\n\
        Label{\n\
            width: Fit, height: Fit\n\
            text: \"this is a mini-app\"\n\
            draw_text +: {\n\
                color: #x1C274C\n\
                text_style: theme.font_regular{font_size: 18}\n\
            }\n\
        }\n\
    }\n\
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
        prompts: AtomicUsize::new(0),
    }))
}

struct ScenarioProvider {
    /// The scenario id (the `--model` value), echoed by `model_id()`.
    model: String,
    /// The number of tool calls emitted so far in this session — used to
    /// uniquify tool-call ids across turns (a repeated id could make octos
    /// pair a tool result with the wrong earlier call).
    calls: AtomicUsize,
    /// The number of fresh user prompts answered so far in this session.
    /// The host-tool role's deterministic sequence keys off this (prompt 0 →
    /// pong, prompt 1 → launch, prompt ≥2 → pong), tracked HERE rather than
    /// by scanning message history because a failed turn can make octos drop
    /// its history — which would otherwise make the next prompt look like a
    /// fresh first turn and repeat "pong" instead of advancing.
    prompts: AtomicUsize,
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

/// The name of the tool the model is being asked to continue after: when the
/// newest message is a Tool result, this call must END the turn, and the
/// finishing behaviour differs by which tool ran. Scans back to the newest
/// assistant message that carried a tool call; when history is trimmed and no
/// call is visible, assumes `launch_splash_app` (its summary is the original
/// host-role behaviour).
fn latest_finished_tool(messages: &[Message]) -> Option<String> {
    if !matches!(messages.last(), Some(m) if m.role == MessageRole::Tool) {
        return None;
    }
    let found = messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Assistant)
        .and_then(|m| m.tool_calls.as_ref())
        .and_then(|calls| calls.last())
        .map(|call| call.name.clone());
    Some(found.unwrap_or_else(|| HOST_LAUNCH_TOOL.to_string()))
}

fn tool_use_response(name: &str, arguments: serde_json::Value, id: String) -> ChatResponse {
    ChatResponse {
        content: None,
        reasoning_content: None,
        tool_calls: vec![ToolCall {
            id,
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
        let offered = tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();

        // Host-tool role: the model decides between Robrix's two chat-session
        // tools, deterministically, per fresh prompt; a call that follows a
        // tool result ends the turn instead.
        if offered.contains(&HOST_LAUNCH_TOOL) {
            // Continuing after a tool result: end the turn.
            if let Some(finished) = latest_finished_tool(messages) {
                if finished == SEND_MESSAGE_TOOL {
                    // The "pong" already went to the room through the tool.
                    // octos rejects an EMPTY end-of-turn response (retries,
                    // then fails the turn), so say it once more briefly — the
                    // Robrix runtime drops this trailing text as redundant, so
                    // a tool turn still leaves exactly one `ai_reply`.
                    return Ok(end_turn_response("pong".to_string()));
                }
                let echo = latest_tool_output(messages).unwrap_or_default();
                return Ok(end_turn_response(format!(
                    "The app was built and launched.\n\nTool result: {echo}"
                )));
            }
            // A fresh user prompt: deterministic per-session sequence —
            //   prompt 0: "pong" via send_message
            //   prompt 1: launch the app via launch_splash_app
            //   prompt ≥2: "pong" again, always
            let ask = latest_user_request(messages);
            let prompt_idx = self.prompts.fetch_add(1, Ordering::SeqCst);
            let id = format!("call-scenario-{}", self.calls.fetch_add(1, Ordering::SeqCst));
            let pong_turn = prompt_idx != 1 && offered.contains(&SEND_MESSAGE_TOOL);
            if pong_turn {
                return Ok(tool_use_response(
                    SEND_MESSAGE_TOOL,
                    serde_json::json!({ "text": "pong" }),
                    id,
                ));
            }
            let description = if ask.is_empty() {
                "make a counter app".to_string()
            } else {
                ask
            };
            return Ok(tool_use_response(
                HOST_LAUNCH_TOOL,
                serde_json::json!({ "description": description }),
                id,
            ));
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

    fn tool_result_message(text: &str, id: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: text.to_string(),
            media: vec![],
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
            client_message_id: None,
            thread_id: None,
            timestamp: chrono::Utc::now(),
        }
    }

    /// The assistant half of a tool turn, as octos's agent appends it after a
    /// model tool_use: an assistant message carrying the call.
    fn assistant_tool_message(name: &str, id: &str) -> Message {
        Message {
            role: MessageRole::Assistant,
            content: String::new(),
            media: vec![],
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: serde_json::json!({}),
                metadata: None,
            }]),
            tool_call_id: None,
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
            prompts: AtomicUsize::new(0),
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
            assistant_tool_message(HOST_LAUNCH_TOOL, "call-scenario-0"),
            tool_result_message("{\"status\":\"installed_and_running\"}", "call-scenario-0"),
        ];
        let resp = p.chat(&messages, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        let text = resp.content.unwrap_or_default();
        assert!(text.contains("installed_and_running"), "summary echoes the tool result: {text}");
    }

    /// The AI-room smoke sequence, end to end in one session: prompt 0
    /// answers "pong" through the `send_message` tool (which Robrix posts as
    /// the room's ai_reply; the turn then ends with a brief pong echo that
    /// Robrix drops — octos itself needs a non-empty response). Prompt 1
    /// launches a Splash app through `launch_splash_app`. Prompt 2 (and any
    /// later) answers "pong" again.
    #[tokio::test]
    async fn ai_room_pong_then_launch_then_always_pong() {
        let p = provider();
        let tools = vec![spec(HOST_LAUNCH_TOOL), spec(SEND_MESSAGE_TOOL)];
        let mut history: Vec<Message> = vec![];

        // Prompt 0: whatever the user wrote, answer "pong" via send_message.
        history.push(user_message("hello"));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let call = &resp.tool_calls[0];
        let pong_id = call.id.clone();
        assert_eq!(call.name, SEND_MESSAGE_TOOL);
        assert_eq!(call.arguments["text"], "pong");

        // The agent ran the tool; the follow-up chat() ends the turn with a
        // NON-EMPTY pong echo (octos rejects an empty end-of-turn response).
        history.push(assistant_tool_message(SEND_MESSAGE_TOOL, &pong_id));
        history.push(tool_result_message("posted", &pong_id));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.content.as_deref(), Some("pong"));

        // Prompt 1: launch_splash_app with the request.
        history.push(user_message("make a counter app"));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let call = &resp.tool_calls[0];
        let launch_id = call.id.clone();
        assert_eq!(call.name, HOST_LAUNCH_TOOL);
        assert_eq!(call.arguments["description"], "make a counter app");

        // The generation ran; the follow-up ends the turn with a summary.
        history.push(assistant_tool_message(HOST_LAUNCH_TOOL, &launch_id));
        history.push(tool_result_message("{\"status\":\"installed_and_running\"}", &launch_id));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        let text = resp.content.unwrap_or_default();
        assert!(text.contains("installed_and_running"), "summary echoes the tool result: {text}");

        // Prompt 2 and beyond: "pong" again, always, via send_message.
        history.push(user_message("again"));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let call = &resp.tool_calls[0];
        let pong2_id = call.id.clone();
        assert_eq!(call.name, SEND_MESSAGE_TOOL);
        assert_eq!(call.arguments["text"], "pong");
        history.push(assistant_tool_message(SEND_MESSAGE_TOOL, &pong2_id));
        history.push(tool_result_message("posted", &pong2_id));
        let resp = p.chat(&history, &tools, &ChatConfig::default()).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.content.as_deref(), Some("pong"));
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
        assert!(text.contains("this is a mini-app"));
        assert!(text.contains("name: Mini App"));
    }
}
