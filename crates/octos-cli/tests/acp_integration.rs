//! End-to-end integration test for the `octos acp` bridge.
//!
//! Drives the real ACP agent (the exact handler wiring `octos acp` uses) with a
//! `MockLlm` — a canned [`LlmProvider`] that returns a fixed assistant reply —
//! through the full `initialize -> session/new -> session/prompt` round-trip,
//! and asserts the streamed `session/update` sequence plus the final stop
//! reason.
//!
//! No network, no subprocess, no OS pipes: the ACP client and the octos ACP
//! agent are wired together **in-process**. [`OctosAcpAgentTransport`] exposes
//! the octos agent as a `ConnectTo<Client>` transport; the client's
//! `connect_with` closure drives the protocol while the agent's handlers run in
//! the background — all on one tokio runtime, exchanging typed JSON-RPC messages
//! directly.

use std::sync::Arc;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, LoadSessionRequest, NewSessionRequest, PromptRequest,
    SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::{Client, ConnectionTo};

use async_trait::async_trait;
use octos_cli::commands::{OctosAcpAgentTransport, TestAgentFactory};
use tokio::sync::Mutex;

/// A canned LLM that always returns the same assistant text and ends the turn —
/// mirrors the `MockLlm` pattern from `chat.rs`'s unit tests.
struct MockLlm {
    reply: String,
}

#[async_trait]
impl octos_llm::LlmProvider for MockLlm {
    async fn chat(
        &self,
        _messages: &[octos_core::Message],
        _tools: &[octos_llm::ToolSpec],
        _config: &octos_llm::ChatConfig,
    ) -> eyre::Result<octos_llm::ChatResponse> {
        Ok(octos_llm::ChatResponse {
            content: Some(self.reply.clone()),
            reasoning_content: None,
            tool_calls: vec![],
            stop_reason: octos_llm::StopReason::EndTurn,
            usage: octos_llm::TokenUsage::default(),
            provider_index: None,
        })
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn model_id(&self) -> &str {
        "mock-1"
    }
}

/// A `MockLlm` that returns a per-call assistant reply and records, for every
/// `chat()` call, the full set of message *contents* it was handed (system +
/// accumulated history + the current user turn). The integration test inspects
/// these snapshots to prove multi-turn history accumulates across prompts.
struct RecordingLlm {
    /// One entry per `chat()` call: the `content` of every incoming message.
    seen: Arc<Mutex<Vec<Vec<String>>>>,
    /// Assistant reply text keyed by call index (0-based); falls back to a
    /// generic reply if the call index is beyond the vec.
    replies: Vec<String>,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl octos_llm::LlmProvider for RecordingLlm {
    async fn chat(
        &self,
        messages: &[octos_core::Message],
        _tools: &[octos_llm::ToolSpec],
        _config: &octos_llm::ChatConfig,
    ) -> eyre::Result<octos_llm::ChatResponse> {
        let idx = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.seen
            .lock()
            .await
            .push(messages.iter().map(|m| m.content.clone()).collect());
        let reply = self
            .replies
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("reply-{idx}"));
        Ok(octos_llm::ChatResponse {
            content: Some(reply),
            reasoning_content: None,
            tool_calls: vec![],
            stop_reason: octos_llm::StopReason::EndTurn,
            usage: octos_llm::TokenUsage::default(),
            provider_index: None,
        })
    }

    fn provider_name(&self) -> &str {
        "recording-mock"
    }

    fn model_id(&self) -> &str {
        "recording-1"
    }
}

/// Pull the text out of an assistant-message `session/update`, if that's what it
/// is.
fn agent_message_text(update: &SessionUpdate) -> Option<String> {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Pull the text out of a user-message `session/update`, if that's what it is.
fn user_message_text(update: &SessionUpdate) -> Option<String> {
    match update {
        SessionUpdate::UserMessageChunk(chunk) => match &chunk.content {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        },
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn should_stream_assistant_message_and_end_turn_when_driven_through_acp_initialize_new_session_prompt()
 {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();

    let reply = "Hello from octos ACP";
    let llm: Arc<dyn octos_llm::LlmProvider> = Arc::new(MockLlm {
        reply: reply.to_string(),
    });
    let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
    let transport = OctosAcpAgentTransport::new(factory);

    // Records every `session/update` the agent streams to the client.
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();

    // The captured stop reason from the prompt turn.
    let stop_reason: Arc<Mutex<Option<StopReason>>> = Arc::new(Mutex::new(None));
    let stop_reason_for_main = stop_reason.clone();

    let prompt_cwd = cwd.clone();

    // Build the in-process ACP CLIENT and drive the protocol from its
    // `connect_with` closure. The octos ACP agent is the transport.
    let client_result = Client
        .builder()
        .name("octos-acp-test-client")
        .on_receive_notification(
            async move |notif: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| {
                updates_for_handler.lock().await.push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            transport,
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                // 1) initialize
                let init = connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                assert_eq!(init.protocol_version, ProtocolVersion::V1);
                // octos advertises text-only prompt capabilities.
                assert!(!init.agent_capabilities.prompt_capabilities.image);

                // 2) session/new
                let new_session = connection
                    .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                    .block_task()
                    .await?;
                let session_id = new_session.session_id;

                // 3) session/prompt
                let prompt = connection
                    .send_request(PromptRequest::new(
                        session_id.clone(),
                        vec![ContentBlock::Text(TextContent::new("hello"))],
                    ))
                    .block_task()
                    .await?;

                *stop_reason_for_main.lock().await = Some(prompt.stop_reason);
                Ok(())
            },
        )
        .await;

    client_result.expect("ACP client run should complete cleanly");

    // The turn ended naturally. `StopReason` is `Copy`, so deref instead of clone.
    let got_stop = *stop_reason.lock().await;
    assert!(
        matches!(got_stop, Some(StopReason::EndTurn)),
        "expected StopReason::EndTurn, got {got_stop:?}"
    );

    // The assistant's canned reply was streamed as an AgentMessageChunk.
    let recorded = updates.lock().await;
    let assistant_texts: Vec<String> = recorded.iter().filter_map(agent_message_text).collect();
    assert!(
        assistant_texts.iter().any(|t| t.contains(reply)),
        "expected an AgentMessageChunk containing {reply:?}; recorded updates: {recorded:?}"
    );
}

/// Multi-turn history must ACCUMULATE across prompts: turn N's `process_message`
/// must see the messages from all earlier turns. This is driven over the real
/// ACP handler wiring (`initialize -> session/new -> prompt x3`) with a
/// `RecordingLlm` that snapshots the messages it is handed each turn.
///
/// `ConversationResponse.messages` for a text-only (no-tool) `EndTurn` is just
/// the turn's user message (the assistant's final text is streamed live and
/// carried in `content`, not re-persisted as a history `Message`), so history
/// accumulation is observable via the USER turns surviving.
///
/// The buggy `*h = resp.messages` (replace) only surfaces from the THIRD turn:
/// after turn 1 the stored history is [user1] and after turn 2 it is REPLACED
/// by [user2], dropping user1 — so turn 3 would no longer see user1. A
/// two-prompt drive can't catch it (turn 2 still sees user1 under the bug); we
/// therefore drive three prompts and assert turn 3 still sees user1, and that
/// the incoming message count grows every turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn should_accumulate_conversation_history_across_multiple_prompts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();

    // Distinct, greppable replies so we can assert turn-1 content survives.
    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm: Arc<dyn octos_llm::LlmProvider> = Arc::new(RecordingLlm {
        seen: seen.clone(),
        replies: vec![
            "ASSISTANT_ONE".to_string(),
            "ASSISTANT_TWO".to_string(),
            "ASSISTANT_THREE".to_string(),
        ],
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
    let transport = OctosAcpAgentTransport::new(factory);

    let prompt_cwd = cwd.clone();

    Client
        .builder()
        .name("octos-acp-history-client")
        .on_receive_notification(
            async move |_notif: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| { Ok(()) },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            transport,
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let new_session = connection
                    .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                    .block_task()
                    .await?;
                let session_id = new_session.session_id;

                for user in ["USER_ONE", "USER_TWO", "USER_THREE"] {
                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new(user))],
                        ))
                        .block_task()
                        .await?;
                    assert!(
                        matches!(prompt.stop_reason, StopReason::EndTurn),
                        "each turn should EndTurn; got {:?}",
                        prompt.stop_reason
                    );
                }
                Ok(())
            },
        )
        .await
        .expect("ACP client run should complete cleanly");

    let snapshots = seen.lock().await;
    assert_eq!(
        snapshots.len(),
        3,
        "exactly one chat() call per prompt turn"
    );

    // Flatten each turn's incoming messages into one blob for content checks.
    let blob = |i: usize| snapshots[i].join("\n");

    // Turn 1 sees only turn-1's user prompt (no prior turns).
    assert!(blob(0).contains("USER_ONE"), "turn 1 sees its own user msg");
    assert!(
        !blob(0).contains("USER_TWO"),
        "turn 1 cannot see a future turn's user msg"
    );

    // Turn 3 MUST still see turn-1's user prompt — the core regression:
    // replacing (instead of appending) history drops user1 by turn 3.
    let t3 = blob(2);
    assert!(
        t3.contains("USER_ONE"),
        "turn 3 must still see turn 1's user msg; history was dropped. turn-3 messages: {:?}",
        snapshots[2]
    );
    assert!(
        t3.contains("USER_TWO"),
        "turn 3 must also see turn 2's user msg. turn-3 messages: {:?}",
        snapshots[2]
    );
    assert!(t3.contains("USER_THREE"), "turn 3 sees its own user msg");

    // codex round-2 regression: the ASSISTANT reply must also persist. A
    // text-only turn carries the final reply in `resp.content`, NOT in
    // `resp.messages`, so without explicitly persisting it the agent remembers
    // what the USER said but not what IT answered. Turns 2 and 3 must see turn
    // 1's assistant reply ("ASSISTANT_ONE").
    assert!(
        blob(1).contains("ASSISTANT_ONE"),
        "turn 2 must see turn 1's ASSISTANT reply; assistant text not persisted. turn-2 messages: {:?}",
        snapshots[1]
    );
    assert!(
        t3.contains("ASSISTANT_ONE") && t3.contains("ASSISTANT_TWO"),
        "turn 3 must see the assistant replies from turns 1 and 2. turn-3 messages: {:?}",
        snapshots[2]
    );

    // The accumulated history the LLM receives grows every turn (user + assistant
    // per turn): [sys, u1] -> [sys, u1, a1, u2] -> [sys, u1, a1, u2, a2, u3].
    assert!(
        snapshots[0].len() < snapshots[1].len() && snapshots[1].len() < snapshots[2].len(),
        "incoming message count must grow across turns: {} < {} < {}",
        snapshots[0].len(),
        snapshots[1].len(),
        snapshots[2].len()
    );
}

/// Regression: `octos acp` speaks ACP JSON-RPC on stdout, so NOTHING else may be
/// written there — a single stray log line makes strict clients (Zed) reject the
/// whole stream with a `-32700` parse error. octos's tracing previously defaulted
/// to stdout for no-log-dir commands; this drives the REAL binary through
/// `initialize` + `session/new` (which loads config → emits startup logs) and
/// asserts every stdout line is valid JSON.
#[test]
fn should_emit_only_valid_json_on_stdout_when_running_acp() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    let tmp = tempfile::tempdir().expect("tempdir");
    // Fully isolate the child from the developer/CI account: its data, config,
    // and auth dirs resolve to temp subdirs so the test can neither read nor
    // mutate real user state (codex). A fresh config home also still exercises
    // startup logging under RUST_LOG=info, so a stdout leak would surface.
    let home = tmp.path().join("home");
    let config = tmp.path().join("config");
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&config).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_octos"))
        .args([
            "acp",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-chat",
            "--cwd",
            tmp.path().to_str().unwrap(),
        ])
        .env("RUST_LOG", "info") // force startup logging so a leak would show
        .env("HOME", &home)
        .env("OCTOS_HOME", &data)
        .env("OCTOS_CONFIG_DIR", &config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn octos acp");

    let mut stdin = child.stdin.take().unwrap();
    // initialize + session/new — the latter triggers config load (the logs that
    // used to leak). No LLM call, so no provider key is needed.
    stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":1,\"clientCapabilities\":{}}}\n")
        .unwrap();
    stdin
        .write_all(format!("{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"session/new\",\"params\":{{\"cwd\":\"{}\",\"mcpServers\":[]}}}}\n", tmp.path().to_str().unwrap()).as_bytes())
        .unwrap();
    stdin.flush().unwrap();

    // Read stdout in a thread and report each response promptly. Waiting for
    // a response rather than sleeping for a fixed startup interval avoids
    // racing a cold binary start on loaded CI workers.
    let stdout = child.stdout.take().unwrap();
    let (line_tx, line_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !line.trim().is_empty() && line_tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut lines = Vec::new();
    while lines.len() < 2 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match line_rx.recv_timeout(remaining) {
            Ok(line) => lines.push(line),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait(); // reap the child so it can't linger as a zombie
    drop(line_rx);
    let _ = handle.join();

    assert!(
        !lines.is_empty(),
        "octos acp should have emitted at least the initialize/session responses on stdout"
    );
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line).unwrap_or_else(|e| {
            panic!("non-JSON line on the ACP stdout stream (would -32700 in Zed): {line:?} ({e})")
        });
    }
}

/// A conversation must survive the process that created it.
///
/// ACP sessions were memory-only: `AcpSession.history` was a `Mutex<Vec<Message>>`
/// and nothing wrote it anywhere. A kill -9 or a supervisor restart left the agent
/// with no idea what it had been doing, while the fleet reported it healthy —
/// silent amnesia mid-task, which is worse than staying down.
///
/// This drives two independent transports over one store: the first holds a
/// conversation, the second loads that session id and must see the earlier turns.
#[tokio::test]
async fn should_restore_a_conversation_through_session_load() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    let sessions_dir = tmp.path().join("sessions");
    std::fs::create_dir_all(&memory_dir).unwrap();

    // ---- first process: one turn, then drop the transport entirely ----
    let seen_one: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm_one: Arc<dyn octos_llm::LlmProvider> = Arc::new(RecordingLlm {
        seen: seen_one.clone(),
        replies: vec!["REMEMBER_THIS".to_string()],
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let factory_one = TestAgentFactory::new(llm_one, memory_dir.clone(), cwd.clone())
        .with_session_store(&sessions_dir);

    let cwd_one = cwd.clone();
    let session_id = Client
        .builder()
        .name("octos-acp-persist-1")
        .on_receive_notification(
            async move |_n: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| Ok(()),
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory_one),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let new_session = connection
                    .send_request(NewSessionRequest::new(cwd_one.clone()))
                    .block_task()
                    .await?;
                let id = new_session.session_id.clone();
                connection
                    .send_request(PromptRequest::new(
                        id.clone(),
                        vec![ContentBlock::from("FIRST_QUESTION")],
                    ))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(id)
            },
        )
        .await
        .expect("first session");

    // ---- second process: same store, same id, fresh everything else ----
    let seen_two: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm_two: Arc<dyn octos_llm::LlmProvider> = Arc::new(RecordingLlm {
        seen: seen_two.clone(),
        replies: vec!["SECOND_REPLY".to_string()],
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let factory_two =
        TestAgentFactory::new(llm_two, memory_dir, cwd.clone()).with_session_store(&sessions_dir);

    let cwd_two = cwd.clone();
    let id_two = session_id.clone();
    Client
        .builder()
        .name("octos-acp-persist-2")
        .on_receive_notification(
            async move |_n: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| Ok(()),
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory_two),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                connection
                    .send_request(LoadSessionRequest::new(id_two.clone(), cwd_two.clone()))
                    .block_task()
                    .await?;
                connection
                    .send_request(PromptRequest::new(
                        id_two.clone(),
                        vec![ContentBlock::from("SECOND_QUESTION")],
                    ))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(())
            },
        )
        .await
        .expect("second session");

    // The second process's LLM must have been handed the first turn as context.
    let calls = seen_two.lock().await;
    let last = calls.last().expect("the reloaded session ran a turn");
    let joined = last.join("\n");
    assert!(
        joined.contains("FIRST_QUESTION"),
        "session/load did not restore the earlier user turn; saw: {joined}"
    );
    assert!(
        joined.contains("REMEMBER_THIS"),
        "session/load did not restore the earlier assistant turn; saw: {joined}"
    );
}

/// #1909: `session/load` must REPLAY the restored transcript as `session/update`
/// notifications — the response carries no history, so without a replay a
/// reconnecting client renders an empty conversation even though the agent
/// restored it (the client then prompts from zero context and the agent
/// answers as if it forgot everything).
///
/// Same two-transport shape as `should_restore_a_conversation_through_session_load`,
/// but the second client RECORDS the updates and asserts the earlier turns were
/// replayed by the time `session/load` responded.
#[tokio::test]
async fn should_replay_stored_history_as_session_updates_on_load() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    let sessions_dir = tmp.path().join("sessions");
    std::fs::create_dir_all(&memory_dir).unwrap();

    // ---- first process: one persisted turn ----
    let llm_one: Arc<dyn octos_llm::LlmProvider> = Arc::new(MockLlm {
        reply: "REMEMBER_THIS".to_string(),
    });
    let factory_one = TestAgentFactory::new(llm_one, memory_dir.clone(), cwd.clone())
        .with_session_store(&sessions_dir);

    let cwd_one = cwd.clone();
    let session_id = Client
        .builder()
        .name("octos-acp-replay-1")
        .on_receive_notification(
            async move |_n: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| Ok(()),
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory_one),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let new_session = connection
                    .send_request(NewSessionRequest::new(cwd_one.clone()))
                    .block_task()
                    .await?;
                let id = new_session.session_id.clone();
                connection
                    .send_request(PromptRequest::new(
                        id.clone(),
                        vec![ContentBlock::from("FIRST_QUESTION")],
                    ))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(id)
            },
        )
        .await
        .expect("first session");

    // ---- second process: same store, load, recording every session/update ----
    let llm_two: Arc<dyn octos_llm::LlmProvider> = Arc::new(MockLlm {
        reply: "SECOND_REPLY".to_string(),
    });
    let factory_two =
        TestAgentFactory::new(llm_two, memory_dir, cwd.clone()).with_session_store(&sessions_dir);

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();
    let id_two = session_id.clone();
    let cwd_two = cwd.clone();
    Client
        .builder()
        .name("octos-acp-replay-2")
        .on_receive_notification(
            async move |n: SessionNotification, _cx: ConnectionTo<agent_client_protocol::Agent>| {
                updates_for_handler.lock().await.push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory_two),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                // By the time session/load RESPONDS, the replay must have
                // landed — the frames precede the response on the connection.
                connection
                    .send_request(LoadSessionRequest::new(id_two.clone(), cwd_two.clone()))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(())
            },
        )
        .await
        .expect("second session");

    let recorded = updates.lock().await;
    let user_texts: Vec<String> = recorded.iter().filter_map(user_message_text).collect();
    let agent_texts: Vec<String> = recorded.iter().filter_map(agent_message_text).collect();
    assert!(
        user_texts.iter().any(|t| t.contains("FIRST_QUESTION")),
        "session/load must replay the user turn as a UserMessageChunk; recorded: {recorded:?}"
    );
    assert!(
        agent_texts.iter().any(|t| t.contains("REMEMBER_THIS")),
        "session/load must replay the assistant turn as an AgentMessageChunk; recorded: {recorded:?}"
    );
}

/// `session/load` must not evict a session that is already live.
///
/// A bare insert dropped the existing entry and with it the `shutdown` flag an
/// in-flight turn watches, so a `session/cancel` for that turn flipped a flag
/// nobody read and cancellation silently stopped working. Loading an id the agent
/// already holds is a no-op.
#[tokio::test]
async fn should_not_evict_a_live_session_on_reload() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    let sessions_dir = tmp.path().join("sessions");
    std::fs::create_dir_all(&memory_dir).unwrap();

    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm: Arc<dyn octos_llm::LlmProvider> = Arc::new(RecordingLlm {
        seen: seen.clone(),
        replies: vec!["FIRST".to_string(), "SECOND".to_string()],
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let factory =
        TestAgentFactory::new(llm, memory_dir, cwd.clone()).with_session_store(&sessions_dir);

    let prompt_cwd = cwd.clone();
    Client
        .builder()
        .name("octos-acp-reload-client")
        .on_receive_notification(
            async move |_n: SessionNotification,
                        _cx: ConnectionTo<agent_client_protocol::Agent>| { Ok(()) },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let new_session = connection
                    .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                    .block_task()
                    .await?;
                let id = new_session.session_id;

                connection
                    .send_request(PromptRequest::new(
                        id.clone(),
                        vec![ContentBlock::from("TURN_ONE")],
                    ))
                    .block_task()
                    .await?;

                // Reload the id the agent is already holding.
                connection
                    .send_request(LoadSessionRequest::new(id.clone(), prompt_cwd.clone()))
                    .block_task()
                    .await?;

                // The session must still work, and still carry its earlier turn.
                connection
                    .send_request(PromptRequest::new(
                        id.clone(),
                        vec![ContentBlock::from("TURN_TWO")],
                    ))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(())
            },
        )
        .await
        .expect("reload should not break the session");

    let calls = seen.lock().await;
    let last = calls.last().expect("second turn ran");
    let joined = last.join("\n");
    assert!(
        joined.contains("TURN_ONE"),
        "reloading a live session lost its history; saw: {joined}"
    );
}

/// #1909: a stored transcript must be SANITIZED on load, before it touches the
/// LLM or the client. The store is append-only JSONL, so a crash can leave an
/// assistant tool_call whose result never landed — feeding that back to the
/// provider is a 400 ("tool_use without tool_result"), and replaying it renders
/// a tool card that never completes in the client. Every other resume path runs
/// `ResumePolicy`; ACP's `session/load` must too.
#[tokio::test]
async fn should_sanitize_stored_history_when_loading_a_session() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cwd = tmp.path().to_path_buf();
    let memory_dir = tmp.path().join("memory");
    let sessions_dir = tmp.path().join("sessions");
    std::fs::create_dir_all(&memory_dir).unwrap();

    let session_id = agent_client_protocol::schema::v1::SessionId::new("octos-seeded");
    let key = octos_core::SessionKey::with_profile(
        octos_core::MAIN_PROFILE_ID,
        "acp",
        session_id.0.as_ref(),
    );

    // Seed the store directly: one healthy user + assistant pair, one assistant
    // tool_call with NO matching result (crash residue), one whitespace-only
    // assistant row. All thread-stamped so the fail-closed write path accepts
    // them.
    {
        let mut mgr = octos_bus::session::SessionManager::open(&sessions_dir).expect("open store");
        let msg = |role: octos_core::MessageRole, content: &str| octos_core::Message {
            role,
            content: content.into(),
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            client_message_id: None,
            thread_id: Some("t-seed".into()),
            timestamp: chrono::Utc::now(),
        };
        mgr.add_message(&key, msg(octos_core::MessageRole::User, "SEED_USER"))
            .await
            .expect("user row");
        let mut ghost = msg(octos_core::MessageRole::Assistant, "");
        ghost.tool_calls = Some(vec![octos_core::ToolCall {
            id: "ghost-call".into(),
            name: "shell".into(),
            arguments: serde_json::json!({}),
            metadata: None,
        }]);
        mgr.add_message(&key, ghost)
            .await
            .expect("ghost tool-call row");
        mgr.add_message(&key, msg(octos_core::MessageRole::Assistant, "   "))
            .await
            .expect("whitespace row");
        mgr.add_message(
            &key,
            msg(octos_core::MessageRole::Assistant, "SEED_ASSISTANT"),
        )
        .await
        .expect("assistant row");
    }

    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm: Arc<dyn octos_llm::LlmProvider> = Arc::new(RecordingLlm {
        seen: seen.clone(),
        replies: vec!["AFTER_LOAD".to_string()],
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let factory =
        TestAgentFactory::new(llm, memory_dir, cwd.clone()).with_session_store(&sessions_dir);

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();
    let id_for_client = session_id.clone();
    let cwd_for_client = cwd.clone();
    Client
        .builder()
        .name("octos-acp-sanitize-client")
        .on_receive_notification(
            async move |n: SessionNotification, _cx: ConnectionTo<agent_client_protocol::Agent>| {
                updates_for_handler.lock().await.push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            OctosAcpAgentTransport::new(factory),
            |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                connection
                    .send_request(LoadSessionRequest::new(
                        id_for_client.clone(),
                        cwd_for_client.clone(),
                    ))
                    .block_task()
                    .await?;
                connection
                    .send_request(PromptRequest::new(
                        id_for_client.clone(),
                        vec![ContentBlock::from("NEXT_QUESTION")],
                    ))
                    .block_task()
                    .await?;
                Ok::<_, agent_client_protocol::Error>(())
            },
        )
        .await
        .expect("load + prompt");

    // The replay must not contain a ToolCall frame for the unresolved call —
    // the client would render a tool card that never completes.
    let recorded = updates.lock().await;
    assert!(
        !recorded
            .iter()
            .any(|u| matches!(u, SessionUpdate::ToolCall(_))),
        "the unresolved tool call must be sanitized away before replay; recorded: {recorded:?}"
    );
    let user_texts: Vec<String> = recorded.iter().filter_map(user_message_text).collect();
    let agent_texts: Vec<String> = recorded.iter().filter_map(agent_message_text).collect();
    assert!(
        user_texts.iter().any(|t| t.contains("SEED_USER")),
        "healthy rows still replay; recorded: {recorded:?}"
    );
    assert!(
        agent_texts.iter().any(|t| t.contains("SEED_ASSISTANT")),
        "healthy rows still replay; recorded: {recorded:?}"
    );

    // And the LLM must never see the orphan rows on the next prompt: no
    // empty/whitespace-only payloads in the handed-over transcript.
    let calls = seen.lock().await;
    let last = calls.last().expect("a turn ran after load");
    assert!(
        last.iter().any(|c| c.contains("SEED_USER"))
            && last.iter().any(|c| c.contains("SEED_ASSISTANT")),
        "healthy rows reach the LLM: {last:?}"
    );
    let blanks = last.iter().filter(|c| c.trim().is_empty()).count();
    assert_eq!(
        blanks, 0,
        "sanitized history must not hand the LLM payload-free rows: {last:?}"
    );
}

// ── Client-advertised MCP servers (`session/new` `mcpServers`) ──────────────
//
// A client can point the agent at MCP servers in `session/new`; octos must
// connect to them per session and expose their tools to the model. E2E over
// the REAL handler wiring (`spawn_acp_agent`, the same one `octos acp`
// serves): the client advertises a canned stdio MCP server (a shell script,
// the same shape as Robrix's `--mcp-bridge` relay), and a scripted LLM calls
// the advertised tool — a full model → registry → rmcp → subprocess → back
// round trip, observable through the ACP `session/update` stream.

#[cfg(unix)]
mod client_mcp_e2e {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    use agent_client_protocol::schema::v1::{McpServer, McpServerStdio, ToolCallContent};
    use octos_llm::{ChatResponse, LlmProvider, TokenUsage};

    /// Canned stdio MCP server (a shell script) that answers `initialize`,
    /// `tools/list` (advertising one tool named `{tool_name}`) and
    /// `tools/call`. Notification frames are skipped.
    fn write_mcp_server(dir: &Path, tool_name: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("client-mcp-server.sh");
        let script = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *initialized*) : ;;
    *initialize*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"fake-mcp","version":"1.0.0"}}}\n' "$id"
      ;;
    *tools/list*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"%s","description":"a fake MCP tool for acp tests","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id" "{tool_name}"
      ;;
    *tools/call*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"fake-mcp-ran"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#
        .replace("{tool_name}", tool_name);
        std::fs::write(&path, script).expect("write client mcp server");
        let mut perms = std::fs::metadata(&path).expect("stat client mcp server").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod client mcp server");
        path
    }

    /// A scripted LLM: the FIRST `chat()` asks for one tool call to
    /// `tool_name`; every later call ends the turn. Records the tool specs
    /// each call was offered, so the test can assert the MCP tool was
    /// registered and advertised to the model.
    struct ToolCallingLlm {
        tool_name: String,
        final_reply: String,
        calls: AtomicUsize,
        /// One entry per `chat()` call: the names of the tools offered.
        seen_tools: Arc<Mutex<Vec<Vec<String>>>>,
    }

    #[async_trait]
    impl LlmProvider for ToolCallingLlm {
        async fn chat(
            &self,
            _messages: &[octos_core::Message],
            tools: &[octos_llm::ToolSpec],
            _config: &octos_llm::ChatConfig,
        ) -> eyre::Result<ChatResponse> {
            self.seen_tools
                .lock()
                .await
                .push(tools.iter().map(|t| t.name.clone()).collect());
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                return Ok(ChatResponse {
                    content: None,
                    reasoning_content: None,
                    tool_calls: vec![octos_core::ToolCall {
                        id: "call-fake-mcp-1".to_string(),
                        name: self.tool_name.clone(),
                        arguments: serde_json::json!({}),
                        metadata: None,
                    }],
                    stop_reason: octos_llm::StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    provider_index: None,
                });
            }
            Ok(ChatResponse {
                content: Some(self.final_reply.clone()),
                reasoning_content: None,
                tool_calls: vec![],
                stop_reason: octos_llm::StopReason::EndTurn,
                usage: TokenUsage::default(),
                provider_index: None,
            })
        }

        fn provider_name(&self) -> &str {
            "tool-calling-mock"
        }

        fn model_id(&self) -> &str {
            "tool-calling-mock-1"
        }
    }

    /// `session/new` advertises an MCP server; the model must be offered its
    /// tool and a prompt that calls it must round trip through the stdio
    /// server child and back to the client as a completed tool call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn should_register_and_call_a_client_mcp_tool_advertised_in_session_new() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cwd = tmp.path().to_path_buf();
        let memory_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();
        let server_script = write_mcp_server(tmp.path(), "fake_client_tool");

        let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn LlmProvider> = Arc::new(ToolCallingLlm {
            tool_name: "fake_client_tool".to_string(),
            final_reply: "the counter app is running".to_string(),
            calls: AtomicUsize::new(0),
            seen_tools: seen.clone(),
        });
        let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
        let transport = OctosAcpAgentTransport::new(factory);

        let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_for_handler = updates.clone();
        let stop_reason: Arc<Mutex<Option<StopReason>>> = Arc::new(Mutex::new(None));
        let stop_for_main = stop_reason.clone();
        let prompt_cwd = cwd.clone();
        let script_for_client = server_script.display().to_string();

        Client
            .builder()
            .name("octos-acp-client-mcp-client")
            .on_receive_notification(
                async move |notif: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| {
                    updates_for_handler.lock().await.push(notif.update);
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                transport,
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;

                    // session/new WITH the client's stdio MCP server.
                    let new_session = connection
                        .send_request(
                            NewSessionRequest::new(prompt_cwd.clone()).mcp_servers(vec![
                                McpServer::Stdio(McpServerStdio::new(
                                    "robrix-tools",
                                    script_for_client.clone(),
                                )),
                            ]),
                        )
                        .block_task()
                        .await?;
                    let session_id = new_session.session_id;

                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new("make a counter app"))],
                        ))
                        .block_task()
                        .await?;
                    *stop_for_main.lock().await = Some(prompt.stop_reason);
                    Ok(())
                },
            )
            .await
            .expect("ACP client run should complete cleanly");

        // The turn ended naturally after the tool round trip.
        assert!(
            matches!(*stop_reason.lock().await, Some(StopReason::EndTurn)),
            "expected EndTurn after the tool round trip"
        );

        // The MCP tool was REGISTERED: the model's first chat() was offered it.
        let snapshots = seen.lock().await;
        assert!(
            snapshots
                .first()
                .is_some_and(|tools| tools.iter().any(|t| t == "fake_client_tool")),
            "the client-advertised MCP tool must be offered to the model; offered: {snapshots:?}"
        );

        // The tool call itself streamed to the client as an ACP ToolCall…
        let recorded = updates.lock().await;
        let tool_calls: Vec<&str> = recorded
            .iter()
            .filter_map(|u| match u {
                SessionUpdate::ToolCall(call) => Some(call.title.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tool_calls.contains(&"fake_client_tool"),
            "the model's call to the MCP tool must stream as a ToolCall update; got {tool_calls:?}"
        );

        // …the fake server's reply reached the model (a completed update
        // carries its text content)…
        let completions: Vec<String> = recorded
            .iter()
            .filter_map(|u| match u {
                SessionUpdate::ToolCallUpdate(u) => u.fields.content.as_ref().and_then(|c| {
                    c.iter().find_map(|chunk| match chunk {
                        ToolCallContent::Content(block) => match &block.content {
                            ContentBlock::Text(t) => Some(t.text.clone()),
                            _ => None,
                        },
                        _ => None,
                    })
                }),
                _ => None,
            })
            .collect();
        assert!(
            completions.iter().any(|t| t.contains("fake-mcp-ran")),
            "the tool result from the stdio server must reach the model; got {completions:?}"
        );

        // …and the turn finished with the model's summary text.
        let agent_texts: Vec<String> = recorded.iter().filter_map(agent_message_text).collect();
        assert!(
            agent_texts.iter().any(|t| t.contains("the counter app is running")),
            "the final assistant reply must stream; got {agent_texts:?}"
        );
    }
}

// ── Host notifications: `session/notify` (octos extension, Robrix P3) ──────
//
// `session/notify` pushes host event context into a live session WITHOUT a
// user turn. These tests drive the real handler wiring over the in-process
// transport: an idle notify must ack without any LLM call, its events must
// reach the model on the next prompt as `system` content (never `user`),
// `auto_respond: true` must start a turn on an idle session, `if_busy`
// drop/fold must behave against an in-flight turn, and notified events must
// survive a `session/load`+resume cycle without ever surfacing as the user.
mod host_notify_e2e {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;

    use agent_client_protocol::schema::v1::SessionId;
    use octos_cli::commands::{NotifyIfBusy, NotifyRequest};
    use octos_core::MessageRole;
    use octos_llm::LlmProvider;

    /// One `chat()` snapshot: every incoming message's role and content, so a
    /// test can prove notified events ride the transcript as `System` context
    /// and never as a `user` turn.
    type SeenMessages = Arc<Mutex<Vec<Vec<(MessageRole, String)>>>>;

    /// A `RecordingLlm` that also snapshots each message's ROLE, so tests can
    /// prove notified events ride the transcript as system context — and never
    /// as a user turn.
    struct RoleRecordingLlm {
        seen: SeenMessages,
        replies: Vec<String>,
        calls: Arc<AtomicUsize>,
        /// When set, the FIRST `chat()` announces entry on `entered` and then
        /// blocks until `release`, so a test can hold a turn in flight while it
        /// drives `session/notify`.
        entered: Option<Arc<tokio::sync::Notify>>,
        release: Option<Arc<tokio::sync::Notify>>,
    }

    #[async_trait]
    impl LlmProvider for RoleRecordingLlm {
        async fn chat(
            &self,
            messages: &[octos_core::Message],
            _tools: &[octos_llm::ToolSpec],
            _config: &octos_llm::ChatConfig,
        ) -> eyre::Result<octos_llm::ChatResponse> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().await.push(
                messages
                    .iter()
                    .map(|m| (m.role, m.content.clone()))
                    .collect(),
            );
            if idx == 0 {
                if let Some(entered) = &self.entered {
                    entered.notify_one();
                    if let Some(release) = &self.release {
                        release.notified().await;
                    }
                }
            }
            let reply = self
                .replies
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("reply-{idx}"));
            Ok(octos_llm::ChatResponse {
                content: Some(reply),
                reasoning_content: None,
                tool_calls: vec![],
                stop_reason: octos_llm::StopReason::EndTurn,
                usage: octos_llm::TokenUsage::default(),
                provider_index: None,
            })
        }

        fn provider_name(&self) -> &str {
            "role-recording-mock"
        }

        fn model_id(&self) -> &str {
            "role-recording-1"
        }
    }

    fn new_role_llm(seen: SeenMessages, replies: Vec<String>) -> Arc<dyn LlmProvider> {
        Arc::new(RoleRecordingLlm {
            seen,
            replies,
            calls: Arc::new(AtomicUsize::new(0)),
            entered: None,
            release: None,
        })
    }

    async fn wait_until(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(15);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Poll the session's busy state by sending probe notifies (default
    /// `if_busy: drop`, so a busy session discards them; once idle the probes
    /// land as harmless System rows) until one reports `busy: false`.
    async fn wait_until_idle(
        connection: &ConnectionTo<agent_client_protocol::Agent>,
        session_id: &SessionId,
    ) -> octos_cli::commands::NotifyResponse {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let ack = connection
                .send_request(NotifyRequest {
                    session_id: session_id.clone(),
                    events: vec!["__idle_probe".to_string()],
                    auto_respond: false,
                    if_busy: NotifyIfBusy::Drop,
                })
                .block_task()
                .await
                .expect("idle probe notify is answered");
            if !ack.busy {
                return ack;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the session to become idle"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// `session/notify` on an idle session is acked immediately (`queued:
    /// true`, `busy: false`) with NO LLM call; the events reach the model on
    /// the next prompt as System-role context and are never surfaced as a user
    /// message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_ack_idle_notify_without_llm_call_and_see_events_on_next_prompt() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cwd = tmp.path().to_path_buf();
        let memory_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();

        let seen: SeenMessages = Arc::new(Mutex::new(Vec::new()));
        let calls: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(RoleRecordingLlm {
            seen: seen.clone(),
            replies: vec!["REPLY_AFTER_NOTIFY".to_string()],
            calls: calls.clone(),
            entered: None,
            release: None,
        });
        let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
        let transport = OctosAcpAgentTransport::new(factory);

        let prompt_cwd = cwd.clone();
        Client
            .builder()
            .name("octos-acp-notify-idle-client")
            .on_receive_notification(
                async move |_n: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| {
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                transport,
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let new_session = connection
                        .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                        .block_task()
                        .await?;
                    let session_id = new_session.session_id;

                    let ack = connection
                        .send_request(NotifyRequest {
                            session_id: session_id.clone(),
                            events: vec!["BUILD_FINISHED_EVENT".to_string()],
                            auto_respond: false,
                            if_busy: NotifyIfBusy::Drop,
                        })
                        .block_task()
                        .await?;
                    assert!(ack.queued, "idle notify must queue the events");
                    assert!(!ack.busy, "idle session must report busy: false");

                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        0,
                        "an idle context-only notify must never start an LLM call"
                    );

                    let prompt = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::from("USER_AFTER_NOTIFY")],
                        ))
                        .block_task()
                        .await?;
                    assert!(
                        matches!(prompt.stop_reason, StopReason::EndTurn),
                        "the prompt after a notify should EndTurn; got {:?}",
                        prompt.stop_reason
                    );
                    Ok(())
                },
            )
            .await
            .expect("ACP client run should complete cleanly");

        let snapshots = seen.lock().await;
        assert_eq!(snapshots.len(), 1, "exactly one chat() call (the prompt)");
        let snapshot = &snapshots[0];
        assert!(
            snapshot.iter().any(|(role, content)| {
                *role == MessageRole::System && content.contains("BUILD_FINISHED_EVENT")
            }),
            "the notified event must reach the model as System content; got: {snapshot:?}"
        );
        assert!(
            snapshot
                .iter()
                .filter(|(role, _)| *role == MessageRole::User)
                .all(|(_, content)| !content.contains("BUILD_FINISHED_EVENT")),
            "a notified event must never be surfaced as a user message; got: {snapshot:?}"
        );
    }

    /// `auto_respond: true` on an idle session starts a turn through the same
    /// path a prompt uses: the ack is immediate, the events are in the model's
    /// context, and the model is told the turn is event-driven (NOT a user
    /// message), with silence allowed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_start_an_auto_respond_turn_when_requested_on_idle() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cwd = tmp.path().to_path_buf();
        let memory_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();

        let seen: SeenMessages = Arc::new(Mutex::new(Vec::new()));
        let calls: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(RoleRecordingLlm {
            seen: seen.clone(),
            replies: vec!["AUTO_REPLY".to_string()],
            calls: calls.clone(),
            entered: None,
            release: None,
        });
        let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
        let transport = OctosAcpAgentTransport::new(factory);

        let prompt_cwd = cwd.clone();
        Client
            .builder()
            .name("octos-acp-notify-auto-respond-client")
            .on_receive_notification(
                async move |_n: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| {
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                transport,
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let new_session = connection
                        .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                        .block_task()
                        .await?;
                    let session_id = new_session.session_id;

                    let ack = connection
                        .send_request(NotifyRequest {
                            session_id: session_id.clone(),
                            events: vec!["NEW_ROOM_MESSAGE_EVENT".to_string()],
                            auto_respond: true,
                            if_busy: NotifyIfBusy::Drop,
                        })
                        .block_task()
                        .await?;
                    assert!(ack.queued, "idle auto_respond notify must queue");
                    assert!(!ack.busy, "idle session must report busy: false");

                    // The turn runs in the background; wait for the model to be
                    // called exactly once by it.
                    wait_until("the auto_respond turn to call the LLM", || {
                        calls.load(Ordering::SeqCst) == 1
                    })
                    .await;

                    // Once the turn has finished, the session is idle again: a
                    // probe notify reports busy: false (the busy counter was
                    // decremented by the auto_respond turn).
                    let idle_ack = wait_until_idle(&connection, &session_id).await;
                    assert!(!idle_ack.busy, "session must be idle after the turn");
                    Ok(())
                },
            )
            .await
            .expect("ACP client run should complete cleanly");

        let snapshots = seen.lock().await;
        assert_eq!(
            snapshots.len(),
            1,
            "the auto_respond notify must cause exactly one chat() call"
        );
        let snapshot = &snapshots[0];
        assert!(
            snapshot.iter().any(|(role, content)| {
                *role == MessageRole::System && content.contains("NEW_ROOM_MESSAGE_EVENT")
            }),
            "the auto_respond turn must see the events as System context; got: {snapshot:?}"
        );
        assert!(
            snapshot
                .iter()
                .filter(|(role, _)| *role == MessageRole::User)
                .any(|(_, content)| content.contains("NOT user messages")),
            "the auto_respond turn's instruction must tell the model the events \
             are not user messages; got: {snapshot:?}"
        );
    }

    /// `if_busy: drop` discards events sent while a turn is in flight;
    /// `if_busy: fold` queues them (bounded) and the NEXT turn sees them.
    /// Both report `busy: true` truthfully in the ack.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_drop_or_fold_events_sent_while_a_turn_is_in_flight() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cwd = tmp.path().to_path_buf();
        let memory_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();

        let seen: SeenMessages = Arc::new(Mutex::new(Vec::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let llm: Arc<dyn LlmProvider> = Arc::new(RoleRecordingLlm {
            seen: seen.clone(),
            replies: vec![
                "FIRST_TURN_REPLY".to_string(),
                "SECOND_TURN_REPLY".to_string(),
            ],
            calls: Arc::new(AtomicUsize::new(0)),
            entered: Some(entered.clone()),
            release: Some(release.clone()),
        });
        let factory = TestAgentFactory::new(llm, memory_dir, cwd.clone());
        let transport = OctosAcpAgentTransport::new(factory);

        let prompt_cwd = cwd.clone();
        Client
            .builder()
            .name("octos-acp-notify-busy-client")
            .on_receive_notification(
                async move |_n: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| {
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                transport,
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let new_session = connection
                        .send_request(NewSessionRequest::new(prompt_cwd.clone()))
                        .block_task()
                        .await?;
                    let session_id = new_session.session_id;

                    // Turn 1: blocks inside chat() until we release it, so a
                    // notify dispatched meanwhile provably sees a busy session.
                    let prompt_conn = connection.clone();
                    let prompt_sid = session_id.clone();
                    let first_turn = tokio::spawn(async move {
                        let prompt = prompt_conn
                            .send_request(PromptRequest::new(
                                prompt_sid.clone(),
                                vec![ContentBlock::from("BLOCKED_PROMPT")],
                            ))
                            .block_task()
                            .await?;
                        Ok::<_, agent_client_protocol::Error>(prompt.stop_reason)
                    });
                    entered.notified().await;

                    let drop_ack = connection
                        .send_request(NotifyRequest {
                            session_id: session_id.clone(),
                            events: vec!["DROPPED_WHILE_BUSY".to_string()],
                            auto_respond: false,
                            if_busy: NotifyIfBusy::Drop,
                        })
                        .block_task()
                        .await?;
                    assert!(drop_ack.busy, "notify during a turn must report busy");
                    assert!(
                        !drop_ack.queued,
                        "if_busy=drop must not queue anything (queued: false)"
                    );

                    let fold_ack = connection
                        .send_request(NotifyRequest {
                            session_id: session_id.clone(),
                            events: vec!["FOLDED_WHILE_BUSY".to_string()],
                            auto_respond: false,
                            if_busy: NotifyIfBusy::Fold,
                        })
                        .block_task()
                        .await?;
                    assert!(fold_ack.busy, "notify during a turn must report busy");
                    assert!(
                        fold_ack.queued,
                        "if_busy=fold must queue the events for the next turn"
                    );

                    release.notify_one();
                    let first_stop = first_turn.await.expect("first turn task joins")?;
                    assert!(
                        matches!(first_stop, StopReason::EndTurn),
                        "turn 1 should EndTurn once released; got {first_stop:?}"
                    );

                    // Turn 2: must see the FOLDED event (injected before this
                    // turn) and must NOT see the dropped one.
                    let second = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::from("AFTER_BUSY_PROMPT")],
                        ))
                        .block_task()
                        .await?;
                    assert!(
                        matches!(second.stop_reason, StopReason::EndTurn),
                        "turn 2 should EndTurn; got {:?}",
                        second.stop_reason
                    );
                    Ok(())
                },
            )
            .await
            .expect("ACP client run should complete cleanly");

        let snapshots = seen.lock().await;
        assert_eq!(snapshots.len(), 2, "two prompt turns, two chat() calls");
        let second = &snapshots[1];
        assert!(
            second.iter().any(|(role, content)| {
                *role == MessageRole::System && content.contains("FOLDED_WHILE_BUSY")
            }),
            "the folded event must be injected before the next turn; got: {second:?}"
        );
        assert!(
            second
                .iter()
                .filter(|(role, _)| *role == MessageRole::User)
                .all(|(_, content)| !content.contains("FOLDED_WHILE_BUSY")),
            "a folded event must still never surface as a user message; got: {second:?}"
        );
        for snapshot in snapshots.iter() {
            for (_role, content) in snapshot {
                assert!(
                    !content.contains("DROPPED_WHILE_BUSY"),
                    "an if_busy=drop event must never reach the model; got: {snapshot:?}"
                );
            }
        }
    }

    /// A notified event survives a `session/load`/resume cycle (it is
    /// persisted at notify time), and neither the reloaded model context nor
    /// the replayed `session/update` stream surfaces it as user content.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_restore_notified_events_through_session_load_never_as_user() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cwd = tmp.path().to_path_buf();
        let memory_dir = tmp.path().join("memory");
        let sessions_dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&memory_dir).unwrap();

        // ---- first process: notify only — no prompt, no user turn at all ----
        let seen_one: SeenMessages = Arc::new(Mutex::new(Vec::new()));
        let llm_one = new_role_llm(seen_one, vec!["SHOULD_NEVER_RUN".to_string()]);
        let factory_one = TestAgentFactory::new(llm_one, memory_dir.clone(), cwd.clone())
            .with_session_store(&sessions_dir);

        let cwd_one = cwd.clone();
        let session_id = Client
            .builder()
            .name("octos-acp-notify-persist-1")
            .on_receive_notification(
                async move |_n: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| Ok(()),
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                OctosAcpAgentTransport::new(factory_one),
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let new_session = connection
                        .send_request(NewSessionRequest::new(cwd_one.clone()))
                        .block_task()
                        .await?;
                    let id = new_session.session_id.clone();
                    let ack = connection
                        .send_request(NotifyRequest {
                            session_id: id.clone(),
                            events: vec!["PERSIST_ME_EVENT".to_string()],
                            auto_respond: false,
                            if_busy: NotifyIfBusy::Drop,
                        })
                        .block_task()
                        .await?;
                    assert!(ack.queued, "the notify must be accepted");
                    Ok::<_, agent_client_protocol::Error>(id)
                },
            )
            .await
            .expect("first process");

        // ---- second process: same store, load, prompt, record everything ----
        let seen_two: SeenMessages = Arc::new(Mutex::new(Vec::new()));
        let llm_two = new_role_llm(seen_two.clone(), vec!["AFTER_LOAD_REPLY".to_string()]);
        let factory_two = TestAgentFactory::new(llm_two, memory_dir, cwd.clone())
            .with_session_store(&sessions_dir);

        let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_for_handler = updates.clone();
        let id_two = session_id.clone();
        let cwd_two = cwd.clone();
        Client
            .builder()
            .name("octos-acp-notify-persist-2")
            .on_receive_notification(
                async move |n: SessionNotification,
                            _cx: ConnectionTo<agent_client_protocol::Agent>| {
                    updates_for_handler.lock().await.push(n.update);
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                OctosAcpAgentTransport::new(factory_two),
                |connection: ConnectionTo<agent_client_protocol::Agent>| async move {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(id_two.clone(), cwd_two.clone()))
                        .block_task()
                        .await?;
                    connection
                        .send_request(PromptRequest::new(
                            id_two.clone(),
                            vec![ContentBlock::from("QUESTION_AFTER_LOAD")],
                        ))
                        .block_task()
                        .await?;
                    Ok::<_, agent_client_protocol::Error>(())
                },
            )
            .await
            .expect("second process");

        // The reloaded session's model context carries the event as a System
        // row, and no User row carries it.
        let calls = seen_two.lock().await;
        let last = calls.last().expect("the reloaded session ran a turn");
        assert!(
            last.iter().any(|(role, content)| {
                *role == MessageRole::System && content.contains("PERSIST_ME_EVENT")
            }),
            "a notified event must survive session/load as System context; got: {last:?}"
        );
        assert!(
            last.iter()
                .filter(|(role, _)| *role == MessageRole::User)
                .all(|(_, content)| !content.contains("PERSIST_ME_EVENT")),
            "a reloaded notified event must never be surfaced as a user message; got: {last:?}"
        );

        // The load replay (all `session/update`s the second client received)
        // must not contain the event as a user message chunk either.
        let recorded = updates.lock().await;
        let user_texts: Vec<String> = recorded.iter().filter_map(user_message_text).collect();
        assert!(
            user_texts.iter().all(|t| !t.contains("PERSIST_ME_EVENT")),
            "session/load must not replay a notified event as user speech; got: {recorded:?}"
        );
    }
}
