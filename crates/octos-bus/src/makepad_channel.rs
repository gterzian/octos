//! Makepad native UI channel — WebSocket-based bidirectional bridge.
//!
//! Listens for WebSocket connections from a Makepad host application.
//! Incoming messages (button clicks, text input, widget events) are
//! forwarded to the agent loop as `InboundMessage`s. Agent responses
//! are sent back over the WebSocket as JSON payloads.
//!
//! This replaces the CRDT-based a2app harness with a direct JSON
//! WebSocket protocol. No third component needed — the profile runtime
//! owns the agent loop and the channel adapter in the same process.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::routing::get;
use chrono::Utc;
use eyre::{Result, WrapErr};
use futures::{SinkExt, StreamExt};
use octos_core::{InboundMessage, OutboundMessage};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};

use crate::channel::{Channel, ChannelHealth};

/// JSON message sent from the Makepad host to octos.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum MakepadInboundMessage {
    /// User typed text or sent a chat message.
    #[serde(rename = "user_input")]
    UserInput {
        content: String,
        #[serde(default)]
        chat_id: Option<String>,
    },
    /// A widget was clicked.
    #[serde(rename = "widget_event")]
    WidgetEvent {
        widget: String,
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        chat_id: Option<String>,
    },
    /// Host requests a fresh app launch.
    #[serde(rename = "launch_request")]
    LaunchRequest {
        #[serde(default)]
        splash_body: Option<String>,
        #[serde(default)]
        chat_id: Option<String>,
    },
    /// Host sends structured user input (e.g. from __pi_response widget).
    #[serde(rename = "pi_response")]
    PiResponse {
        content: String,
        #[serde(default)]
        chat_id: Option<String>,
    },
    /// Host sends widget tree snapshot or debug info.
    #[serde(rename = "debug_info")]
    DebugInfo {
        #[serde(default)]
        widget_tree: Option<serde_json::Value>,
        #[serde(default)]
        message: Option<String>,
    },
    /// Ping / keepalive.
    #[serde(rename = "ping")]
    Ping {
        #[serde(default)]
        ts: Option<f64>,
    },
}

/// JSON message sent from octos to the Makepad host.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum MakepadOutboundMessage {
    /// Launch a Splash app in the host window.
    #[serde(rename = "launch")]
    Launch {
        splash_body: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        app_id: Option<String>,
    },
    /// Text response from the agent (displayed in __ai_text widget).
    #[serde(rename = "response")]
    Response {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_final: Option<bool>,
    },
    /// Streaming text delta (incremental update to __ai_text).
    #[serde(rename = "stream_delta")]
    StreamDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        sequence: Option<u64>,
    },
    /// Stream complete — finalize rendering.
    #[serde(rename = "stream_complete")]
    StreamComplete {
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
    },
    /// Snapshot of the widget tree (response to debug_info).
    #[serde(rename = "widget_snapshot_reply")]
    WidgetSnapshotReply {
        tree: serde_json::Value,
    },
    /// Pong response.
    #[serde(rename = "pong")]
    Pong {
        #[serde(skip_serializing_if = "Option::is_none")]
        ts: Option<f64>,
    },
    /// Error message.
    #[serde(rename = "error")]
    Error {
        message: String,
    },
}

/// Shared state for the Makepad channel's connection handlers.
struct MakepadState {
    /// Sender to the agent loop's inbound message bus.
    inbound_tx: mpsc::Sender<InboundMessage>,
    /// Map of chat_id → WebSocket sender for broadcasting responses.
    connections: RwLock<HashMap<String, mpsc::UnboundedSender<String>>>,
    /// Default chat ID for messages without an explicit one.
    default_chat_id: String,
    /// Whether to allow multiple connections (true) or only one (false).
    allow_multiple: bool,
    /// Channel to send launch commands to the harness WS client.
    harness_tx: RwLock<Option<mpsc::UnboundedSender<String>>>,
}

/// Makepad native UI channel adapter.
pub struct MakepadChannel {
    /// Port to listen on for WebSocket connections.
    port: u16,
    /// Host to bind to.
    host: String,
    /// Default chat ID for routing.
    default_chat_id: String,
    /// Whether to allow multiple simultaneous connections.
    allow_multiple: bool,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Path to the Makepad host binary (optional).
    host_binary: Option<String>,
    /// Path to the harness binary.
    harness_binary: Option<String>,
    /// Bound address after server starts.
    bound_addr: Arc<RwLock<Option<SocketAddr>>>,
}

impl MakepadChannel {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            host: "127.0.0.1".into(),
            default_chat_id: "makepad".into(),
            allow_multiple: true,
            shutdown: Arc::new(AtomicBool::new(false)),
            host_binary: None,
            harness_binary: None,
            bound_addr: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    pub fn with_default_chat_id(mut self, chat_id: &str) -> Self {
        self.default_chat_id = chat_id.to_string();
        self
    }

    pub fn with_allow_multiple(mut self, allow: bool) -> Self {
        self.allow_multiple = allow;
        self
    }

    pub fn with_host_binary(mut self, binary: &str) -> Self {
        self.host_binary = Some(binary.to_string());
        self
    }

    pub fn with_harness_binary(mut self, binary: &str) -> Self {
        self.harness_binary = Some(binary.to_string());
        self
    }

    /// Get the address the server is bound to (available after `start`).
    pub async fn bound_address(&self) -> Option<SocketAddr> {
        *self.bound_addr.read().await
    }
}

#[async_trait]
impl Channel for MakepadChannel {
    fn name(&self) -> &str {
        "makepad"
    }

    async fn start(&self, inbound_tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        let state = Arc::new(MakepadState {
            inbound_tx: inbound_tx.clone(),
            connections: RwLock::new(HashMap::new()),
            default_chat_id: self.default_chat_id.clone(),
            allow_multiple: self.allow_multiple,
            harness_tx: RwLock::new(None),
        });

        // ── Launch a2app harness (which spawns the Makepad host) ─────────
        if let Some(harness_path) = &self.harness_binary {
            info!(
                channel = "makepad",
                harness = %harness_path,
                "Launching a2app harness + Makepad host"
            );
            match Command::new(harness_path)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
            {
                Ok(_child) => {
                    // Leak the child — it will be killed when the process exits
                    Box::leak(Box::new(_child));
                    info!(channel = "makepad", "Harness launched, waiting for services...");
                    tokio::time::sleep(Duration::from_secs(4)).await;

                    // ── Connect to harness JSON WS as a client ─────
                    let state_clone = state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = connect_to_harness_ws(state_clone).await {
                            warn!(channel = "makepad", "Harness WS client: {e}");
                        }
                    });
                }
                Err(e) => {
                    warn!(
                        channel = "makepad",
                        harness = %harness_path,
                        error = %e,
                        "Failed to launch harness — continuing without Makepad host"
                    );
                }
            }
        }

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .route("/health", get(health_handler))
            .with_state(state);

        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .wrap_err_with(|| format!("invalid address: {}:{}", self.host, self.port))?;

        let listener = TcpListener::bind(addr)
            .await
            .wrap_err_with(|| format!("failed to bind to {addr}"))?;

        let bound_addr = listener.local_addr()?;
        {
            let mut addr_w = self.bound_addr.write().await;
            *addr_w = Some(bound_addr);
        }

        info!(
            channel = "makepad",
            addr = %bound_addr,
            "Makepad channel WebSocket server started"
        );

        let shutdown = self.shutdown.clone();
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                while !shutdown.load(Ordering::SeqCst) {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
                info!(channel = "makepad", "Makepad channel shutting down");
            })
            .await
            .wrap_err("Makepad channel server failed")?;

        Ok(())
    }

    async fn send(&self, _msg: &OutboundMessage) -> Result<()> {
        warn!(
            channel = "makepad",
            "send() called directly — use WebSocket connection for responses"
        );
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        info!(channel = "makepad", "Makepad channel stop requested");
        Ok(())
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        let bound = *self.bound_addr.read().await;
        match bound {
            Some(_addr) => Ok(ChannelHealth::Healthy),
            None => Ok(ChannelHealth::Degraded("not yet bound".into())),
        }
    }

    fn max_message_length(&self) -> usize {
        // Makepad channel can handle large messages — no practical limit.
        100_000
    }

    fn supports_edit(&self) -> bool {
        true
    }
}

/// WebSocket upgrade handler.
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<MakepadState>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Health check endpoint.
async fn health_handler(State(state): State<Arc<MakepadState>>) -> &'static str {
    if state.connections.read().await.is_empty() {
        "ok (no connections)"
    } else {
        "ok"
    }
}

/// Handle a single WebSocket connection from the Makepad host.
async fn handle_socket(socket: WebSocket, state: Arc<MakepadState>) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Create a channel for sending messages to this connection
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Register this connection
    {
        let mut conns = state.connections.write().await;
        // If multiple connections not allowed, clear existing ones
        if !state.allow_multiple {
            conns.clear();
        }
        // Use a unique connection ID based on count
        let conn_id = format!("conn-{}", conns.len() + 1);
        conns.insert(conn_id, tx.clone());
    }

    info!(channel = "makepad", "Makepad host connected");

    // Spawn a task to forward messages from our channel to the WebSocket.
    // The ws_sender is moved into this task — all outbound communication
    // (including pong responses) goes through the tx/rx channel pair.
    let ws_send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_sender.send(WsMessage::Text(msg.into())).await.is_err() {
                break;
            }
        }
        let _ = ws_sender.close().await;
    });

    // Process incoming WebSocket messages
    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        handle_incoming_ws_text(&text, &state, &tx).await;
                    }
                    Some(Ok(WsMessage::Binary(data))) => {
                        debug!(channel = "makepad", "received binary message ({} bytes)", data.len());
                    }
                    Some(Ok(WsMessage::Ping(_))) => {
                        // Axum handles Ping/Pong at the protocol level automatically
                    }
                    Some(Ok(WsMessage::Close(_))) => {
                        info!(channel = "makepad", "Makepad host disconnected");
                        break;
                    }
                    Some(Err(e)) => {
                        warn!(channel = "makepad", "WebSocket error: {e}");
                        break;
                    }
                    None => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup: remove this connection
    // We can't easily remove the specific one since we don't track IDs back.
    // For simplicity with single-connection mode, just clear.
    if !state.allow_multiple {
        state.connections.write().await.clear();
    }

    ws_send_task.abort();
    info!(channel = "makepad", "Makepad connection handler exiting");
}

/// Parse and handle an incoming WebSocket text message from the Makepad host.
async fn handle_incoming_ws_text(
    text: &str,
    state: &MakepadState,
    _response_tx: &mpsc::UnboundedSender<String>,
) {
    let parsed: Result<MakepadInboundMessage, _> = serde_json::from_str(text);
    let msg = match parsed {
        Ok(m) => m,
        Err(e) => {
            warn!(channel = "makepad", "failed to parse incoming message: {e}");
            let _ = _response_tx.send(
                serde_json::to_string(&MakepadOutboundMessage::Error {
                    message: format!("invalid message format: {e}"),
                })
                .unwrap(),
            );
            return;
        }
    };

    match msg {
        MakepadInboundMessage::Ping { ts } => {
            let pong = serde_json::to_string(&MakepadOutboundMessage::Pong { ts }).unwrap();
            let _ = _response_tx.send(pong);
        }
        MakepadInboundMessage::UserInput { content, chat_id }
        | MakepadInboundMessage::PiResponse { content, chat_id } => {
            let chat_id = chat_id
                .unwrap_or_else(|| state.default_chat_id.clone());
            let inbound = InboundMessage {
                channel: "makepad".into(),
                sender_id: "user".into(),
                chat_id: chat_id.clone(),
                content,
                timestamp: Utc::now(),
                media: vec![],
                metadata: serde_json::json!({}),
                message_id: None,
                origin: octos_core::MessageOrigin::ExternalUser,
            };
            if let Err(e) = state.inbound_tx.send(inbound).await {
                error!(channel = "makepad", chat_id = %chat_id, "failed to send inbound message: {e}");
            }
        }
        MakepadInboundMessage::WidgetEvent {
            widget,
            value,
            chat_id,
        } => {
            let chat_id = chat_id
                .unwrap_or_else(|| state.default_chat_id.clone());
            let content = match &value {
                Some(v) => format!("Widget '{widget}' event with value: {v}"),
                None => format!("Widget '{widget}' event"),
            };
            let inbound = InboundMessage {
                channel: "makepad".into(),
                sender_id: "user".into(),
                chat_id: chat_id.clone(),
                content,
                timestamp: Utc::now(),
                media: vec![],
                metadata: serde_json::json!({
                    "widget": widget,
                    "value": value,
                }),
                message_id: None,
                origin: octos_core::MessageOrigin::ExternalUser,
            };
            if let Err(e) = state.inbound_tx.send(inbound).await {
                error!(channel = "makepad", chat_id = %chat_id, "failed to send widget event: {e}");
            }
        }
        MakepadInboundMessage::LaunchRequest {
            splash_body: _,
            chat_id,
        } => {
            let chat_id = chat_id
                .unwrap_or_else(|| state.default_chat_id.clone());
            info!(channel = "makepad", chat_id = %chat_id, "launch request received");
            // Launch requests from the host are acknowledged but handled
            // via the agent's tools (launch_app tool sends the splash body).
            let inbound = InboundMessage {
                channel: "makepad".into(),
                sender_id: "user".into(),
                chat_id: chat_id.clone(),
                content: "Request to launch a Splash app".into(),
                timestamp: Utc::now(),
                media: vec![],
                metadata: serde_json::json!({"type": "launch_request"}),
                message_id: None,
                origin: octos_core::MessageOrigin::ExternalUser,
            };
            if let Err(e) = state.inbound_tx.send(inbound).await {
                error!(channel = "makepad", chat_id = %chat_id, "failed to send launch request: {e}");
            }
        }
        MakepadInboundMessage::DebugInfo {
            widget_tree,
            message,
        } => {
            let msg = message.unwrap_or_else(|| "debug info received".into());
            info!(channel = "makepad", debug_msg = %msg, widget_tree = ?widget_tree.is_some(), "debug info from host");
        }
    }
}

/// Convenience function to build a Splash launch message.
pub fn build_launch_message(splash_body: &str, app_id: Option<String>) -> String {
    serde_json::to_string(&MakepadOutboundMessage::Launch {
        splash_body: splash_body.to_string(),
        app_id,
    })
    .unwrap()
}

/// Convenience function to build a streaming delta message.
pub fn build_stream_delta(delta: &str, sequence: Option<u64>) -> String {
    serde_json::to_string(&MakepadOutboundMessage::StreamDelta {
        delta: delta.to_string(),
        sequence,
    })
    .unwrap()
}

/// Convenience function to build a final response message.
pub fn build_response_message(content: &str, stream_final: Option<bool>) -> String {
    serde_json::to_string(&MakepadOutboundMessage::Response {
        content: content.to_string(),
        stream_final,
    })
    .unwrap()
}

/// Connect to the a2app harness's JSON WebSocket as a client.
/// Forwards launch commands from the agent to the harness, which
/// passes them to the Makepad host via CRDT.
async fn connect_to_harness_ws(state: Arc<MakepadState>) -> Result<()> {
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message as TungsteniteMsg;

    let ws_url = "ws://127.0.0.1:2341";
    info!(channel = "makepad", url = %ws_url, "Connecting to harness JSON WS...");

    loop {
        match connect_async(ws_url).await {
            Ok((ws_stream, _)) => {
                info!(channel = "makepad", "Connected to harness");
                let (mut ws_tx, mut ws_rx) = ws_stream.split();

                // Read welcome
                if let Some(Ok(TungsteniteMsg::Text(welcome))) = ws_rx.next().await {
                    debug!(channel = "makepad", welcome = %welcome, "Harness welcome");
                }

                // Store the sender so our launch handler can use it
                {
                    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
                    *state.harness_tx.write().await = Some(tx);

                    // Forward messages from the internal channel → harness WS
                    let forward = tokio::spawn(async move {
                        while let Some(msg) = rx.recv().await {
                            if ws_tx.send(TungsteniteMsg::Text(msg.into())).await.is_err() {
                                break;
                            }
                        }
                    });

                    // Listen for responses from harness → our inbound channel
                    loop {
                        tokio::select! {
                            msg = ws_rx.next() => {
                                match msg {
                                    Some(Ok(TungsteniteMsg::Text(text))) => {
                                        let parsed: serde_json::Value =
                                            serde_json::from_str(&text).unwrap_or_default();
                                        let typ = parsed.get("type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("?");

                                        match typ {
                                            "user_response" => {
                                                if let Some(response) = parsed.get("response")
                                                    .and_then(|v| v.as_str())
                                                {
                                                    let inbound = InboundMessage {
                                                        channel: "makepad".into(),
                                                        sender_id: "user".into(),
                                                        chat_id: "makepad".into(),
                                                        content: response.to_string(),
                                                        timestamp: Utc::now(),
                                                        media: vec![],
                                                        metadata: serde_json::json!({
                                                            "source": "pi_response"
                                                        }),
                                                        message_id: None,
                                                        origin: octos_core::MessageOrigin::ExternalUser,
                                                    };
                                                    let _ = state.inbound_tx.send(inbound).await;
                                                }
                                            }
                                            "error" => {
                                                if let Some(msg_text) = parsed.get("message")
                                                    .and_then(|v| v.as_str())
                                                {
                                                    warn!(channel = "makepad", error = %msg_text);
                                                }
                                            }
                                            "status" | "debug_response" | "doc_state" => {
                                                debug!(channel = "makepad", msg = %text);
                                            }
                                            other => {
                                                debug!(channel = "makepad", type = %other);
                                            }
                                        }
                                    }
                                    Some(Ok(TungsteniteMsg::Close(_))) => {
                                        info!(channel = "makepad", "Harness WS closed");
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        warn!(channel = "makepad", "Harness WS error: {e}");
                                        break;
                                    }
                                    None => break,
                                    _ => {}
                                }
                            }
                        }
                    }

                    forward.abort();
                    *state.harness_tx.write().await = None;
                }
            }
            Err(e) => {
                warn!(channel = "makepad", "Failed to connect to harness: {e}");
                tokio::time::sleep(Duration::from_millis(2000)).await;
                continue;
            }
        }

        // Reconnect loop
        tokio::time::sleep(Duration::from_millis(2000)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn should_serialize_launch_message() {
        let body = r#"App { Window { Body { Text { text: "Hello" } } } }"#;
        let json = build_launch_message(body, Some("app-1".into()));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "launch");
        assert_eq!(parsed["splash_body"], body);
        assert_eq!(parsed["app_id"], "app-1");
    }

    #[test]
    fn should_serialize_stream_delta() {
        let json = build_stream_delta("Hello ", Some(1));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "stream_delta");
        assert_eq!(parsed["delta"], "Hello ");
        assert_eq!(parsed["sequence"], 1);
    }

    #[test]
    fn should_serialize_response() {
        let json = build_response_message("Done!", Some(true));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "response");
        assert_eq!(parsed["content"], "Done!");
        assert_eq!(parsed["stream_final"], true);
    }

    #[test]
    fn should_deserialize_user_input() {
        let json = r#"{"type":"user_input","content":"hello world"}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::UserInput { content, chat_id } => {
                assert_eq!(content, "hello world");
                assert!(chat_id.is_none());
            }
            _ => panic!("expected UserInput"),
        }
    }

    #[test]
    fn should_deserialize_widget_event() {
        let json = r#"{"type":"widget_event","widget":"increment","value":null}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::WidgetEvent { widget, value, .. } => {
                assert_eq!(widget, "increment");
                assert!(value.is_none());
            }
            _ => panic!("expected WidgetEvent"),
        }
    }

    #[test]
    fn should_deserialize_widget_event_with_value() {
        let json = r#"{"type":"widget_event","widget":"text_input","value":"hello"}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::WidgetEvent { widget, value, .. } => {
                assert_eq!(widget, "text_input");
                assert_eq!(value, Some("hello".into()));
            }
            _ => panic!("expected WidgetEvent"),
        }
    }

    #[test]
    fn should_deserialize_pi_response() {
        let json = r#"{"type":"pi_response","content":"ai:ask:what is a CRDT?"}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::PiResponse { content, .. } => {
                assert_eq!(content, "ai:ask:what is a CRDT?");
            }
            _ => panic!("expected PiResponse"),
        }
    }

    #[test]
    fn should_deserialize_launch_request() {
        let json = r#"{"type":"launch_request"}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::LaunchRequest { .. } => {} // ok
            _ => panic!("expected LaunchRequest"),
        }
    }

    #[test]
    fn should_deserialize_ping() {
        let json = r#"{"type":"ping","ts":12345.0}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::Ping { ts } => {
                assert_eq!(ts, Some(12345.0));
            }
            _ => panic!("expected Ping"),
        }
    }

    #[test]
    fn should_deserialize_debug_info() {
        let json = r#"{"type":"debug_info","message":"widget tree synced"}"#;
        let parsed: MakepadInboundMessage = serde_json::from_str(json).unwrap();
        match parsed {
            MakepadInboundMessage::DebugInfo { message, widget_tree } => {
                assert_eq!(message, Some("widget tree synced".into()));
                assert!(widget_tree.is_none());
            }
            _ => panic!("expected DebugInfo"),
        }
    }

    #[test]
    fn should_return_error_on_invalid_json() {
        let json = r#"not valid json"#;
        let parsed: Result<MakepadInboundMessage, _> = serde_json::from_str(json);
        assert!(parsed.is_err());
    }

    #[test]
    fn should_serialize_pong() {
        let pong = serde_json::to_string(&MakepadOutboundMessage::Pong { ts: Some(123.0) }).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&pong).unwrap();
        assert_eq!(parsed["type"], "pong");
        assert_eq!(parsed["ts"], 123.0);
    }

    #[test]
    fn should_serialize_error_message() {
        let err = serde_json::to_string(&MakepadOutboundMessage::Error {
            message: "something broke".into(),
        })
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&err).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["message"], "something broke");
    }

    #[test]
    fn should_serialize_widget_snapshot_reply() {
        let tree = serde_json::json!({"widgets": [{"id": "btn1", "type": "Button"}]});
        let reply = serde_json::to_string(&MakepadOutboundMessage::WidgetSnapshotReply {
            tree: tree.clone(),
        })
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(parsed["type"], "widget_snapshot_reply");
        assert_eq!(parsed["tree"], tree);
    }

    #[tokio::test]
    async fn should_handle_incoming_user_input() {
        let (inbound_tx, mut inbound_rx) = mpsc::channel(100);
        let state = MakepadState {
            inbound_tx,
            connections: RwLock::new(HashMap::new()),
            default_chat_id: "makepad".into(),
            allow_multiple: true,
            harness_tx: RwLock::new(None),
        };
        let state = Arc::new(state);
        let (_tx, _rx) = mpsc::unbounded_channel();

        let text = r#"{"type":"user_input","content":"hello from makepad"}"#;
        handle_incoming_ws_text(text, &state, &_tx).await;

        let msg = inbound_rx.recv().await.unwrap();
        assert_eq!(msg.content, "hello from makepad");
        assert_eq!(msg.channel, "makepad");
        assert_eq!(msg.chat_id, "makepad");
    }

    #[tokio::test]
    async fn should_handle_widget_event_as_inbound_message() {
        let (inbound_tx, mut inbound_rx) = mpsc::channel(100);
        let state = MakepadState {
            inbound_tx,
            connections: RwLock::new(HashMap::new()),
            default_chat_id: "makepad".into(),
            allow_multiple: true,
            harness_tx: RwLock::new(None),
        };
        let state = Arc::new(state);
        let (_tx, _rx) = mpsc::unbounded_channel();

        let text = r#"{"type":"widget_event","widget":"increment","value":"1"}"#;
        handle_incoming_ws_text(text, &state, &_tx).await;

        let msg = inbound_rx.recv().await.unwrap();
        assert!(msg.content.contains("increment"));
        assert_eq!(msg.metadata["widget"], "increment");
    }

    #[tokio::test]
    async fn should_handle_ping_with_pong_reply() {
        let (inbound_tx, _inbound_rx) = mpsc::channel(100);
        let state = MakepadState {
            inbound_tx,
            connections: RwLock::new(HashMap::new()),
            default_chat_id: "makepad".into(),
            allow_multiple: true,
            harness_tx: RwLock::new(None),
        };
        let state = Arc::new(state);
        let (tx, mut rx) = mpsc::unbounded_channel();

        let text = r#"{"type":"ping","ts":42.0}"#;
        handle_incoming_ws_text(text, &state, &tx).await;

        let response = rx.recv().await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(parsed["type"], "pong");
        assert_eq!(parsed["ts"], 42.0);
    }

    #[test]
    fn test_makepad_channel_name() {
        let channel = MakepadChannel::new(9001);
        assert_eq!(channel.name(), "makepad");
    }

    #[test]
    fn test_makepad_channel_max_message_length() {
        let channel = MakepadChannel::new(9001);
        assert_eq!(channel.max_message_length(), 100_000);
    }

    #[test]
    fn test_makepad_channel_supports_edit() {
        let channel = MakepadChannel::new(9001);
        assert!(channel.supports_edit());
    }

    #[tokio::test]
    async fn test_makepad_channel_builder() {
        let channel = MakepadChannel::new(9001)
            .with_host("0.0.0.0")
            .with_default_chat_id("test-ui")
            .with_allow_multiple(false)
            .with_host_binary("makepad-host");
        assert_eq!(channel.name(), "makepad");
        assert_eq!(channel.default_chat_id, "test-ui");
        assert!(!channel.allow_multiple);
        assert_eq!(channel.host_binary, Some("makepad-host".into()));

        let _ = channel.stop().await;
    }
}
