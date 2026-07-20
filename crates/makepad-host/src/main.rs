//! Makepad host — native UI window that renders Splash DSL apps.
//!
//! Connects to octos's Makepad channel via JSON WebSocket (port 2341 by default).
//! No CRDT, no separate harness — direct bidirectional JSON messaging.
//!
//! Protocol:
//!   ← octos: {"type":"launch","splash_body":"...","app_id":"..."}
//!   ← octos: {"type":"send_pi_response","data":"..."}
//!   → octos: {"type":"pi_response","content":"..."}
//!   → octos: {"type":"widget_event","widget":"increment"}
//!
//! Env vars:
//!   MAKEPAD_WS_URL    — WebSocket URL (default: ws://127.0.0.1:2341/ws)
//!   MAKEPAD_WS_TOKEN  — optional auth token

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use makepad_widgets::makepad_platform::thread::SignalToUI;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing::{error, info, warn};

mod agent_splash;
mod app;

/// Shared splash body — set by the WS listener thread, read by the UI thread.
pub static SPLASH_BODY: OnceLock<Arc<Mutex<Option<String>>>> = OnceLock::new();

/// Shared app ID — identifies the current rendered app.
pub static APP_ID: OnceLock<Arc<Mutex<Option<String>>>> = OnceLock::new();

/// Shared error message — set by the UI thread when splash evaluation fails.
pub static ERROR_MSG: OnceLock<Arc<Mutex<Option<String>>>> = OnceLock::new();

/// Sender for responses from the UI thread back to the WS connection.
pub static RESPONSE_TX: OnceLock<mpsc::UnboundedSender<String>> = OnceLock::new();

/// Shared should_exit flag.
pub static SHOULD_EXIT: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Streaming delta channel: background thread sends deltas, UI thread receives them.
pub static STREAMING_RX: OnceLock<Mutex<mpsc::UnboundedReceiver<String>>> = OnceLock::new();

const DEFAULT_WS_URL: &str = "ws://127.0.0.1:2341/ws";
const CONNECT_RETRY_MS: u64 = 1000;

fn main() {
    // Initialize shared state
    SPLASH_BODY.set(Arc::new(Mutex::new(None))).ok();
    APP_ID.set(Arc::new(Mutex::new(None))).ok();
    ERROR_MSG.set(Arc::new(Mutex::new(None))).ok();
    SHOULD_EXIT.set(Arc::new(AtomicBool::new(false))).ok();

    // Create the response channel
    let (resp_tx, resp_rx) = mpsc::unbounded_channel::<String>();
    RESPONSE_TX.set(resp_tx).ok();

    // Create the streaming channel
    let (delta_tx, delta_rx) = mpsc::unbounded_channel::<String>();
    STREAMING_RX.set(Mutex::new(delta_rx)).ok();

    // Start the background WebSocket listener on a separate thread
    std::thread::spawn(move || {
        let rt = Runtime::new().expect("create tokio runtime");
        rt.block_on(ws_background(resp_rx, delta_tx));
    });

    // Write ready marker so the launcher knows we're up
    if let Ok(marker_path) = std::env::var("MAKEPAD_HOST_READY_MARKER") {
        let _ = std::fs::write(&marker_path, "ready\n");
    }

    // Run the Makepad app on the main thread
    app::app_main();
}

/// Background task: connect to octos's Makepad channel WebSocket and
/// listen for incoming JSON messages.
async fn ws_background(
    mut resp_rx: mpsc::UnboundedReceiver<String>,
    delta_tx: mpsc::UnboundedSender<String>,
) {
    let ws_url = std::env::var("MAKEPAD_WS_URL").unwrap_or_else(|_| DEFAULT_WS_URL.to_string());

    loop {
        match connect_async(&ws_url).await {
            Ok((ws_stream, _response)) => {
                info!("Connected to octos at {ws_url}");
                let (mut ws_tx, mut ws_rx) = ws_stream.split();

                // Spawn a task to forward responses from the UI thread to the WS
                let forward_handle = tokio::spawn(async move {
                    while let Some(msg) = resp_rx.recv().await {
                        if ws_tx.send(WsMessage::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                });

                // Main WS receive loop
                loop {
                    tokio::select! {
                        msg = ws_rx.next() => {
                            match msg {
                                Some(Ok(WsMessage::Text(text))) => {
                                    handle_ws_message(&text, &delta_tx).await;
                                }
                                Some(Ok(WsMessage::Close(_))) => {
                                    info!("WebSocket closed by server");
                                    break;
                                }
                                Some(Err(e)) => {
                                    warn!("WebSocket error: {e}");
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

                forward_handle.abort();

                // Check if we should exit
                if SHOULD_EXIT.get().map(|f| f.load(Ordering::SeqCst)).unwrap_or(false) {
                    break;
                }

                warn!("Disconnected, reconnecting in {CONNECT_RETRY_MS}ms...");
                tokio::time::sleep(Duration::from_millis(CONNECT_RETRY_MS)).await;
            }
            Err(e) => {
                warn!("Failed to connect to {ws_url}: {e}");
                tokio::time::sleep(Duration::from_millis(CONNECT_RETRY_MS)).await;
                continue;
            }
        }
    }

    info!("WS background task exiting");
}

/// Handle an incoming JSON message from the octos channel.
async fn handle_ws_message(text: &str, delta_tx: &mpsc::UnboundedSender<String>) {
    let parsed: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse WS message: {e}");
            return;
        }
    };

    match parsed.get("type").and_then(|v| v.as_str()) {
        Some("launch") => {
            let splash_body = parsed
                .get("splash_body")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let app_id = parsed
                .get("app_id")
                .and_then(|v| v.as_str())
                .unwrap_or("app");

            if let Some(body) = SPLASH_BODY.get() {
                if let Ok(mut guard) = body.lock() {
                    *guard = Some(splash_body.to_string());
                }
            }
            if let Some(id) = APP_ID.get() {
                if let Ok(mut guard) = id.lock() {
                    *guard = Some(app_id.to_string());
                }
            }

            // Clear any previous error
            if let Some(err) = ERROR_MSG.get() {
                if let Ok(mut guard) = err.lock() {
                    *guard = None;
                }
            }

            info!(app_id = app_id, splash_len = splash_body.len(), "Launching splash app");
            SignalToUI::set_ui_signal();
        }
        Some("clear") => {
            if let Some(body) = SPLASH_BODY.get() {
                if let Ok(mut guard) = body.lock() {
                    *guard = None;
                }
            }
            if let Some(id) = APP_ID.get() {
                if let Ok(mut guard) = id.lock() {
                    *guard = None;
                }
            }
            SignalToUI::set_ui_signal();
        }
        Some("send_pi_response") => {
            // Data sent from octos to the splash app
            if let Some(data) = parsed.get("data").and_then(|v| v.as_str()) {
                // Store for the UI thread to read
                // Currently handled via app.rs sync
                info!(data_len = data.len(), "Received pi_response from octos");
            }
            SignalToUI::set_ui_signal();
        }
        Some("exit") => {
            if let Some(exit) = SHOULD_EXIT.get() {
                exit.store(true, Ordering::SeqCst);
            }
            SignalToUI::set_ui_signal();
        }
        Some("stream_delta") => {
            if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
                let _ = delta_tx.send(delta.to_string());
                SignalToUI::set_ui_signal();
            }
        }
        Some("stream_complete") => {
            // Signal that streaming is done
            SignalToUI::set_ui_signal();
        }
        other => {
            warn!("Unknown message type: {:?}", other);
        }
    }
}
