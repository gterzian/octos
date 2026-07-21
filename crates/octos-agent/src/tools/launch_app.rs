//! Launch App tool — sends a Splash DSL body to the Makepad host for rendering.
//!
//! Auto-launches the a2app harness (which spawns the Makepad host) if it's
//! not already running. Kills the harness when the tool is dropped (on
//! chat/gateway shutdown).

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use async_trait::async_trait;
use eyre::{Result, WrapErr};
use serde::Deserialize;
use serde_json::json;

use super::{Tool, ToolResult};

/// Default path to the a2app harness binary.
const DEFAULT_HARNESS_BIN: &str =
    "/Users/Gregory/Projects/a2app_harness/target/debug/harness";

/// Tool that launches a Splash app on the Makepad host.
/// Stores the harness child handle and kills it on Drop.
pub struct LaunchAppTool {
    /// Handle to the spawned harness child (if any).
    child: Mutex<Option<Child>>,
}

impl LaunchAppTool {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
        }
    }
}

impl Drop for LaunchAppTool {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        // Also pkill any leftover harness/makepad-host processes
        let _ = Command::new("pkill")
            .args(["-f", "harness"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = Command::new("pkill")
            .args(["-f", "makepad-host"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Try to connect to the harness. Returns true if it's running.
async fn is_harness_running() -> bool {
    use tokio::net::TcpStream;
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        TcpStream::connect("127.0.0.1:2341"),
    )
    .await
    .ok()
    .and_then(|r| r.ok())
    .is_some()
}

/// Kill any existing harness, then launch a fresh one. Returns the child.
fn launch_harness() -> Result<Child> {
    // Kill any leftovers
    let _ = Command::new("pkill")
        .args(["-f", "harness"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = Command::new("pkill")
        .args(["-f", "makepad-host"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    std::thread::sleep(std::time::Duration::from_millis(500));

    let harness_bin = std::env::var("HARNESS_BINARY")
        .unwrap_or_else(|_| DEFAULT_HARNESS_BIN.to_string());

    let child = Command::new(&harness_bin)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .wrap_err_with(|| format!("Failed to launch harness at {harness_bin}"))?;

    Ok(child)
}

#[derive(Deserialize)]
struct Input {
    splash_body: String,
    #[serde(default = "default_app_id")]
    app_id: String,
}

fn default_app_id() -> String {
    "splash-app".to_string()
}

#[async_trait]
impl Tool for LaunchAppTool {
    fn name(&self) -> &str {
        "launch_app"
    }

    fn description(&self) -> &str {
        "Launch a Splash DSL mini-app on the Makepad native UI host."
    }

    fn tags(&self) -> &[&str] {
        &["gateway", "makepad"]
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "splash_body": {
                    "type": "string",
                    "description": "The Splash DSL body (Makepad widget syntax)."
                },
                "app_id": {
                    "type": "string",
                    "description": "Optional app identifier (default: splash-app)"
                }
            },
            "required": ["splash_body"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let input: Input = serde_json::from_value(input.clone())
            .wrap_err("Invalid launch_app input")?;

        let splash_body = input.splash_body;
        let app_id = input.app_id;

        // Check if harness is already running
        if !is_harness_running().await {
            tracing::info!(target: "launch_app", "Launching fresh harness");
            match launch_harness() {
                Ok(child) => {
                    // Store the child so Drop kills it on shutdown
                    if let Ok(mut guard) = self.child.lock() {
                        *guard = Some(child);
                    }
                }
                Err(e) => {
                    return Ok(ToolResult {
                        output: format!("Failed to start harness: {e}"),
                        success: false,
                        file_modified: None,
                        files_to_send: vec![],
                        tokens_used: None,
                        structured_metadata: None,
                        named_outputs: None,
                    });
                }
            }

            // Wait for harness to be ready
            let mut ready = false;
            for _ in 0..30 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if is_harness_running().await {
                    tracing::info!(target: "launch_app", "Harness is ready");
                    ready = true;
                    break;
                }
            }
            if !ready {
                return Ok(ToolResult {
                    output: "Harness did not become ready within 15 seconds".to_string(),
                    success: false,
                    file_modified: None,
                    files_to_send: vec![],
                    tokens_used: None,
                    structured_metadata: None,
                    named_outputs: None,
                });
            }
        }

        // Send launch to harness
        let ws_url = "ws://127.0.0.1:2341";
        let launch_msg = json!({
            "type": "launch",
            "app_id": app_id,
            "splash_body": splash_body
        });

        match send_launch_to_harness(ws_url, &launch_msg.to_string()).await {
            Ok(responses) => {
                let summary = if responses.is_empty() {
                    "App launched successfully".to_string()
                } else {
                    format!("App launched. {}", responses.join("; "))
                };
                Ok(ToolResult {
                    output: summary,
                    success: true,
                    file_modified: None,
                    files_to_send: vec![],
                    tokens_used: None,
                    structured_metadata: None,
                    named_outputs: None,
                })
            }
            Err(e) => Ok(ToolResult {
                output: format!("Failed to launch app: {e}"),
                success: false,
                file_modified: None,
                files_to_send: vec![],
                tokens_used: None,
                structured_metadata: None,
                named_outputs: None,
            }),
        }
    }
}

/// Connect to the harness JSON WS, send launch, collect responses.
async fn send_launch_to_harness(ws_url: &str, message: &str) -> Result<Vec<String>> {
    use futures::{SinkExt, StreamExt};
    use std::time::Duration;
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    let (ws_stream, _) = connect_async(ws_url).await?;
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    let mut responses = Vec::new();

    tokio::time::sleep(Duration::from_millis(200)).await;
    loop {
        tokio::select! {
            msg = ws_rx.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => responses.push(text.to_string()),
                    Some(Ok(WsMessage::Close(_))) => break,
                    _ => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(200)) => break,
        }
    }

    ws_tx.send(WsMessage::Text(message.into())).await?;

    tokio::time::sleep(Duration::from_millis(500)).await;
    loop {
        tokio::select! {
            msg = ws_rx.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        responses.push(text.to_string());
                        if text.contains("\"status\"") || text.contains("\"user_response\"") {
                            break;
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) => break,
                    _ => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(1000)) => break,
        }
    }

    Ok(responses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_have_correct_name() {
        let tool = LaunchAppTool::new();
        assert_eq!(tool.name(), "launch_app");
    }

    #[test]
    fn should_have_valid_input_schema() {
        let tool = LaunchAppTool::new();
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"].as_array().unwrap().contains(&"splash_body".into()));
    }
}
