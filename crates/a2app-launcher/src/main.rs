//! A2App launcher — starts the harness + Makepad host and launches a Splash app.
//!
//! Usage:
//!   cargo run --bin a2app-launch
//!
//! This starts the a2app harness (which spawns the Makepad host window),
//! connects via WebSocket, and sends the Splash counter DSL to be rendered.

use std::process::{Command, Stdio};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;

const A2APP_DIR: &str = "/Users/Gregory/Projects/a2app_harness";
const JSON_WS_URL: &str = "ws://127.0.0.1:2341";

/// Correct Splash DSL counter app using Makepad widget syntax.
const SPLASH_COUNTER: &str = r#"let count = 0
display := Label{text:"0" draw_text.text_style.font_size:24}
ButtonFlat{text:"-" on_click:||{count -= 1; ui.display.set_text("" + count)}}
ButtonFlat{text:"+" on_click:||{count += 1; ui.display.set_text("" + count)}}
ButtonFlat{text:"Reset" on_click:||{count = 0; ui.display.set_text("0")}}
btn_hello := ButtonFlat{text:"Send to Pi" on_click:||{__pi_response.set_text("Hello from octos! Count is " + count)}}"#;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    println!("=== A2App Launcher ===");

    // 1. Kill any leftover processes
    println!("[1/5] Cleaning up...");
    let _ = Command::new("pkill").args(["-f", "harness"]).status();
    let _ = Command::new("pkill").args(["-f", "makepad-host"]).status();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // 2. Start the harness (spawns Makepad host)
    println!("[2/5] Starting harness + Makepad host...");
    let harness_path = format!("{A2APP_DIR}/target/debug/harness");
    let mut harness_child = Command::new(&harness_path)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to start harness");

    // Wait for servers to be ready
    println!("[3/5] Waiting for services...");
    tokio::time::sleep(Duration::from_secs(4)).await;

    // 3. Connect to harness JSON WS
    println!("[4/5] Connecting to {}...", JSON_WS_URL);
    let (ws_stream, _) = connect_async(JSON_WS_URL).await?;
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    // Read welcome message
    if let Some(Ok(msg)) = ws_rx.next().await {
        let text = msg.to_string();
        println!("  ← Welcome: {}", &text[..text.len().min(80)]);
    }

    // 4. Send the Splash counter DSL
    println!("[5/5] Launching Splash counter app...");
    let launch = serde_json::json!({
        "type": "launch",
        "app_id": "counter-1",
        "splash_body": SPLASH_COUNTER
    });
    ws_tx.send(WsMessage::Text(launch.to_string().into())).await?;
    println!("  → Launch sent ({} chars of Splash DSL)", SPLASH_COUNTER.len());

    // 5. Listen for responses for a few seconds
    println!("\n  Listening for responses (5s)...");
    for _ in 0..10 {
        tokio::select! {
            msg = ws_rx.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                        let typ = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                        match typ {
                            "status" => println!("  ← Status: {}", parsed.get("status").and_then(|v| v.as_str()).unwrap_or("")),
                            "user_response" => println!("  ← User response: {}", parsed.get("response").and_then(|v| v.as_str()).unwrap_or("")),
                            "error" => println!("  ← ERROR: {}", parsed.get("message").and_then(|v| v.as_str()).unwrap_or("")),
                            "debug_response" => {
                                let result = parsed.get("result").and_then(|v| v.as_str()).unwrap_or("");
                                println!("  ← Debug: {}...", &result[..result.len().min(100)]);
                            }
                            other => println!("  ← {}: {}", other, &text[..text.len().min(120)]),
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) => {
                        println!("  Connection closed");
                        break;
                    }
                    Some(Err(e)) => {
                        println!("  Error: {e}");
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
        }
    }

    println!("\n=== The Makepad Host window should now show the counter app! ===");
    println!("  (Check your screen for a native window titled 'Makepad Host')");
    println!("\nPress Ctrl+C to stop the harness and exit.\n");

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    println!("\nShutting down...");
    let _ = harness_child.kill();
    let _ = harness_child.wait();

    Ok(())
}
