//! Makepad native UI channel adapter registration.

use std::sync::Arc;

use octos_bus::ChannelManager;

use crate::config::ChannelEntry;

pub fn register(
    channel_mgr: &mut ChannelManager,
    entry: &ChannelEntry,
) -> eyre::Result<()> {
    let port = entry
        .settings
        .get("port")
        .and_then(|v| v.as_u64())
        .unwrap_or(2341) as u16;

    let host = entry
        .settings
        .get("host")
        .and_then(|v| v.as_str())
        .unwrap_or("127.0.0.1");

    let default_chat_id = entry
        .settings
        .get("default_chat_id")
        .and_then(|v| v.as_str())
        .unwrap_or("makepad");

    let allow_multiple = entry
        .settings
        .get("allow_multiple")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut channel = octos_bus::MakepadChannel::new(port)
        .with_host(host)
        .with_default_chat_id(default_chat_id)
        .with_allow_multiple(allow_multiple);

    if let Some(binary) = entry
        .settings
        .get("host_binary")
        .and_then(|v| v.as_str())
    {
        channel = channel.with_host_binary(binary);
    }

    if let Some(binary) = entry
        .settings
        .get("harness_binary")
        .and_then(|v| v.as_str())
    {
        channel = channel.with_harness_binary(binary);
    }

    channel_mgr.register(Arc::new(channel));
    Ok(())
}
