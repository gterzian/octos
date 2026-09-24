//! C ABI for running Octos's **host-managed** ACP loop over a transport the
//! caller supplies.
//!
//! Unlike [`crate::octos_run_task`] (the embedded task runtime, which owns the
//! provider and its credentials), this drives the exact loop
//! `octos acp --host-managed` runs: the embedding host stays the authority for
//! model completions and tool calls, so no provider credential enters this
//! process. It exists for hosts whose transport is not stdio — an Apple
//! ExtensionFoundation `.appex` forwarding frames over XPC is the motivating
//! case.
//!
//! Wire discipline: one JSON-RPC message per [`octos_host_managed_feed`] call
//! (a trailing newline is added if absent); outbound messages are delivered to
//! the [`OctosHostManagedSend`] callback without their trailing newline. The
//! loop ends when the host stops feeding — [`octos_host_managed_free`] drops
//! the feed channel, the reader sees EOF, and `serve` returns.
//!
//! SAFETY matches the rest of the crate: the handle is an opaque
//! `Box::into_raw` pointer freed only by [`octos_host_managed_free`], every
//! pointer is NULL-checked, the callback's `ctx` must stay valid until free,
//! and every body runs inside the crate's panic firewall.

use std::ffi::c_void;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use agent_client_protocol::ByteStreams;
use futures::io::{AsyncRead, AsyncWrite};
use libc::c_char;
use tokio::sync::mpsc;

use super::{clear_last_error, cstr_to_str, guard, set_last_error};

/// Delivers one outbound host-managed frame to the embedding host.
///
/// `ctx` is the opaque pointer passed to [`octos_host_managed_start`]; `data`
/// points at `len` bytes valid for the duration of the call. Return non-zero
/// on success, zero if the frame could not be delivered (which ends the loop).
pub type OctosHostManagedSend =
    unsafe extern "C" fn(ctx: *mut c_void, data: *const u8, len: usize) -> i32;

/// The host-owned callback plus its context, made `Send` so the Octos runtime
/// may invoke it from any worker thread.
struct Callback {
    ctx: *mut c_void,
    send: OctosHostManagedSend,
}

// SAFETY: the C contract requires `ctx` to remain valid until the handle is
// freed and the callback itself to be callable from any thread; the runtime
// only ever invokes it from its own worker threads.
unsafe impl Send for Callback {}

impl Callback {
    fn send(&self, frame: &[u8]) -> io::Result<()> {
        // SAFETY: the caller contract holds `ctx` valid and `send` live for the
        // handle's lifetime; `frame` is a live slice for the call.
        let ok = unsafe { (self.send)(self.ctx, frame.as_ptr(), frame.len()) };
        if ok == 0 {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "host send callback failed",
            ))
        } else {
            Ok(())
        }
    }
}

/// Inbound bytes fed from the host, presented as a `futures::io::AsyncRead`
/// for `agent_client_protocol`'s byte-stream transport.
struct Reader {
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    pending: Vec<u8>,
    offset: usize,
    done: bool,
}

impl AsyncRead for Reader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        loop {
            if this.offset < this.pending.len() {
                let n = (this.pending.len() - this.offset).min(buf.len());
                buf[..n].copy_from_slice(&this.pending[this.offset..this.offset + n]);
                this.offset += n;
                return Poll::Ready(Ok(n));
            }
            if this.done {
                return Poll::Ready(Ok(0));
            }
            match this.rx.poll_recv(cx) {
                Poll::Ready(Some(chunk)) => {
                    this.pending = chunk;
                    this.offset = 0;
                }
                Poll::Ready(None) => {
                    this.done = true;
                    return Poll::Ready(Ok(0));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Outbound frames delivered to the host callback, presented as a
/// `futures::io::AsyncWrite`.
struct Writer {
    callback: Callback,
}

impl AsyncWrite for Writer {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        // `ByteStreams` writes a whole line including its newline. XPC is
        // message-oriented, so the host receives one JSON-RPC message with no
        // trailing separator.
        let frame = buf.strip_suffix(b"\n").unwrap_or(buf);
        match self.callback.send(frame) {
            Ok(()) => Poll::Ready(Ok(buf.len())),
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Opaque host-managed session handle. Freed only by
/// [`octos_host_managed_free`].
pub struct OctosHostManaged {
    runtime: Option<tokio::runtime::Runtime>,
    feed: Option<mpsc::UnboundedSender<Vec<u8>>>,
}

/// Starts a host-managed session and returns its handle, or NULL on error
/// (see [`crate::octos_last_error`]). `sandbox` is reported to the broker in
/// the `initialize` capabilities; the process compartment is the caller's
/// responsibility, so an embedding host passes the name of the compartment it
/// provides (e.g. an app-extension sandbox).
#[unsafe(no_mangle)]
pub extern "C" fn octos_host_managed_start(
    max_iterations: u32,
    sandbox: *const c_char,
    send: Option<OctosHostManagedSend>,
    ctx: *mut c_void,
) -> *mut OctosHostManaged {
    guard(std::ptr::null_mut(), "octos_host_managed_start", || {
        clear_last_error();
        let Some(send) = send else {
            set_last_error("a send callback is required");
            return std::ptr::null_mut();
        };
        // SAFETY: `cstr_to_str` NULL-checks and validates UTF-8.
        let sandbox = match unsafe { cstr_to_str(sandbox) } {
            Ok(value) => value,
            Err(error) => {
                set_last_error(error);
                return std::ptr::null_mut();
            }
        };
        // `serve` requires `&'static str` (it is captured by an `Fn` request
        // handler). The sandbox label is a short, fixed, one-per-session
        // string, so interning it here is a bounded, negligible leak.
        let sandbox: &'static str = Box::leak(sandbox.to_string().into_boxed_str());

        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(4)
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024)
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                set_last_error(format!("failed to build tokio runtime: {error}"));
                return std::ptr::null_mut();
            }
        };

        let (feed, inbound) = mpsc::unbounded_channel::<Vec<u8>>();
        let transport = ByteStreams::new(
            Writer {
                callback: Callback { ctx, send },
            },
            Reader {
                rx: inbound,
                pending: Vec::new(),
                offset: 0,
                done: false,
            },
        );
        runtime.spawn(async move {
            if let Err(error) =
                octos_cli::commands::acp::host_managed::serve(max_iterations, sandbox, transport)
                    .await
            {
                set_last_error(format!("host-managed ACP connection ended: {error}"));
            }
        });

        Box::into_raw(Box::new(OctosHostManaged {
            runtime: Some(runtime),
            feed: Some(feed),
        }))
    })
}

/// Feeds one inbound JSON-RPC frame. Returns 1 on success, 0 if the handle is
/// invalid/closed or the frame could not be queued.
#[unsafe(no_mangle)]
pub extern "C" fn octos_host_managed_feed(
    handle: *mut OctosHostManaged,
    data: *const u8,
    len: usize,
) -> i32 {
    guard(0, "octos_host_managed_feed", || {
        if handle.is_null() || data.is_null() {
            return 0;
        }
        // SAFETY: the caller guarantees `data` points at `len` readable bytes.
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        // SAFETY: the handle is a live pointer returned by
        // `octos_host_managed_start` and not yet freed.
        let handle = unsafe { &*handle };
        let Some(feed) = handle.feed.as_ref() else {
            return 0;
        };
        let mut frame = bytes.to_vec();
        if !frame.ends_with(b"\n") {
            frame.push(b'\n');
        }
        if feed.send(frame).is_ok() { 1 } else { 0 }
    })
}

/// Stops the session and frees the handle. Safe to call once with a pointer
/// from [`octos_host_managed_start`]; NULL is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn octos_host_managed_free(handle: *mut OctosHostManaged) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` came from `Box::into_raw` in
    // `octos_host_managed_start` and is freed exactly once here.
    let mut handle = unsafe { Box::from_raw(handle) };
    // Dropping the sender EOFs the reader so `serve` returns promptly.
    handle.feed.take();
    if let Some(runtime) = handle.runtime.take() {
        runtime.shutdown_timeout(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// start → feed → callback → free, without Swift: the loop must accept a
    /// frame and answer on the callback. A capability-less `initialize` is
    /// rejected by the host-managed handler, so the reply is a JSON-RPC error
    /// — enough to prove the transport is wired.
    #[test]
    fn host_managed_bridge_round_trips_a_frame() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        unsafe extern "C" fn send(ctx: *mut c_void, data: *const u8, len: usize) -> i32 {
            // SAFETY: `ctx` is the `Arc<Mutex<Vec<String>>>` leaked below.
            let sink = unsafe { &*(ctx as *const Mutex<Vec<String>>) };
            // SAFETY: `data`/`len` describe a live slice for this call.
            let bytes = unsafe { std::slice::from_raw_parts(data, len) };
            sink.lock()
                .unwrap()
                .push(String::from_utf8_lossy(bytes).into_owned());
            1
        }
        let ctx = Arc::into_raw(sink.clone()) as *mut c_void;
        let sandbox = c"xpc-extension";
        let handle = octos_host_managed_start(4, sandbox.as_ptr(), Some(send), ctx);
        assert!(!handle.is_null(), "start returned NULL");

        let request = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#;
        assert_eq!(
            octos_host_managed_feed(handle, request.as_ptr(), request.len()),
            1
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        octos_host_managed_free(handle);
        // SAFETY: reclaim the context exactly once, after the loop is stopped.
        let _ = unsafe { Arc::from_raw(ctx as *const Mutex<Vec<String>>) };

        let frames = seen.lock().unwrap();
        assert!(
            frames.iter().any(|frame| frame.contains("\"error\"")),
            "expected an initialize error reply, saw {frames:?}"
        );
    }
}
