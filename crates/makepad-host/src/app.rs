//! Makepad host application — renders Splash DSL apps sent via WebSocket.
//!
//! The UI thread reads shared state (set by the WS background thread) and
//! renders/re-renders the Splash app on Signal events.

use std::cell::Cell;

use makepad_widgets::*;

use crate::{APP_ID, ERROR_MSG, SHOULD_EXIT, SPLASH_BODY};

app_main!(MakepadHostApp);

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    mod.widgets.AgentSplashBase = #(crate::agent_splash::AgentSplash::register_widget(vm))

    mod.widgets.AgentSplash = set_type_default() do mod.widgets.AgentSplashBase{
        width: Fill height: Fit
    }

    startup() do #(MakepadHostApp::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(980, 760)
                window.title: "Makepad Host — octos"
                body +: {
                    width: Fill
                    height: Fill
                    flow: Down
                    spacing: 8
                    padding: 14

                    status_line := Label {
                        text: "Waiting for app launch…"
                        draw_text.color: #xccddff
                        draw_text.text_style.font_size: 11
                    }

                    error_line := Label {
                        text: ""
                        draw_text.color: #xff8888
                        draw_text.text_style.font_size: 10
                    }

                    splash_holder := RoundedView {
                        width: Fill height: Fit padding: 12
                        draw_bg.color: #1f232e
                        draw_bg.border_radius: 8.0
                        splash := mod.widgets.AgentSplash{width: Fill height: Fit}
                    }

                    source := TextInput {
                        width: Fill height: 0
                        is_read_only: true is_multiline: true
                        visible: false
                    }
                }
            }
        }
    }
}

/// Deferred UI changes applied on the next Draw event.
struct PendingUiUpdate {
    splash_body: Option<String>,
    source_body: Option<String>,
    status: Option<String>,
    error_msg: Option<String>,
}

#[derive(Script, ScriptHook)]
pub struct MakepadHostApp {
    #[live]
    ui: WidgetRef,
    #[rust]
    last_app_id: String,
    #[rust]
    last_splash_body: String,
    #[rust]
    pending_update: Option<PendingUiUpdate>,
    #[rust]
    last_error_msg: String,
}

impl MakepadHostApp {
    /// Read the current state from shared globals (set by WS thread) and
    /// prepare UI updates to be applied on the next Draw event.
    fn sync_from_state(&mut self, cx: &mut Cx) {
        let splash_body = SPLASH_BODY.get().and_then(|m| m.lock().ok().map(|g| g.clone())).flatten();
        let app_id = APP_ID.get().and_then(|m| m.lock().ok().map(|g| g.clone())).flatten();
        let error_msg = ERROR_MSG.get().and_then(|m| m.lock().ok().map(|g| g.clone())).flatten();
        let should_exit = SHOULD_EXIT.get().map(|f| f.load(std::sync::atomic::Ordering::SeqCst)).unwrap_or(false);

        if should_exit {
            std::process::exit(0);
        }

        // Build UI update
        let mut update = PendingUiUpdate {
            splash_body: None,
            source_body: None,
            status: None,
            error_msg: None,
        };

        let error_text = error_msg
            .as_ref()
            .map(|e| format!("⚠ {e}"))
            .unwrap_or_default();

        if splash_body.is_none() && app_id.is_none() {
            // No app — clear display
            if self.last_app_id.is_empty() && self.last_splash_body.is_empty() {
                return; // Already cleared
            }
            update.splash_body = Some(String::new());
            update.source_body = Some(String::new());
            update.status = Some("Waiting for app launch…".to_string());
            self.last_app_id.clear();
            self.last_splash_body.clear();
            self.pending_update = Some(update);
            return;
        }

        // Early return if nothing changed
        let splash = splash_body.clone().unwrap_or_default();
        let id = app_id.clone().unwrap_or_default();
        if splash == self.last_splash_body && id == self.last_app_id && error_text == self.last_error_msg {
            return;
        }

        self.last_error_msg = error_text.clone();
        update.error_msg = Some(error_text);

        if splash != self.last_splash_body || id != self.last_app_id {
            update.splash_body = Some(splash.clone());
            update.source_body = Some(splash.clone());
            update.status = Some(format!("App: {}", id));
            self.last_app_id = id;
            self.last_splash_body = splash;
        }

        self.pending_update = Some(update);
    }

    /// Apply deferred UI updates during the Draw phase.
    fn apply_pending_updates(&mut self, cx: &mut Cx) {
        let Some(update) = self.pending_update.take() else {
            return;
        };

        if let Some(err) = &update.error_msg {
            self.ui.label(cx, ids!(error_line)).set_text(cx, err);
        }

        if let Some(body) = &update.splash_body {
            self.ui.widget(cx, ids!(splash)).set_text(cx, body);
        }
        if let Some(body) = &update.source_body {
            self.ui.widget(cx, ids!(source)).set_text(cx, body);
        }
        if let Some(status) = &update.status {
            self.ui.label(cx, ids!(status_line)).set_text(cx, status);
        }
    }
}

impl AppMain for MakepadHostApp {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn after_new_from_script(_vm: &mut ScriptVm, app: &mut Self) {
        app.last_app_id = String::new();
        app.last_splash_body = String::new();
        app.last_error_msg = String::new();
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        // Early exit on WindowClosed
        if matches!(event, Event::WindowClosed(_)) {
            std::process::exit(0);
        }

        // Wrap in catch_unwind to prevent panics from propagating
        // through the #[no_unwind] macOS NSTimer callback
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Apply deferred UI updates before rendering
            if matches!(event, Event::Draw(_)) {
                self.apply_pending_updates(cx);
            }

            self.ui.handle_event(cx, event, &mut Scope::empty());

            match event {
                Event::Startup => {
                    self.sync_from_state(cx);
                }
                Event::Signal => {
                    self.sync_from_state(cx);
                }
                _ => {}
            }
        }));

        if let Err(e) = result {
            eprintln!("[app] handle_event panicked: {:?}", e);
        }
    }
}
