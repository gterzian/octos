//! AgentSplash widget — evaluates and renders Splash DSL mini-apps.
//!
//! This widget is injected into every splash body. It handles:
//! - Evaluating Splash DSL code and rendering the widget tree
//! - Sending user responses back to octos (via __pi_response.set_text)
//! - Receiving streaming deltas and rendering them inline
//! - Receiving pi_response data from octos

use makepad_widgets::*;

use crate::{ERROR_MSG, RESPONSE_TX, SPLASH_BODY};

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    mod.widgets.AgentSplashBase = #(AgentSplash::register_widget(vm))

    mod.widgets.AgentSplash = set_type_default() do mod.widgets.AgentSplashBase{
        width: Fill height: Fit
    }
}

/// Prefix and suffix injected around every splash body for Makepad compatibility.
const SPLASH_PREFIX: &str = "use mod.prelude.widgets.*View{width:Fill height:Fit flow:Down ";
const SPLASH_SUFFIX: &str = r#"  __run_splash := mod.widgets.AgentSplash{width:Fill height:Fit is_root:false}
  __pi_status := Label{text:" " height:Fit width:Fill}
  __ai_text := Label{text:" " height:Fit width:Fill}
  __pi_response := Label{text:"" visible:false}
  __pi_data := Label{text:" " visible:false}"#;

#[derive(Script, ScriptHook, Widget)]
pub struct AgentSplash {
    #[source]
    source: ScriptObjectRef,
    #[deref]
    pub view: View,
    #[live]
    body: ArcStringMut,
    #[rust]
    render_ok: bool,
    #[rust]
    last_response: String,
    #[rust]
    last_pi_data: String,
    #[rust]
    last_streaming_text: String,
    #[live(true)]
    is_root: bool,
}

impl AgentSplash {
    fn self_id(&self) -> usize {
        self as *const Self as usize
    }

    fn render_body(&mut self, cx: &mut Cx, body: &str) -> bool {
        let self_id = self.self_id();
        let widget_uid = self.widget_uid();
        let code = format!("{}{}{}", SPLASH_PREFIX, body, SPLASH_SUFFIX);
        let script_mod = ScriptMod {
            cargo_manifest_path: String::new(),
            module_path: String::new(),
            file: String::new(),
            line: self_id,
            column: 0,
            code: String::new(),
            values: vec![],
        };

        cx.with_vm(|vm| {
            let value = vm.eval_with_append_source(script_mod, &code, NIL.into());
            if value.is_err() || value.is_nil() || !value.is_object() {
                return false;
            }
            self.view = View::script_from_value(vm, value);
            vm.cx_mut().widget_tree_mark_dirty(widget_uid);
            true
        })
    }

    fn eval_body(&mut self, cx: &mut Cx) -> bool {
        let body = self.body.as_ref().to_string();
        if body.is_empty() {
            let code = "use mod.prelude.widgets.*View{width:Fill height:Fit}".to_string();
            let script_mod = ScriptMod {
                cargo_manifest_path: String::new(),
                module_path: String::new(),
                file: String::new(),
                line: self.self_id(),
                column: 0,
                code: String::new(),
                values: vec![],
            };
            cx.with_vm(|vm| {
                let value = vm.eval_with_append_source(script_mod, &code, NIL.into());
                if value.is_object() {
                    self.view = View::script_from_value(vm, value);
                }
            });
            return true;
        }
        self.render_body(cx, &body)
    }

    /// Sync streaming text from octos into the injected __ai_text widget.
    fn sync_streaming_text(&mut self, cx: &mut Cx) {
        let incoming = crate::STREAMING_RX.get().and_then(|m| {
            m.lock().ok().map(|mut guard| {
                let mut texts = Vec::new();
                while let Ok(text) = guard.try_recv() {
                    texts.push(text);
                }
                texts
            })
        });

        let Some(deltas) = incoming else {
            return;
        };

        if deltas.is_empty() {
            return;
        }

        // Concatenate all pending deltas
        let new_text = deltas.join("");
        self.last_streaming_text = new_text.clone();

        // Update the __ai_text label
        if let Ok(ai_text) = self.view.child_by_name("__ai_text") {
            let _ = ai_text.set_text_and_redraw(cx, &new_text);
        }

        // Extract and render ```runsplash code blocks
        if let Ok(run_splash) = self.view.child_by_name("__run_splash") {
            for line in new_text.lines() {
                if line.contains("```runsplash") {
                    // Extract the code block and set it on the nested splash
                    if let Some(code_start) = new_text.find("```runsplash") {
                        let after_prefix = &new_text[code_start + "```runsplash".len()..];
                        if let Some(code_end) = after_prefix.find("```") {
                            let code = &after_prefix[..code_end];
                            let _ = run_splash.set_text_and_redraw(cx, code);
                        }
                    }
                }
            }
        }
    }

    /// Sync pi_response from octos into the injected __pi_data label.
    fn sync_pi_data_to_splash(&mut self, cx: &mut Cx) {
        // pi_response is handled through the splash_body mechanism
        // When a send_pi_response arrives, SPLASH_BODY is updated and
        // the UI thread re-renders
    }

    /// Send a user response back to octos via the RESPONSE_TX channel.
    fn send_response(&self, response: &str) {
        if let Some(tx) = RESPONSE_TX.get() {
            let msg = serde_json::json!({
                "type": "pi_response",
                "content": response
            });
            let _ = tx.send(msg.to_string());
        }
    }
}

impl Widget for AgentSplash {
    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        // Handle streaming text from octos
        if matches!(event, Event::Signal) {
            self.sync_streaming_text(cx);
            self.sync_pi_data_to_splash(cx);
        }

        // Delegate to inner view
        self.view.handle_event(cx, event, scope);

        match event {
            Event::Startup => {
                self.eval_body(cx);
            }
            Event::BeforeFirstDraw(_) => {
                if !self.render_ok && !self.body.as_ref().to_string().is_empty() {
                    self.render_ok = self.render_body(cx, self.body.as_ref());
                    if !self.render_ok {
                        // Set error message for the main app to display
                        if let Some(err) = ERROR_MSG.get() {
                            if let Ok(mut guard) = err.lock() {
                                *guard = Some("Failed to evaluate Splash DSL body".to_string());
                            }
                        }
                    }
                }
            }
            Event::MouseUp(mu) => {
                // Track clicks on the __pi_response label: the splash app calls
                // __pi_response.set_text("...") which the host detects here.
                if let Ok(pi_response) = self.view.child_by_name("__pi_response") {
                    // Try to read the text on every event type (not just click)
                    // so we catch set_text() calls from any interaction
                }
            }
            _ => {}
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        // Check for response updates every draw cycle
        if let Ok(pi_response) = self.view.child_by_name("__pi_response") {
            let text = pi_response.text(cx);
            if !text.is_empty() && text != self.last_response {
                self.last_response = text.clone();
                self.send_response(&text);
            }
        }

        self.view.draw_walk(cx, scope, walk)
    }

    fn text(&self) -> String {
        self.body.as_ref().to_string()
    }

    fn set_text(&mut self, cx: &mut Cx, text: &str) {
        self.body.as_ref().to_string();
        let mut body = self.body.borrow_mut();
        let set = text.to_string();
        if *body != set {
            *body = set;
            self.render_ok = false;
        }
    }

    fn set_text_and_redraw(&mut self, cx: &mut Cx, text: &str) {
        self.set_text(cx, text);
        cx.redraw(self.view.area());
    }
}
