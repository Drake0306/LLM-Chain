use std::path::PathBuf;
use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
struct SidecarState {
    port: Mutex<Option<u16>>,
    // Keep the spawned child handle alive for the lifetime of the app so
    // dropping it can never close the subprocess pipes prematurely.
    child: Mutex<Option<CommandChild>>,
}

#[tauri::command]
fn sidecar_port(state: tauri::State<SidecarState>) -> Option<u16> {
    *state.port.lock().unwrap()
}

fn settings_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".llm-chain").join("desktop-settings.json"))
}

fn read_settings_file() -> Option<serde_json::Value> {
    let path = settings_path()?;
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

#[tauri::command]
fn save_desktop_settings(settings: serde_json::Value) -> Result<(), String> {
    let path = settings_path().ok_or_else(|| "no HOME/USERPROFILE in env".to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let body = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    std::fs::write(path, body).map_err(|e| e.to_string())?;
    Ok(())
}

/// F-D17: spawn a new WebView window pointed at ``route`` so the
/// user can pin Train / RunDetail / Library side by side. The
/// resulting window inherits the sidecar (single backend process)
/// and shares localStorage with the original, which is how the
/// per-window selection state stays in sync.
#[tauri::command]
fn open_in_new_window(
    app: tauri::AppHandle,
    label: String,
    route: String,
) -> Result<(), String> {
    use tauri::{WebviewUrl, WebviewWindowBuilder};

    // Strip a leading slash so we can join cleanly with the dev
    // server URL or the bundled index.html. Tauri's WebviewUrl::App
    // resolves relative paths against the configured devUrl /
    // distDir, so "train" maps to "/train" once loaded.
    let normalised = route.trim_start_matches('/').to_string();
    WebviewWindowBuilder::new(&app, &label, WebviewUrl::App(normalised.into()))
        .title(format!("LLM-Chain · {}", route))
        .inner_size(1100.0, 750.0)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(SidecarState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            eprintln!("[sidecar] spawning binaries/llm-chain-sidecar");

            // Apply the user's desktop-settings.json (if any) by injecting
            // LLM_CHAIN_RUNS_DIR into the sidecar's env. The sidecar reads
            // this env var at startup; changing it requires an app restart.
            let mut command = handle.shell().sidecar("llm-chain-sidecar")?;
            if let Some(settings) = read_settings_file() {
                if let Some(out_dir) = settings
                    .get("defaultOutputDir")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    eprintln!("[sidecar] LLM_CHAIN_RUNS_DIR={out_dir}");
                    command = command.env("LLM_CHAIN_RUNS_DIR", out_dir);
                }
            }
            let (mut rx, child) = command.spawn()?;
            eprintln!("[sidecar] spawned pid={}", child.pid());
            // Park the child handle in app state so it isn't dropped here.
            handle.state::<SidecarState>().child.lock().unwrap().replace(child);

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let s = String::from_utf8_lossy(&line);
                            let trimmed = s.trim_end();
                            eprintln!("[sidecar stdout] {trimmed}");
                            if let Some(rest) = trimmed.strip_prefix("LLM_CHAIN_SIDECAR_PORT=") {
                                if let Ok(p) = rest.trim().parse::<u16>() {
                                    handle
                                        .state::<SidecarState>()
                                        .port
                                        .lock()
                                        .unwrap()
                                        .replace(p);
                                    eprintln!("[sidecar] resolved port={p}");
                                } else {
                                    eprintln!("[sidecar] could not parse port from {rest:?}");
                                }
                            }
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!(
                                "[sidecar stderr] {}",
                                String::from_utf8_lossy(&line).trim_end()
                            );
                        }
                        CommandEvent::Error(e) => eprintln!("[sidecar error] {e}"),
                        CommandEvent::Terminated(t) => {
                            eprintln!("[sidecar terminated] code={:?} signal={:?}", t.code, t.signal);
                            // Clear the cached port. The frontend's
                            // useSidecarStatus hook polls sidecar_port to
                            // detect this — once port goes back to None
                            // the UI shows a "sidecar dead" banner instead
                            // of letting subsequent fetches silently
                            // connect-refuse against the dead address.
                            handle.state::<SidecarState>().port.lock().unwrap().take();
                        }
                        _ => {}
                    }
                }
                eprintln!("[sidecar] event loop exited");
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar_port,
            save_desktop_settings,
            open_in_new_window,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
