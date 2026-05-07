use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
struct SidecarState {
    port: Mutex<Option<u16>>,
}

#[tauri::command]
fn sidecar_port(state: tauri::State<SidecarState>) -> Option<u16> {
    *state.port.lock().unwrap()
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
            let (mut rx, _child) = handle
                .shell()
                .sidecar("llm-chain-sidecar")?
                .spawn()?;
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        let s = String::from_utf8_lossy(&line);
                        // Sidecar prints "LLM_CHAIN_SIDECAR_PORT=<n>" on startup
                        if let Some(rest) = s.strip_prefix("LLM_CHAIN_SIDECAR_PORT=") {
                            if let Ok(p) = rest.trim().parse::<u16>() {
                                let state: tauri::State<SidecarState> = handle.state();
                                *state.port.lock().unwrap() = Some(p);
                            }
                        }
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_port])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
