import socket

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from . import __version__
    from .api.routes import router as api_router
except ImportError:
    from llm_chain_sidecar import __version__
    from llm_chain_sidecar.api.routes import router as api_router

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)
# The sidecar binds to 127.0.0.1 only and only ever serves three known origins:
# the macOS Tauri WebView (tauri://localhost), the Windows Tauri WebView
# (http://tauri.localhost), and the Vite dev server during `npm run tauri dev`
# (http://localhost:1420). Allow-listing those specifically — instead of
# allow_origins=["*"] — narrows the DNS-rebinding attack surface: a malicious
# page that rebound DNS to point at 127.0.0.1 used to be allowed to read our
# JSON; now it's rejected at the CORS preflight.
_ALLOWED_ORIGINS = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def run() -> None:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    print(f"LLM_CHAIN_SIDECAR_PORT={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run()
