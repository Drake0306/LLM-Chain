import socket

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router as api_router

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)
# The sidecar binds to 127.0.0.1 only; the only consumers are the Tauri WebView
# (origin: tauri://localhost or http://tauri.localhost on Windows) and the Vite
# dev server (origin: http://localhost:1420). Both are cross-origin from FastAPI's
# perspective, and the WebView will silently drop requests without CORS headers.
# Allowing * here is fine because nothing outside the local machine can reach us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
