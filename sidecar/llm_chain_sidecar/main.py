import socket

import uvicorn
from fastapi import FastAPI

from . import __version__
from .api.routes import router as api_router

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)
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
