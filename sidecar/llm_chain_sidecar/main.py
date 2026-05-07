from fastapi import FastAPI
import uvicorn
from . import __version__

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=0)
