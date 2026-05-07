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
    uvicorn.run(app, host="127.0.0.1", port=0)
