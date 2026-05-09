"""Cloud burst (F-C13).

Send a run to a rented GPU at one of the supported providers
(Modal, RunPod, Lambda) instead of training locally. Contradicts
the project's local-first ethos by design — every layer requires
explicit opt-in:

  1. Sidecar starts with cloud burst DISABLED. The user must set
     ``LLM_CHAIN_CLOUD_BURST_ENABLED=1`` (or toggle it in Settings)
     to expose any of the cloud routes.
  2. Per-provider credentials live in
     ``~/.llm-chain/cloud-credentials.json`` and never leave that
     file — the route layer reads them on demand and never logs
     them to events / state files.
  3. Per-run confirmation modal in the UI shows the cost estimate
     and the provider-specific terms before the user can click
     "Submit to cloud".

This package owns the wire format + credential management + the
provider adapter Protocol. Each concrete provider (modal.py,
runpod.py, lambda_labs.py) implements the adapter and pulls in
its own SDK lazily so a CPU-only sidecar build doesn't drag the
provider deps onto the import path.

Status: scaffolding only. The provider adapters return a clean
"not wired yet" error. Wire them in a future session per provider
once you've validated the cost / API surface against your account.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

CloudProvider = Literal["modal", "runpod", "lambda"]
SUPPORTED_PROVIDERS: tuple[CloudProvider, ...] = ("modal", "runpod", "lambda")

# Runtime values that the RunConfig.runtime field can take.
# "local" is the default and the only one that hits the existing
# trainer code paths. "cloud:<provider>" routes through this package.
LocalRuntime = Literal["local"]
CloudRuntime = Literal["cloud:modal", "cloud:runpod", "cloud:lambda"]


def _credentials_path() -> Path:
    """Honour the test override env so unit tests don't write into
    the user's real home directory."""
    env = os.environ.get("LLM_CHAIN_CLOUD_CREDENTIALS_PATH")
    if env:
        return Path(env)
    return Path.home() / ".llm-chain" / "cloud-credentials.json"


def is_enabled() -> bool:
    """The hard outer gate. Defaults to False so a fresh install
    can't accidentally submit a run to a cloud account before the
    user has explicitly opted in.
    """
    raw = os.environ.get("LLM_CHAIN_CLOUD_BURST_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_credentials() -> dict:
    """Read the credentials file. Returns ``{}`` when missing — the
    route layer maps "missing credentials" onto a clean 400 with a
    pointer at the Settings UI rather than a generic file-not-found.
    """
    p = _credentials_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_credentials(creds: dict) -> None:
    """Persist credentials atomically. The file is mode 0o600 because
    it carries API keys; readable only by the running user. Tmp-then-
    rename so a concurrent reader can't catch a torn file.
    """
    p = _credentials_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(creds, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        # Filesystem may not support chmod (e.g. Windows). The
        # protection is best-effort; the API key still lives on
        # the user's local disk regardless.
        pass
    os.replace(tmp, p)


def has_provider_credentials(provider: CloudProvider) -> bool:
    creds = load_credentials()
    return bool(creds.get(provider))


def runtime_provider(runtime: str) -> CloudProvider | None:
    """Parse a runtime string like ``"cloud:modal"`` into the
    provider piece. Returns None for the local case."""
    if not runtime or runtime == "local":
        return None
    if not runtime.startswith("cloud:"):
        raise ValueError(f"Unknown runtime: {runtime!r}")
    p = runtime.removeprefix("cloud:")
    if p not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown cloud provider {p!r}. Pick one of {list(SUPPORTED_PROVIDERS)}."
        )
    return p  # type: ignore[return-value]


@dataclass
class CostEstimate:
    """What the UI shows before the user confirms cloud kickoff."""

    provider: CloudProvider
    estimated_minutes: float
    estimated_usd: float
    notes: str = ""


def estimate_cost(
    provider: CloudProvider,
    *,
    estimated_minutes: float,
    rate_usd_per_hour: float | None = None,
) -> CostEstimate:
    """Best-effort cost estimate. Each provider has its own published
    rate that drifts over time; the values below are conservative
    defaults from late-2025 pricing pages and the user can pass an
    override when they know better.

    Returned with a clear "estimated" caveat so the UI doesn't
    accidentally promise an exact bill.
    """
    if rate_usd_per_hour is None:
        defaults = {"modal": 1.10, "runpod": 0.39, "lambda": 1.10}
        rate_usd_per_hour = defaults.get(provider, 1.0)
    cost = (estimated_minutes / 60.0) * rate_usd_per_hour
    return CostEstimate(
        provider=provider,
        estimated_minutes=estimated_minutes,
        estimated_usd=round(cost, 2),
        notes=(
            "Estimate only — actual billing depends on the provider's "
            "current rates and any cold-start / data transfer charges."
        ),
    )


class CloudAdapter(Protocol):
    """Per-provider adapter contract. Each provider implements its
    own submission + polling + download flow; this Protocol pins
    the surface so the route layer can dispatch generically."""

    def submit(
        self,
        *,
        run_dir: Path,
        config_dict: dict,
        credentials: dict,
    ) -> str: ...

    def status(
        self,
        *,
        job_id: str,
        credentials: dict,
    ) -> dict: ...

    def download_adapter(
        self,
        *,
        job_id: str,
        run_dir: Path,
        credentials: dict,
    ) -> None: ...


class ProviderNotWiredError(NotImplementedError):
    """Raised by the stub adapters until a future session lands the
    real provider SDK integration. The route maps this to a 501
    with a pointer at the open task in HANDOFF."""


@dataclass
class _StubAdapter:
    """Placeholder adapter for unwired providers. Returns a clear
    error from every method so the route layer can surface "this
    provider isn't wired yet" without faking work."""

    provider: CloudProvider

    def submit(self, **_kwargs) -> str:
        raise ProviderNotWiredError(
            f"Cloud burst: the {self.provider} adapter isn't wired yet. "
            "The scaffolding (RunConfig.runtime, credentials, gating, "
            "cost estimate) is in place; the provider SDK call lands "
            "in a follow-up session — see HANDOFF F-C13."
        )

    def status(self, **_kwargs) -> dict:
        raise ProviderNotWiredError(
            f"Cloud burst {self.provider} status check isn't wired yet."
        )

    def download_adapter(self, **_kwargs) -> None:
        raise ProviderNotWiredError(
            f"Cloud burst {self.provider} download isn't wired yet."
        )


def get_adapter(provider: CloudProvider) -> CloudAdapter:
    """Resolve the adapter for ``provider``. Today every provider
    returns the stub; future sessions wire each one independently."""
    return _StubAdapter(provider=provider)


__all__ = [
    "CloudAdapter",
    "CloudProvider",
    "CostEstimate",
    "ProviderNotWiredError",
    "SUPPORTED_PROVIDERS",
    "estimate_cost",
    "get_adapter",
    "has_provider_credentials",
    "is_enabled",
    "load_credentials",
    "runtime_provider",
    "save_credentials",
]
