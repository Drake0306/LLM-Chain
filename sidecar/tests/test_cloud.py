"""F-C13: cloud burst scaffolding tests.

The actual provider SDK calls are stubbed in this commit; tests
exercise the gating logic, credential file management, runtime
parsing, and cost estimation. Integration tests live with each
provider adapter once they're wired in a future session.
"""
import json
import os
from pathlib import Path

import pytest

from llm_chain_sidecar import cloud


@pytest.fixture
def cred_file(tmp_path: Path, monkeypatch):
    p = tmp_path / "cloud-credentials.json"
    monkeypatch.setenv("LLM_CHAIN_CLOUD_CREDENTIALS_PATH", str(p))
    return p


# --- is_enabled -------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("LLM_CHAIN_CLOUD_BURST_ENABLED", raising=False)
    assert cloud.is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
def test_is_enabled_recognises_truthy_strings(monkeypatch, value):
    monkeypatch.setenv("LLM_CHAIN_CLOUD_BURST_ENABLED", value)
    assert cloud.is_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_is_enabled_treats_other_values_as_off(monkeypatch, value):
    monkeypatch.setenv("LLM_CHAIN_CLOUD_BURST_ENABLED", value)
    assert cloud.is_enabled() is False


# --- runtime parsing -------------------------------------------------


def test_runtime_provider_returns_none_for_local():
    assert cloud.runtime_provider("local") is None
    assert cloud.runtime_provider("") is None


@pytest.mark.parametrize("provider", ["modal", "runpod", "lambda"])
def test_runtime_provider_parses_supported_cloud(provider):
    assert cloud.runtime_provider(f"cloud:{provider}") == provider


def test_runtime_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown cloud provider"):
        cloud.runtime_provider("cloud:made-up")


def test_runtime_provider_rejects_unparseable():
    with pytest.raises(ValueError, match="Unknown runtime"):
        cloud.runtime_provider("cumulus")


# --- credentials -----------------------------------------------------


def test_load_credentials_returns_empty_when_missing(cred_file):
    assert cloud.load_credentials() == {}


def test_save_then_load_credentials_round_trips(cred_file):
    cloud.save_credentials({"modal": {"token": "t1"}, "runpod": {"key": "k"}})
    loaded = cloud.load_credentials()
    assert loaded == {"modal": {"token": "t1"}, "runpod": {"key": "k"}}


def test_save_credentials_atomic(cred_file):
    cloud.save_credentials({"modal": {"token": "x"}})
    leftovers = list(cred_file.parent.glob("*.tmp"))
    assert leftovers == []


def test_save_credentials_chmods_to_user_only(cred_file):
    """File holds API keys — verify it's not world-readable on
    POSIX (Windows skipped by the chmod try/except in the writer)."""
    if os.name == "nt":
        pytest.skip("Windows skips chmod")
    cloud.save_credentials({"modal": {"token": "x"}})
    mode = cred_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_load_credentials_resilient_to_corrupt_file(cred_file):
    cred_file.write_text("not json")
    assert cloud.load_credentials() == {}


def test_load_credentials_resilient_to_non_object(cred_file):
    cred_file.write_text("[1, 2, 3]")
    assert cloud.load_credentials() == {}


def test_has_provider_credentials_reflects_file(cred_file):
    assert cloud.has_provider_credentials("modal") is False
    cloud.save_credentials({"modal": {"token": "t"}})
    assert cloud.has_provider_credentials("modal") is True
    assert cloud.has_provider_credentials("runpod") is False


# --- cost estimate ----------------------------------------------------


def test_estimate_cost_uses_default_rate_per_provider():
    e = cloud.estimate_cost("modal", estimated_minutes=60)
    assert e.provider == "modal"
    assert e.estimated_minutes == 60
    assert e.estimated_usd > 0
    assert "Estimate only" in e.notes


def test_estimate_cost_honours_override_rate():
    e = cloud.estimate_cost(
        "modal", estimated_minutes=30, rate_usd_per_hour=2.0,
    )
    # 30 min at $2/hr → $1.00.
    assert e.estimated_usd == 1.0


# --- adapter dispatch -------------------------------------------------


@pytest.mark.parametrize("provider", ["modal", "runpod", "lambda"])
def test_get_adapter_returns_stub_for_every_provider(provider):
    """Today every provider returns a stub. The Protocol is the
    long-term contract; the test locks the per-provider lookup so
    a future wiring lands behind the same key."""
    a = cloud.get_adapter(provider)
    with pytest.raises(cloud.ProviderNotWiredError):
        a.submit(run_dir=Path("/tmp"), config_dict={}, credentials={})
    with pytest.raises(cloud.ProviderNotWiredError):
        a.status(job_id="x", credentials={})
    with pytest.raises(cloud.ProviderNotWiredError):
        a.download_adapter(job_id="x", run_dir=Path("/tmp"), credentials={})


def test_provider_not_wired_error_is_notimplemented_subclass():
    """Future code can ``except NotImplementedError`` to handle the
    "not yet wired" case generically alongside other unwired feature
    errors. Lock the inheritance so a refactor doesn't break that."""
    assert issubclass(cloud.ProviderNotWiredError, NotImplementedError)
