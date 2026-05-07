from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.exports import hub as hub_mod
from llm_chain_sidecar.exports.hub import HubAuthError, push_to_hub


def _make_run_dir(tmp_path: Path, run_id: str) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "adapter_model.safetensors").write_bytes(b"")
    return run_dir


def test_push_to_hub_raises_when_not_signed_in(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    _make_run_dir(tmp_path, "run-1")

    with patch.object(hub_mod, "_resolve_token", return_value=None):
        with pytest.raises(HubAuthError, match="Not signed in"):
            push_to_hub("run-1", "user/repo", runs_root=tmp_path)


def test_push_to_hub_forwards_repo_id_and_private(tmp_path: Path, monkeypatch):
    _make_run_dir(tmp_path, "run-1")

    api = MagicMock()
    fake_hf_module = MagicMock()
    fake_hf_module.HfApi.return_value = api

    with patch.object(hub_mod, "_resolve_token", return_value="hf_xxx"):
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf_module)
        url = push_to_hub("run-1", "user/my-adapter", runs_root=tmp_path, private=False)

    assert url == "https://huggingface.co/user/my-adapter"
    fake_hf_module.HfApi.assert_called_once_with(token="hf_xxx")
    api.create_repo.assert_called_once()
    create_kwargs = api.create_repo.call_args.kwargs
    assert create_kwargs["repo_id"] == "user/my-adapter"
    assert create_kwargs["private"] is False
    assert create_kwargs["exist_ok"] is True

    api.upload_folder.assert_called_once()
    upload_kwargs = api.upload_folder.call_args.kwargs
    assert upload_kwargs["repo_id"] == "user/my-adapter"
    assert Path(upload_kwargs["folder_path"]) == tmp_path / "run-1"


def test_push_to_hub_uses_merged_subdir_when_requested(tmp_path: Path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, "run-1")
    merged = run_dir / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")

    api = MagicMock()
    fake_hf_module = MagicMock()
    fake_hf_module.HfApi.return_value = api
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf_module)

    with patch.object(hub_mod, "_resolve_token", return_value="hf_xxx"):
        push_to_hub("run-1", "user/repo", runs_root=tmp_path, folder="merged")

    upload_kwargs = api.upload_folder.call_args.kwargs
    assert Path(upload_kwargs["folder_path"]) == merged


def test_push_to_hub_404s_when_run_dir_missing(tmp_path: Path):
    with patch.object(hub_mod, "_resolve_token", return_value="hf_xxx"):
        with pytest.raises(FileNotFoundError, match="run does-not-exist not found"):
            push_to_hub("does-not-exist", "user/repo", runs_root=tmp_path)


def test_push_to_hub_rejects_unknown_folder(tmp_path: Path):
    _make_run_dir(tmp_path, "run-1")
    with patch.object(hub_mod, "_resolve_token", return_value="hf_xxx"):
        with pytest.raises(ValueError, match="unknown folder"):
            push_to_hub("run-1", "user/repo", runs_root=tmp_path, folder="bogus")


def test_push_to_hub_404s_when_merged_dir_absent(tmp_path: Path):
    _make_run_dir(tmp_path, "run-1")
    with patch.object(hub_mod, "_resolve_token", return_value="hf_xxx"):
        with pytest.raises(FileNotFoundError, match="merged"):
            push_to_hub("run-1", "user/repo", runs_root=tmp_path, folder="merged")


def test_resolve_token_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "from_env")
    # Force HfFolder.get_token to return None to confirm env fallback works.
    fake_hf = MagicMock()
    fake_hf.HfFolder.get_token.return_value = None
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf)

    assert hub_mod._resolve_token() == "from_env"
