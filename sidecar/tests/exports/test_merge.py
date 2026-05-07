import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.exports import gguf as gguf_mod
from llm_chain_sidecar.exports.gguf import find_latest_adapter, merge_adapter


def _write_run(runs_root: Path, run_id: str, model_id: str = "tiny/model") -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "id": run_id,
                "status": "succeeded",
                "config": {"model_id": model_id, "backend": "cuda", "technique": "lora"},
            }
        )
    )
    return run_dir


def test_find_latest_adapter_picks_highest_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    for step in (50, 200, 100):
        cp = run_dir / f"checkpoint-{step}"
        cp.mkdir()
    assert find_latest_adapter(run_dir).name == "checkpoint-200"


def test_find_latest_adapter_uses_run_dir_when_adapter_present(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "adapter_model.safetensors").write_bytes(b"")
    assert find_latest_adapter(run_dir) == run_dir


def test_find_latest_adapter_raises_when_empty(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        find_latest_adapter(run_dir)


def test_merge_adapter_returns_existing_dir_if_already_merged(tmp_path: Path):
    run_dir = _write_run(tmp_path, "abc123")
    merged = run_dir / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")

    # Idempotent: should NOT touch peft/transformers when merged dir is present.
    with patch.object(gguf_mod, "PeftModel", create=True) as p_peft:
        result = merge_adapter("abc123", tmp_path)
    assert result == merged
    p_peft.assert_not_called()


def test_merge_adapter_invokes_peft_pipeline(tmp_path: Path, monkeypatch):
    run_dir = _write_run(tmp_path, "abc123", model_id="tiny/model")
    (run_dir / "adapter_model.safetensors").write_bytes(b"")

    saved_dirs: list[Path] = []
    merged_obj = MagicMock()
    merged_obj.save_pretrained.side_effect = lambda d: saved_dirs.append(Path(d))

    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged_obj

    fake_peft = MagicMock()
    fake_peft.PeftModel.from_pretrained.return_value = peft_model

    fake_transformers = MagicMock()
    fake_tok = MagicMock()
    fake_tok.save_pretrained.side_effect = lambda d: saved_dirs.append(Path(d))
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()

    monkeypatch.setitem(__import__("sys").modules, "peft", fake_peft)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    result = merge_adapter("abc123", tmp_path)
    assert result == run_dir / "merged"
    fake_transformers.AutoTokenizer.from_pretrained.assert_called_with("tiny/model")
    fake_transformers.AutoModelForCausalLM.from_pretrained.assert_called_with("tiny/model")
    fake_peft.PeftModel.from_pretrained.assert_called_once()
    peft_model.merge_and_unload.assert_called_once()
    # Both the merged model and the tokenizer are saved into the merged dir.
    assert saved_dirs.count(run_dir / "merged") == 2
