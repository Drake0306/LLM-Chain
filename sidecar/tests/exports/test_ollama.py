import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_chain_sidecar.exports import ollama


# --- validate_name ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["my-adapter", "qwen3.lora", "snake_case", "v1", "a" * 128],
)
def test_validate_name_accepts_well_formed_names(name):
    ollama.validate_name(name)  # should not raise


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Capital",  # uppercase
        "name with space",
        "trailing/slash",
        "a" * 200,
        "$cmd",
        "\\name",
        "name;rm -rf",
    ],
)
def test_validate_name_rejects_unsafe_or_malformed(name):
    with pytest.raises(ollama.OllamaInvalidNameError):
        ollama.validate_name(name)


# --- render_modelfile -------------------------------------------------


def test_render_modelfile_emits_required_directives(tmp_path: Path):
    gguf = tmp_path / "merged-q4_k_m.gguf"
    gguf.write_bytes(b"fake")
    text = ollama.render_modelfile(gguf, ollama.ModelfileOptions())
    assert text.startswith(f"FROM {gguf}\n")
    assert "PARAMETER temperature 0.7" in text
    assert "PARAMETER top_p 0.95" in text
    assert "PARAMETER num_ctx 4096" in text


def test_render_modelfile_emits_one_stop_per_token(tmp_path: Path):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    opts = ollama.ModelfileOptions(stop_tokens=["<|im_end|>", "<|eot_id|>"])
    text = ollama.render_modelfile(gguf, opts)
    assert text.count("PARAMETER stop") == 2
    assert '"<|im_end|>"' in text
    assert '"<|eot_id|>"' in text


def test_render_modelfile_escapes_quotes_in_stop_tokens(tmp_path: Path):
    """A stop token containing a literal double-quote could break the
    Modelfile's PARAMETER line and produce a confusing parse error
    inside ``ollama create``. We escape with ``\\"`` so the parser
    sees a single literal."""
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    opts = ollama.ModelfileOptions(stop_tokens=['weird"token'])
    text = ollama.render_modelfile(gguf, opts)
    assert 'weird\\"token' in text


def test_render_modelfile_includes_system_block(tmp_path: Path):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    opts = ollama.ModelfileOptions(system="Be brisk.")
    text = ollama.render_modelfile(gguf, opts)
    assert 'SYSTEM """' in text
    assert "Be brisk." in text


# --- register / unregister --------------------------------------------


def _ok_runner(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _failing_runner(stderr: str = "ollama: bad", code: int = 1):
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=code, stdout="", stderr=stderr)
    return runner


def test_register_writes_modelfile_and_records_name(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: True)

    result = ollama.register(
        run_dir=tmp_path,
        gguf_path=gguf,
        name="my-adapter",
        runner=_ok_runner,
    )
    assert result.run_command == "ollama run my-adapter"
    assert (tmp_path / "Modelfile").exists()
    assert ollama.list_registrations(tmp_path) == ["my-adapter"]


def test_register_invokes_argv_style_subprocess(tmp_path: Path, monkeypatch):
    """The subprocess.run argv must be a list, not a shell string —
    that's what protects against shell metacharacters even before our
    name validator runs. Lock the argv shape so a future refactor
    can't regress to ``shell=True``."""
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    captured: dict = {}

    def capturing_runner(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ollama.register(
        run_dir=tmp_path,
        gguf_path=gguf,
        name="my-adapter",
        runner=capturing_runner,
    )
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "ollama"
    assert argv[1] == "create"
    assert argv[2] == "my-adapter"
    assert "shell" not in captured["kwargs"] or captured["kwargs"]["shell"] is False


def test_register_rejects_when_ollama_not_installed(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    with pytest.raises(ollama.OllamaNotInstalledError):
        ollama.register(
            run_dir=tmp_path,
            gguf_path=gguf,
            name="my-adapter",
            runner=_ok_runner,
        )


def test_register_rejects_invalid_name_before_subprocess(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    called = {"n": 0}

    def runner(*args, **kwargs):
        called["n"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ollama.OllamaInvalidNameError):
        ollama.register(
            run_dir=tmp_path,
            gguf_path=gguf,
            name="BAD NAME",
            runner=runner,
        )
    assert called["n"] == 0


def test_register_propagates_subprocess_failure(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    with pytest.raises(ollama.OllamaCommandError, match="bad"):
        ollama.register(
            run_dir=tmp_path,
            gguf_path=gguf,
            name="x",
            runner=_failing_runner("bad"),
        )


def test_register_rejects_when_gguf_missing(tmp_path: Path, monkeypatch):
    """A registration request for a run that hasn't completed its
    GGUF export should fail at the file check, not produce a Modelfile
    with a ``FROM /nonexistent`` line that would confuse ollama
    create downstream."""
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    with pytest.raises(FileNotFoundError):
        ollama.register(
            run_dir=tmp_path,
            gguf_path=tmp_path / "missing.gguf",
            name="x",
            runner=_ok_runner,
        )


def test_unregister_removes_record_and_invokes_rm(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    ollama.register(
        run_dir=tmp_path,
        gguf_path=gguf,
        name="my-adapter",
        runner=_ok_runner,
    )
    ollama.unregister(run_dir=tmp_path, name="my-adapter", runner=_ok_runner)
    assert ollama.list_registrations(tmp_path) == []


def test_unregister_idempotent_on_missing_tag(tmp_path: Path, monkeypatch):
    """A double-click on the cleanup button should not raise — Ollama
    surfaces "model not found" when the tag is already gone, and we
    treat that as "already cleaned up"."""
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    runner = _failing_runner(stderr="Error: model not found", code=1)
    # Pre-record a registration so the function has something to drop.
    (tmp_path / ollama.REGISTRATION_FILE).write_text(
        json.dumps({"names": ["my-adapter"]})
    )
    ollama.unregister(run_dir=tmp_path, name="my-adapter", runner=runner)
    assert ollama.list_registrations(tmp_path) == []


def test_list_registrations_empty_when_no_file(tmp_path: Path):
    assert ollama.list_registrations(tmp_path) == []
