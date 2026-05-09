"""Ollama integration (F-B9).

Once a run has a GGUF on disk, the user can register it with their
local Ollama install in one click — a Modelfile is generated, the
sidecar shells out to ``ollama create``, and the user gets the literal
``ollama run <name>`` command they can paste into Terminal.

Why a separate module: keeping the subprocess invocation, name
validation, and registration tracking out of routes.py makes the
unit tests simple (mock subprocess.run, assert what we'd shell out)
without spinning up FastAPI.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ollama model names follow the same convention HF uses for repo
# segments — lowercase alphanumerics plus a small set of separators.
# Validating up front means we never pass a user-controlled string
# into the subprocess argv that could carry shell metacharacters.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")

# Where we persist registrations. One JSON per run dir; the route
# layer reads this on DELETE /runs/{id} to offer ``ollama rm`` as
# follow-up cleanup.
REGISTRATION_FILE = "ollama-registrations.json"


class OllamaNotInstalledError(RuntimeError):
    """Raised when ``ollama`` isn't on PATH. The route maps this to a
    400 with install instructions instead of a generic 500."""


class OllamaInvalidNameError(ValueError):
    """Raised for names that don't match Ollama's tag rules."""


class OllamaCommandError(RuntimeError):
    """Raised when ``ollama create`` exits non-zero. Carries stderr so
    the route can surface it."""


@dataclass
class ModelfileOptions:
    """Per-registration knobs the UI lets the user override.

    Defaults match the playground's GenerationConfig so an Ollama-
    served model produces similar output to the in-app inference.
    ``stop_tokens`` is empty by default — the user adds family-
    specific tokens (``<|im_end|>`` for Qwen, ``<|eot_id|>`` for
    Llama-3) when they care.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    num_ctx: int = 4096
    stop_tokens: list[str] = None  # type: ignore[assignment]
    system: str | None = None

    def __post_init__(self) -> None:
        if self.stop_tokens is None:
            self.stop_tokens = []


@dataclass
class RegistrationResult:
    name: str
    run_command: str
    modelfile_path: str


def is_installed() -> bool:
    """Return True iff ``ollama`` resolves on PATH.

    ``shutil.which`` is the right primitive here — it walks PATH the
    same way the user's shell does and respects PATHEXT on Windows.
    Used by the route to short-circuit before subprocess.run.
    """
    return shutil.which("ollama") is not None


def validate_name(name: str) -> None:
    """Raise ``OllamaInvalidNameError`` if the name isn't safe to use
    as an Ollama tag.

    The regex enforces lowercase alphanumeric prefix + dotted/dashed/
    underscored body up to 128 chars. Without this, a name like
    ``"foo; rm -rf ~"`` would be passed verbatim into our subprocess
    argv — argv-style invocations are immune to shell injection, but
    a clearly-malformed name should fail loudly here rather than
    surface as a confusing ``ollama create`` error.
    """
    if not isinstance(name, str) or not name:
        raise OllamaInvalidNameError("Ollama name must be a non-empty string.")
    if not _NAME_RE.match(name):
        raise OllamaInvalidNameError(
            "Ollama name must be lowercase, start with a letter or digit, "
            "and contain only letters, digits, dots, dashes, or underscores "
            "(max 128 chars)."
        )


def render_modelfile(gguf_path: Path, opts: ModelfileOptions) -> str:
    """Build the Modelfile text Ollama's ``create`` reads.

    Single ``FROM`` line points at the absolute GGUF path; one
    ``PARAMETER`` line per knob. Stop tokens are emitted as separate
    ``PARAMETER stop`` lines because Ollama supports multiple stops.
    Any ``"`` in a stop string is escaped — Ollama's parser strips
    surrounding quotes.
    """
    lines: list[str] = [f"FROM {gguf_path}"]
    lines.append(f"PARAMETER temperature {opts.temperature}")
    lines.append(f"PARAMETER top_p {opts.top_p}")
    lines.append(f"PARAMETER num_ctx {opts.num_ctx}")
    for stop in opts.stop_tokens or []:
        if not stop:
            continue
        # Modelfile is line-oriented — a literal newline in a stop
        # token would terminate the PARAMETER line and Ollama would
        # parse the rest as garbage. Escape \\, ", \n, \r so anything
        # the user pasted survives intact (Ollama interprets the
        # standard JSON-style backslash escapes inside the quotes).
        escaped = (
            stop.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        lines.append(f'PARAMETER stop "{escaped}"')
    if opts.system:
        # Triple-quoted form so multi-line system prompts pass through.
        lines.append("SYSTEM \"\"\"")
        lines.append(opts.system.rstrip())
        lines.append("\"\"\"")
    return "\n".join(lines) + "\n"


def register(
    *,
    run_dir: Path,
    gguf_path: Path,
    name: str,
    opts: ModelfileOptions | None = None,
    runner: callable | None = None,  # type: ignore[type-arg]
) -> RegistrationResult:
    """Generate a Modelfile + invoke ``ollama create``.

    Side-effects:
    - Writes ``<run_dir>/Modelfile`` (overwrites any prior).
    - Appends the registered name to
      ``<run_dir>/ollama-registrations.json`` so a later cleanup
      step knows which Ollama tags this run owns.

    Raises:
    - OllamaNotInstalledError: ``ollama`` not on PATH.
    - OllamaInvalidNameError: name fails ``validate_name``.
    - OllamaCommandError: subprocess returned non-zero.
    """
    if not is_installed():
        raise OllamaNotInstalledError(
            "Ollama isn't installed (or isn't on PATH). Install from "
            "https://ollama.com and re-run."
        )
    validate_name(name)
    if not gguf_path.is_file():
        raise FileNotFoundError(
            f"GGUF file not found at {gguf_path}. Run the GGUF export first."
        )
    opts = opts or ModelfileOptions()
    modelfile_text = render_modelfile(gguf_path, opts)
    modelfile_path = run_dir / "Modelfile"
    modelfile_path.write_text(modelfile_text, encoding="utf-8")

    # Resolve at call time, not as a default-arg binding — that's the
    # only shape that lets a test-side ``monkeypatch.setattr("subprocess.run", ...)``
    # take effect.
    invoke = runner if runner is not None else subprocess.run
    # ``ollama create <name> -f <modelfile>`` — argv-style invocation
    # so a hostile name (already rejected by validate_name) couldn't
    # carry shell metacharacters into the kernel's argv anyway.
    completed = invoke(
        ["ollama", "create", name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        # stderr contains Ollama's own message ("model already exists",
        # "invalid path", etc.). Bubble verbatim so the user can act.
        raise OllamaCommandError(
            (completed.stderr or completed.stdout or "ollama create failed").strip()
        )

    _record_registration(run_dir, name)
    return RegistrationResult(
        name=name,
        run_command=f"ollama run {name}",
        modelfile_path=str(modelfile_path),
    )


def unregister(
    *,
    run_dir: Path,
    name: str,
    runner: callable | None = None,  # type: ignore[type-arg]
) -> None:
    """Remove an Ollama tag this run created.

    Soft on missing-binary / missing-tag — both surface as
    ``OllamaCommandError`` only when the local Ollama install
    refuses outright; tag-not-found is treated as already-cleaned-up
    so a double-click on the cleanup button is idempotent.
    """
    if not is_installed():
        raise OllamaNotInstalledError(
            "Ollama isn't installed (or isn't on PATH)."
        )
    validate_name(name)
    invoke = runner if runner is not None else subprocess.run
    completed = invoke(
        ["ollama", "rm", name],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").lower()
        if "not found" in stderr or "no such" in stderr:
            # Already gone — clear our local record and return clean.
            pass
        else:
            raise OllamaCommandError(
                (completed.stderr or completed.stdout or "ollama rm failed").strip()
            )
    _drop_registration(run_dir, name)


def list_registrations(run_dir: Path) -> list[str]:
    p = run_dir / REGISTRATION_FILE
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    return list(data.get("names", []))


def _atomic_write_text(path: Path, content: str) -> None:
    """Tmp-then-rename writer. Mirrors the pattern used by the gguf
    state writer so concurrent registrations can't expose a torn
    file to ``list_registrations``.
    """
    import os as _os

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    _os.replace(tmp, path)


def _record_registration(run_dir: Path, name: str) -> None:
    p = run_dir / REGISTRATION_FILE
    current = list_registrations(run_dir)
    if name not in current:
        current.append(name)
    _atomic_write_text(p, json.dumps({"names": current}, indent=2))


def _drop_registration(run_dir: Path, name: str) -> None:
    p = run_dir / REGISTRATION_FILE
    current = [n for n in list_registrations(run_dir) if n != name]
    if not current and p.exists():
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        return
    _atomic_write_text(p, json.dumps({"names": current}, indent=2))


__all__ = [
    "ModelfileOptions",
    "OllamaCommandError",
    "OllamaInvalidNameError",
    "OllamaNotInstalledError",
    "REGISTRATION_FILE",
    "RegistrationResult",
    "is_installed",
    "list_registrations",
    "register",
    "render_modelfile",
    "unregister",
    "validate_name",
]
