"""Inference playground: load a trained adapter on top of its base model
and stream generated tokens back to the UI.

Two backends mirror the trainers:
  - MLX runs use ``mlx_lm.utils.load`` (with ``adapter_path``) +
    ``mlx_lm.generate.stream_generate`` for token-by-token streaming
    on Apple Silicon.
  - HF / CPU runs use ``transformers``' ``TextIteratorStreamer`` with
    ``peft.PeftModel.from_pretrained`` over the base model.

A single-slot in-process cache keeps the most recently used model
warm so a follow-up prompt doesn't pay the multi-second load cost
again. Loading is gated by a lock so two concurrent ``/generate``
calls can't tear the cache while a load is in flight.

This module is import-light by design — the heavy frameworks
(``transformers``, ``mlx_lm``) are imported lazily inside the loader
functions, not at module load. The route layer can import this
module to dispatch generate calls without paying torch/MLX import
cost on a sidecar that never serves a generate request.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationConfig:
    """Inputs the SSE endpoint forwards to the underlying generator.

    All fields have safe defaults so the UI can fire a prompt without
    surfacing every knob; advanced users can override per request.
    """
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass(frozen=True)
class GenerationToken:
    """One frame of streamed output.

    The SSE wire carries three event types built from this one shape so
    the consumer doesn't need parallel channels:
      - ``token`` (default): ``text`` is the decoded delta to append.
      - ``status``: ``status`` is informational ("loading model…",
        "queued behind run X") that the UI shows as a transient hint
        and does NOT append to the generated text.
      - ``done``: ``done=True`` marks the end of the stream.
    Exactly one of ``text`` / ``status`` / ``done`` is meaningful per
    frame.
    """
    text: str = ""
    done: bool = False
    status: str | None = None


class _LoadedModel(Protocol):
    """Whatever the backend-specific loader returned. Held by the cache
    and passed back to the generator function — opaque from the
    cache's perspective."""


@dataclass
class _CacheEntry:
    run_id: str
    backend: str
    obj: Any


_cache_lock = threading.Lock()
_cache: _CacheEntry | None = None


def _free_cache_locked() -> None:
    """Drop the cached model + tokenizer references and invite the GC.

    Called when we're about to load a different run's model — without
    explicitly clearing, the old objects stay alive until ``_cache`` is
    overwritten, briefly doubling memory during the load. Caller must
    hold ``_cache_lock``.
    """
    global _cache
    _cache = None
    # Best-effort: free GPU/MPS allocations the previous model held.
    # Empty try/except: if torch isn't installed (pure-MLX boxes) or
    # the runtime doesn't expose empty_cache(), no-op.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_for_run(run: dict, runs_root: Path) -> _CacheEntry:
    """Resolve the right loader for the run's backend and return the
    cache entry. Cheaper than a separate dispatch table because each
    backend's import is only paid when that backend is selected.

    When ``run["output_dir"]`` is the empty string we treat that as
    a "load base model only, no adapter" signal — the eval suite
    uses this to compare the trained adapter against its base. A
    proper run always has a non-empty output_dir, so this isn't
    ambiguous in normal use.
    """
    backend = run["config"]["backend"]
    run_id = run["id"]
    output_dir = run["output_dir"]
    model_id = run["config"]["model_id"]
    base_only = not output_dir

    if backend in ("mlx", "mlx_vlm"):
        # mlx_lm's load() reads adapters.safetensors via adapter_path.
        # Lazy import: don't pay mlx_lm cost on non-MLX hosts. Pass
        # adapter_path=None (not "") for the base-only path so
        # mlx_lm knows to skip adapter loading.
        from mlx_lm.utils import load as mlx_load

        model, tokenizer = mlx_load(
            model_id, adapter_path=output_dir if not base_only else None,
        )
        return _CacheEntry(
            run_id=run_id, backend=backend, obj={"model": model, "tokenizer": tokenizer}
        )

    if backend in ("cuda", "cpu", "rocm", "cuda_vlm"):
        # PeftModel can read the adapter dir directly. We default to
        # CPU device when no GPU is available (the playground might be
        # used long after the original training device went away —
        # e.g. a cloud GPU run reviewed locally).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)

        if base_only:
            base.eval()
            return _CacheEntry(
                run_id=run_id,
                backend=backend,
                obj={"model": base, "tokenizer": tok, "device": device},
            )

        from peft import PeftModel

        adapter_dir = _find_hf_adapter_dir(Path(output_dir))
        if adapter_dir is None:
            raise FileNotFoundError(
                f"No HF adapter found under {output_dir}. The run may have "
                "failed before saving."
            )
        model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
        model.eval()
        return _CacheEntry(
            run_id=run_id,
            backend=backend,
            obj={"model": model, "tokenizer": tok, "device": device},
        )

    raise ValueError(f"Inference unsupported for backend {backend!r}")


def _find_hf_adapter_dir(run_dir: Path) -> Path | None:
    """Same probing logic as the GGUF merge step but for inference: the
    adapter may live at the run dir itself, or under the latest
    ``checkpoint-N/`` if HF Trainer saved checkpoints instead.
    """
    if (run_dir / "adapter_model.safetensors").exists():
        return run_dir
    checkpoints = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if checkpoints and (checkpoints[-1] / "adapter_model.safetensors").exists():
        return checkpoints[-1]
    return None


def _build_chat_prompt(tokenizer, prompt: str) -> str:
    """Prefer the chat template when the tokenizer has one — that's
    what the model was fine-tuned for. Fall back to the raw prompt if
    no template is present (base/completion models).
    """
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate_stream(
    run_dict: dict,
    cfg: GenerationConfig,
    runs_root: Path,
    cancel_event: threading.Event | None = None,
) -> Iterator[GenerationToken]:
    """Stream generated tokens for the given run + prompt.

    Yields ``GenerationToken`` frames: zero or more ``status`` hints
    (loading / queued), one or more ``token`` deltas with the decoded
    text, and finally a single ``done=True`` frame. The route layer
    wraps each yield as the corresponding SSE event.

    Reuses the most-recently-used model when the same run id is
    requested twice in a row; otherwise loads fresh under the cache
    lock. Concurrent callers serialize on the lock — for a different
    run_id the second caller has to wait for the first model to be
    swapped out, which can take a couple of seconds; we surface that
    via an early ``status`` frame so the UI shows progress instead
    of staring at a spinner.

    ``cancel_event``: when set, a HF model.generate is told to stop at
    the next step boundary (via StoppingCriteria) and the stream
    terminates cleanly with a ``done`` frame. The route sets this on
    SSE-client disconnect so a closed playground tab doesn't leave a
    multi-thousand-token generation churning in the background.
    """
    global _cache
    run_id = run_dict["id"]

    # First — synchronously read the cache state so we can surface a
    # "loading…" hint to the UI before we block on the lock. Doing it
    # outside the lock is fine: another thread might have changed
    # state since this read, but at worst we emit a status frame and
    # then take a fast path; the lock ensures correctness.
    with _cache_lock:
        cached = _cache
    needs_load = cached is None or cached.run_id != run_id
    if needs_load:
        if cached is None:
            yield GenerationToken(status="Loading model into memory…")
        else:
            yield GenerationToken(
                status=f"Switching cached model from run {cached.run_id} to {run_id}…"
            )

    with _cache_lock:
        # Re-check under the lock — a concurrent request could have
        # finished the load already.
        if _cache is None or _cache.run_id != run_id:
            _free_cache_locked()
            _cache = _load_for_run(run_dict, runs_root)
        entry = _cache

    if entry.backend in ("mlx", "mlx_vlm"):
        yield from _stream_mlx(entry, cfg, cancel_event)
    else:
        yield from _stream_hf(entry, cfg, cancel_event)


def _stream_mlx(
    entry: _CacheEntry,
    cfg: GenerationConfig,
    cancel_event: threading.Event | None = None,
) -> Iterator[GenerationToken]:
    """mlx_lm.stream_generate yields one decoded token chunk per step.

    Since mlx_lm versions ship the streaming API under slightly
    different names, we try the common spellings before giving up.

    Cancel: stream_generate is itself a generator running in this
    thread (no background worker), so abandoning the iteration via
    GeneratorExit naturally terminates the model's per-token loop on
    the next yield. The explicit cancel_event check below short-
    circuits one step earlier when we know we're already canceled,
    which keeps the wind-down within a single token instead of two.
    """
    model = entry.obj["model"]
    tok = entry.obj["tokenizer"]
    prompt = _build_chat_prompt(tok, cfg.prompt)

    # Try the modern API first (mlx_lm 0.21+); fall back to the older
    # location if needed. Either way the function yields chunks with
    # a ``.text`` attribute or directly returns text.
    try:
        from mlx_lm import stream_generate
    except ImportError:
        from mlx_lm.generate import stream_generate  # type: ignore[no-redef]

    try:
        for response in stream_generate(
            model, tok, prompt=prompt, max_tokens=cfg.max_tokens
        ):
            if cancel_event is not None and cancel_event.is_set():
                break
            # mlx_lm 0.21+ yields a ``GenerationResponse`` whose
            # ``.text`` is the decoded delta for this step. That
            # delta is LEGITIMATELY empty when the current token's
            # bytes haven't yet accumulated into a complete UTF-8
            # sequence (BPE tokenization can split a multi-byte
            # UTF-8 char across two tokens). The previous code
            # collapsed ``getattr(...) or str(response)``, which
            # turned every empty delta into a full repr dump like
            # ``GenerationResponse(text='', token=216, …)`` showing
            # up in the user's chat. We only fall back to ``str()``
            # when the response object genuinely lacks a ``.text``
            # attribute — that's the contract for older mlx_lm
            # versions that yielded raw strings.
            if hasattr(response, "text"):
                text = response.text or ""
            else:
                text = str(response)
            if text:
                yield GenerationToken(text=text)
    finally:
        yield GenerationToken(done=True)


def _stream_hf(
    entry: _CacheEntry,
    cfg: GenerationConfig,
    cancel_event: threading.Event | None = None,
) -> Iterator[GenerationToken]:
    """Bridge transformers' ``TextIteratorStreamer`` (push-based, fed
    by ``model.generate`` running in a background thread) to a
    pull-based generator. The streamer's iterator yields decoded
    deltas as soon as each token clears the model's forward pass.

    Cancel: model.generate runs in a daemon thread, so abandoning the
    consumer iteration won't stop the worker on its own. We attach a
    StoppingCriteria that polls ``cancel_event``; on the next forward
    pass the model returns and the worker exits cleanly. The route
    sets the event when the SSE consumer disconnects, so closing a
    playground tab doesn't leave a 2k-token generation churning in
    the background.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    model = entry.obj["model"]
    tok = entry.obj["tokenizer"]
    device = entry.obj["device"]
    prompt = _build_chat_prompt(tok, cfg.prompt)
    inputs = tok(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tok, skip_prompt=True, skip_special_tokens=True
    )

    # Local cancel event the route sets via cancel_event_ref. We
    # always create our own internally so the StoppingCriteria has
    # something to poll even when the route didn't pass one in (e.g.
    # synthetic test invocations).
    local_cancel = cancel_event if cancel_event is not None else threading.Event()

    class _CancelOnEvent(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):  # noqa: D401
            return local_cancel.is_set()

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=cfg.max_tokens,
        do_sample=cfg.temperature > 0,
        temperature=max(cfg.temperature, 1e-5),
        top_p=cfg.top_p,
        streamer=streamer,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        stopping_criteria=StoppingCriteriaList([_CancelOnEvent()]),
    )

    # Channel for surfacing model.generate exceptions back to the
    # consumer. Without this, an exception in the worker thread would
    # leave the streamer iterator hanging (HF's TextIteratorStreamer
    # only ends when on_finish() fires) and the user would see a
    # truncated response with no error context.
    err_holder: dict[str, BaseException | None] = {"err": None}

    def _runner() -> None:
        try:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        except BaseException as e:  # noqa: BLE001 — re-raised on consumer side
            err_holder["err"] = e
        finally:
            # Always close the streamer's queue so its iterator
            # terminates. Without this, the consumer's ``for chunk in
            # streamer`` blocks forever when generate raises before
            # any token is emitted.
            try:
                streamer.end()
            except Exception:
                pass

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    try:
        for chunk in streamer:
            if chunk:
                yield GenerationToken(text=chunk)
    finally:
        # Tell the worker to stop at the next step. If the consumer
        # abandoned mid-stream (GeneratorExit), the StoppingCriteria
        # sees this on the next forward pass and model.generate
        # returns; without it the worker would run to max_tokens and
        # hold the GPU/MPS for many more seconds.
        local_cancel.set()
        thread.join(timeout=5)
    if err_holder["err"] is not None:
        raise err_holder["err"]
    yield GenerationToken(done=True)


def free_cache() -> None:
    """Public hook for tests / shutdown to drop the cached model."""
    with _cache_lock:
        _free_cache_locked()


def cached_run_id() -> str | None:
    with _cache_lock:
        return _cache.run_id if _cache else None
