"""Adapter-level model merging (F-C12).

Combine two or more LoRA adapters that share a base model into a new
adapter. Three methods supported:

- ``linear``: weighted average of LoRA A/B matrices. Simplest and most
  predictable; the user supplies one weight per source adapter.
- ``ties``: TIES-merging — sign-aware redundant-parameter trim, then
  weighted average. Less prone to "averaging out" complementary
  capabilities than plain linear.
- ``dare``: DARE (Drop And REscale) — random sparsification of each
  adapter's delta before averaging. Cheap, often surprisingly good.

The merged result is saved as a new run with ``purpose: "merged"``
so the rest of the app (Library, playground, eval, GGUF export)
treats it like any other completed adapter without special-casing.

Why adapter-only and not full-model merging: full-model mergekit
needs many GB of base weights resident in memory and produces a
new merged base that can't be hot-swapped at inference time. The
adapter-merge story is more honest about what we can do well in a
local-first desktop app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

MergeMethod = Literal["linear", "ties", "dare"]
SUPPORTED_METHODS: tuple[MergeMethod, ...] = ("linear", "ties", "dare")


@dataclass
class MergeInput:
    """One source adapter + the weight it contributes to the merge."""

    run_id: str
    adapter_dir: Path
    weight: float


@dataclass
class MergeResult:
    """What the merger reports back to the route layer."""

    output_dir: Path
    method: MergeMethod
    sources: list[str]
    weights: list[float]
    tensor_count: int


def _load_safetensors(path: Path) -> dict:
    """Read every tensor from ``adapter_model.safetensors`` into a
    plain dict. Lazy import keeps safetensors off the import-cost
    path on hosts that never merge.
    """
    from safetensors.torch import load_file

    return load_file(str(path))


def _save_safetensors(tensors: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def _load_adapter_config(adapter_dir: Path) -> dict:
    """The PEFT adapter_config.json carries the LoRA shape — rank,
    alpha, target_modules. We need it to (a) verify shape parity
    across the inputs and (b) write a matching config for the merged
    output so PeftModel.from_pretrained can load it back."""
    p = adapter_dir / "adapter_config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p} — adapter merging needs the PEFT config "
            "alongside the safetensors weights."
        )
    return json.loads(p.read_text())


def _validate_inputs(inputs: list[MergeInput]) -> dict:
    """Check shape parity across every source adapter.

    Returns the shared adapter_config (the first one — they all match
    so the choice is arbitrary). Raises ``ValueError`` with a precise
    diff message when configs disagree on the dimensions that affect
    weight shape.
    """
    if len(inputs) < 2:
        raise ValueError(
            f"Need at least 2 source adapters to merge; got {len(inputs)}."
        )
    configs = [_load_adapter_config(inp.adapter_dir) for inp in inputs]
    base_id_field = "base_model_name_or_path"
    base = configs[0]
    for cfg, inp in zip(configs[1:], inputs[1:]):
        for field in ("r", "lora_alpha", base_id_field):
            if cfg.get(field) != base.get(field):
                raise ValueError(
                    f"Adapter {inp.run_id} disagrees with the first input on "
                    f"{field!r}: {cfg.get(field)!r} vs {base.get(field)!r}. "
                    "Merging requires identical base model + LoRA shape."
                )
        # target_modules can be a list, a set-like, or the string
        # "all-linear" (PEFT shorthand for "every Linear layer in
        # the model"). Two adapters that agree at the *string* level
        # are trivially equal; two that agree as a set are equal once
        # we normalise to a frozenset. The tricky case is "all-linear"
        # (string) vs the expanded list — same logical target, but
        # we can't expand here without loading the base model.
        # We treat the bare-string form as a wildcard that matches
        # itself only; mismatches between "all-linear" and an
        # explicit list raise so the user can re-save one of the
        # adapters with the matching format.
        a = cfg.get("target_modules")
        b = base.get("target_modules")

        def _norm(v):
            if isinstance(v, (list, tuple, set)):
                return frozenset(v)
            return v

        if _norm(a) != _norm(b):
            raise ValueError(
                f"Adapter {inp.run_id} targets different modules: "
                f"{a!r} vs {b!r}. Merging requires identical "
                "target_modules. If one adapter shipped 'all-linear' "
                "and another the expanded list, re-save the expanded "
                "one as 'all-linear' (or vice versa) so the strings "
                "agree."
            )
    total = sum(inp.weight for inp in inputs)
    if total <= 0:
        raise ValueError(
            "Sum of merge weights must be > 0 (got "
            f"{[inp.weight for inp in inputs]})."
        )
    return base


def _normalise_weights(inputs: list[MergeInput]) -> list[float]:
    total = sum(inp.weight for inp in inputs)
    return [inp.weight / total for inp in inputs]


def _merge_linear(weights: list[float], tensors: list[dict]) -> dict:
    """Element-wise weighted sum across the per-adapter tensor dicts.

    All inputs share the same key set (verified by ``_validate_inputs``);
    iterate over the first dict's keys and combine matching tensors.
    """
    out: dict = {}
    for key in tensors[0].keys():
        # Stack with the first tensor's dtype/device — torch will
        # broadcast sensibly when adapters were saved at different
        # precisions on the same hardware family.
        first = tensors[0][key]
        accum = first * weights[0]
        for w, td in zip(weights[1:], tensors[1:]):
            if key not in td:
                raise ValueError(
                    f"Key {key!r} present in adapter 0 but missing in "
                    f"adapter {tensors.index(td)} — adapter shape mismatch."
                )
            accum = accum + td[key] * w
        out[key] = accum
    return out


def _merge_ties(
    weights: list[float], tensors: list[dict], density: float = 0.2,
) -> dict:
    """TIES-merging on adapter deltas.

    1. Trim: keep top-``density`` fraction of |Δ| in each adapter,
       zero the rest.
    2. Elect sign: per-element, sum the trimmed deltas and take the
       sign of the result.
    3. Disjoint merge: average only the elements whose sign matches
       the elected sign; zero everything else.

    The density default (0.2) follows the original TIES paper. Lower
    values trim more aggressively; higher values trend toward the
    plain linear merge.
    """
    import torch

    out: dict = {}
    for key in tensors[0].keys():
        stacked = torch.stack([td[key] for td in tensors], dim=0)
        weighted = stacked * torch.tensor(weights).view(
            -1, *([1] * (stacked.dim() - 1)),
        )
        # 1. Trim each adapter independently to its top-density |Δ|.
        flat = weighted.view(weighted.shape[0], -1)
        k = max(1, int(density * flat.shape[1]))
        thresholds = flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1).values
        keep_mask = flat.abs() >= thresholds.unsqueeze(1)
        trimmed = (flat * keep_mask).view(weighted.shape)
        # 2. Elect sign element-wise.
        elected = trimmed.sum(dim=0).sign()
        # 3. Disjoint merge: average elements whose sign matches.
        # Edge case: when every adapter trims a position to 0,
        # ``elected`` is 0 and ``sign_match = (0 == 0)`` is True
        # everywhere, so the position averages 0/N = 0. Outcome is
        # the desired zero, but to make the intent explicit (and
        # robust to a future refactor that flips the sign-match
        # semantics) we mask sign_match to False on positions where
        # nothing survived the trim.
        nonzero_elected = (elected != 0).unsqueeze(0)
        sign_match = (trimmed.sign() == elected.unsqueeze(0)) & nonzero_elected
        kept = trimmed * sign_match
        denom = sign_match.float().sum(dim=0).clamp_min(1)
        out[key] = kept.sum(dim=0) / denom
    return out


def _merge_dare(
    weights: list[float],
    tensors: list[dict],
    drop_p: float = 0.5,
    seed: int = 0,
) -> dict:
    """DARE (Drop And REscale) merge.

    For each adapter, randomly zero ``drop_p`` of the elements and
    rescale the survivors by ``1/(1-drop_p)``. Combine with the
    weighted sum at the end. The dropout is per-element rather than
    structured, which preserves expectations cheaply.

    Reproducibility: takes a ``seed`` so the same recipe + seed
    produces the same merged adapter. Without this, ``torch.rand_like``
    pulled from the global RNG made every re-run produce a slightly
    different artifact — bad for "did my eval improve?" debugging.
    The audit file records the resolved seed so a re-run is
    bit-identical when the user asks for one.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    out: dict = {}
    survival = 1.0 - drop_p
    for key in tensors[0].keys():
        first = tensors[0][key]
        accum = torch.zeros_like(first)
        for w, td in zip(weights, tensors):
            t = td[key]
            # Generate the random tensor on CPU then move to t.device
            # so the seed produces consistent results regardless of
            # whether the adapter weights live on CPU / CUDA / MPS.
            mask = (
                (
                    torch.rand(t.shape, generator=gen) > drop_p
                ).to(t.dtype).to(t.device)
            )
            accum = accum + (t * mask / survival) * w
        out[key] = accum
    return out


def merge_adapters(
    inputs: list[MergeInput],
    *,
    method: MergeMethod,
    output_dir: Path,
    method_options: dict | None = None,
) -> MergeResult:
    """Compute a merged adapter under ``output_dir``.

    Side effects:
    - Writes ``output_dir/adapter_model.safetensors`` with the merged
      tensors.
    - Copies the first input's ``adapter_config.json`` so PeftModel
      can read the merged result back at inference time.
    - Writes a sibling ``merge.json`` with the recipe (method,
      sources, weights) for audit.

    Raises:
    - ValueError: shape mismatch, < 2 inputs, weights sum to 0,
      unknown method.
    - FileNotFoundError: missing adapter_config.json or
      adapter_model.safetensors in any input.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown merge method {method!r}; pick one of {list(SUPPORTED_METHODS)}."
        )
    base_cfg = _validate_inputs(inputs)
    weights = _normalise_weights(inputs)

    tensors_per_input: list[dict] = []
    for inp in inputs:
        weights_path = inp.adapter_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing {weights_path} — adapter merging needs the "
                "safetensors weights file alongside adapter_config.json."
            )
        tensors_per_input.append(_load_safetensors(weights_path))

    opts = method_options or {}
    # Persist the *resolved* method-options (with defaults filled in)
    # so the audit file captures exactly what was applied — including
    # the DARE seed, which is the only knob that flips the result
    # under fixed inputs.
    resolved_opts: dict = {}
    if method == "linear":
        merged = _merge_linear(weights, tensors_per_input)
    elif method == "ties":
        density = float(opts.get("density", 0.2))
        resolved_opts["density"] = density
        merged = _merge_ties(weights, tensors_per_input, density=density)
    else:  # dare
        drop_p = float(opts.get("drop_p", 0.5))
        seed = int(opts.get("seed", 0))
        resolved_opts["drop_p"] = drop_p
        resolved_opts["seed"] = seed
        merged = _merge_dare(
            weights, tensors_per_input, drop_p=drop_p, seed=seed,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_safetensors(merged, output_dir / "adapter_model.safetensors")
    (output_dir / "adapter_config.json").write_text(json.dumps(base_cfg, indent=2))
    (output_dir / "merge.json").write_text(
        json.dumps(
            {
                "method": method,
                # Resolved options carry defaults the user didn't pass
                # (DARE seed in particular) so re-running the recipe
                # is bit-identical when the file is replayed.
                "method_options": resolved_opts,
                "sources": [inp.run_id for inp in inputs],
                "raw_weights": [inp.weight for inp in inputs],
                "weights": weights,
                "tensor_count": len(merged),
            },
            indent=2,
        )
    )
    return MergeResult(
        output_dir=output_dir,
        method=method,
        sources=[inp.run_id for inp in inputs],
        weights=weights,
        tensor_count=len(merged),
    )


__all__ = [
    "MergeInput",
    "MergeMethod",
    "MergeResult",
    "SUPPORTED_METHODS",
    "merge_adapters",
]
