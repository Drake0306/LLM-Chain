"""Per-layer adapter diff (F-D18).

Computes the Frobenius norm of ``A - B`` for every matching tensor
key across two LoRA adapters. The result is a flat dict the
frontend renders as a vertical heatmap so the user can answer
"where did the fine-tune actually change the model?" without
loading weights into a notebook.

Why Frobenius (i.e. ``sqrt(sum((A - B) ** 2))``): it's the natural
matrix norm for "how far apart are these weight matrices on
average?", scales with the number of elements in a way that lets
cross-layer comparison make sense, and matches what mergekit /
PEFT pretty-printers tend to use. Per-layer mean and max are also
returned so the heatmap can normalise without re-scanning.

Adapter-only — we don't merge into the base or load the base at
all, so this is fast (single-digit seconds for any reasonable
LoRA) and works without GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LayerDiff:
    key: str
    frobenius: float
    abs_max: float
    shape: tuple[int, ...]


@dataclass
class DiffResult:
    base_model: str
    layers: list[LayerDiff]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "layers": [
                {
                    "key": ld.key,
                    "frobenius": ld.frobenius,
                    "abs_max": ld.abs_max,
                    "shape": list(ld.shape),
                }
                for ld in self.layers
            ],
            "summary": self.summary,
        }


def _load_adapter(adapter_dir: Path) -> tuple[dict, dict]:
    """Read the adapter's tensors + config. Returns (tensors, config).

    Raises FileNotFoundError when either file is missing — the
    route layer maps that to a 400 with a hint pointing at the
    adapter file.
    """
    weights = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    if not weights.exists():
        raise FileNotFoundError(
            f"No adapter_model.safetensors at {adapter_dir}. The diff "
            "view needs the safetensors weights file alongside the "
            "config — re-run the trainer or pick a different run."
        )
    if not config.exists():
        raise FileNotFoundError(
            f"No adapter_config.json at {adapter_dir}. Diff needs both "
            "files."
        )
    from safetensors.torch import load_file

    return load_file(str(weights)), json.loads(config.read_text())


def diff_adapters(a_dir: Path, b_dir: Path) -> DiffResult:
    """Compute per-layer ``||A - B||_F`` across matching tensor keys.

    Validation:
      - Both adapters must share base_model_name_or_path + r + alpha.
        Cross-base diffs would compare weights of incompatible
        shapes; the route layer also gates this but we re-validate
        from disk as defence-in-depth.
      - Keys present in only one adapter are reported under the
        summary's ``unmatched_keys`` field rather than silently
        dropped — the user might want to know why the diff under-
        represents differences.

    Returns a DiffResult with one LayerDiff per matched key (sorted
    by Frobenius norm descending, so the heatmap top is the layer
    that changed most) plus a summary block.
    """
    import torch

    a_tensors, a_cfg = _load_adapter(a_dir)
    b_tensors, b_cfg = _load_adapter(b_dir)

    for field in ("base_model_name_or_path", "r", "lora_alpha"):
        if a_cfg.get(field) != b_cfg.get(field):
            raise ValueError(
                f"Adapters disagree on {field!r}: "
                f"{a_cfg.get(field)!r} vs {b_cfg.get(field)!r}. "
                "Diff requires identical base model + LoRA shape."
            )

    matched = sorted(set(a_tensors.keys()) & set(b_tensors.keys()))
    only_a = sorted(set(a_tensors.keys()) - set(b_tensors.keys()))
    only_b = sorted(set(b_tensors.keys()) - set(a_tensors.keys()))

    layers: list[LayerDiff] = []
    for key in matched:
        a_t = a_tensors[key]
        b_t = b_tensors[key]
        if a_t.shape != b_t.shape:
            # Shape mismatch on a matched key shouldn't happen
            # given the config check, but if it does, skip rather
            # than crash — record under unmatched.
            only_a.append(key)
            continue
        delta = (a_t - b_t).to(torch.float32)
        # Frobenius norm via flatten so non-2D tensors (1D LoRA bias,
        # embedding rows) work with the same code path. ``norm("fro")``
        # is only defined for 2D inputs in newer torch; the flattened
        # L2 norm is mathematically identical for matrices and well-
        # defined for arbitrary shapes.
        frob = float(torch.linalg.norm(delta.flatten()).item())
        absmax = float(delta.abs().max().item()) if delta.numel() else 0.0
        layers.append(
            LayerDiff(
                key=key,
                frobenius=frob,
                abs_max=absmax,
                shape=tuple(a_t.shape),
            )
        )

    # Sort by Frobenius descending so the user sees the layers that
    # changed most at the top of the heatmap.
    layers.sort(key=lambda ld: ld.frobenius, reverse=True)

    summary = {
        "matched_count": len(layers),
        "unmatched_keys": {"only_a": only_a, "only_b": only_b},
        "max_frobenius": layers[0].frobenius if layers else 0.0,
        "mean_frobenius": (
            sum(ld.frobenius for ld in layers) / len(layers)
            if layers
            else 0.0
        ),
    }
    return DiffResult(
        base_model=a_cfg.get("base_model_name_or_path", "unknown"),
        layers=layers,
        summary=summary,
    )


__all__ = [
    "DiffResult",
    "LayerDiff",
    "diff_adapters",
]
