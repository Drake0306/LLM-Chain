"""Recipe manifest loader (F-B7).

A recipe is a YAML descriptor that pre-fills the Train page: model id,
technique, dataset reference (curated id, synth spec, or bring-your-own),
and hyperparameter defaults. The frontend reads /api/recipes, the user
clicks one, and the selection state is populated end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

# Bumped whenever the recipe schema changes shape — recipes whose
# min_app_version is higher than this are surfaced as "needs newer
# app" rather than parsed and possibly mis-applied. Synced with the
# package's pyproject.toml at ship time; the runtime check uses
# tuple comparison so semver pre-release tags are tolerated.
APP_VERSION = "0.1.0"


@dataclass
class RecipeDataset:
    """How a recipe expresses its dataset preference.

    Exactly one branch should be set — ``curated_id`` references a
    F-B6 entry, ``synth`` describes a F-A2 generator preset, and
    ``bring_your_own`` flags a recipe that intentionally leaves the
    dataset blank for the user to fill in.
    """

    curated_id: str | None = None
    synth_topic: str | None = None
    synth_style: str | None = None
    bring_your_own: bool = False

    @property
    def kind(self) -> str:
        if self.curated_id:
            return "curated"
        if self.synth_topic is not None:
            return "synth"
        if self.bring_your_own:
            return "bring_your_own"
        return "unknown"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "curated_id": self.curated_id,
            "synth_topic": self.synth_topic,
            "synth_style": self.synth_style,
            "bring_your_own": self.bring_your_own,
        }


@dataclass
class RecipeHyperparameters:
    epochs: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 1

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.batch_size,
        }


@dataclass
class Recipe:
    id: str
    name: str
    description: str
    model: str
    technique: str
    dataset: RecipeDataset
    hyperparameters: RecipeHyperparameters
    min_app_version: str = "0.0.0"
    suggested_backend: str | None = None
    notes: str = ""
    # Set by load_manifest when the recipe's min_app_version exceeds
    # the running APP_VERSION. The frontend renders these greyed-out
    # with a "needs newer app version" hint instead of hiding them
    # entirely so the user knows the option exists.
    needs_upgrade: bool = field(default=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "technique": self.technique,
            "dataset": self.dataset.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "min_app_version": self.min_app_version,
            "suggested_backend": self.suggested_backend,
            "notes": self.notes,
            "needs_upgrade": self.needs_upgrade,
        }


_KNOWN_TECHNIQUES = {"lora", "qlora"}


def _parse_version(s: str) -> tuple[int, ...]:
    """Lenient semver parse — splits on the first non-numeric run so
    pre-release tags like '0.1.0-alpha.5' compare against numeric
    counterparts cleanly. Missing components default to 0.
    """
    parts: list[int] = []
    cur = ""
    for ch in s.strip():
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                parts.append(int(cur))
                cur = ""
            if ch != "." and ch != "-":
                # Stop at any non-separator, non-digit — we ignore
                # pre-release suffixes for the gating check.
                break
    if cur:
        parts.append(int(cur))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _is_compatible(recipe_min: str, app_version: str) -> bool:
    return _parse_version(recipe_min) <= _parse_version(app_version)


def _parse_dataset(raw: Any) -> RecipeDataset:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "recipe dataset must be a mapping with one of: curated_id, "
            "synth, bring_your_own"
        )
    branches = sum(
        1
        for k in ("curated_id", "synth", "bring_your_own")
        if k in raw and raw[k]
    )
    if branches != 1:
        raise ValueError(
            "recipe dataset must set exactly one of curated_id, synth, "
            "or bring_your_own"
        )
    if "curated_id" in raw and raw["curated_id"]:
        return RecipeDataset(curated_id=str(raw["curated_id"]))
    if "synth" in raw and raw["synth"]:
        synth = raw["synth"]
        if not isinstance(synth, dict):
            raise ValueError("synth branch must be a mapping with topic/style")
        return RecipeDataset(
            synth_topic=str(synth.get("topic", "")).strip() or None,
            synth_style=str(synth.get("style", "")).strip() or None,
        )
    return RecipeDataset(bring_your_own=True)


def _parse_hyperparameters(raw: Any) -> RecipeHyperparameters:
    if raw is None:
        return RecipeHyperparameters()
    if not isinstance(raw, dict):
        raise ValueError("hyperparameters must be a mapping")
    return RecipeHyperparameters(
        epochs=int(raw.get("epochs", 1)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        lora_rank=int(raw.get("lora_rank", 16)),
        lora_alpha=int(raw.get("lora_alpha", 32)),
        batch_size=int(raw.get("batch_size", 1)),
    )


def load_manifest(
    path: Path | None = None,
    app_version: str = APP_VERSION,
) -> list[Recipe]:
    """Parse the in-package recipes.yaml into typed entries.

    Each entry is validated for technique enum, hyperparameter ranges,
    and the dataset branch invariant (exactly one of curated/synth/
    BYO). Recipes whose ``min_app_version`` is newer than the running
    app are marked ``needs_upgrade=True`` rather than dropped — the
    UI shows them disabled with a hint.
    """
    if path is None:
        with resources.as_file(
            resources.files(__package__) / "recipes.yaml",
        ) as p:
            text = Path(p).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")

    raw = yaml.safe_load(text) or {}
    recipes: list[Recipe] = []
    seen_ids: set[str] = set()

    for raw_entry in raw.get("recipes", []) or []:
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"recipes.yaml: each entry must be a mapping, got "
                f"{type(raw_entry).__name__}"
            )
        try:
            entry = Recipe(
                id=raw_entry["id"],
                name=raw_entry["name"],
                description=raw_entry.get("description", ""),
                model=raw_entry["model"],
                technique=raw_entry["technique"],
                dataset=_parse_dataset(raw_entry.get("dataset")),
                hyperparameters=_parse_hyperparameters(
                    raw_entry.get("hyperparameters"),
                ),
                min_app_version=str(raw_entry.get("min_app_version", "0.0.0")),
                suggested_backend=(
                    (raw_entry.get("device") or {}).get("suggested_backend")
                    if raw_entry.get("device") is not None
                    else None
                ),
                notes=raw_entry.get("notes", "").strip(),
            )
        except KeyError as e:
            raise ValueError(
                f"recipes.yaml: entry missing required field {e.args[0]!r}"
            ) from e
        if entry.technique not in _KNOWN_TECHNIQUES:
            raise ValueError(
                f"recipes.yaml: entry {entry.id!r} has unknown technique "
                f"{entry.technique!r}; expected one of {sorted(_KNOWN_TECHNIQUES)}"
            )
        if entry.id in seen_ids:
            raise ValueError(f"recipes.yaml: duplicate id {entry.id!r}")
        seen_ids.add(entry.id)
        entry.needs_upgrade = not _is_compatible(
            entry.min_app_version, app_version,
        )
        recipes.append(entry)
    return recipes


def find_recipe(recipes: list[Recipe], recipe_id: str) -> Recipe | None:
    return next((r for r in recipes if r.id == recipe_id), None)


__all__ = [
    "APP_VERSION",
    "Recipe",
    "RecipeDataset",
    "RecipeHyperparameters",
    "find_recipe",
    "load_manifest",
]
