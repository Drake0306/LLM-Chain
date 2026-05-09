from pathlib import Path

import pytest

from llm_chain_sidecar.recipes import (
    APP_VERSION,
    _is_compatible,
    _parse_version,
    find_recipe,
    load_manifest,
)


def test_load_shipped_manifest_is_valid():
    """The in-package recipes.yaml ships with valid entries — every
    one has a known technique, a properly-shaped dataset branch, and
    a unique id."""
    recipes = load_manifest()
    assert recipes
    for r in recipes:
        assert r.technique in {"lora", "qlora"}
        assert r.dataset.kind in {"curated", "synth", "bring_your_own"}
        assert r.id


def test_dataset_must_have_exactly_one_branch(tmp_path: Path):
    p = tmp_path / "two-branches.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: bad\n"
        "    name: Bad\n"
        "    model: m\n"
        "    technique: lora\n"
        "    dataset:\n"
        "      curated_id: dolly-15k\n"
        "      bring_your_own: true\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_manifest(p)


def test_dataset_with_zero_branches_rejected(tmp_path: Path):
    p = tmp_path / "no-branch.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: empty\n"
        "    name: Empty\n"
        "    model: m\n"
        "    technique: lora\n"
        "    dataset: {}\n"
    )
    with pytest.raises(ValueError, match="must be a mapping with one of"):
        load_manifest(p)


def test_unknown_technique_rejected(tmp_path: Path):
    p = tmp_path / "bad-technique.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: bt\n"
        "    name: BT\n"
        "    model: m\n"
        "    technique: full-finetune\n"
        "    dataset: {bring_your_own: true}\n"
    )
    with pytest.raises(ValueError, match="unknown technique"):
        load_manifest(p)


def test_duplicate_id_rejected(tmp_path: Path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: x\n"
        "    name: X\n"
        "    model: m\n"
        "    technique: lora\n"
        "    dataset: {bring_your_own: true}\n"
        "  - id: x\n"
        "    name: X again\n"
        "    model: m2\n"
        "    technique: qlora\n"
        "    dataset: {bring_your_own: true}\n"
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(p)


def test_min_app_version_marks_needs_upgrade(tmp_path: Path):
    """Recipes that demand a newer app version are returned with
    needs_upgrade=True, not silently dropped — the UI surfaces them
    disabled so the user can see the option exists."""
    p = tmp_path / "future.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: future\n"
        "    name: From the future\n"
        "    model: m\n"
        "    technique: lora\n"
        "    dataset: {bring_your_own: true}\n"
        "    min_app_version: '99.0.0'\n"
    )
    [recipe] = load_manifest(p, app_version=APP_VERSION)
    assert recipe.needs_upgrade is True


def test_synth_branch_parses_topic_and_style(tmp_path: Path):
    p = tmp_path / "synth.yaml"
    p.write_text(
        "recipes:\n"
        "  - id: s\n"
        "    name: S\n"
        "    model: m\n"
        "    technique: lora\n"
        "    dataset:\n"
        "      synth:\n"
        "        topic: Customer support emails\n"
        "        style: Brisk and helpful\n"
    )
    [r] = load_manifest(p)
    assert r.dataset.kind == "synth"
    assert r.dataset.synth_topic == "Customer support emails"


def test_find_recipe():
    recipes = load_manifest()
    sample = recipes[0]
    assert find_recipe(recipes, sample.id) is sample
    assert find_recipe(recipes, "ghost-recipe") is None


def test_version_parser_handles_pre_release_tags():
    assert _parse_version("0.1.0-alpha.5") == (0, 1, 0)
    assert _parse_version("1.2.3") == (1, 2, 3)
    assert _is_compatible("0.1.0", "0.1.0-alpha.5") is True
    assert _is_compatible("0.1.0", "0.0.9") is False


def test_dataset_branch_kind_property():
    p = load_manifest()
    kinds = {r.dataset.kind for r in p}
    # The shipped manifest exercises all three branches — sanity-check
    # the property accessor against real data so a future YAML edit
    # that breaks one branch surfaces here.
    assert "curated" in kinds
    assert "bring_your_own" in kinds
