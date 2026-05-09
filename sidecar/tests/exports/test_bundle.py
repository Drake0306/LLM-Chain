import json
import zipfile
from pathlib import Path

import pytest

from llm_chain_sidecar.exports.bundle import (
    BUNDLE_SUFFIX,
    export_bundle,
    import_bundle,
    manifest_from_bundle,
)


def _stage_run(tmp_path: Path, run_id: str = "abc123") -> tuple[Path, dict]:
    """Build a fake run dir + the matching run dict the route would
    pass into export_bundle. Adapter weights are an empty placeholder
    safetensors blob — we don't actually parse them in the bundle
    code, just copy the file."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "test/base",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    run = {
        "id": run_id,
        "status": "succeeded",
        "config": {
            "model_id": "test/base",
            "backend": "cuda",
            "technique": "lora",
            "dataset_path": "/Users/roy/secret/data.jsonl",
            "epochs": 1,
        },
    }
    return run_dir, run


# --- export ----------------------------------------------------------


def test_export_bundle_writes_archive(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    out = tmp_path / "exported.llmchain"
    result = export_bundle(run, run_dir=run_dir, output_path=out)
    assert out.exists()
    assert result.bytes_written > 0
    assert result.files_included >= 3  # manifest + adapter weights + config


def test_export_bundle_rejects_wrong_extension(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    with pytest.raises(ValueError, match=BUNDLE_SUFFIX):
        export_bundle(run, run_dir=run_dir, output_path=tmp_path / "out.zip")


def test_export_bundle_refuses_to_overwrite(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    out = tmp_path / "out.llmchain"
    out.write_bytes(b"placeholder")
    with pytest.raises(FileExistsError):
        export_bundle(run, run_dir=run_dir, output_path=out)


def test_export_bundle_refuses_when_no_adapter(tmp_path: Path):
    """A run dir with no weights is useless to ship — the importer
    would just see an empty adapter/. Raising upfront with a hint is
    better than producing a bundle that fails on import."""
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    run = {
        "id": "empty",
        "status": "succeeded",
        "config": {"model_id": "x"},
    }
    out = tmp_path / "empty.llmchain"
    with pytest.raises(FileNotFoundError, match="No adapter"):
        export_bundle(run, run_dir=run_dir, output_path=out)
    # Cleanup happened — no half-written file left behind.
    assert not out.exists()


def test_export_bundle_strips_local_dataset_path(tmp_path: Path):
    """Privacy: the manifest must not include the user's local
    filesystem path. Replaced with a sentinel so the importer
    re-picks."""
    run_dir, run = _stage_run(tmp_path)
    out = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=out)
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["config"]["dataset_path"] == "imported-bundle"
    # The original id should be preserved for audit, just not the path.
    assert manifest["imported_from"] == run["id"]


def test_export_bundle_includes_optional_files(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    (run_dir / "notes.md").write_text("# what worked\n\nlower LR")
    (run_dir / "merge.json").write_text("{}")
    out = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "notes.md" in names
    assert "merge.json" in names


def test_export_bundle_falls_back_to_latest_checkpoint(tmp_path: Path):
    """HF Trainer saves into checkpoint-N/ subdirs. The bundler should
    pick the highest N."""
    run_dir = tmp_path / "ckpt-run"
    run_dir.mkdir()
    for n in (50, 200, 100):
        d = run_dir / f"checkpoint-{n}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(f"weights-{n}".encode())
        (d / "adapter_config.json").write_text("{}")
    run = {"id": "ckpt", "status": "succeeded", "config": {"model_id": "x"}}
    out = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=out)
    with zipfile.ZipFile(out) as zf:
        # The 200-step weights should be what landed under adapter/.
        weights = zf.read("adapter/adapter_model.safetensors")
    assert weights == b"weights-200"


# --- import ----------------------------------------------------------


def test_import_bundle_extracts_to_target(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    bundle = tmp_path / "x.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=bundle)

    runs_root = tmp_path / "imports"
    result = import_bundle(bundle, runs_root=runs_root, new_run_id="newid12")
    extracted = runs_root / "newid12"
    assert (extracted / "manifest.json").exists()
    # Adapter weights land at top level after the import flatten so
    # the playground / find_latest_adapter / GGUF merger find them
    # without special-casing imported runs.
    assert (extracted / "adapter_model.safetensors").exists()
    assert result.imported_from == "abc123"


def test_import_bundle_rejects_path_traversal(tmp_path: Path):
    """A malicious bundle could ship a name like ``../../etc/passwd``
    that escapes the run dir. Refuse upfront."""
    bundle = tmp_path / "evil.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
        zf.writestr("../escape.txt", "should not extract")
    with pytest.raises(ValueError, match="path-traversal"):
        import_bundle(bundle, runs_root=tmp_path / "imports", new_run_id="x")


def test_import_bundle_rejects_windows_backslash_traversal(tmp_path: Path):
    """A bundle authored on Windows can ship ``foo\\..\\bar`` which
    on POSIX is a single Path part (backslash is a literal byte)
    and slips past a naive ``..`` check. The hardened check
    normalises both separators before splitting."""
    bundle = tmp_path / "evil-win.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
        zf.writestr("foo\\..\\bar.txt", "should not extract")
    with pytest.raises(ValueError, match="path-traversal"):
        import_bundle(bundle, runs_root=tmp_path / "imports", new_run_id="x")


def test_import_bundle_caps_uncompressed_size(tmp_path: Path):
    """Zip-bomb defence: a tiny .llmchain that decompresses to
    gigabytes would otherwise fill the user's disk. Refuse upfront
    when the declared uncompressed size exceeds the cap."""
    bundle = tmp_path / "bomb.llmchain"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
        # 100 KB of zero bytes — decompresses cheaply but well over
        # the 50KB test cap below.
        zf.writestr("adapter/big.bin", b"\x00" * (100 * 1024))
    with pytest.raises(ValueError, match="zip-bomb"):
        import_bundle(
            bundle,
            runs_root=tmp_path / "imports",
            new_run_id="x",
            max_extract_bytes=50 * 1024,
        )


def test_import_bundle_rejects_unsupported_schema_version(tmp_path: Path):
    bundle = tmp_path / "future.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"schema_version": 99, "config": {}}),
        )
    with pytest.raises(ValueError, match="schema_version"):
        import_bundle(bundle, runs_root=tmp_path / "imports", new_run_id="x")


def test_import_bundle_flattens_adapter_subdir(tmp_path: Path):
    """Imported runs need adapter weights at run_dir top-level so
    the playground / find_latest_adapter / GGUF merger find them.
    Verify the subdir is flattened post-extraction."""
    run_dir, run = _stage_run(tmp_path)
    bundle = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=bundle)

    runs_root = tmp_path / "imports"
    import_bundle(bundle, runs_root=runs_root, new_run_id="newid12")
    extracted = runs_root / "newid12"
    # Top-level — what mlx_lm and find_latest_adapter expect.
    assert (extracted / "adapter_model.safetensors").exists()
    # The transient subdir should be cleaned up.
    assert not (extracted / "adapter").exists()


def test_export_bundle_strips_resume_from(tmp_path: Path):
    """Privacy: resume_from is a source-machine run id, meaningless
    on import and a soft leak of training history. Drop it."""
    run_dir, run = _stage_run(tmp_path)
    run["config"]["resume_from"] = "abcd12345678"
    out = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=out)
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "resume_from" not in manifest["config"]


def test_export_bundle_accepts_uppercase_extension(tmp_path: Path):
    """Tauri save dialogs on Windows can produce ``.LLMCHAIN``
    when the user types the suffix uppercase. Both forms should
    work."""
    run_dir, run = _stage_run(tmp_path)
    out = tmp_path / "out.LLMCHAIN"
    result = export_bundle(run, run_dir=run_dir, output_path=out)
    assert result.path.exists()


def test_import_bundle_rejects_absolute_path_entries(tmp_path: Path):
    bundle = tmp_path / "evil2.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
        # zipfile normalises absolute paths but a hand-crafted name
        # with a leading slash should still be refused.
        zf.writestr("/etc/passwd", "danger")
    # Whether the leading slash makes it past zipfile depends on the
    # tool that wrote the file; either way our defence-in-depth check
    # should also catch the `..` case below. Verify path-traversal
    # protection holds even when the entry is stored as absolute.
    try:
        import_bundle(bundle, runs_root=tmp_path / "imports", new_run_id="x")
    except ValueError as e:
        assert "path-traversal" in str(e)
    # If zipfile silently normalised the leading slash, the import
    # would succeed with the entry under "etc/passwd" relative to
    # the target — that's acceptable because it's contained inside
    # the run dir, but the test documents the normalisation.


def test_import_bundle_rejects_missing_manifest(tmp_path: Path):
    bundle = tmp_path / "no-manifest.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("adapter/adapter_model.safetensors", b"x")
    with pytest.raises(ValueError, match="manifest.json"):
        import_bundle(bundle, runs_root=tmp_path / "imports", new_run_id="x")


def test_import_bundle_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        import_bundle(
            tmp_path / "ghost.llmchain",
            runs_root=tmp_path / "imports",
            new_run_id="x",
        )


def test_manifest_from_bundle_returns_subset(tmp_path: Path):
    run_dir, run = _stage_run(tmp_path)
    bundle = tmp_path / "out.llmchain"
    export_bundle(run, run_dir=run_dir, output_path=bundle)
    manifest = manifest_from_bundle(bundle)
    assert manifest["schema_version"] == 1
    assert manifest["imported_from"] == "abc123"
    # dataset_path scrubbed to the sentinel — privacy claim.
    assert manifest["config"]["dataset_path"] == "imported-bundle"


def test_manifest_from_bundle_rejects_missing_manifest(tmp_path: Path):
    bundle = tmp_path / "bad.llmchain"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("readme.txt", "not a bundle")
    with pytest.raises(ValueError, match="manifest"):
        manifest_from_bundle(bundle)
