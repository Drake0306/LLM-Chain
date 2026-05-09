"""Portable run bundle (F-D19).

A single-file ``.llmchain`` archive (just a renamed ``.zip``) the
user can email or upload to a colleague, who imports it back into
their own LLM-Chain install. Contents:

  - manifest.json   — subset of run.json (config + status + audit)
  - adapter/        — the adapter weights + adapter_config.json
  - prompts.jsonl   — eval prompts (optional, F-A3)
  - notes.md        — run notes (optional, F-A5)

The archive deliberately excludes anything that would leak local
context: dataset path, events log, GGUF artifacts, run.json
verbatim. Re-import creates a fresh run id on the destination
machine; the original id is preserved as ``manifest.imported_from``
so the audit trail survives.

Adapter formats supported: HF/peft (``adapter_model.safetensors``)
and MLX (``adapters.safetensors``). Both are fileset-equivalent so
the export side just copies whatever's in the run dir; the import
side relies on the same ``find_latest_adapter`` logic the rest of
the app uses to discover the layout.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Files we DO copy into the bundle's adapter/ directory. Anything
# else (run.json, events.jsonl, the staged dataset under _mlx_data/)
# is left out — same exclude philosophy as the HF Hub push.
_ADAPTER_FILE_GLOBS: tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapters.safetensors",
    "adapter_config.json",
    "tokenizer*",
    "special_tokens_map.json",
)
# Top-level files we copy as-is (when present) for the optional
# bundle pieces.
_OPTIONAL_TOP_LEVEL: tuple[str, ...] = (
    "notes.md",
    "merge.json",
)
# The eval prompts file lives at <run_dir>/eval-prompts.jsonl when
# the user saves their suite. We don't currently write that — F-A3
# eval is in-memory — but reserve the slot so a future eval-save
# feature can roundtrip through the bundle without a schema change.
_EVAL_PROMPTS_FILENAME = "eval-prompts.jsonl"

# Bundle file extension. Plain zip — ``.llmchain`` is the friendly
# label so the file manager surfaces "this is an LLM-Chain artifact"
# and the import flow can validate by sniffing.
BUNDLE_SUFFIX = ".llmchain"


@dataclass
class BundleResult:
    path: Path
    bytes_written: int
    files_included: int


@dataclass
class ImportResult:
    run_id: str
    imported_from: str | None
    files_extracted: int


def _build_manifest(run: dict) -> dict:
    """Subset of run.json that's safe to ship.

    Keeps the config (so the importer can recreate the trainer
    setup) and preserves the source run id for the audit trail.
    Strips fields that leak local-machine context or have no
    meaning on the destination machine:
      - ``dataset_path``: filesystem path → sentinel
      - ``output_dir``: local path → dropped (the store will set
        a fresh one on import)
      - ``resume_from``: source-machine run id → dropped
    """
    cfg = dict(run.get("config", {}))
    if "dataset_path" in cfg:
        cfg["dataset_path"] = "imported-bundle"
    cfg.pop("output_dir", None)
    cfg.pop("resume_from", None)
    return {
        "schema_version": 1,
        "imported_from": run.get("id"),
        "exported_status": run.get("status"),
        "config": cfg,
    }


def _add_adapter_files(zf: zipfile.ZipFile, run_dir: Path) -> int:
    """Copy whatever adapter layout is on disk into the bundle's
    adapter/ subdir.

    Three layouts seen in the wild (same as gguf.find_latest_adapter):
      - peft direct save: adapter_model.safetensors at run_dir
      - mlx_lm: adapters.safetensors at run_dir
      - HF Trainer checkpoint: <run_dir>/checkpoint-N/
    The first two are flat. The checkpoint case uses the latest
    checkpoint dir (max N) so the bundle ships the most-trained
    weights, not the earliest sample.
    """
    count = 0
    # Top-level adapter files first (if present).
    for pattern in _ADAPTER_FILE_GLOBS:
        for src in run_dir.glob(pattern):
            if not src.is_file():
                continue
            zf.write(src, f"adapter/{src.name}")
            count += 1
    # If nothing landed at top level, fall through to the highest
    # checkpoint dir — HF Trainer saves there.
    if count == 0:
        checkpoints = sorted(
            (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-", 1)[1])
            if p.name.split("-", 1)[1].isdigit()
            else -1,
        )
        if checkpoints:
            ckpt = checkpoints[-1]
            for pattern in _ADAPTER_FILE_GLOBS:
                for src in ckpt.glob(pattern):
                    if not src.is_file():
                        continue
                    zf.write(src, f"adapter/{src.name}")
                    count += 1
    return count


def export_bundle(
    run: dict,
    *,
    run_dir: Path,
    output_path: Path,
) -> BundleResult:
    """Write a ``.llmchain`` archive at ``output_path``.

    Refuses to overwrite — same rationale as the curated download
    refusing to overwrite. The user can delete and retry.

    Raises:
    - FileNotFoundError: run dir doesn't exist or contains no adapter.
    - FileExistsError: ``output_path`` already exists.
    - ValueError: output_path doesn't end in ``.llmchain``.
    """
    # Case-insensitive: a Tauri save dialog on Windows can yield
    # ``.LLMCHAIN`` if the user types the suffix uppercase. Both
    # forms are accepted.
    if output_path.suffix.lower() != BUNDLE_SUFFIX:
        raise ValueError(
            f"Bundle path must end in {BUNDLE_SUFFIX} (got {output_path.suffix!r})."
        )
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run dir doesn't exist: {run_dir}. Can't bundle a missing run."
        )
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Delete it on disk if you "
            "want to overwrite."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files_included = 0
    # Stage the manifest in-memory; everything else copies from disk.
    manifest = _build_manifest(run)

    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        files_included += 1

        adapter_count = _add_adapter_files(zf, run_dir)
        if adapter_count == 0:
            # Roll back: a bundle without weights is useless and
            # importing it would surface a confusing "no adapter"
            # error. Refuse upfront instead.
            output_path.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"No adapter files found in {run_dir}. The run may "
                "have failed before saving — bundle aborted."
            )
        files_included += adapter_count

        for name in _OPTIONAL_TOP_LEVEL:
            src = run_dir / name
            if src.is_file():
                zf.write(src, name)
                files_included += 1

        prompts = run_dir / _EVAL_PROMPTS_FILENAME
        if prompts.is_file():
            zf.write(prompts, _EVAL_PROMPTS_FILENAME)
            files_included += 1

    bytes_written = output_path.stat().st_size
    return BundleResult(
        path=output_path,
        bytes_written=bytes_written,
        files_included=files_included,
    )


_DEFAULT_MAX_EXTRACT_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


def _is_unsafe_entry_name(name: str) -> bool:
    """Path-traversal / absolute-path detector that handles both
    POSIX and Windows separators.

    A bundle authored on Windows can ship an entry like ``foo\\..\\bar``
    which on POSIX has a single Path part (the backslash is a literal
    byte) and would slip past a naive ``..`` check. Normalise both
    separators before splitting so the same bundle is rejected on
    both platforms.
    """
    if not name:
        return True
    if name.startswith("/") or name.startswith("\\"):
        return True
    # Windows drive letters: "C:\\foo".
    if len(name) >= 2 and name[1] == ":":
        return True
    parts = name.replace("\\", "/").split("/")
    return any(p == ".." for p in parts)


def import_bundle(
    bundle_path: Path,
    *,
    runs_root: Path,
    new_run_id: str,
    max_extract_bytes: int | None = None,
) -> ImportResult:
    """Extract ``bundle_path`` into a new run dir under ``runs_root``.

    The caller mints ``new_run_id`` (typically via ``RunStore.create``);
    we just unpack the archive into ``<runs_root>/<new_run_id>/``.

    Adapter file layout: bundles ship the adapter under ``adapter/``
    inside the zip (so the manifest sits at the top level), but the
    rest of the app (playground, find_latest_adapter, GGUF merger)
    expects the adapter weights at the run dir's top level. The
    import flattens the adapter/ dir so an imported MLX run with
    ``adapter/adapters.safetensors`` ends up with ``adapters.safetensors``
    directly under the run dir — same layout as a freshly-trained
    run. Without this flatten step, mlx_lm's loader would silently
    fall back to the base model.

    Defence-in-depth:
    - Path-traversal entries (``..`` segments, absolute paths,
      Windows drive letters, backslash-encoded ``..``) are refused
      before any extraction.
    - Total uncompressed size is capped (default 5 GiB) so a zip-bomb
      can't fill the disk. Python 3.11's ``extractall`` has no
      built-in mitigation; we sum ``file_size`` ourselves.

    Raises:
    - FileNotFoundError: bundle doesn't exist.
    - ValueError: archive doesn't include manifest.json, has the
      wrong schema_version, has a path-traversal entry, or exceeds
      the size cap.
    """
    if not bundle_path.exists() or not bundle_path.is_file():
        raise FileNotFoundError(
            f"Bundle file doesn't exist: {bundle_path}"
        )
    cap = max_extract_bytes or _DEFAULT_MAX_EXTRACT_BYTES
    target = runs_root / new_run_id
    target.mkdir(parents=True, exist_ok=True)
    files_extracted = 0
    imported_from: str | None = None

    with zipfile.ZipFile(bundle_path, mode="r") as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if "manifest.json" not in names:
            raise ValueError(
                f"{bundle_path} doesn't look like a valid LLM-Chain "
                "bundle (no manifest.json)."
            )
        for info in infos:
            if _is_unsafe_entry_name(info.filename):
                raise ValueError(
                    f"Bundle has unsafe entry name {info.filename!r}. "
                    "Refusing to import to avoid path-traversal."
                )
        total_size = sum(info.file_size for info in infos)
        if total_size > cap:
            raise ValueError(
                f"Bundle's uncompressed size ({total_size} bytes) "
                f"exceeds the {cap}-byte cap. Refusing to import — "
                "this looks like a zip-bomb. Pass a larger "
                "max_extract_bytes if you trust the source."
            )

        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Bundle uses schema_version="
                f"{manifest.get('schema_version')!r}; this version of "
                "LLM-Chain only supports schema_version=1. Update the "
                "app or pick a compatible bundle."
            )
        imported_from = manifest.get("imported_from")
        zf.extractall(target)
        files_extracted = len(names)

    # Flatten the adapter/ subdir so imported runs use the same on-disk
    # layout as freshly-trained runs. mlx_lm and find_latest_adapter
    # both look at the run dir's top level for adapter*.safetensors.
    adapter_subdir = target / "adapter"
    if adapter_subdir.is_dir():
        for child in adapter_subdir.iterdir():
            if not child.is_file():
                continue
            dest = target / child.name
            if dest.exists():
                # Don't clobber the store's run.json or any other
                # top-level file the import already wrote.
                continue
            shutil.move(str(child), str(dest))
        # Remove the now-empty subdir so the run dir doesn't carry
        # vestigial structure.
        try:
            adapter_subdir.rmdir()
        except OSError:
            # Subdir wasn't empty (e.g. a future tokenizer file we
            # didn't move) — leave it; the loader can still find what
            # it needs at the top level.
            pass

    return ImportResult(
        run_id=new_run_id,
        imported_from=imported_from,
        files_extracted=files_extracted,
    )


def manifest_from_bundle(bundle_path: Path) -> dict:
    """Peek at the manifest without extracting. Used by the import
    UI to show a "this bundle came from run X, with config Y" preview
    before the user commits to the import.
    """
    with zipfile.ZipFile(bundle_path, mode="r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(
                f"{bundle_path} doesn't include a manifest.json."
            )
        return json.loads(zf.read("manifest.json"))


__all__ = [
    "BUNDLE_SUFFIX",
    "BundleResult",
    "ImportResult",
    "export_bundle",
    "import_bundle",
    "manifest_from_bundle",
]
