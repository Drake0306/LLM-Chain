from pathlib import Path

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def test_text_dir_loads_all_txt_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("first")
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "skip.md").write_text("ignored")
    src = DatasetSource(format=DatasetFormat.TEXT_DIR, path=str(tmp_path))
    ds = load_dataset(src)
    assert len(ds) == 2
    assert {r["text"] for r in ds} == {"first", "second"}
