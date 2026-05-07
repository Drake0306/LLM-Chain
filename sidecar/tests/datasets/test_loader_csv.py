from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def test_csv_loads_with_text_column(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("text,label\nhello,greet\nbye,farewell\n")
    src = DatasetSource(format=DatasetFormat.CSV, path=str(p), text_column="text")
    ds = load_dataset(src)
    assert len(ds) == 2
    assert ds[0]["text"] == "hello"


def test_csv_requires_text_column(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("foo,bar\n1,2\n")
    src = DatasetSource(format=DatasetFormat.CSV, path=str(p), text_column="text")
    with pytest.raises(ValueError, match="column 'text' not found"):
        load_dataset(src)
