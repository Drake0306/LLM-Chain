from unittest.mock import patch

from llm_chain_sidecar.datasets.loader import load_dataset, make_source
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def test_hf_hub_dispatches_to_datasets_library():
    fake_rows = [{"text": "x"}, {"text": "y"}]
    with patch("llm_chain_sidecar.datasets.loader._hf_load") as m:
        m.return_value = fake_rows
        src = DatasetSource(format=DatasetFormat.HF_HUB, hf_id="acme/dataset", split="train")
        ds = load_dataset(src)
        assert ds == fake_rows
        m.assert_called_once_with("acme/dataset", "train")


def test_make_source_routes_hf_id_for_hf_hub():
    """The trainer carries a single dataset_path string in its config; for
    HF Hub it's the dataset id, not a filesystem path. make_source has to
    route it to the hf_id field — the previous trainer code stuck it into
    src.path and the loader silently saw hf_id=None, so HF Hub training
    had been broken since that path was added."""
    src = make_source(DatasetFormat.HF_HUB, "acme/dataset")
    assert src.hf_id == "acme/dataset"
    assert src.path is None


def test_make_source_routes_filesystem_path_for_local_formats():
    src = make_source(DatasetFormat.JSONL_CHAT, "/data/x.jsonl")
    assert src.path == "/data/x.jsonl"
    assert src.hf_id is None


def test_make_source_carries_text_column_for_csv():
    src = make_source(DatasetFormat.CSV, "/data/rows.csv", text_column="body")
    assert src.text_column == "body"


def test_make_source_accepts_string_format():
    src = make_source("hf_hub", "acme/dataset")
    assert src.format == DatasetFormat.HF_HUB
    assert src.hf_id == "acme/dataset"
