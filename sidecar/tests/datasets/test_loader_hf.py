from unittest.mock import patch

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def test_hf_hub_dispatches_to_datasets_library():
    fake_rows = [{"text": "x"}, {"text": "y"}]
    with patch("llm_chain_sidecar.datasets.loader._hf_load") as m:
        m.return_value = fake_rows
        src = DatasetSource(format=DatasetFormat.HF_HUB, hf_id="acme/dataset", split="train")
        ds = load_dataset(src)
        assert ds == fake_rows
        m.assert_called_once_with("acme/dataset", "train")
