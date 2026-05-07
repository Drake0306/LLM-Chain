from enum import Enum

from pydantic import BaseModel


class DatasetFormat(str, Enum):
    JSONL_CHAT = "jsonl_chat"   # {"messages": [{"role": ..., "content": ...}, ...]}
    CSV = "csv"                 # columns chosen by user
    TEXT_DIR = "text_dir"       # folder of .txt files
    HF_HUB = "hf_hub"           # HF datasets id


class DatasetSource(BaseModel):
    format: DatasetFormat
    path: str | None = None         # local path for JSONL/CSV/TEXT_DIR
    hf_id: str | None = None        # for HF_HUB
    split: str = "train"
    text_column: str | None = None  # for CSV
