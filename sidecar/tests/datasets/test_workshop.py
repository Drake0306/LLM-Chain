import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource
from llm_chain_sidecar.datasets.workshop import (
    BuildStats,
    CleaningOptions,
    SchemaMapping,
    apply_schema,
    clean,
    detect_schema,
    parse_text,
    safe_filename,
    write_jsonl,
)


# --- parse_text -------------------------------------------------------


def test_parse_csv_yields_dicts_keyed_by_header():
    text = "user,assistant\nhi,hello\nbye,goodbye\n"
    rows = parse_text(text, "csv")
    assert rows == [
        {"user": "hi", "assistant": "hello"},
        {"user": "bye", "assistant": "goodbye"},
    ]


def test_parse_tsv_uses_tab_delimiter():
    text = "user\tassistant\nhi\thello\n"
    rows = parse_text(text, "tsv")
    assert rows == [{"user": "hi", "assistant": "hello"}]


def test_parse_jsonl_returns_dicts():
    text = (
        json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "b"}]}) + "\n"
    )
    rows = parse_text(text, "jsonl")
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "a"


def test_parse_empty_text_returns_empty_list():
    assert parse_text("", "csv") == []
    assert parse_text("   \n  ", "jsonl") == []


def test_parse_jsonl_invalid_row_points_at_line():
    text = '{"messages": []}\nthis is not json\n'
    with pytest.raises(ValueError, match="Row 2"):
        parse_text(text, "jsonl")


def test_parse_jsonl_rejects_non_object_lines():
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_text("[1, 2, 3]\n", "jsonl")


def test_parse_csv_returns_empty_on_blank_input():
    """Blank/whitespace-only paste shouldn't raise — the live preview
    would flash a parser error on every keystroke if it did. Routes
    layer catches the empty-rows case as a 400 separately."""
    assert parse_text("\n", "csv") == []


# --- detect_schema ----------------------------------------------------


def test_detect_schema_finds_user_assistant_columns():
    rows = [{"user": "a", "assistant": "b"}]
    s = detect_schema(rows)
    assert s.target == "chat"
    assert s.user_field == "user"
    assert s.assistant_field == "assistant"


def test_detect_schema_handles_synonyms_case_insensitively():
    rows = [{"Question": "?", "Answer": "!"}]
    s = detect_schema(rows)
    assert s.user_field == "Question"
    assert s.assistant_field == "Answer"


def test_detect_schema_picks_completion_target_for_prompt_completion_columns():
    rows = [{"prompt": "p", "completion": "c"}]
    s = detect_schema(rows)
    assert s.target == "completion"
    assert s.prompt_field == "prompt"
    assert s.completion_field == "completion"


def test_detect_schema_passthrough_when_already_chat_shaped():
    rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    s = detect_schema(rows)
    assert s.passthrough_chat is True


def test_detect_schema_handles_empty_input():
    s = detect_schema([])
    assert s.target == "chat"
    assert not s.passthrough_chat


# --- apply_schema -----------------------------------------------------


def test_apply_schema_chat_maps_columns_to_messages():
    schema = SchemaMapping(target="chat", user_field="u", assistant_field="a")
    rows = [{"u": "hi", "a": "hello"}, {"u": "bye", "a": "goodbye"}]
    mapped = apply_schema(rows, schema)
    assert mapped == [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "goodbye"},
            ]
        },
    ]


def test_apply_schema_drops_rows_missing_either_field():
    schema = SchemaMapping(target="chat", user_field="u", assistant_field="a")
    rows = [{"u": "hi"}, {"a": "lonely"}, {"u": "x", "a": "y"}]
    mapped = apply_schema(rows, schema)
    assert len(mapped) == 1


def test_apply_schema_passthrough_keeps_only_chat_rows():
    schema = SchemaMapping(target="chat", passthrough_chat=True)
    rows = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"foo": "bar"},
    ]
    assert len(apply_schema(rows, schema)) == 1


def test_apply_schema_chat_target_requires_both_fields():
    schema = SchemaMapping(target="chat", user_field="u")
    with pytest.raises(ValueError, match="user_field and assistant_field"):
        apply_schema([{"u": "x"}], schema)


# --- clean ------------------------------------------------------------


def _chat(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_clean_passes_through_well_formed_rows():
    rows = [_chat("a", "b"), _chat("c", "d")]
    survivors, stats = clean(rows, CleaningOptions())
    assert len(survivors) == 2
    assert stats.output_rows == 2
    assert stats.dropped_duplicate == 0


def test_clean_drops_empty_rows():
    rows = [_chat("", ""), _chat("hi", "hello")]
    survivors, stats = clean(rows, CleaningOptions(drop_empty=True))
    assert len(survivors) == 1
    assert stats.dropped_empty == 1


def test_clean_dedupes_by_content_hash():
    rows = [_chat("a", "b"), _chat("a", "b"), _chat("c", "d")]
    survivors, stats = clean(rows, CleaningOptions(dedupe=True, role_balance=False))
    assert len(survivors) == 2
    assert stats.dropped_duplicate == 1


def test_clean_role_balance_drops_assistant_first():
    bad = {
        "messages": [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]
    }
    rows = [bad, _chat("ok", "fine")]
    survivors, stats = clean(rows, CleaningOptions(role_balance=True))
    assert len(survivors) == 1
    assert stats.dropped_role_violation == 1


def test_clean_role_balance_allows_leading_system_message():
    """system → user → assistant should pass; the system message is
    metadata and shouldn't disqualify the row."""
    row = {
        "messages": [
            {"role": "system", "content": "you are nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    survivors, stats = clean([row], CleaningOptions(role_balance=True))
    assert len(survivors) == 1
    assert stats.dropped_role_violation == 0


def test_clean_max_chars_filters_long_rows():
    rows = [_chat("a", "b"), _chat("x" * 200, "y" * 200)]
    survivors, stats = clean(
        rows, CleaningOptions(max_chars=100, role_balance=False, dedupe=False)
    )
    assert len(survivors) == 1
    assert stats.dropped_length == 1


def test_clean_drops_malformed_rows_under_empty_bucket():
    """A row missing 'messages' is structurally broken, not just empty;
    counting it under dropped_empty keeps the stats schema flat without
    inventing a third bucket nobody'll click on."""
    rows = [{"foo": "bar"}, _chat("a", "b")]
    survivors, stats = clean(rows, CleaningOptions())
    assert len(survivors) == 1
    assert stats.dropped_empty == 1


# --- write_jsonl + safe_filename + roundtrip --------------------------


def test_safe_filename_sanitises_slashes_and_spaces():
    assert safe_filename("My Cool Dataset!") == "my-cool-dataset"
    assert safe_filename("../etc/passwd") == "etc-passwd"
    assert safe_filename("") == "dataset"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "nested" / "dir" / "out.jsonl"
    rows = [_chat("a", "b")]
    write_jsonl(rows, target)
    assert target.exists()
    assert target.read_text().strip() == json.dumps(rows[0], ensure_ascii=False)


def test_workshop_output_loads_back_through_jsonl_chat_loader(tmp_path: Path):
    """Acceptance: a workshop-built file must train cleanly through the
    same JSONL chat loader the trainer uses. Otherwise we shipped a
    feature that produces 'valid-looking' JSONL the trainer can't
    actually read."""
    target = tmp_path / "out.jsonl"
    rows = [_chat("hi", "hello"), _chat("bye", "goodbye")]
    write_jsonl(rows, target)
    loaded = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(target))
    )
    assert len(loaded) == 2
    assert loaded[0]["messages"][0]["content"] == "hi"


# --- end-to-end pipeline ---------------------------------------------


def test_full_pipeline_paste_csv_through_to_jsonl(tmp_path: Path):
    text = "user,assistant\nhi,hello\nhi,hello\n,empty\nbye,goodbye\n"
    rows = parse_text(text, "csv")
    schema = detect_schema(rows)
    mapped = apply_schema(rows, schema)
    survivors, stats = clean(mapped, CleaningOptions())
    target = tmp_path / "out.jsonl"
    write_jsonl(survivors, target)

    loaded = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(target))
    )
    assert len(loaded) == 2  # dedupe drops the duplicate; empty drops the blank-user row
    assert stats.dropped_duplicate >= 1
