"""Test-wide setup that runs before any test module is imported.

Ensures the api routes module's default RunStore root points at a tmp dir
instead of ~/.llm-chain/runs, so the test suite never writes to the user's
real home directory.
"""

import json
import os
import tempfile

import pytest

os.environ.setdefault(
    "LLM_CHAIN_RUNS_DIR",
    tempfile.mkdtemp(prefix="llm-chain-test-runs-"),
)


@pytest.fixture(scope="session")
def existing_jsonl_chat(tmp_path_factory) -> str:
    """A real, on-disk JSONL chat file. Many route tests build a run config
    that references a dataset path; the API now validates the path exists at
    create-time, so synthetic '/tmp/x.jsonl' paths return 400 instead of the
    test's expected status. This fixture gives every test a path that's
    guaranteed to exist for the session.
    """
    p = tmp_path_factory.mktemp("ds") / "chat.jsonl"
    p.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            }
        )
        + "\n"
    )
    return str(p)


@pytest.fixture(scope="session")
def existing_csv(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("ds") / "rows.csv"
    p.write_text("text\nhello\nworld\n")
    return str(p)
