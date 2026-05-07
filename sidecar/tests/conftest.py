"""Test-wide setup that runs before any test module is imported.

Ensures the api routes module's default RunStore root points at a tmp dir
instead of ~/.llm-chain/runs, so the test suite never writes to the user's
real home directory.
"""

import os
import tempfile

os.environ.setdefault(
    "LLM_CHAIN_RUNS_DIR",
    tempfile.mkdtemp(prefix="llm-chain-test-runs-"),
)
