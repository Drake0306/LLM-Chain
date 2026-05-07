# Contributing to LLM-Chain

Thanks for considering a contribution! This is an early-stage project — small fixes and v1.1 backend ports (AMD ROCm, Intel XPU, CPU) are especially welcome.

## Filing issues

- **Bugs:** use the bug-report template. Please include OS + GPU, the exact command you ran, and the relevant log lines (sidecar terminal output is most useful).
- **Feature requests:** use the feature-request template. If it's already on the v1.1 / v1.2 list in the [README](README.md), comment on the existing thread instead.

## Local dev setup

See [`setup.md`](setup.md) for the canonical "How to run it" steps. The short version:

```bash
git clone https://github.com/Drake0306/LLM-Chain.git && cd LLM-Chain
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e './sidecar[dev,mlx]'        # or [dev,cuda]
cd apps/desktop && npm install && cd ..
./scripts/build-sidecar.sh --dev
cd apps/desktop && npm run tauri dev
```

## Running the test suite

```bash
# Python sidecar (fast, slow tests excluded by default)
cd sidecar && pytest -v

# Real training smoke tests (requires a GPU/MLX)
cd sidecar && pytest -v -m slow

# Frontend
cd apps/desktop && npm test

# Rust shell
cd apps/desktop/src-tauri && cargo check
```

CI runs the same commands on Ubuntu / macOS / Windows. Open a PR only after the local tests pass.

The sidecar suite is currently **51 tests** (45 fast + 6 download-progress unit tests, plus 2 slow real-training tests gated behind `-m slow`).

For background on how a training run flows through the system, see [`docs/data-flow.md`](docs/data-flow.md) — useful before touching the executor, trainers, or SSE plumbing.

## Commit message style

Conventional commits, scoped:

```
feat(scope): short imperative summary

Optional body that explains the *why* not the *what*. Wrap at 72 cols.
```

Scopes used so far: `hardware`, `models`, `datasets`, `runs`, `trainers`, `api`, `desktop`, `ui`, `ci`. Use `fix(...)` for bug fixes, `chore(...)` for non-behavioral changes, `docs(...)` for documentation, `test(...)` for test-only changes, `refactor(...)` for code reshuffling.

Example:

```
fix(trainers): stage user JSONL into mlx_lm.lora's train/valid layout

mlx_lm.lora's --data flag expects a directory containing train.jsonl and
valid.jsonl, not a single file. _stage_data() splits the user's JSONL
90/10 (with at least one row per split, duplicating when there's only
one row available).
```

## Pull request flow

1. Fork the repo and create a topic branch off `main`.
2. Match the existing code style. Python: ruff + mypy ready (`pip install -e './sidecar[dev]'` pulls them). TS: vitest + tsc strict. Rust: `cargo fmt`.
3. Add a test for any behavior change. The repo already has decent coverage; new code without tests is unlikely to merge.
4. Use the PR template; explain what changed and what you tested.
5. CI must pass before review.

## Coding conventions

- **No emojis in code or comments.** README/issue threads are fine.
- **Don't add features beyond what the task requires.** A bug fix doesn't need surrounding cleanup; a one-shot operation doesn't need a helper. Three similar lines is better than a premature abstraction.
- **Trust internal callers.** Only validate at boundaries (HTTP routes, file IO, subprocess output).
- **Default to no comments.** Add one only when the *why* is non-obvious — a hidden constraint, a workaround for a specific bug, behavior that would surprise a reader.

## License

By submitting code, you agree to license it under [Apache 2.0](LICENSE) — the same license as the rest of the repo.
