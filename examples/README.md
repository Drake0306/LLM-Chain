# Example datasets

Tiny demo data so you can validate the Train flow on first launch without preparing your own corpus.

## `tiny-chat.jsonl`

10-row JSONL chat dataset in the format LLM-Chain's `JSONL chat` loader expects:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

This is way too small to produce a useful model — it's enough to verify the full pipeline (load → tokenize → train → save adapter → stream events back to the UI) on a small base model like `Qwen/Qwen3-0.6B` or `mlx-community/Qwen3-0.6B-4bit`.

### Try it from the UI

1. Launch the app: `cd apps/desktop && npm run tauri dev`
2. **Dashboard** — pick your GPU.
3. **Model** — pick **Qwen3 0.6B** (or **Pythia 70M** if you want it to finish in seconds).
4. **Dataset** — Format: `JSONL chat` → Browse → select `examples/tiny-chat.jsonl`.
5. **Train** — accept defaults, hit **Start training**.
6. **Runs** — watch the loss chart light up.

### Try it from the command line

```bash
source .venv/bin/activate
python -c "
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.runs.executor import RunExecutor
from pathlib import Path

store = RunStore(root=Path('/tmp/llm-chain-demo'))
cfg = RunConfig(
    model_id='mlx-community/Qwen3-0.6B-4bit',
    backend='mlx', technique='qlora',
    dataset_path='examples/tiny-chat.jsonl',
    epochs=1, batch_size=1,
)
run = store.create(cfg)
for ev in RunExecutor(store).execute(run.id):
    print(ev)
"
```

(Swap `backend='mlx'` for `'cuda'` and `mlx-community/Qwen3-0.6B-4bit` for `Qwen/Qwen3-0.6B` on an NVIDIA box.)
