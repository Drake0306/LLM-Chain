# Example datasets

Tiny demo data so you can validate the Train flow on first launch without preparing your own corpus.

## `llm-chain-assistant.jsonl` (recommended)

30-row JSONL chat dataset that teaches the model to be the **LLM-Chain helper** — answers questions about local LLM training in a consistent style. Tailored for `HuggingFaceTB/SmolLM2-360M-Instruct`, but works with any chat-capable model in the registry.

Why this one and not `tiny-chat.jsonl` for a smoke test:
- **Consistent persona.** All 30 assistant turns end up in the same voice, so the fine-tuned model produces visibly different output from the base. Easy to verify training "worked".
- **Tight domain.** Q&A about LLM-Chain itself — short answers, narrow vocabulary. The loss drops cleanly on a 360M model in 1–2 minutes on Apple Silicon.
- **Demonstrable.** After training, ask the model "Hello, who are you?" — base SmolLM2 says something generic; the fine-tuned one answers as "the LLM-Chain helper".

### Try it from the UI

1. Launch the app: `cd apps/desktop && npm run tauri dev`
2. **Dashboard** — pick your GPU.
3. **Dataset** — Format: `JSONL chat` → Browse → select `examples/llm-chain-assistant.jsonl`. (Picking the dataset before the model gates the picker; the order doesn't matter for correctness any more, but the chat-only filter on the Models page is more obvious this way.)
4. **Models** — pick **SmolLM2 360M Instruct**. Base models are filtered out automatically for chat datasets.
5. **Train** — accept defaults, hit **Start training**. ~1–5 minutes on an M-series Mac.
6. **Runs** — watch the loss chart drop. The system-stats bar in the top-right shows live CPU / RAM / GPU usage.

## `tiny-chat.jsonl`

10-row generic-Q&A JSONL chat dataset. Same format as above:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Smaller and more eclectic than `llm-chain-assistant.jsonl` — useful when you just want to verify the full pipeline (load → tokenize → train → save adapter → stream events back to the UI) and don't care about the trained behavior.

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
