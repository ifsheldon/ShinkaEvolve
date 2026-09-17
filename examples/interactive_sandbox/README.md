# Interactive sandbox

Test interactive evolution with local mock LLM, embedding, and pricing providers.
The evaluator assigns random scores in [0, 10]; these runs illustrate the interface and do not represent benchmark results.
No embedding or generation API calls are required.
Mock summary responses are routed using the same canonical system prompts as the three-step summarizer, for both synchronous and asynchronous calls.
The global overview is explicitly labeled as illustrative and explains that random scores do not measure code quality.

## Start an interactive demo

With the EvolVis release datasets installed, run this from the EvolVis repository root:

```bash
uv run poe start-mock-interactive
```

The launcher creates a fresh working copy of `mock-demo`, connects its runner, and starts the frontend and backend.
The five initial nodes remain unchanged until you click **Start**.
The release snapshots stay fixed.
Opening **Guide** temporarily switches to the populated, read-only `mock-guide` example and returns to the original dataset when you exit.

## Create or resume a standalone run

From the ShinkaEvolve root:

```bash
# Initialize five island seeds without starting evolution.
uv run python examples/interactive_sandbox/run_evo.py --init-only --results-dir /tmp/my-mock-demo

# Attach an interactive runner to those existing seeds.
uv run python examples/interactive_sandbox/run_evo.py --resume --results-dir /tmp/my-mock-demo
```

Omit `--init-only` to initialize a fresh interactive run and wait for a start command.
A fresh run requires a destination that does not exist; the script never deletes an existing run.
Use `--resume` to retain existing nodes, including runs containing only generation-zero seeds.
`--resume` and `--init-only` are mutually exclusive.
The runner rejects directories marked `mock-guide` in `dataset.json`.

To connect an existing run to the EvolVis UI, run this from `evomaestro-interface`:

```bash
uv run python start.py --shinka-search-root /tmp/my-mock-demo \
  --example-runner ../ShinkaEvolve/examples/interactive_sandbox/run_evo.py \
  --runner-args '--resume --results-dir /tmp/my-mock-demo' \
  --frontend-port 3000 --backend-port 8001
```

## What to try

| Feature | Action |
|---|---|
| Start / Continue / Pause | Start mock generations, pause, and continue the runner. |
| Expert suggestion | Right-click a node, choose Suggest, and enter guidance. |
| Multi-parent merge | Select two or more nodes and open Merge. |
| Live updates | Watch new nodes appear through WebSocket updates. |
| Summary | Wait for the configured summarization interval; a seed-only run has no summary. |

## Files

| File | Purpose |
|---|---|
| `initial.py` | Seed program. |
| `evaluate.py` | Local evaluator. |
| `run_evo.py` | Mock providers, configuration, and command-line entry point. |
| `runtime.py` | Offline pricing snapshot and seed-only initialization. |
