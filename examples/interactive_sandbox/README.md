# Interactive Sandbox

Minimal example for testing the interactive evolution features without
real LLM API keys. The mock LLM prints every prompt it receives and returns
trivially modified programs; the evaluator assigns random scores in [0, 10].

## Quick start

```bash
# 1. Run the mock evolution  (from ShinkaEvolve root)
cd examples/interactive_sandbox
uv run python run_evo.py

# 2. In a second terminal, start the evolve-shell backend
cd evolve-shell/backend
SHINKA_SEARCH_ROOT=../../ShinkaEvolve/examples/interactive_sandbox uv run python -m uvicorn main:app --reload --port 8000

# 3. In a third terminal, start the evolve-shell frontend
cd evolve-shell
npm run dev
```

Open <http://localhost:3000> and select the `interactive_sandbox` database.

## What to try

| Feature            | How                                            |
|--------------------|------------------------------------------------|
| Pause / Resume     | Click the pause/play buttons in the control bar |
| Stop               | Click stop (the runner will drain active jobs)  |
| Expert Suggestion  | Right-click a node → Suggest → type guidance    |
| Multi-parent Merge | Ctrl-click 2–3 nodes → fill the merge panel     |
| Real-time updates  | Watch the tree grow via WebSocket push           |

## Files

| File           | Description                                |
|----------------|--------------------------------------------|
| `initial.py`   | Seed program (trivial math function)       |
| `evaluate.py`  | Evaluator that returns random scores       |
| `run_evo.py`   | Runner with mock LLM (prints all prompts)  |
