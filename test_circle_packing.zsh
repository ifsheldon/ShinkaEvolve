# Load GEMINI_API_KEY and OPENAI_API_KEY from ../.env and export them
set -a; source ../.env; set +a
uv run shinka_launch variant=circle_packing_example_test