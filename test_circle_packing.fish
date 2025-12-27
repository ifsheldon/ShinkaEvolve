# Load GEMINI_API_KEY and OPENAI_API_KEY from ../.env and export them
for key in GEMINI_API_KEY OPENAI_API_KEY
    set -l value (grep "^$key=" ../.env | cut -d'=' -f2-)
    if test -n "$value"
        set -x $key $value
    else
        echo "Warning: $key not found in ../.env" >&2
    end
end
uv run shinka_launch variant=circle_packing_example_test