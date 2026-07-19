from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_github_actions_remain_disabled_on_interactive_branch() -> None:
    for workflow in ("ci.yml", "integration.yml", "docs-release.yml"):
        assert not (ROOT / ".github" / "workflows" / workflow).exists()


def test_pytest_markers_are_registered() -> None:
    pyproject = _read("pyproject.toml")

    assert 'addopts = "--strict-markers"' in pyproject
    assert "integration: live external/provider integration coverage" in pyproject
    assert "models_dev_live: live models.dev catalog contract coverage" in pyproject
    assert (
        "requires_secrets: tests that need CI secrets or private credentials"
        in pyproject
    )
