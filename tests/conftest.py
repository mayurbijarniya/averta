import pytest


@pytest.fixture(autouse=True)
def isolated_transcripts(tmp_path_factory, monkeypatch):
    """Keep every test away from the real `~/.claude/projects`.

    Transcript discovery follows `$CLAUDE_CONFIG_DIR`, so pointing it at an
    empty directory means a test that forgets to set up its own transcripts
    finds none, rather than silently reading the developer's sessions and
    passing or failing on whatever happens to be there.
    """
    config = tmp_path_factory.mktemp("claude-config")
    (config / "projects").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    monkeypatch.delenv("AVERTA_SESSION", raising=False)
    monkeypatch.delenv("AVERTA_PROJECT", raising=False)
    return config / "projects"
