"""Every test gets an isolated projects dir: sessions autosave on each script,
and tests must neither litter the repo nor see each other's saved scenes."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "projects"))
