from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture()
def write_bicep(tmp_path):
    """Write Bicep source to a temp directory and return the directory."""

    def _write(source: str, name: str = "main.bicep") -> Path:
        (tmp_path / name).write_text(source, encoding="utf-8")
        return tmp_path

    return _write
