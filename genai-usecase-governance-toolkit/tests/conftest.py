from pathlib import Path

import pytest

from genai_governance.models import load_portfolio

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def portfolio_path() -> Path:
    return ROOT / "examples" / "portfolio.yaml"


@pytest.fixture()
def portfolio(portfolio_path):
    return load_portfolio(portfolio_path)


@pytest.fixture(scope="session")
def predictions_path() -> Path:
    return ROOT / "examples" / "data" / "predictions_synthetic.csv"
