from pathlib import Path

import pytest

from evalh.dataset import Dataset, load_dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset(ROOT / "data" / "cases.jsonl", ROOT / "data" / "kb")
