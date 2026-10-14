from pathlib import Path

import pytest

from sqlassist.data.generate import build
from sqlassist.semantic import SemanticLayer

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("data") / "demo.db")


@pytest.fixture(scope="session")
def semantic() -> SemanticLayer:
    return SemanticLayer.load(ROOT / "semantic_layer.yaml")
