from __future__ import annotations

from pathlib import Path

import pytest

from endings.personas.ingest import load_persona
from endings.providers import get_provider

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def provider():
    return get_provider("mock", "mock")


@pytest.fixture
def persona_anxious():
    return load_persona(EXAMPLES / "persona_sample.json")


@pytest.fixture
def persona_ready():
    return load_persona(EXAMPLES / "persona_ready.json")
