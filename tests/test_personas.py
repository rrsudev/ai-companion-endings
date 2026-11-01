from __future__ import annotations

import pytest

from endings.personas.ingest import derive_attachment_style, from_raw, load_persona
from endings.schemas import Level, ReadinessStage


def test_sample_persona_fields(persona_anxious):
    p = persona_anxious
    assert p.id == "rn_sample_01"
    assert p.attachment.style == "anxious"  # high anxiety, low avoidance -> derived
    assert p.dependency.salience == Level.HIGH
    assert p.readiness == ReadinessStage.CONTEMPLATION
    assert p.escalates_when_bounded is True


@pytest.mark.parametrize("anx,avo,expected", [
    (6.5, 2.0, "anxious"), (2.0, 6.0, "avoidant"),
    (6.0, 6.0, "fearful"), (2.0, 2.0, "secure"),
])
def test_derive_attachment_style(anx, avo, expected):
    assert derive_attachment_style(anx, avo) == expected


def test_unknown_keys_go_to_raw():
    p = from_raw({"id": "x", "display_name": "X", "some_curated_field": 123})
    assert p.raw["some_curated_field"] == 123  # strict schema not broken by extra keys


def test_missing_id_raises():
    with pytest.raises(ValueError):
        from_raw({"display_name": "no id"})


def test_load_persona_file(persona_ready):
    assert persona_ready.companion_name == "Ela"
    assert persona_ready.readiness == ReadinessStage.PREPARATION
