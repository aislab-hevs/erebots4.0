"""Tests for ``SkillRegistry``."""

from __future__ import annotations

from erebots.agents.skill import Skill, SkillRegistry


def test_register_and_get():
    registry = SkillRegistry()
    skill = Skill(name="foo", description="Foo skill")
    registry.register(skill)

    assert registry.get("foo") is skill
    assert registry.get("missing") is None


def test_register_overwrites_same_name():
    registry = SkillRegistry()
    registry.register(Skill(name="x", description="first"))
    registry.register(Skill(name="x", description="second"))

    assert registry.get("x").description == "second"
    assert len(registry.list_skills()) == 1


def test_to_a2a_skills_shape():
    registry = SkillRegistry()
    registry.register(Skill(name="ping", description="ping me"))

    assert registry.to_a2a_skills() == [
        {
            "id": "ping",
            "name": "ping",
            "description": "ping me",
            "tags": [],
        }
    ]


def test_a2a_skill_dicts_carry_only_what_the_card_reads():
    """``_build_card`` reads these four keys and ignores anything else.

    Stated as its own invariant because the alternative is silent: a key added
    here looks advertised and reaches nothing. Both ``parameters`` and the
    per-skill mode lists were exactly that.
    """
    registry = SkillRegistry()
    registry.register(Skill(name="ping", description="ping me"))

    assert set(registry.to_a2a_skills()[0]) == {"id", "name", "description", "tags"}
