"""The conversion specialist's deterministic tool.

The point of this agent is that its answers are arithmetic, not generation, so
the tests pin the numbers and the rejections rather than any model behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from specialists.conversion import ConversionAgent, build_conversion_mcp, convert
from specialists.conversion.skills import get_conversion_skills


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (70, "kg", "lb", 154.323583529),
        (1, "lb", "kg", 0.45359237),
        (100, "kcal", "kj", 418.4),
        (418.4, "kj", "kcal", 100.0),
        (6, "ft", "cm", 182.88),
        (1, "mi", "km", 1.609344),
        (500, "ml", "l", 0.5),
        (2.5, "kg", "g", 2500.0),
    ],
)
def test_known_conversions(value, source, target, expected):
    result, _, _ = convert(value, source, target)
    assert result == pytest.approx(expected)


def test_conversion_round_trips():
    there, _, _ = convert(83.5, "kg", "lb")
    back, _, _ = convert(there, "lb", "kg")
    assert back == pytest.approx(83.5)


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [("Kilograms", "kg"), ("POUNDS", "lb"), ("inches", "in"), ("kilojoule", "kj"), ("in.", "in")],
)
def test_unit_spellings_resolve_to_one_canonical_unit(spelling, canonical):
    _, source, target = convert(1, spelling, spelling)
    assert (source, target) == (canonical, canonical)


def test_mixing_dimensions_is_refused():
    with pytest.raises(ValueError, match="Cannot convert kg \\(mass\\) to m \\(length\\)"):
        convert(1, "kg", "m")


def test_an_unsupported_unit_is_refused_and_names_the_supported_ones():
    with pytest.raises(ValueError, match="Unsupported unit 'stone'") as exc:
        convert(1, "kg", "stone")
    # The model has to be able to retry without guessing.
    assert "kcal" in str(exc.value)


def test_calories_are_not_silently_read_as_kilocalories():
    """"cal" strictly means 1/1000 kcal; resolving the ambiguity is not ours."""
    with pytest.raises(ValueError, match="Unsupported unit"):
        convert(1, "cal", "kj")


async def test_the_mcp_tool_reports_the_conversion():
    server = build_conversion_mcp()
    result = await server.call_tool("convert_units", {"value": 70, "from_unit": "kg", "to_unit": "lb"})
    assert "70 kg = 154.324 lb" in str(result)


def test_the_card_advertises_exactly_the_published_tools():
    published = {t.name for t in asyncio.run(build_conversion_mcp().list_tools())}
    assert {s.name for s in get_conversion_skills()} == published


def test_the_agent_advertises_its_skill():
    agent = ConversionAgent(api_key="EMPTY", api_base_url="http://x/v1", model="local")
    assert [s.name for s in agent.skills.list_skills()] == ["convert_units"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused_rather_than_converted(value):
    """A deterministic tool must not launder nonsense into an answer: the LLM
    fills these arguments in, and "inf kg = inf lb" reads like a result."""
    with pytest.raises(ValueError, match="finite"):
        convert(value, "kg", "lb")
