"""Conversion-agent skill descriptors mirroring the MCP tools.

Metadata only: it is what the A2A card advertises, while execution happens on
the MCP server (see :mod:`specialists.conversion.tools`). The argument schema
lives there, on the tool itself, where a caller can actually read it — the A2A
card has no field for one.
"""

from __future__ import annotations

from erebots.agents.skill import Skill


def get_conversion_skills() -> list[Skill]:
    """Return the descriptors of the conversion MCP tools."""
    return [
        Skill(
            name="convert_units",
            description=(
                "Convert a quantity between units of the same kind: mass "
                "(kg, g, mg, lb, oz), length (m, cm, mm, km, in, ft, mi), "
                "energy (kcal, kJ) or volume (l, dl, ml)."
            ),
        ),
    ]
