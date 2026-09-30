"""The conversion agent's MCP tool server.

One deterministic tool: :func:`convert_units`. No model, no network, no state —
the same arguments always give the same answer, which is the point. It exists so
a specialist can hand off an arithmetic step instead of letting an LLM guess at
it, and so the demonstrator has a second capability that is trivially verifiable.

Units are converted through a base unit per dimension (kilogram, metre,
kilocalorie, litre). Cooking measures (cup, tablespoon) are deliberately absent:
their size depends on the convention, so they cannot be converted deterministically.
"""

from __future__ import annotations

import math

from mcp.server.fastmcp import FastMCP

# unit → (dimension, how many base units it is worth)
_UNITS: dict[str, tuple[str, float]] = {
    "kg": ("mass", 1.0),
    "g": ("mass", 1e-3),
    "mg": ("mass", 1e-6),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "m": ("length", 1.0),
    "cm": ("length", 1e-2),
    "mm": ("length", 1e-3),
    "km": ("length", 1e3),
    "in": ("length", 0.0254),
    "ft": ("length", 0.3048),
    "mi": ("length", 1609.344),
    "kcal": ("energy", 1.0),
    # 1 kcal = 4.184 kJ exactly (thermochemical calorie).
    "kj": ("energy", 1.0 / 4.184),
    "l": ("volume", 1.0),
    "dl": ("volume", 1e-1),
    "ml": ("volume", 1e-3),
}

# Spellings a model is likely to emit. "cal" is *not* mapped to kcal: in
# nutrition it colloquially means kcal but strictly means 1/1000 of one, and a
# deterministic tool must not resolve that ambiguity silently.
_ALIASES: dict[str, str] = {
    "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
    "gram": "g", "grams": "g",
    "milligram": "mg", "milligrams": "mg",
    "pound": "lb", "pounds": "lb", "lbs": "lb",
    "ounce": "oz", "ounces": "oz",
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",
    "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "inch": "in", "inches": "in",
    "foot": "ft", "feet": "ft",
    "mile": "mi", "miles": "mi",
    "kilocalorie": "kcal", "kilocalories": "kcal",
    "kilojoule": "kj", "kilojoules": "kj",
    "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "decilitre": "dl", "decilitres": "dl", "deciliter": "dl", "deciliters": "dl",
    "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml",
}


def _resolve(unit: str) -> tuple[str, str, float]:
    """Normalize one unit to ``(canonical name, dimension, factor)``.

    Raises:
        ValueError: The unit is blank or not supported, naming what is.
    """
    key = unit.strip().lower().rstrip(".")
    key = _ALIASES.get(key, key)
    if key not in _UNITS:
        raise ValueError(
            f"Unsupported unit {unit!r}. Supported: {', '.join(sorted(_UNITS))}."
        )
    dimension, factor = _UNITS[key]
    return key, dimension, factor


def convert(value: float, from_unit: str, to_unit: str) -> tuple[float, str, str]:
    """Convert ``value``, returning ``(result, canonical from, canonical to)``.

    The tool below is a thin wrapper over this: keeping the arithmetic callable
    without an MCP server is what lets it be tested directly.

    Raises:
        ValueError: The value is not a finite number, a unit is unsupported, or
            the two units measure different things.
    """
    # The LLM fills these arguments in, so ``nan``/``inf`` are reachable
    # through a JSON literal or a stray division. They would convert happily
    # and put "inf kg = inf lb" in front of the user as if it were an answer:
    # a deterministic tool has to refuse instead of laundering nonsense.
    if not math.isfinite(value):
        raise ValueError(f"Value must be a finite number, got {value!r}.")
    source, source_dim, source_factor = _resolve(from_unit)
    target, target_dim, target_factor = _resolve(to_unit)
    if source_dim != target_dim:
        raise ValueError(
            f"Cannot convert {source} ({source_dim}) to {target} ({target_dim})."
        )
    return value * source_factor / target_factor, source, target


def build_conversion_mcp() -> FastMCP:
    """MCP server exposing the conversion tool.

    Stateless: it holds nothing between calls, so ``stateless_http`` matches how
    :class:`~erebots.agents.mcp.client.MCPClient` calls it.
    """
    server = FastMCP("conversion", stateless_http=True)

    @server.tool()
    async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a quantity between units of the same kind.

        Supports mass (kg, g, mg, lb, oz), length (m, cm, mm, km, in, ft, mi),
        energy (kcal, kJ) and volume (l, dl, ml). Use it for every conversion
        instead of computing one yourself.

        Args:
            value: The amount to convert.
            from_unit: Unit the value is expressed in (e.g. "kg", "pounds").
            to_unit: Unit to convert it to (e.g. "lb", "kilograms").

        Returns:
            The converted quantity, e.g. ``70 kg = 154.324 lb``.
        """
        result, source, target = convert(value, from_unit, to_unit)
        return f"{value:.6g} {source} = {result:.6g} {target}"

    return server
