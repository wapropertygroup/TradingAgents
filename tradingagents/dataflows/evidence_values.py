"""Shared provenance-carrying scalar and safe-coercion helpers.

Extracted from :mod:`tradingagents.dataflows.earnings_models`, which built this
first and had nothing earnings-specific about it: every "fetch a numeric field
from a vendor, and be honest when it is absent" analyst needs the same
``Value`` wrapper and the same coercion functions, so a second and third
analyst (Quality, Valuation) reuse this module rather than re-implementing
~150 lines. ``earnings_models`` re-exports these names for backward
compatibility, so nothing that already imports from it needs to change.

Three rules are load-bearing, unchanged from where they were written:

**A missing value is missing, never zero.** Every scalar travels as a
:class:`Value` carrying its own ``source``/``as_of`` and, when absent, an
``unavailable_reason`` string. ``0.0`` is a measurement; ``None`` is an
admission.

**NaN must never reach a comparison.** ``safe_float`` rejects NaN and ±inf
explicitly: a NaN that survives into a weighted-mean score poisons it silently
(NaN compares false against every band boundary, so a chain of ``>=`` tests
falls through to the most negative band without erroring).

**A bounded score keeps the arithmetic reproducible.** ``bounded`` maps a raw
value onto ``[-1, 1]`` by saturating at ``±scale`` — the same shape every
signal in a weighted-mean score needs, so each analyst's scoring module states
its own scale per signal rather than re-deriving the clamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

#: Below this, a denominator is treated as indistinguishable from zero.
EPSILON = 1e-9


def safe_float(raw: Any) -> float | None:
    """Coerce to ``float``, mapping every unusable input to ``None``.

    Rejects NaN and ±inf as well as the obvious non-numerics. pandas hands out
    ``nan`` for a blank cell and ``numpy.float64`` for a filled one, and a NaN
    that survives into scoring silently poisons a weighted mean.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def safe_int(raw: Any) -> int | None:
    """Coerce to ``int`` via :func:`safe_float`, so NaN counts become ``None``."""
    value = safe_float(raw)
    if value is None:
        return None
    return int(round(value))


def safe_date(raw: Any) -> str | None:
    """Normalize a date-ish value to ``YYYY-MM-DD``, or ``None``.

    Accepts ``date``/``datetime`` (yfinance's ``calendar`` returns
    ``datetime.date`` objects), an ISO string, or a pandas ``Timestamp``, which
    subclasses ``datetime``. A timestamp is reduced to its date component in
    whatever zone it arrives in; callers that care about zone alignment
    normalize before calling.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    text = str(raw).strip()
    if not text or text.lower() in {"none", "nan", "nat", "-", "null"}:
        return None
    try:
        return datetime.fromisoformat(text[:19].replace("Z", "")).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%b %d, %Y"):
        try:
            return datetime.strptime(text[: len(fmt) + 6], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator vanishes."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < EPSILON:
        return None
    result = numerator / denominator
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def bounded(value: float | None, scale: float) -> float | None:
    """Map a raw value onto ``[-1, 1]``, saturating at ``±scale``."""
    if value is None:
        return None
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    return max(-1.0, min(1.0, value / scale))


def piecewise_score(value: float | None, low: float, mid: float, high: float) -> float | None:
    """Map ``value`` onto ``[-1, 1]`` via three anchor points, clamped beyond them.

    ``low`` scores -1, ``mid`` scores 0, ``high`` scores +1, linearly
    interpolated between and clamped outside — the shape a Quality/Valuation
    checklist item needs when its "good" zone is not centered on zero (a
    current ratio of 1.5 is neutral, not 0; below 1.0 is bad, above 2.5 is
    excellent). ``low``/``mid``/``high`` need not be increasing: pass a
    decreasing triple to invert a signal (e.g. leverage, where lower is
    better) without a separate negation step at the call site.
    """
    if value is None:
        return None
    if mid == low or high == mid:
        raise ValueError(f"anchors must be distinct: low={low!r} mid={mid!r} high={high!r}")
    increasing = high > low
    if increasing:
        if value <= low:
            return -1.0
        if value <= mid:
            return -1.0 + (value - low) / (mid - low)
        if value <= high:
            return (value - mid) / (high - mid)
        return 1.0
    else:
        if value >= low:
            return -1.0
        if value >= mid:
            return -1.0 + (value - low) / (mid - low)
        if value >= high:
            return (value - mid) / (high - mid)
        return 1.0


def weighted_mean_score(
    signals: dict[str, float | None],
    weights: dict[str, float],
    min_available_weight: float,
) -> tuple[float | None, dict[str, float], dict[str, float], float, list[str]]:
    """Renormalize a set of ``[-1, 1]`` signals over whatever is available.

    Shared shape behind every band-scoring analyst here (earnings' momentum,
    Quality's tier, Valuation's tier): only present signals vote, weights are
    renormalized over those present so one missing signal cannot masquerade as
    a neutral vote, and below ``min_available_weight`` the honest answer is "no
    score" rather than a number built from too little evidence.

    Returns ``(score_or_None, used_signals, weights_used, available_weight,
    missing_signal_names)``.
    """
    used = {name: value for name, value in signals.items() if value is not None}
    missing = sorted(name for name, value in signals.items() if value is None)
    weights_used = {name: weights[name] for name in used}
    available_weight = sum(weights_used.values())

    if not used or available_weight < min_available_weight:
        return None, used, weights_used, available_weight, missing

    score = sum(used[name] * weights_used[name] for name in used) / available_weight
    score = max(-1.0, min(1.0, score))
    return score, used, weights_used, available_weight, missing


@dataclass(frozen=True)
class Value:
    """One scalar plus everything needed to judge whether to trust it.

    ``available`` is derived from ``value is not None`` rather than stored, so
    the two can never disagree. ``unavailable_reason`` is required reading when
    the value is absent: the difference between "this provider does not
    publish X" and "this company has no X" is the difference between a tooling
    gap and a fact about the company.
    """

    value: float | None = None
    unit: str = "number"
    currency: str | None = None
    source: str | None = None
    as_of: str | None = None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None

    @classmethod
    def missing(cls, reason: str, *, unit: str = "number", source: str | None = None) -> Value:
        return cls(value=None, unit=unit, source=source, unavailable_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "currency": self.currency,
            "source": self.source,
            "as_of": self.as_of,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Value:
        if not isinstance(raw, dict):
            return cls.missing("malformed serialized value")
        return cls(
            value=safe_float(raw.get("value")),
            unit=str(raw.get("unit") or "number"),
            currency=raw.get("currency"),
            source=raw.get("source"),
            as_of=raw.get("as_of"),
            unavailable_reason=raw.get("unavailable_reason"),
        )


def _fmt(value: Value, *, digits: int = 2) -> str:
    """Render a :class:`Value` for a report, currency-suffixed when known."""
    if not value.available:
        return "unavailable"
    number = value.value
    assert number is not None
    if value.unit == "pct_dec":
        return f"{number * 100:+.2f}%"
    if value.unit == "count":
        return f"{int(round(number))}"
    if value.unit == "ratio":
        return f"{number:+.3f}"
    if value.unit == "currency_large":
        return _fmt_large(number, value.currency)
    text = f"{number:,.{digits}f}"
    return f"{text} {value.currency}" if value.currency else text


def _fmt_large(number: float, currency: str | None) -> str:
    """Compact a large figure (e.g. revenue); providers report these in absolute units."""
    magnitude = abs(number)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= divisor:
            text = f"{number / divisor:,.2f}{suffix}"
            break
    else:
        text = f"{number:,.0f}"
    return f"{text} {currency}" if currency else text
