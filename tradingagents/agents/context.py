"""Prompt context shared by the agents: instrument identity, output language and portfolio."""

import functools
import logging
from collections.abc import Mapping
from typing import Any

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    # The labelled lines are read by the program, so they keep their English
    # label and value: a translated rating line leaves the reader prose to
    # search, where a negated rating ("not a Sell") reads as the call (#1435).
    return (
        f" Write your entire response in {lang}, except the labelled lines the format"
        f" asks for (the \"**Rating**:\" line, \"FINAL TRANSACTION PROPOSAL:\"):"
        f" keep their label and value in English, exactly as specified."
    )


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """Opponent's latest argument, or an explicit opening marker when empty.

    The first speaker in each debate round receives an empty opponent response;
    interpolating it into a "refute the opponent" prompt makes the model
    fabricate the other side's position. Returning a clear "has not spoken yet"
    marker instead lets it open with its own case (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. An answer
    is cached for the process; a failed lookup is asked again next time.

    Identity resolves for the same instrument the price path fetches
    (``XAUUSD`` -> ``GC=F``, #983).
    """
    try:
        return _identity(ticker)
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}


@functools.lru_cache(maxsize=256)
def _identity(ticker: str) -> dict:
    """The vendor's identity fields for ``ticker``; raises if the lookup fails."""
    info = get_company_profile(ticker)
    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
    trade_date: str | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).

    That profile carries no historical vintage: it describes the company today.
    A run dated earlier gets the current name alone, as a way to tell the
    company apart from others rather than as what it was called then; a sector,
    industry or exchange it holds today is not given, since it may not have held
    on the analysis date.
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "The tools serve this instrument; refer to it by this exact ticker in every report and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    identity = identity or {}
    name = identity.get("company_name") or identity.get("name")
    label = "Name" if is_crypto else "Company"
    details = []
    if is_historical(trade_date):
        if name:
            details.append(
                f"{label}: {name} (its current name, given only to identify it; "
                f"on {trade_date} it may have been named differently)"
            )
    else:
        if name:
            details.append(f"{label}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_portfolio_context_from_state(state: Mapping[str, Any]) -> str:
    """The caller's holdings and limit verdicts, or "" when none was supplied.

    Returned verbatim. The block is produced by a deterministic engine on the
    caller's side (ystocker's ``exposure.render_block``), and every number in it —
    the floors, the ceilings, the before/after deltas — was computed there
    precisely so no model has to do the arithmetic. Reformatting or summarising it
    here would put that arithmetic back in reach of a language model, which is the
    one thing the block exists to prevent.

    Empty is the common case and means *unknown*, not *flat*. A consumer must not
    turn silence into an assumption that nothing is held.
    """
    context = state.get("portfolio_context")
    if isinstance(context, str) and context.strip():
        return context
    return ""


#: Framing for the holdings block, written once so five prompts cannot drift.
#: The rules are the load-bearing part. Two failure modes they exist to stop:
#: computing a headroom figure from a floor (the floor is a lower bound, so the
#: difference to the limit is not a number the data supports), and reading an
#: absent block as an empty portfolio.
_PORTFOLIO_RULES = (
    "Rules for using it. Every figure was computed by a deterministic engine "
    "before this run began — quote them as given and never recompute, re-derive "
    "or round them. A company-level exposure is a measured FLOOR because funds "
    "disclose only their largest holdings, so each name carries a floor and a "
    "ceiling and the true value lies between them; do not treat the midpoint as "
    "an estimate and do not compute remaining headroom against the floor. The "
    "verdicts are three-valued and only PASS means a limit was verified to hold: "
    "BREACH means the floor alone exceeds the limit, and INDETERMINATE means the "
    "disclosures cannot rule out a breach — which is a real finding that should "
    "reduce conviction, not a gap to resolve with an assumption. Limits are the "
    "holder's own stated policy; a limit that is absent was never set and must "
    "not be invented. Recommend a size that respects a BREACH rather than "
    "arguing around it."
)


def get_risk_gate_block(state: Mapping[str, Any]) -> str:
    """The deterministic risk-gate ruling as text, or "" before the gate has run.

    Rendered by :mod:`tradingagents.risk_engine`, not here, so the four prompts
    that carry it cannot phrase the same ruling four ways. Empty means the node has
    not executed yet -- which is the normal state for every agent upstream of it --
    and must not be read as approval.
    """
    from tradingagents import risk_engine

    decision = state.get("risk_gate")
    if not isinstance(decision, Mapping) or not decision:
        return ""
    return risk_engine.render(decision)


def get_portfolio_block(state: Mapping[str, Any], heading: str) -> str:
    """The holdings block wrapped in its heading and rules, or "" when absent.

    Empty means the caller supplied no portfolio, which is *unknown* and not
    *flat*. Returning "" rather than a "no holdings" sentence is deliberate: a
    sentence would license the reader to size as though opening from zero, and
    most callers of this framework genuinely have no portfolio attached.
    """
    context = get_portfolio_context_from_state(state)
    if not context:
        return ""
    return f"\n{heading}\n{context}\n\n{_PORTFOLIO_RULES}\n"


def get_market_context_from_state(state: Mapping[str, Any]) -> str:
    """The caller's macro/regime notes, or "" when none was supplied.

    Returned verbatim, same rationale as ``get_portfolio_context_from_state``:
    the block is assembled by the caller (ystocker) from its own cached market
    data before the run starts, so re-deriving or summarising any of it here
    would put arithmetic no agent can independently check back in reach of a
    language model.
    """
    context = state.get("market_context")
    if isinstance(context, str) and context.strip():
        return context
    return ""


def get_relative_strength_from_state(state: Mapping[str, Any]) -> str:
    """The caller's peer-comparison notes, or "" when none was supplied.

    Empty is the common case: a ticker outside every configured peer group, or
    one Yahoo covers with no analyst estimates, has nothing to compare.
    """
    context = state.get("relative_strength_context")
    if isinstance(context, str) and context.strip():
        return context
    return ""


#: Framing for both blocks below, written once so five prompts cannot drift —
#: same reasoning as ``_PORTFOLIO_RULES`` above. Deliberately shorter: these
#: are descriptive market data with no compliance obligation attached, unlike
#: the portfolio block's three-valued BREACH/INDETERMINATE/PASS verdicts, so
#: the risk here is a model over-weighting one macro paragraph into a
#: company-specific conclusion, not misreading an arithmetic result.
_MARKET_CONTEXT_RULES = (
    "This is context, not a directive: it describes the broad market or the "
    "peer group, not this specific company, and every figure in it was "
    "computed before this run began — quote figures as given rather than "
    "re-deriving them."
)


def get_market_context_block(state: Mapping[str, Any], heading: str) -> str:
    """The macro/regime notes wrapped in a heading and rules, or "" when absent."""
    context = get_market_context_from_state(state)
    if not context:
        return ""
    return f"\n{heading}\n{context}\n\n{_MARKET_CONTEXT_RULES}\n"


def get_relative_strength_block(state: Mapping[str, Any], heading: str) -> str:
    """The peer-comparison notes wrapped in a heading and rules, or "" when absent."""
    context = get_relative_strength_from_state(state)
    if not context:
        return ""
    return f"\n{heading}\n{context}\n\n{_MARKET_CONTEXT_RULES}\n"


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def report_or_absent(text: str, source: str) -> str:
    """An analyst's report, or a marker saying it was never produced.

    A report is empty when its analyst was not selected, refused, or returned
    nothing. Interpolating that into a labelled section presents an absence as a
    blank finding, and the reading agent fills it in from nothing, the same way
    an empty opponent argument used to invite an invented rebuttal (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(No {source} report in this run: it is not available, not an empty finding.)"
