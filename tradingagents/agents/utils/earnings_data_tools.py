"""LangChain tools for the Earnings & Estimate Revision Analyst.

Two tools, both returning a stable machine-readable payload rather than prose.

:func:`get_earnings_evidence` returns JSON. That is unusual for a tool in this
project — every other one returns a formatted report — and it is the point: the
Earnings Analyst renders its own numbers from these fields with code, so the
model never re-transcribes a consensus estimate. A digit dropped while copying
"8.81249" out of a markdown table is indistinguishable, downstream, from an
analyst revision.

:func:`get_earnings_commentary` returns text, because a transcript *is* text and
there is nothing to compute from it.
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_earnings_evidence(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve analyst earnings-estimate evidence for a ticker as a JSON document.

    Covers, where the configured vendor publishes them: consensus EPS and revenue
    per forecast period; the consensus trend at 7, 30, 60 and 90 days ago;
    analyst up/down revision counts; analyst coverage counts; the next earnings
    date; reported-versus-consensus surprise history; post-earnings drift; and a
    computed earnings-momentum band.

    Every field carries its own source, as-of date and availability flag. A field
    that is absent is reported absent with a reason — it is never zero, and it
    must not be filled in from prior knowledge. The momentum band and all
    numeric values are computed by code and must be reported exactly as given.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A JSON document of normalized earnings evidence
    """
    return route_to_vendor("get_earnings_evidence", ticker, curr_date)


@tool
def get_earnings_commentary(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve the most recent already-published earnings call transcript.

    Optional enrichment. Returns speaker-attributed excerpts of management
    commentary from the latest call that had actually taken place on or before
    curr_date. When no transcript is available — no key configured, the vendor's
    entitlement does not cover it, or the call has not happened — the result
    begins with EARNINGS_COMMENTARY_UNAVAILABLE or DATA_UNAVAILABLE. In that case
    report that management commentary is unavailable; do not characterise what
    management said, and do not substitute news coverage for a transcript.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: Transcript excerpts, or an explicit unavailable marker
    """
    return route_to_vendor("get_earnings_commentary", ticker, curr_date)
