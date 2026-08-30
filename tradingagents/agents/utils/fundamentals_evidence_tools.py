"""LangChain tools for the Quality and Valuation Analysts.

Both return JSON, for the same reason ``get_earnings_evidence`` does: the
analyst renders its own numbers from these fields with code, so the model
never re-transcribes a ratio it could get wrong in a way that looks right.
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_quality_evidence(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve normalized business-quality evidence for a ticker as a JSON document.

    Covers, where the configured vendor publishes them: return on equity,
    operating margin, profit margin, return on assets, debt-to-equity, current
    ratio, free cash flow and free-cash-flow margin, several years of operating
    margin history, and a computed quality tier.

    Every field carries its own source and availability flag. A field that is
    absent is reported absent with a reason — it is never zero, and it must not
    be filled in from prior knowledge. The quality tier and all numeric values
    are computed by code and must be reported exactly as given.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A JSON document of normalized business-quality evidence
    """
    return route_to_vendor("get_quality_evidence", ticker, curr_date)


@tool
def get_valuation_evidence(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve normalized valuation evidence for a ticker as a JSON document.

    Covers, where the configured vendor publishes them: trailing and forward
    P/E, PEG ratio, price-to-book, dividend yield, market cap, and a computed
    valuation tier.

    Every field carries its own source and availability flag. A field that is
    absent is reported absent with a reason — a negative or undefined P/E
    (commonly a negative-earnings company) is reported absent, never as a real
    multiple. The valuation tier and all numeric values are computed by code
    and must be reported exactly as given.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A JSON document of normalized valuation evidence
    """
    return route_to_vendor("get_valuation_evidence", ticker, curr_date)
