"""A-share signal-data tools used by specialized analysts."""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_profit_forecast(ticker: Annotated[str, "Six-digit A-share code"], curr_date: Annotated[str, "YYYY-MM-DD analysis date"]) -> str:
    """Return consensus EPS forecasts and forward valuation evidence."""
    return route_to_vendor("get_profit_forecast", ticker, curr_date)


@tool
def get_hot_stocks(curr_date: Annotated[str, "YYYY-MM-DD analysis date"] = "") -> str:
    """Return strong A-shares and their attributed themes."""
    return route_to_vendor("get_hot_stocks", curr_date)


@tool
def get_northbound_flow(curr_date: Annotated[str, "YYYY-MM-DD analysis date"], include_history: bool = False) -> str:
    """Return northbound capital-flow evidence."""
    return route_to_vendor("get_northbound_flow", curr_date, include_history)


@tool
def get_concept_blocks(ticker: Annotated[str, "Six-digit A-share code"]) -> str:
    """Return the stock's industry and concept memberships."""
    return route_to_vendor("get_concept_blocks", ticker)


@tool
def get_fund_flow(ticker: Annotated[str, "Six-digit A-share code"], curr_date: Annotated[str, "YYYY-MM-DD analysis date"], include_history: bool = True) -> str:
    """Return main-force and retail order-flow evidence."""
    return route_to_vendor("get_fund_flow", ticker, curr_date, include_history)


@tool
def get_dragon_tiger_board(ticker: Annotated[str, "Six-digit A-share code"], curr_date: Annotated[str, "YYYY-MM-DD analysis date"], look_back_days: int = 30) -> str:
    """Return Dragon-Tiger List appearances and seat details."""
    return route_to_vendor("get_dragon_tiger_board", ticker, curr_date, look_back_days)


@tool
def get_lockup_expiry(ticker: Annotated[str, "Six-digit A-share code"], curr_date: Annotated[str, "YYYY-MM-DD analysis date"], forward_days: int = 90) -> str:
    """Return historical and upcoming restricted-share unlock events."""
    return route_to_vendor("get_lockup_expiry", ticker, curr_date, forward_days)


@tool
def get_industry_comparison(ticker: Annotated[str, "Six-digit A-share code"], curr_date: Annotated[str, "YYYY-MM-DD analysis date"]) -> str:
    """Return A-share industry performance and capital-flow comparison."""
    return route_to_vendor("get_industry_comparison", ticker, curr_date)
