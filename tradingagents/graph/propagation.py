# TradingAgents/graph/propagation.py

from typing import Any

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        portfolio_context: str = "",
        portfolio_data: dict[str, Any] | None = None,
        market_context: str = "",
        relative_strength_context: str = "",
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``portfolio_context`` is a caller-supplied block describing what the
        holder already owns and which of their stated limits the holding already
        breaches. Empty is the normal case — most callers have no portfolio — and
        empty must read as *unknown*, not as *no holdings*: an agent told nothing
        about a portfolio must not conclude the position is being opened from
        flat. Only the decision-making agents receive it; the analysts do not, so
        that a fundamentals or news read cannot be shaded by what the reader
        happens to hold.

        ``market_context`` and ``relative_strength_context`` are caller-supplied
        macro/regime and peer-comparison notes respectively, on the same
        decision-agents-only distribution as ``portfolio_context`` and for the
        same reason: they describe the moment and the peer group, not this
        specific company's fundamentals or news, so an analyst's read of the
        company itself must not be shaded by them either.
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "portfolio_context": portfolio_context,
            "portfolio_data": dict(portfolio_data or {}),
            "market_context": market_context,
            "relative_strength_context": relative_strength_context,
            "risk_gate": {},
            "pm_levels": {},
            "gate_compliance": {},
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "trader_levels": {},
            "market_report": "",
            "fundamentals_report": "",
            "earnings_report": "",
            "quality_report": "",
            "valuation_report": "",
            "sentiment_report": "",
            "news_report": "",
            "policy_report": "",
            "hot_money_report": "",
            "lockup_report": "",
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
