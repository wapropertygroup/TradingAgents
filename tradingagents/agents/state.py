from typing import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]
    history: Annotated[str, "Conversation history"]
    current_response: Annotated[str, "Latest response"]
    count: Annotated[int, "Length of the current conversation"]


# Risk management team state
class RiskDebateState(TypedDict):
    aggressive_history: Annotated[
        str, "Aggressive Agent's Conversation history"
    ]
    conservative_history: Annotated[
        str, "Conservative Agent's Conversation history"
    ]
    neutral_history: Annotated[
        str, "Neutral Agent's Conversation history"
    ]
    history: Annotated[str, "Conversation history"]
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[
        str, "Latest response by the aggressive analyst"
    ]
    current_conservative_response: Annotated[
        str, "Latest response by the conservative analyst"
    ]
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]
    count: Annotated[int, "Length of the current conversation"]


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    asset_type: Annotated[str, "Asset type under analysis such as stock or crypto"]
    instrument_context: Annotated[str, "Deterministic ticker identity resolved at run start"]
    trade_date: Annotated[str, "The analysis date; data is served as of it"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    news_report: Annotated[str, "Report from the News Analyst on company and world news"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst"]
    earnings_report: Annotated[str, "Report from the Earnings & Estimate Revision Analyst"]
    quality_report: Annotated[str, "Report from the Quality Analyst"]
    valuation_report: Annotated[str, "Report from the Valuation Analyst"]
    policy_report: Annotated[str, "Report from the A-share Policy Analyst"]
    hot_money_report: Annotated[str, "Report from the A-share Hot Money Tracker"]
    lockup_report: Annotated[str, "Report from the A-share Lock-up Monitor"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Investment plan from the Research Manager"]

    trader_investment_plan: Annotated[str, "Transaction proposal from the Trader"]
    trader_levels: Annotated[dict, "Typed numeric levels from the Trader's proposal (entry/stop/target/size, derived reward:risk, consistency flags). Empty dict when the structured call fell back to free text, which means unstated rather than zero."]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision from the Portfolio Manager"]
    final_rating: Annotated[str, "The Portfolio Manager's 5-tier rating, or REVIEW when it has none"]
    past_context: Annotated[str, "Memory log context for the Portfolio Manager (same-ticker decisions + cross-ticker lessons), written by the Memory Log step"]
    memory_note: Annotated[str, "What the Memory Log step could not settle or read this run, for the report; empty when all went well"]
    pm_levels: Annotated[dict, "Typed numeric fields from the Portfolio Manager's decision (size/entry/stop/target, consistency flags). Empty when the structured call fell back to free text."]
    gate_compliance: Annotated[dict, "Computed check of whether the final size honoured the risk gate's ruling. Reported, never corrected."]
    risk_gate: Annotated[dict, "Deterministic risk-gate ruling on the Trader's proposal (pass/clamped/blocked, approved size, reasons). Empty until the gate node runs; not_evaluated when no portfolio was supplied."]
    portfolio_data: Annotated[dict, "Caller-supplied size ladder the gate checks against; empty when the caller has no portfolio."]
    portfolio_context: Annotated[str, "Caller-supplied holdings and limit verdicts injected at run start; empty when the caller has no portfolio"]
    market_context: Annotated[str, "Caller-supplied macro/regime notes (index valuation percentile, breadth, CTA positioning, Fed policy odds) injected at run start; empty when unavailable or disabled"]
    relative_strength_context: Annotated[str, "Caller-supplied peer-comparison notes (EPS-revision direction, recommendation shift, price-target spread vs. peer group) injected at run start; empty when the ticker has no peer-group coverage"]
