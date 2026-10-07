from collections.abc import Iterable
from dataclasses import dataclass

from tradingagents.agents.analysts import (
    earnings_analyst,
    fundamentals_analyst,
    hot_money_tracker,
    lockup_watcher,
    market_analyst,
    news_analyst,
    policy_analyst,
    quality_analyst,
    valuation_analyst,
)


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    report_key: str
    tools: tuple = ()


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        report_key="market_report",
        tools=market_analyst.TOOLS,
    ),
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        agent_node="Sentiment Analyst",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        report_key="news_report",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        tools=fundamentals_analyst.TOOLS,
    ),
    "earnings": AnalystNodeSpec(
        key="earnings",
        agent_node="Earnings Analyst",
        report_key="earnings_report",
        tools=earnings_analyst.TOOLS,
    ),
    "quality": AnalystNodeSpec(
        key="quality",
        agent_node="Quality Analyst",
        report_key="quality_report",
        tools=quality_analyst.TOOLS,
    ),
    "valuation": AnalystNodeSpec(
        key="valuation",
        agent_node="Valuation Analyst",
        report_key="valuation_report",
        tools=valuation_analyst.TOOLS,
    ),
    "policy": AnalystNodeSpec(
        key="policy",
        agent_node="Policy Analyst",
        report_key="policy_report",
        tools=policy_analyst.TOOLS,
    ),
    "hot_money": AnalystNodeSpec(
        key="hot_money",
        agent_node="Hot Money Tracker",
        report_key="hot_money_report",
        tools=hot_money_tracker.TOOLS,
    ),
    "lockup": AnalystNodeSpec(
        key="lockup",
        agent_node="Lock-up Monitor",
        report_key="lockup_report",
        tools=lockup_watcher.TOOLS,
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
) -> AnalystExecutionPlan:
    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs)


