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
    clear_node: str
    report_key: str
    tools: tuple = ()

    @property
    def tool_node(self) -> str | None:
        """The node that runs this analyst's tool calls; None when it has no tools."""
        return f"tools_{self.key}" if self.tools else None


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        clear_node="Msg Clear Market",
        report_key="market_report",
        tools=market_analyst.TOOLS,
    ),
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        agent_node="Sentiment Analyst",
        clear_node="Msg Clear Sentiment",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        clear_node="Msg Clear News",
        report_key="news_report",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        clear_node="Msg Clear Fundamentals",
        report_key="fundamentals_report",
        tools=fundamentals_analyst.TOOLS,
    ),
    "earnings": AnalystNodeSpec(
        key="earnings",
        agent_node="Earnings Analyst",
        clear_node="Msg Clear Earnings",
        report_key="earnings_report",
        tools=earnings_analyst.TOOLS,
    ),
    "quality": AnalystNodeSpec(
        key="quality",
        agent_node="Quality Analyst",
        clear_node="Msg Clear Quality",
        report_key="quality_report",
        tools=quality_analyst.TOOLS,
    ),
    "valuation": AnalystNodeSpec(
        key="valuation",
        agent_node="Valuation Analyst",
        clear_node="Msg Clear Valuation",
        report_key="valuation_report",
        tools=valuation_analyst.TOOLS,
    ),
    "policy": AnalystNodeSpec(
        key="policy",
        agent_node="Policy Analyst",
        clear_node="Msg Clear Policy",
        report_key="policy_report",
        tools=policy_analyst.TOOLS,
    ),
    "hot_money": AnalystNodeSpec(
        key="hot_money",
        agent_node="Hot Money Tracker",
        clear_node="Msg Clear Hot Money",
        report_key="hot_money_report",
        tools=hot_money_tracker.TOOLS,
    ),
    "lockup": AnalystNodeSpec(
        key="lockup",
        agent_node="Lock-up Monitor",
        clear_node="Msg Clear Lock-up",
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


