"""The deterministic risk gate as a graph node.

Sits between the Trader and the risk debate so its ruling is in front of every
agent that could argue about size. Placed *before* the debate rather than after it
deliberately: a gate that ran last would be overruling a conclusion three agents
had already committed to in prose, and the report would contain both.

Runs no model. It reads two mappings off the state and writes one back.
"""

from __future__ import annotations

from tradingagents import risk_engine


def create_risk_gate():
    """A node that rules on the Trader's proposal. Takes no LLM: it is arithmetic."""

    def risk_gate_node(state) -> dict:
        decision = risk_engine.evaluate(state.get("trader_levels"),
                                        state.get("portfolio_data"))
        return {"risk_gate": decision}

    return risk_gate_node
