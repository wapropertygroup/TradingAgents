"""Portfolio context: reaches the decision agents, never the analysts.

The caller (ystocker's ``exposure`` engine) computes a holdings-and-limits block
deterministically and hands it in at run start. Two properties are asserted here
because both fail silently:

* **The analysts must not see it.** A fundamentals or news read shaded by what the
  reader happens to hold is the confirmation-bias failure, and nothing in the
  output would reveal it. Checked structurally against the analyst sources rather
  than by running a node, so it holds for an analyst added later too.
* **Absent must mean unknown, not flat.** Most callers have no portfolio. An agent
  that reads silence as "nothing is held" will size a position as though opening
  from zero.

No network and no API key: the LLM is a stub that records the prompt it was given.
"""

from __future__ import annotations

import pathlib

import pytest

from tradingagents.agents.utils.agent_utils import (
    get_portfolio_block,
    get_portfolio_context_from_state,
)
from tradingagents.graph.propagation import Propagator

SENTINEL = "ZZ_PORTFOLIO_SENTINEL_ZZ"
RULES_MARKER = "headroom against the floor"

#: Every agent whose prompt is allowed to carry the block, by module path.
DECISION_AGENTS = (
    "tradingagents/agents/trader/trader.py",
    "tradingagents/agents/managers/portfolio_manager.py",
    "tradingagents/agents/risk_mgmt/aggressive_debator.py",
    "tradingagents/agents/risk_mgmt/conservative_debator.py",
    "tradingagents/agents/risk_mgmt/neutral_debator.py",
)

_REPO = pathlib.Path(__file__).resolve().parents[1]


class _FakeLLM:
    """Records the prompt it is asked to run. No provider, no key, no network."""

    def __init__(self):
        self.seen: list[str] = []

    def invoke(self, arg):
        self.seen.append(str(arg))

        class _Reply:
            content = "ok"
            tool_calls: list = []

        return _Reply()

    def bind_tools(self, *_a, **_k):
        return self

    # Deliberately no with_structured_output: invoke_structured_or_freetext then
    # takes its documented free-text path, which is the one this asserts against.


def _state(portfolio_context: str = ""):
    state = Propagator().create_initial_state(
        "NVDA", "2026-08-30", portfolio_context=portfolio_context)
    state.update({
        "investment_plan": "plan", "trader_investment_plan": "trader plan",
        "market_report": "m", "sentiment_report": "s", "news_report": "n",
        "fundamentals_report": "f", "policy_report": "p",
        "hot_money_report": "h", "lockup_report": "l", "earnings_report": "e",
    })
    state["investment_debate_state"]["judge_decision"] = "plan"
    return state


def _decision_nodes():
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
    from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
    from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
    from tradingagents.agents.trader.trader import create_trader

    return {
        "aggressive": create_aggressive_debator,
        "conservative": create_conservative_debator,
        "neutral": create_neutral_debator,
        "trader": create_trader,
        "portfolio_manager": create_portfolio_manager,
    }


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------

def test_state_carries_the_block_verbatim():
    # Verbatim matters: every number in it was computed by a deterministic engine
    # precisely so no model has to, and reformatting would put the arithmetic back
    # within reach of one.
    state = _state(SENTINEL)
    assert state["portfolio_context"] == SENTINEL
    assert get_portfolio_context_from_state(state) == SENTINEL


def test_absent_context_is_empty_not_a_sentence():
    state = _state()
    assert state["portfolio_context"] == ""
    assert get_portfolio_context_from_state(state) == ""
    assert get_portfolio_block(state, "Holdings:") == ""


@pytest.mark.parametrize("value", [None, 42, [], {}, "   ", "\n\t"])
def test_non_string_or_blank_context_degrades_to_empty(value):
    # A malformed state must not raise inside a prompt build, half way through a
    # run that has already spent its analyst calls.
    assert get_portfolio_context_from_state({"portfolio_context": value}) == ""


def test_block_carries_the_rules_and_the_heading():
    block = get_portfolio_block(_state(SENTINEL), "Holdings:")
    assert block.startswith("\nHoldings:\n")
    assert SENTINEL in block
    for phrase in ("measured FLOOR", RULES_MARKER, "only PASS",
                   "INDETERMINATE", "must not be invented"):
        assert phrase in block, phrase


def test_graph_constructor_accepts_and_stores_the_context():
    # Checked without building a graph: instantiating TradingAgentsGraph resolves
    # LLM providers and would need a key.
    import inspect

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sig = inspect.signature(TradingAgentsGraph.__init__)
    assert "portfolio_context" in sig.parameters
    assert sig.parameters["portfolio_context"].default == ""


def test_run_signature_ignores_the_portfolio():
    # A portfolio moves daily. Folding it into the checkpoint key would turn every
    # resume of the same ticker into a cold start.
    source = (_REPO / "tradingagents/graph/trading_graph.py").read_text(encoding="utf-8")
    start = source.index("def _run_signature")
    body = source[start:source.index("def ", start + 10)]
    assert "portfolio" not in body


# ---------------------------------------------------------------------------
# Who sees it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_decision_nodes()))
def test_decision_agent_prompt_carries_the_block(name):
    llm = _FakeLLM()
    node = _decision_nodes()[name](llm)
    node(_state(SENTINEL))
    joined = "\n".join(llm.seen)
    assert SENTINEL in joined, f"{name} did not receive the block"
    assert RULES_MARKER in joined, f"{name} received the block without its rules"


@pytest.mark.parametrize("name", sorted(_decision_nodes()))
def test_decision_agent_prompt_omits_it_when_absent(name):
    # No stray heading, and no "no holdings" sentence invented on the way past.
    llm = _FakeLLM()
    node = _decision_nodes()[name](llm)
    node(_state())
    joined = "\n".join(llm.seen)
    assert RULES_MARKER not in joined
    assert "current portfolio and stated limits" not in joined


@pytest.mark.parametrize("path", sorted(
    str(p.relative_to(_REPO))
    for p in (_REPO / "tradingagents/agents/analysts").glob("*.py")
    if p.name != "__init__.py"
))
def test_analyst_sources_never_reference_the_portfolio(path):
    # Structural, not behavioural: this must also hold for an analyst added after
    # this test was written, and running every analyst node needs a toolkit.
    source = (_REPO / path).read_text(encoding="utf-8")
    for banned in ("portfolio_context", "get_portfolio_block"):
        assert banned not in source, (
            f"{path} reads the holder's portfolio. An analyst's job is to read the "
            f"security; knowing what the reader already owns invites it to justify "
            f"the position rather than assess it.")


def test_only_the_declared_decision_agents_reference_it():
    # Catches the block spreading by copy-paste into a sixth prompt without the
    # design decision being revisited.
    hits = []
    for path in (_REPO / "tradingagents/agents").rglob("*.py"):
        rel = str(path.relative_to(_REPO))
        if rel.endswith("utils/agent_utils.py"):
            continue        # where the helper lives
        if "get_portfolio_block" in path.read_text(encoding="utf-8"):
            hits.append(rel)
    assert sorted(hits) == sorted(DECISION_AGENTS), sorted(hits)
