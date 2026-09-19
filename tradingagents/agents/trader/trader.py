"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_market_context_block,
    get_portfolio_block,
    get_relative_strength_block,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured,
)
from tradingagents.dataflows.a_stock import is_a_share


_A_SHARE_CONSTRAINTS = """
For this mainland China A-share, obey market mechanics: T+1 settlement for
shares bought today; board-specific daily price limits (normally 10%, 20% for
STAR/ChiNext, 5% for ST names, subject to listing-day exceptions); orders in
100-share lots except permitted odd-lot sales; Shanghai/Shenzhen trading
sessions and auction windows; suspension/delisting risk; and margin eligibility.
Do not invent an executable price or imply same-day round trips are available.
"""


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        market_constraints = _A_SHARE_CONSTRAINTS if is_a_share(company_name) else ""
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""
        # Included directly, not left to the Research Manager's plan to relay.
        # The plan is a summary, and estimate revision direction is exactly the
        # kind of dated, countable detail a summary drops first.
        portfolio_block = get_portfolio_block(
            state, "The holder's current portfolio and stated limits:")
        market_block = get_market_context_block(state, "Market regime notes:")
        relative_strength_block = get_relative_strength_block(
            state, "Relative strength vs. peers:")
        earnings_report = state.get("earnings_report", "")
        earnings_block = (
            f"\n\nEarnings & estimate-revision evidence:\n{earnings_report}\n\n"
            "Every figure and the momentum band above were computed from provider "
            "data; report them as given and do not recompute or round them. Fields "
            "marked unavailable, and a band of Insufficient Data, mean the coverage "
            "does not exist — treat them as unknown rather than neutral, and do not "
            "size a position as though the signal were confirmed."
            if earnings_report
            else ""
        )
        # Same reasoning as earnings_block: the plan is a summary, and a quality
        # or valuation tier is exactly the kind of countable detail a summary
        # drops first when the debate spent its words elsewhere.
        quality_report = state.get("quality_report", "")
        quality_block = (
            f"\n\nBusiness-quality evidence:\n{quality_report}\n\n"
            "Every ratio and the quality tier above were computed from provider "
            "data; report them as given and do not recompute or round them. The "
            "tier is final -- do not upgrade, downgrade or re-label it. Insufficient "
            "Data means signal coverage does not exist, not that quality is neutral."
            if quality_report
            else ""
        )
        valuation_report = state.get("valuation_report", "")
        valuation_block = (
            f"\n\nValuation evidence:\n{valuation_report}\n\n"
            "Every multiple and the valuation tier above were computed from "
            "provider data; report them as given and do not recompute or round "
            "them. A missing trailing P/E is commonly a negative-earnings company "
            "-- absent, not evidence the stock is cheap or expensive. The tier is "
            "final -- do not upgrade, downgrade or re-label it."
            if valuation_report
            else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + market_constraints
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\n"
                    f"{report_section}"
                    f"Proposed Investment Plan: {investment_plan}\n"
                    f"{earnings_block}\n"
                    f"{quality_block}\n"
                    f"{valuation_block}\n"
                    f"{portfolio_block}\n"
                    f"{market_block}\n"
                    f"{relative_strength_block}\n"
                    f"Leverage these insights to make an informed and strategic decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Target Price**, **Position Size**, "
                    "**Position Sizing**: when you can state them"
                ),
            },
        ]

        trader_plan, proposal = invoke_structured(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        # The typed levels travel on the state as well as inside the rendered
        # markdown, so a deterministic risk engine downstream reads numbers rather
        # than regex-scraping prose. Empty on the free-text fallback path, where
        # the numbers genuinely do not exist -- a consumer must read {} as
        # "unstated" and not as zeros.
        levels = proposal.levels() if proposal is not None else {}

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "trader_levels": levels,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
