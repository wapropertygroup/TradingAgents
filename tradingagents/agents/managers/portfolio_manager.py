"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_portfolio_block,
    get_risk_gate_block,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.a_stock import is_a_share


_A_SHARE_FINAL_CONSTRAINTS = """
This is a mainland China A-share. The final action must respect T+1, the
applicable 5%/10%/20% daily price limit, 100-share buy lots, market sessions,
suspension/delisting status, and margin eligibility. An unlock is potential
supply rather than proof of a sale. Do not state an executable price unless the
provided evidence supports it.
"""


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        market_constraints = (
            _A_SHARE_FINAL_CONSTRAINTS
            if is_a_share(state["company_of_interest"])
            else ""
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )
        # Placed in the final synthesis directly. By this point the signal has
        # passed through a debate summary, a research plan, a trader proposal and
        # a risk debate; anything that survives only by being quoted at each hop
        # is gone. The instruction below is the load-bearing part: the one thing a
        # final decision must not do is convert an admitted gap into a verdict.
        portfolio_block = get_portfolio_block(
            state, "**The holder's current portfolio and stated limits:**")
        risk_gate_block = get_risk_gate_block(state)
        earnings_report = state.get("earnings_report", "")
        earnings_block = (
            "\n**Earnings & estimate-revision evidence:**\n"
            f"{earnings_report}\n\n"
            "Rules for using it. Every number and the momentum band were computed "
            "from provider data — quote them as given, never recomputed or rounded. "
            "A field marked unavailable and a band of Insufficient Data are "
            "statements that the analyst coverage or vendor history does not exist; "
            "they are not neutral readings and must not be resolved with an estimate, "
            "an assumption, or prior knowledge of this company. If the decision rests "
            "on earnings evidence that is absent or low-confidence, say so in the "
            "rationale and let it reduce conviction rather than filling the hole.\n"
            if earnings_report
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
{earnings_block}
{portfolio_block}
{risk_gate_block}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.

{market_constraints}

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
