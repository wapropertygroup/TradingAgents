"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        astock_reports = "\n".join([
            "Policy report: " + state.get("policy_report", ""),
            "Hot money / capital-flow report: " + state.get("hot_money_report", ""),
            "Lock-up / insider-reduction report: " + state.get("lockup_report", ""),
        ])
        # Read straight off the state rather than relying on the debate to have
        # carried it. The debate summary is a lossy channel — a revision that
        # neither researcher happened to quote would otherwise vanish before the
        # decision, which is the same failure the A-share block above exists to
        # prevent.
        #
        # Gated on presence, unlike the A-share block above, because earnings is
        # opt-in and off by default: an unconditional block would print a heading
        # with nothing under it plus a paragraph of instructions about a report
        # that does not exist, on every default run. That both spends tokens and
        # invites the model to comment on an absence.
        earnings_report = state.get("earnings_report", "")
        earnings_block = (
            f"""**Earnings & estimate-revision evidence:**
{earnings_report}

When this report is present, reconcile it explicitly against your rating rather
than mentioning it in passing. Address each of: which way consensus EPS has moved
over 7, 30 and 90 days and for which fiscal period; how broad the analyst
participation behind that move was, and whether the up and down counts agree with
the trend; what management actually guided, as distinct from what analysts
inferred; the recent surprise record and whether beats came against a rising or
falling bar; and what post-earnings drift did. State where your rating disagrees
with the revision direction and why — a Buy against broadly falling estimates
needs a reason, and so does a Sell against broadly rising ones.

The momentum band is computed, not argued: report it as given. A band of
Insufficient Data, or a field marked unavailable, means the coverage or vendor
history does not exist — it is not a neutral reading, and it must not be filled in
from prior knowledge of the company.
"""
            if earnings_report
            else ""
        )
        # Same reasoning as the earnings block above: read straight off the
        # state rather than trusting the bull/bear debate to have carried it,
        # since a report neither side happened to quote otherwise vanishes
        # before the decision that most needs it.
        quality_report = state.get("quality_report", "")
        quality_block = (
            f"""**Business-quality evidence:**
{quality_report}

Reconcile it explicitly: does the quality tier support or cut against the bull/
bear case you are weighing? The tier and every ratio behind it are computed, not
argued — report them as given, and do not let a debate-side characterization of
"strong margins" or "weak balance sheet" override the published numbers. A tier
of Insufficient Data means signal coverage does not exist, not that quality is
neutral.
"""
            if quality_report
            else ""
        )
        valuation_report = state.get("valuation_report", "")
        valuation_block = (
            f"""**Valuation evidence:**
{valuation_report}

Reconcile it explicitly: a bull case argued purely on business quality while the
valuation tier is Expensive or Extreme Premium needs to say so and address it,
not omit it. The tier and every multiple behind it are computed, not argued. A
missing trailing P/E is commonly a negative-earnings company — absent, not
evidence of cheapness or expense.
"""
            if valuation_report
            else ""
        )

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Debate History:**
{history}

**A-share specialist evidence (when present):**
{astock_reports}

Explicitly reconcile policy direction, speculative capital flow, and potential
unlock/reduction supply shocks; do not let an omitted debate reference erase a
specialist report.

{earnings_block}
{quality_block}
{valuation_block}
{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
