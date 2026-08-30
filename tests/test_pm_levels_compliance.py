"""Numeric size on PortfolioDecision, and gate compliance as a computed fact.

Before this, the Portfolio Manager's size existed only inside the executive
summary's prose. An A/B on this pipeline found the model does honour a binding
ruling — 12/12 across four arms on gemini-3.1-pro-preview at high thinking — but
that is only a useful claim if it is checked on every run rather than assumed from
a sample, and a decision ledger cannot record obedience it has to read out of a
paragraph.

Three properties these tests hold:

* **A violation is reported, never corrected.** Overwriting the size while the
  narrative still argued for the original produces a report that contradicts
  itself, and a reader believes the narrative.
* **Unverifiable is not compliant.** No size stated, or no ruling to compare
  against, is a gap in the record — not a control that was exercised.
* **No cross-field validator raises.** Same trap as TraderProposal: the free-text
  retry discards the whole structured object, so a validator refusing an
  out-of-limit size would destroy the number the check needs.

No network, no LLM, no API key.
"""

from __future__ import annotations

import pytest

from tradingagents import risk_engine
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.risk_engine import (
    COMPLY_OK,
    COMPLY_UNVERIFIABLE,
    COMPLY_VIOLATED,
    COMPLIANCE_TEXT,
    check_compliance,
    render_compliance,
)


def _dec(**over):
    base = {"rating": "Buy", "executive_summary": "s", "investment_thesis": "t"}
    base.update(over)
    return PortfolioDecision(**base)


def _gate(verdict="clamped", approved=5.0, proposed=20.0):
    return {"verdict": verdict, "approved_size_pct": approved,
            "proposed_size_pct": proposed, "reasons": [], "binding": True,
            "symbol": "NVDA", "level_flags": [], "rungs_tested": 10}


# ---------------------------------------------------------------------------
# The fields
# ---------------------------------------------------------------------------

def test_the_numeric_fields_exist():
    for name in ("position_size_pct", "entry_price", "stop_loss", "price_target"):
        assert name in PortfolioDecision.model_fields, name


@pytest.mark.parametrize("junk", ["", "N/A", "none", "NULL", "-", "TBD", "unknown"])
def test_placeholder_strings_coerce_rather_than_raise(junk):
    d = _dec(position_size_pct=junk, entry_price=junk, stop_loss=junk,
             price_target=junk)
    assert d.position_size_pct is None
    assert d.entry_price is None
    assert d.stop_loss is None


def test_no_model_validator_was_added():
    # Guards the decision, not just the behaviour: a cross-field validator would
    # re-introduce the discard-everything failure via the free-text retry.
    assert not PortfolioDecision.__pydantic_decorators__.model_validators


def test_unstated_size_is_none_not_zero():
    # "I did not state a size" and "hold no position" are different decisions.
    assert _dec().levels()["position_size_pct"] is None


def test_zero_size_is_a_stated_value():
    assert _dec(rating="Hold", position_size_pct=0.0).levels()[
        "position_size_pct"] == 0.0


@pytest.mark.parametrize("value", [0.05, 0.5, 1.0])
def test_the_hundred_x_hazard_is_flagged_at_this_end_too(value):
    # Same trap as TraderProposal: "0.05" could be five percent or one hundredth.
    levels = _dec(position_size_pct=value).levels()
    assert "size_ambiguous" in levels["flags"]
    assert levels["position_size_pct"] == value      # not rescaled


def test_a_size_above_one_hundred_is_flagged():
    assert "size_out_of_range" in _dec(position_size_pct=250.0).levels()["flags"]


@pytest.mark.parametrize("rating", ["Buy", "Overweight"])
def test_zero_size_on_a_buy_is_flagged(rating):
    # The rating reads as an instruction and the size quietly cancels it.
    assert "size_zero_on_a_buy" in _dec(rating=rating,
                                        position_size_pct=0.0).levels()["flags"]


@pytest.mark.parametrize("rating", ["Hold", "Sell", "Underweight"])
def test_zero_size_is_fine_on_a_non_buy(rating):
    assert _dec(rating=rating, position_size_pct=0.0).levels()["flags"] == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_renderer_emits_the_numbers():
    text = render_pm_decision(_dec(position_size_pct=5.0, entry_price=100.0,
                                  stop_loss=95.0, price_target=115.0))
    assert "**Position Size**: 5.0% of portfolio" in text
    assert "**Entry Price**: 100.0" in text
    assert "**Stop Loss**: 95.0" in text


def test_renderer_omits_what_was_not_stated():
    text = render_pm_decision(_dec())
    for absent in ("Position Size", "Entry Price", "Stop Loss", "Price Target"):
        assert absent not in text, absent


def test_renderer_keeps_the_headers_downstream_parsers_read():
    text = render_pm_decision(_dec(position_size_pct=5.0))
    for header in ("**Rating**:", "**Executive Summary**:", "**Investment Thesis**:"):
        assert header in text, header


def test_renderer_names_an_inconsistency():
    text = render_pm_decision(_dec(position_size_pct=0.05))
    assert "**Level Warnings**:" in text
    assert "ambiguous" in text


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

def test_a_size_at_the_approved_ceiling_is_compliant():
    r = check_compliance(_dec(position_size_pct=5.0).levels(), _gate())
    assert r["status"] == COMPLY_OK
    assert r["violated"] is False


def test_a_size_below_the_ceiling_is_compliant():
    assert check_compliance(_dec(position_size_pct=2.0).levels(),
                            _gate())["status"] == COMPLY_OK


def test_a_size_above_the_ceiling_is_a_violation():
    r = check_compliance(_dec(position_size_pct=20.0).levels(), _gate())
    assert r["status"] == COMPLY_VIOLATED
    assert r["violated"] is True
    assert "size_exceeds_approved" in r["reasons"]


def test_a_size_on_a_blocked_trade_is_a_violation():
    r = check_compliance(_dec(position_size_pct=4.0).levels(),
                         _gate(verdict="blocked", approved=None))
    assert r["status"] == COMPLY_VIOLATED
    assert "size_on_a_blocked_trade" in r["reasons"]


def test_zero_size_on_a_blocked_trade_is_compliant():
    r = check_compliance(_dec(rating="Hold", position_size_pct=0.0).levels(),
                         _gate(verdict="blocked", approved=None))
    assert r["status"] == COMPLY_OK


def test_no_size_stated_is_unverifiable_not_compliant():
    # A gap in the record is not a control that was exercised.
    r = check_compliance(_dec().levels(), _gate())
    assert r["status"] == COMPLY_UNVERIFIABLE
    assert r["violated"] is False
    assert "no_size_stated" in r["reasons"]


@pytest.mark.parametrize("gate", [None, {}, {"verdict": "not_evaluated",
                                            "approved_size_pct": None}])
def test_no_ruling_is_unverifiable(gate):
    r = check_compliance(_dec(position_size_pct=20.0).levels(), gate)
    assert r["status"] == COMPLY_UNVERIFIABLE
    assert "no_ruling" in r["reasons"]


def test_a_pass_with_no_recorded_ceiling_is_unverifiable():
    r = check_compliance(_dec(position_size_pct=20.0).levels(),
                         _gate(verdict="pass", approved=None))
    assert r["status"] == COMPLY_UNVERIFIABLE


def test_a_pass_still_checks_the_size_against_its_ceiling():
    # The gate passing the Trader's proposal does not licence the PM to invent a
    # larger one.
    r = check_compliance(_dec(position_size_pct=30.0).levels(),
                         _gate(verdict="pass", approved=20.0))
    assert r["status"] == COMPLY_VIOLATED


def test_free_text_levels_are_unverifiable():
    r = check_compliance({}, _gate())
    assert r["status"] == COMPLY_UNVERIFIABLE


def test_decision_level_flags_are_carried_into_the_reasons():
    r = check_compliance(_dec(position_size_pct=0.05).levels(), _gate())
    assert "decision_levels_inconsistent" in r["reasons"]


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------

def test_a_compliant_decision_gets_no_banner():
    # A "complied" note on every run is noise; the ruling is already in the report.
    assert render_compliance(
        check_compliance(_dec(position_size_pct=5.0).levels(), _gate())) == ""


def test_a_violation_notice_says_the_decision_was_not_altered():
    text = render_compliance(
        check_compliance(_dec(position_size_pct=20.0).levels(), _gate()))
    assert "RISK GATE VIOLATION" in text
    assert "has NOT been altered" in text
    assert "reader believes the" in text
    assert "Final size stated: 20.0%" in text
    assert "Size approved by the gate: 5.0%" in text


def test_an_unverifiable_result_under_a_binding_ruling_is_announced():
    # A binding constraint that went unchecked is a real gap in the record.
    text = render_compliance(check_compliance(_dec().levels(), _gate()))
    assert "NOT VERIFIED" in text
    assert "VIOLATION" not in text


def test_an_unverifiable_result_with_no_binding_ruling_is_silent():
    # There was no constraint to enforce. A banner here would be noise on every
    # run without a portfolio, and it would break the documented contract that the
    # free-text fallback passes the model's text through unchanged.
    for gate in (None, {}, {"verdict": "not_evaluated", "binding": False},
                 {"verdict": "pass", "approved_size_pct": 20.0, "binding": False}):
        result = check_compliance(_dec().levels(), gate)
        assert result["status"] == COMPLY_UNVERIFIABLE
        assert render_compliance(result) == "", gate


def test_the_free_text_path_appends_nothing_without_a_binding_ruling(self=None):
    from tradingagents.agents.managers.portfolio_manager import (
        create_portfolio_manager)

    class _LLM:
        def with_structured_output(self, *_a, **_k):
            raise AttributeError("unsupported")

        def invoke(self, _p):
            class _R:
                content = "**Rating**: Sell\n\nExit ahead of guidance."
            return _R()

    out = create_portfolio_manager(_LLM())(_pm_state({}))
    assert out["final_trade_decision"] == "**Rating**: Sell\n\nExit ahead of guidance."


def test_every_compliance_reason_has_text():
    produced = set()
    for levels, gate in (
            (_dec(position_size_pct=20.0).levels(), _gate()),
            (_dec(position_size_pct=4.0).levels(), _gate(verdict="blocked",
                                                         approved=None)),
            (_dec().levels(), _gate()),
            (_dec(position_size_pct=20.0).levels(), None),
            (_dec(position_size_pct=0.05).levels(), _gate())):
        produced.update(check_compliance(levels, gate)["reasons"])
    assert produced
    assert not produced - set(COMPLIANCE_TEXT), produced - set(COMPLIANCE_TEXT)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class _Structured:
    def __init__(self, result):
        self._result = result

    def invoke(self, _p):
        return self._result


def _pm_state(gate):
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("NVDA", "2026-08-30")
    state.update({"investment_plan": "p", "trader_investment_plan": "tp",
                  "market_report": "m", "sentiment_report": "s", "news_report": "n",
                  "fundamentals_report": "f", "earnings_report": "e",
                  "policy_report": "", "hot_money_report": "", "lockup_report": "",
                  "risk_gate": gate})
    return state


def _run_pm(decision, gate):
    from tradingagents.agents.managers.portfolio_manager import (
        create_portfolio_manager)

    class _LLM:
        def with_structured_output(self, *_a, **_k):
            return _Structured(decision)

        def invoke(self, _p):
            class _R:
                content = "prose"
            return _R()

    return create_portfolio_manager(_LLM())(_pm_state(gate))


def test_the_node_returns_both_ledger_fields():
    out = _run_pm(_dec(position_size_pct=5.0, entry_price=100.0), _gate())
    assert out["pm_levels"]["position_size_pct"] == 5.0
    assert out["gate_compliance"]["status"] == COMPLY_OK


def test_a_violation_is_appended_to_the_report_not_silently_recorded():
    out = _run_pm(_dec(position_size_pct=20.0), _gate())
    assert out["gate_compliance"]["violated"] is True
    assert "RISK GATE VIOLATION" in out["final_trade_decision"]


def test_the_reported_size_is_left_as_the_model_wrote_it():
    # Reported, never corrected: the structured field must still show what the
    # decision actually said, or the record of the violation is itself wrong.
    out = _run_pm(_dec(position_size_pct=20.0), _gate())
    assert out["pm_levels"]["position_size_pct"] == 20.0


def test_a_compliant_run_appends_nothing():
    out = _run_pm(_dec(position_size_pct=5.0), _gate())
    assert "RISK GATE VIOLATION" not in out["final_trade_decision"]
    assert "NOT VERIFIED" not in out["final_trade_decision"]


def test_the_free_text_path_yields_empty_levels_and_unverifiable():
    from tradingagents.agents.managers.portfolio_manager import (
        create_portfolio_manager)

    class _LLM:
        def with_structured_output(self, *_a, **_k):
            raise AttributeError("unsupported")

        def invoke(self, _p):
            class _R:
                content = "prose only"
            return _R()

    out = create_portfolio_manager(_LLM())(_pm_state(_gate()))
    assert out["pm_levels"] == {}
    assert out["gate_compliance"]["status"] == COMPLY_UNVERIFIABLE


def test_state_declares_both_fields_and_they_start_empty():
    from tradingagents.agents.utils.agent_states import AgentState
    from tradingagents.graph.propagation import Propagator

    for field in ("pm_levels", "gate_compliance"):
        assert field in AgentState.__annotations__, field
    init = Propagator().create_initial_state("NVDA", "2026-08-30")
    assert init["pm_levels"] == {}
    assert init["gate_compliance"] == {}
