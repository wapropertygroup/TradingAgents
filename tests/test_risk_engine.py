"""The deterministic risk gate: clamp on breach, block on unverifiable levels.

The two rules the user chose, and the ones these tests exist to hold:

* **A breach clamps.** A run costs fifteen minutes and real API spend, so the
  largest allowed size is returned rather than the analysis thrown away.
* **Missing or inconsistent levels block.** There is nothing trustworthy to clamp
  *to*, and sizing down an unverifiable proposal produces a smaller position
  justified by the same unchecked numbers.

Two properties matter more than either, because both have already been got wrong
once in this pipeline:

* **Clamping is downward only.** A larger trade can pass where a smaller one
  breaches — dilution rescues an existing concentration — and a gate that enlarged
  a position to clear a risk limit would be inventing a trade nobody proposed.
* **Absence is not zero.** No portfolio is "not evaluated", not a pass. An unstated
  cash balance caps nothing; a stated zero balance caps everything.

Pure: no network, no LLM, no fixtures beyond plain dicts.
"""

from __future__ import annotations

import pytest

from tradingagents.risk_engine import (
    GATE_BLOCKED,
    GATE_CLAMPED,
    GATE_NOT_EVALUATED,
    GATE_PASS,
    REASON_TEXT,
    evaluate,
    render,
)


def _levels(**over):
    base = {"action": "Buy", "entry_price": 100.0, "stop_loss": 95.0,
            "target_price": 115.0, "position_size_pct": 4.0,
            "position_sizing_text": None, "reward_risk": 3.0,
            "complete": True, "flags": []}
    base.update(over)
    return base


def _ladder(passing_upto=10.0, total=100_000.0, cash=None, baseline="pass"):
    """A ladder where every rung up to *passing_upto* passes and the rest breach."""
    rungs = []
    for pct in (0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0):
        if pct == 0.0:
            verdict = baseline
        else:
            verdict = "pass" if pct <= passing_upto else "breach"
        rungs.append({"pct": pct, "value": total * pct / 100.0,
                      "verdict": verdict, "breach_count": 0 if verdict == "pass" else 1,
                      "worst_symbol": "MSFT", "worst_floor_pct": pct,
                      "worst_ceiling_pct": pct})
    return {"symbol": "SOXQ", "total_value": total, "cash": cash,
            "max_single_name_pct": 8.0, "max_issuer_pct": None,
            "rungs": rungs, "rung_count": len(rungs), "max_rung_pct": 25.0}


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("portfolio", [None, {}, {"rungs": []},
                                      {"rungs": "not a list"}])
def test_no_portfolio_is_not_evaluated_rather_than_passed(portfolio):
    # Most callers of this framework have no holdings. Blocking them all would make
    # the gate a bug; passing them would claim a limit was checked.
    d = evaluate(_levels(), portfolio)
    assert d["verdict"] == GATE_NOT_EVALUATED
    assert d["binding"] is False


def test_not_evaluated_is_stated_as_not_an_approval():
    text = render(evaluate(_levels(), None))
    assert "not evaluated" in text
    assert "not an approval" in text


def test_unstated_cash_caps_nothing():
    # The bug that shipped once on the ystocker side: cash defaulting to zero made
    # every buy a liquidity breach.
    d = evaluate(_levels(position_size_pct=4.0), _ladder(cash=None))
    assert d["verdict"] == GATE_PASS
    assert "clamped_to_cash" not in d["reasons"]


def test_stated_zero_cash_blocks_a_buy():
    # Clamping to 0% would be a refusal wearing the word "approved", and a
    # Portfolio Manager reading CLAMPED describes it as permission at a smaller
    # size. A zero-size position is not a trade.
    d = evaluate(_levels(position_size_pct=4.0), _ladder(cash=0.0))
    assert d["verdict"] == GATE_BLOCKED
    assert d["approved_size_pct"] is None
    assert "clamped_to_cash" in d["reasons"]


def test_cash_clamps_below_the_limit_allowance():
    # $2,000 of a $100,000 portfolio is 2%, tighter than the 10% the limits allow.
    d = evaluate(_levels(position_size_pct=8.0), _ladder(cash=2_000.0))
    assert d["verdict"] == GATE_CLAMPED
    assert d["approved_size_pct"] == pytest.approx(2.0)
    assert "clamped_to_cash" in d["reasons"]


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_a_size_within_the_limits_passes_unchanged():
    d = evaluate(_levels(position_size_pct=4.0), _ladder(passing_upto=10.0))
    assert d["verdict"] == GATE_PASS
    assert d["approved_size_pct"] == 4.0
    assert d["binding"] is False


def test_a_breaching_size_is_clamped_to_the_largest_passing_rung():
    d = evaluate(_levels(position_size_pct=20.0), _ladder(passing_upto=8.0))
    assert d["verdict"] == GATE_CLAMPED
    assert d["approved_size_pct"] == 8.0
    assert "clamped_to_limit" in d["reasons"]
    assert d["binding"] is True


def test_clamping_never_enlarges_the_position():
    # The dilution case: only rungs at 15% and above pass, and the Trader asked for
    # 4%. Enlarging to 15% would be inventing a trade nobody proposed on the
    # reasoning that a bigger bet fixes a risk limit.
    rungs = [{"pct": p, "verdict": "pass" if p >= 15.0 else "breach"}
             for p in (0.0, 4.0, 10.0, 15.0, 20.0)]
    d = evaluate(_levels(position_size_pct=4.0),
                 {"symbol": "BND", "total_value": 10_000.0, "cash": None,
                  "rungs": rungs})
    assert d["verdict"] == GATE_BLOCKED
    assert d["approved_size_pct"] is None
    assert d["passing_rungs_above_proposal"] == [15.0, 20.0]


def test_larger_passing_sizes_are_reported_but_disclaimed():
    rungs = [{"pct": p, "verdict": "pass" if p >= 15.0 else "breach"}
             for p in (0.0, 4.0, 15.0)]
    text = render(evaluate(_levels(position_size_pct=4.0),
                           {"symbol": "BND", "total_value": 1.0, "cash": None,
                            "rungs": rungs}))
    assert "For information only" in text
    assert "is not risk" in text          # "increasing a position ... is not risk management"


def test_no_passing_rung_at_or_below_blocks():
    d = evaluate(_levels(position_size_pct=4.0),
                 _ladder(passing_upto=-1.0, baseline="breach"))
    assert d["verdict"] == GATE_BLOCKED
    assert "no_allowed_size" in d["reasons"]

def test_only_the_zero_rung_passing_is_a_block():
    # baseline passes but every real size breaches: there is no trade to approve.
    d = evaluate(_levels(position_size_pct=4.0), _ladder(passing_upto=-1.0))
    assert d["verdict"] == GATE_BLOCKED
    assert d["approved_size_pct"] is None


def test_an_already_breached_portfolio_is_named_as_such():
    # The finding is about the portfolio, not the proposal: no size of this trade
    # fixes a limit that was already broken.
    d = evaluate(_levels(position_size_pct=4.0),
                 _ladder(passing_upto=-1.0, baseline="breach"))
    assert "already_breached" in d["reasons"]


def test_an_indeterminate_rung_is_not_treated_as_passing():
    # An unverifiable limit is not a satisfied one, matching exposure.verdict_for.
    rungs = [{"pct": 0.0, "verdict": "pass"},
             {"pct": 4.0, "verdict": "indeterminate"}]
    d = evaluate(_levels(position_size_pct=4.0),
                 {"symbol": "X", "total_value": 1.0, "cash": None, "rungs": rungs})
    # Only the do-nothing rung is acceptable, which is a refusal.
    assert d["verdict"] == GATE_BLOCKED
    assert d["approved_size_pct"] is None


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [None, {}])
def test_free_text_levels_block(levels):
    # No structured output means no numbers at all.
    d = evaluate(levels, _ladder())
    assert d["verdict"] == GATE_BLOCKED
    assert "no_structured_levels" in d["reasons"]


def test_inconsistent_levels_block_rather_than_clamp():
    # Sizing down an unverifiable proposal produces a smaller position justified by
    # the same unchecked figures.
    d = evaluate(_levels(flags=["stop_not_below_entry"]), _ladder())
    assert d["verdict"] == GATE_BLOCKED
    assert "inconsistent_levels" in d["reasons"]
    assert d["approved_size_pct"] is None


def test_an_ambiguous_size_blocks():
    # The 100x hazard: "0.05" could be five percent or one hundredth of it.
    d = evaluate(_levels(position_size_pct=0.05, flags=["size_ambiguous"]),
                 _ladder())
    assert d["verdict"] == GATE_BLOCKED
    assert "inconsistent_levels" in d["reasons"]


def test_incomplete_levels_block():
    d = evaluate(_levels(target_price=None, complete=False), _ladder())
    assert d["verdict"] == GATE_BLOCKED
    assert "incomplete_levels" in d["reasons"]


def test_inconsistency_is_reported_before_incompleteness():
    # A contradiction is the more specific fault and the more useful report.
    d = evaluate(_levels(complete=False, flags=["target_wrong_side"]), _ladder())
    assert d["reasons"] == ["inconsistent_levels"]


def test_a_missing_size_blocks_rather_than_reading_as_zero():
    d = evaluate(_levels(position_size_pct=None, complete=True), _ladder())
    assert d["verdict"] == GATE_BLOCKED
    assert "no_size_proposed" in d["reasons"]


# ---------------------------------------------------------------------------
# Hold
# ---------------------------------------------------------------------------

def test_a_hold_is_not_gated():
    # Leaving a position alone is the one action that never needs permission.
    d = evaluate(_levels(action="Hold", position_size_pct=None, complete=False),
                 _ladder())
    assert d["verdict"] == GATE_PASS
    assert d["binding"] is False


def test_a_hold_is_not_gated_even_on_a_breached_portfolio():
    d = evaluate(_levels(action="Hold", position_size_pct=None, complete=False),
                 _ladder(passing_upto=-1.0, baseline="breach"))
    assert d["verdict"] == GATE_PASS


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_the_ruling_is_framed_as_binding():
    # A constraint phrased as a suggestion is one a model weighs against the
    # argument rather than obeys.
    text = render(evaluate(_levels(position_size_pct=20.0), _ladder(passing_upto=8.0)))
    assert "BINDING" in text
    assert "do not recommend a size above" in text


def test_a_block_says_the_trade_must_not_be_placed():
    text = render(evaluate(_levels(flags=["stop_not_below_entry"]), _ladder()))
    assert "Approved size: NONE" in text
    assert "must not be placed" in text


def test_the_approved_size_is_called_a_ceiling_not_a_target():
    text = render(evaluate(_levels(position_size_pct=20.0), _ladder(passing_upto=8.0)))
    assert "ceiling, not a target" in text


def test_untested_sizes_are_disclosed():
    # A silent cap reads as full coverage.
    text = render(evaluate(_levels(), _ladder()))
    assert "were not tested" in text
    assert "Sizes tested:" in text


def test_reasons_render_as_sentences_not_codes():
    text = render(evaluate(_levels(position_size_pct=20.0), _ladder(passing_upto=8.0)))
    assert "clamped_to_limit" not in text
    assert REASON_TEXT["clamped_to_limit"] in text


def test_every_reason_code_has_text():
    # A code with no entry renders as a raw identifier in a report.
    produced = set()
    for levels, ladder in (
            (_levels(), None),
            ({}, _ladder()),
            (_levels(flags=["x"]), _ladder()),
            (_levels(complete=False), _ladder()),
            (_levels(position_size_pct=None), _ladder()),
            (_levels(position_size_pct=20.0), _ladder(passing_upto=8.0)),
            (_levels(position_size_pct=8.0), _ladder(cash=2_000.0)),
            (_levels(), _ladder(passing_upto=-1.0, baseline="breach"))):
        produced.update(evaluate(levels, ladder)["reasons"])
    assert produced
    assert not produced - set(REASON_TEXT), produced - set(REASON_TEXT)


def test_render_is_delimited():
    text = render(evaluate(_levels(), _ladder()))
    assert text.startswith("<start_of_risk_gate>")
    assert text.rstrip().endswith("<end_of_risk_gate>")


# ---------------------------------------------------------------------------
# Malformed input must not raise mid-run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rungs", [
    [{"pct": "abc", "verdict": "pass"}],
    [{"verdict": "pass"}],
    [{"pct": None, "verdict": "pass"}],
    "not a list",
    None,
])
def test_a_malformed_ladder_does_not_raise(rungs):
    # Raising here would kill a run that has already spent every analyst call.
    d = evaluate(_levels(), {"symbol": "X", "total_value": 1.0, "cash": None,
                             "rungs": rungs})
    assert d["verdict"] in (GATE_BLOCKED, GATE_NOT_EVALUATED)
    assert isinstance(d["reasons"], list)


def test_a_non_numeric_cash_is_ignored():
    d = evaluate(_levels(), _ladder(cash="lots"))
    assert d["verdict"] == GATE_PASS


def test_a_zero_total_value_does_not_divide_by_zero():
    d = evaluate(_levels(), _ladder(total=0.0, cash=100.0))
    assert d["verdict"] in (GATE_PASS, GATE_CLAMPED, GATE_BLOCKED)
