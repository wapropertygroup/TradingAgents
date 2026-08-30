"""Numeric trade levels on TraderProposal — the precondition for a risk engine.

A deterministic engine can only check numbers, and before this the Trader's size
was free text (``'5% of portfolio'``) with no target price at all, so a
reward-to-risk ratio could not be computed and a size could not be compared to a
limit. These tests pin the three properties that make the numbers trustworthy:

* **No cross-field validator raises.** ``invoke_structured_or_freetext`` catches
  *any* structured failure and retries as free text, so a validator that rejected
  an inconsistent proposal would discard every number in it. Inconsistency is
  therefore reported by :meth:`TraderProposal.levels`, not enforced.
* **Derived figures are computed, never asked for.** A reward-to-risk ratio a model
  wrote is a plausible number nobody can audit.
* **Absent means unstated, never zero.** Both in the fields and on the free-text
  path, where ``trader_levels`` is ``{}``.

No network and no API key: the LLM is a stub.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.schemas import (
    TraderProposal,
    render_trader_proposal,
)
from tradingagents.agents.utils.structured import (
    invoke_structured,
    invoke_structured_or_freetext,
)


def _prop(**over):
    base = {"action": "Buy", "reasoning": "because"}
    base.update(over)
    return TraderProposal(**base)


# ---------------------------------------------------------------------------
# The fields exist and coerce rather than reject
# ---------------------------------------------------------------------------

def test_the_four_numeric_fields_exist():
    fields = TraderProposal.model_fields
    for name in ("entry_price", "stop_loss", "target_price", "position_size_pct"):
        assert name in fields, name


@pytest.mark.parametrize("junk", ["", "N/A", "n/a", "none", "NULL", "-", "TBD",
                                  "unknown", "  na  "])
def test_placeholder_strings_coerce_to_none_on_every_numeric_field(junk):
    # #1058: models write a placeholder instead of omitting the field. Raising
    # here would drop the whole structured object, numbers included.
    p = _prop(entry_price=junk, stop_loss=junk, target_price=junk,
              position_size_pct=junk)
    assert p.entry_price is None
    assert p.stop_loss is None
    assert p.target_price is None
    assert p.position_size_pct is None


def test_numeric_strings_still_parse():
    p = _prop(entry_price="189.5", position_size_pct="4")
    assert p.entry_price == 189.5
    assert p.position_size_pct == 4.0


def test_an_inconsistent_proposal_still_validates():
    # The load-bearing one. A stop above entry on a Buy is wrong, and it must
    # construct anyway so the flag survives to a consumer.
    p = _prop(entry_price=100.0, stop_loss=110.0)
    assert "stop_not_below_entry" in p.levels()["flags"]


def test_no_model_validator_was_added():
    # Guards the decision rather than the behaviour: a future cross-field
    # validator would silently re-introduce the discard-everything failure.
    assert not getattr(TraderProposal, "__pydantic_decorators__").model_validators


# ---------------------------------------------------------------------------
# levels(): unstated vs zero
# ---------------------------------------------------------------------------

def test_an_empty_proposal_reports_unstated_not_zero():
    levels = _prop().levels()
    for key in ("entry_price", "stop_loss", "target_price",
                "position_size_pct", "reward_risk"):
        assert levels[key] is None, key
    assert levels["complete"] is False
    assert levels["flags"] == []


def test_complete_is_true_only_with_all_four_numbers():
    assert _prop(entry_price=100.0, stop_loss=95.0, target_price=115.0,
                 position_size_pct=4.0).levels()["complete"] is True
    # Any one missing and the proposal cannot be fully checked.
    assert _prop(entry_price=100.0, stop_loss=95.0,
                 target_price=115.0).levels()["complete"] is False
    assert _prop(entry_price=100.0, stop_loss=95.0,
                 position_size_pct=4.0).levels()["complete"] is False


def test_zero_is_a_stated_value_and_not_absence():
    # A 0% size is a real statement ("do not take a position"), distinct from
    # having said nothing about size.
    levels = _prop(position_size_pct=0.0).levels()
    assert levels["position_size_pct"] == 0.0
    assert "size_ambiguous" not in levels["flags"]


# ---------------------------------------------------------------------------
# The 100x hazard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.05, 0.5, 1.0, 0.01])
def test_a_size_at_or_below_one_is_flagged_ambiguous_not_rescaled(value):
    # "0.05" is a very plausible way for a model to write five percent, and the
    # two readings differ by 100x. Rescaling would risk a position a hundred times
    # the intended one; accepting silently would risk one a hundredth the size.
    levels = _prop(position_size_pct=value).levels()
    assert "size_ambiguous" in levels["flags"]
    assert levels["position_size_pct"] == value      # untouched, not rescaled


@pytest.mark.parametrize("value", [1.01, 4.0, 50.0, 100.0])
def test_an_unambiguous_size_is_not_flagged(value):
    assert "size_ambiguous" not in _prop(position_size_pct=value).levels()["flags"]


def test_a_size_above_one_hundred_is_flagged():
    levels = _prop(position_size_pct=250.0).levels()
    assert "size_out_of_range" in levels["flags"]
    assert levels["position_size_pct"] == 250.0      # reported, not clamped


# ---------------------------------------------------------------------------
# Reward:risk is computed, and refuses to emit a meaningless number
# ---------------------------------------------------------------------------

def test_reward_risk_for_a_long():
    # (115 - 100) / (100 - 95) == 3
    assert _prop(entry_price=100.0, stop_loss=95.0,
                 target_price=115.0).levels()["reward_risk"] == 3.0


def test_reward_risk_for_a_short_uses_the_other_sign():
    # A Sell profits downward: (100 - 85) / (105 - 100) == 3
    levels = TraderProposal(action="Sell", reasoning="r", entry_price=100.0,
                            stop_loss=105.0, target_price=85.0).levels()
    assert levels["reward_risk"] == 3.0
    assert levels["flags"] == []


def test_reward_risk_is_none_when_a_level_is_missing():
    for kwargs in ({"entry_price": 100.0, "stop_loss": 95.0},
                   {"entry_price": 100.0, "target_price": 115.0},
                   {"stop_loss": 95.0, "target_price": 115.0}):
        assert _prop(**kwargs).levels()["reward_risk"] is None, kwargs


def test_reward_risk_is_none_when_the_stop_is_on_the_wrong_side():
    # Dividing by a negative risk leg would emit a number that reads as a ratio.
    levels = _prop(entry_price=100.0, stop_loss=110.0, target_price=115.0).levels()
    assert levels["reward_risk"] is None
    assert "stop_not_below_entry" in levels["flags"]


def test_reward_risk_is_none_when_the_stop_equals_the_entry():
    # Zero risk leg: a division by zero, and a "risk-free trade" claim.
    levels = _prop(entry_price=100.0, stop_loss=100.0, target_price=115.0).levels()
    assert levels["reward_risk"] is None


def test_reward_risk_is_none_when_the_target_is_the_wrong_side():
    levels = _prop(entry_price=100.0, stop_loss=95.0, target_price=90.0).levels()
    assert levels["reward_risk"] is None
    assert "target_wrong_side" in levels["flags"]


def test_levels_on_a_hold_with_prices_is_flagged():
    # "Entry" and "stop" have no side on a Hold, so no side can be checked.
    levels = TraderProposal(action="Hold", reasoning="r", entry_price=100.0,
                            stop_loss=95.0).levels()
    assert "levels_without_direction" in levels["flags"]


def test_a_bare_hold_is_not_flagged():
    assert TraderProposal(action="Hold", reasoning="r").levels()["flags"] == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_renderer_shows_the_new_numbers_and_the_derived_ratio():
    text = render_trader_proposal(
        _prop(entry_price=100.0, stop_loss=95.0, target_price=115.0,
              position_size_pct=4.0))
    assert "**Target Price**: 115.0" in text
    assert "**Position Size**: 4.0% of portfolio" in text
    assert "**Reward:Risk**: 3.0:1" in text


def test_renderer_omits_what_was_not_stated():
    text = render_trader_proposal(_prop())
    for absent in ("Target Price", "Position Size", "Reward:Risk", "Entry Price",
                   "Stop Loss"):
        assert absent not in text, absent


def test_renderer_keeps_the_legacy_grep_line():
    # External code and the analyst stop-signal text grep for this exact line.
    assert render_trader_proposal(_prop()).rstrip().endswith(
        "FINAL TRANSACTION PROPOSAL: **BUY**")


def test_renderer_emits_no_ratio_when_it_cannot_be_computed():
    text = render_trader_proposal(_prop(entry_price=100.0, stop_loss=110.0,
                                        target_price=115.0))
    assert "Reward:Risk" not in text


def test_free_text_sizing_is_kept_alongside_the_number():
    levels = _prop(position_size_pct=4.0,
                   position_sizing="scale in over three tranches").levels()
    assert levels["position_size_pct"] == 4.0
    assert levels["position_sizing_text"] == "scale in over three tranches"


# ---------------------------------------------------------------------------
# The object survives to the state
# ---------------------------------------------------------------------------

class _StubStructured:
    def __init__(self, result):
        self._result = result

    def invoke(self, _prompt):
        return self._result


class _StubPlain:
    def invoke(self, _prompt):
        class _R:
            content = "free text answer"
        return _R()


def test_invoke_structured_returns_both_markdown_and_object():
    proposal = _prop(entry_price=100.0, stop_loss=95.0)
    text, obj = invoke_structured(_StubStructured(proposal), _StubPlain(),
                                  "p", render_trader_proposal, "Trader")
    assert "**Entry Price**: 100.0" in text
    assert obj is proposal


def test_free_text_path_returns_none_for_the_object():
    # There are no numbers on that path. A caller must read None as unstated.
    text, obj = invoke_structured(None, _StubPlain(), "p",
                                  render_trader_proposal, "Trader")
    assert text == "free text answer"
    assert obj is None


def test_a_structured_failure_falls_back_and_returns_none():
    class _Boom:
        def invoke(self, _p):
            raise ValueError("malformed")

    text, obj = invoke_structured(_Boom(), _StubPlain(), "p",
                                  render_trader_proposal, "Trader")
    assert text == "free text answer"
    assert obj is None


def test_the_old_helper_still_returns_just_a_string():
    # Three other agents call it; the tuple must not leak into their state.
    out = invoke_structured_or_freetext(
        _StubStructured(_prop()), _StubPlain(), "p",
        render_trader_proposal, "Trader")
    assert isinstance(out, str)


def test_trader_node_puts_levels_on_the_state():
    from tradingagents.agents.trader.trader import create_trader

    proposal = _prop(entry_price=100.0, stop_loss=95.0, target_price=115.0,
                     position_size_pct=4.0)

    class _LLM:
        def with_structured_output(self, *_a, **_k):
            return _StubStructured(proposal)

        def invoke(self, _p):
            raise AssertionError("should not reach the free-text path")

    state = {"company_of_interest": "NVDA", "investment_plan": "plan",
             "instrument_context": "ctx", "earnings_report": "",
             "portfolio_context": "", "messages": []}
    out = create_trader(_LLM())(state)
    assert out["trader_levels"]["reward_risk"] == 3.0
    assert out["trader_levels"]["complete"] is True
    assert out["trader_levels"]["flags"] == []


def test_trader_node_reports_empty_levels_on_the_free_text_path():
    from tradingagents.agents.trader.trader import create_trader

    class _LLM:
        def with_structured_output(self, *_a, **_k):
            raise AttributeError("unsupported")

        def invoke(self, _p):
            class _R:
                content = "prose only"
            return _R()

    state = {"company_of_interest": "NVDA", "investment_plan": "plan",
             "instrument_context": "ctx", "earnings_report": "",
             "portfolio_context": "", "messages": []}
    out = create_trader(_LLM())(state)
    assert out["trader_levels"] == {}


def test_state_declares_the_field_and_it_starts_empty():
    from tradingagents.agents.utils.agent_states import AgentState
    from tradingagents.graph.propagation import Propagator

    assert "trader_levels" in AgentState.__annotations__
    assert Propagator().create_initial_state("NVDA", "2026-08-30")[
        "trader_levels"] == {}

# ---------------------------------------------------------------------------
# Flags are surfaced, not just recorded
# ---------------------------------------------------------------------------

def test_renderer_names_an_inconsistency_in_words():
    # Without this the levels render as plausible numbers and the ratio just goes
    # missing, so the reader downstream has nothing to notice.
    text = render_trader_proposal(_prop(entry_price=100.0, stop_loss=110.0,
                                        target_price=115.0))
    assert "**Level Warnings**:" in text
    assert "stop-loss is not below the entry" in text


def test_renderer_names_the_ambiguous_size():
    text = render_trader_proposal(_prop(position_size_pct=0.05))
    assert "ambiguous" in text
    assert "treat the size as unstated" in text


def test_renderer_has_no_warning_line_when_clean():
    text = render_trader_proposal(_prop(entry_price=100.0, stop_loss=95.0,
                                        target_price=115.0, position_size_pct=4.0))
    assert "Level Warnings" not in text


def test_every_flag_has_human_text():
    # A flag with no entry would render as its raw identifier in a report.
    from tradingagents.agents import schemas

    produced = set()
    for kwargs in ({"position_size_pct": 0.05}, {"position_size_pct": 250.0},
                   {"entry_price": 100.0, "stop_loss": 110.0},
                   {"entry_price": 100.0, "target_price": 90.0}):
        produced.update(_prop(**kwargs).levels()["flags"])
    produced.update(TraderProposal(action="Sell", reasoning="r", entry_price=100.0,
                                   stop_loss=95.0).levels()["flags"])
    produced.update(TraderProposal(action="Hold", reasoning="r",
                                   entry_price=100.0).levels()["flags"])
    assert produced, "no flags exercised"
    missing = produced - set(schemas._FLAG_TEXT)
    assert not missing, missing
