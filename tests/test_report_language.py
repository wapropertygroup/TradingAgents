"""A Chinese run's report is Chinese in the parts code writes, too.

``get_language_instruction()`` only reaches the model. Before
``report_language`` the evidence tables, verdicts, data gaps and decision
headers of a Chinese report were English, and the model quoted the English
verdict into its Chinese prose. Measured on an INTC run on 2026-10-05: the
Earnings, Quality and Valuation sections each opened with an English report.

Three things are pinned here:

- every renderer writes Chinese on a Chinese run, checked by looking for Latin
  words left over rather than for the Chinese it should contain, so a new label
  added in English only fails;
- the Portfolio Manager's Chinese rating reads back as the English step it
  names, because every consumer of the run's signal keys on those;
- English output is the same with the module as without it, which the existing
  renderer tests already assert byte for byte.

No network, no LLM.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tradingagents.agents.rating import RATINGS_5_TIER, extract_rating, parse_rating
from tradingagents.agents.schemas import (
    _FLAG_TEXT,
    _FLAG_TEXT_ZH,
    EarningsNarrative,
    PortfolioDecision,
    PortfolioRating,
    QualityNarrative,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderProposal,
    ValuationNarrative,
    render_earnings_narrative,
    render_pm_decision,
    render_quality_narrative,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
    render_valuation_narrative,
)
from tradingagents.dataflows import earnings_models, quality_models, valuation_models
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.earnings_models import (
    DriftObservation,
    EarningsCalendar,
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    PeriodEvidence,
    RevisionBreadth,
    SurpriseEvent,
    Value,
    finalize_evidence,
    render_evidence_report,
)
from tradingagents.dataflows.quality_models import QualityEvidence, render_quality_report
from tradingagents.dataflows.valuation_models import ValuationEvidence, render_valuation_report
from tradingagents.report_language import CHINESE, ENGLISH, for_language, is_chinese
from tradingagents.risk_engine import COMPLIANCE_TEXT, COMPLIANCE_TEXT_ZH, render_compliance

ZH = "Simplified Chinese (简体中文)"
ROOT = Path(__file__).resolve().parents[1]

# Latin words a Chinese report may still contain: acronyms a Chinese financial
# text writes in Latin letters, vendor and endpoint names, the ticker and
# currency of the fixtures, and the symmetric-change formula's variables.
_ALLOWED_LATIN = {
    "EPS", "PEG", "ETF", "USD",
    "Alpha", "Vantage", "EARNINGS", "EARNINGS_ESTIMATES", "EARNINGS_CALENDAR",
    "yfinance", "Yahoo", "Finance", "jqka",
    "INTC", "US", "SPY", "XYZ",
    "new", "old",
}


def _latin_left(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z][A-Za-z_]+", text)} - _ALLOWED_LATIN


def _assert_chinese(text: str) -> None:
    left = _latin_left(text)
    assert not left, f"English left in a Chinese render: {sorted(left)}\n\n{text}"


# ---------------------------------------------------------------------------
# Which language
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("language", [ZH, "Chinese", "中文", "简体中文", "zh", "zh-CN", "ZH_cn"])
def test_simplified_chinese_settings(language):
    assert is_chinese(language)


@pytest.mark.unit
@pytest.mark.parametrize("language", ["English", "", None, "Japanese",
                                      "Traditional Chinese (繁體中文)", "zh-TW", "zh-Hant"])
def test_everything_else_keeps_english(language):
    assert not is_chinese(language)


@pytest.mark.unit
def test_the_run_config_decides_when_no_language_is_passed():
    assert for_language() is ENGLISH
    with run_config({"output_language": ZH}):
        assert for_language() is CHINESE


# ---------------------------------------------------------------------------
# The rating reads back
# ---------------------------------------------------------------------------


def _decision(rating: PortfolioRating, **over) -> PortfolioDecision:
    fields = dict(rating=rating, executive_summary="分批减仓。", investment_thesis="利润率承压。")
    fields.update(over)
    return PortfolioDecision(**fields)


@pytest.mark.unit
@pytest.mark.parametrize("rating", list(PortfolioRating))
def test_a_chinese_decision_reads_back_as_its_english_step(rating):
    """The coupling that matters: render in Chinese, parse, get the step back."""
    text = render_pm_decision(_decision(rating), language=ZH)
    assert parse_rating(text) == rating.value


@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    ("**评级**：减持", "Underweight"),
    ("**评级**: 增持", "Overweight"),
    ("- **最终评级**：**买入**", "Buy"),
    ("### 投资评级：卖出", "Sell"),
    ("评级：持有（Hold）", "Hold"),
    ("评级：超配", "Overweight"),
    ("评级：减持建议", "Underweight"),
    ("**评级**：Underweight", "Underweight"),
    ("Rating: 减持", "Underweight"),
])
def test_a_chinese_label_or_word_is_read(text, expected):
    assert extract_rating(f"讨论了多空双方。\n\n{text}\n\n分批执行。") == expected


@pytest.mark.unit
def test_the_last_chinese_label_wins():
    assert extract_rating("**评级**：买入\n\n重新权衡之后：\n\n**评级**：减持") == "Underweight"


@pytest.mark.unit
def test_a_rating_quoted_mid_sentence_is_not_the_call():
    assert extract_rating("研究经理给出的评级：增持，但交易员不同意。") is None


@pytest.mark.unit
def test_the_chinese_scale_is_a_legend_not_a_call():
    legend = "评级标准：买入、增持、持有、减持、卖出"
    assert extract_rating(legend) is None
    assert extract_rating(f"{legend}\n\n**评级**：持有") == "Hold"


@pytest.mark.unit
def test_chinese_prose_without_a_label_needs_review():
    """持有 is also "to own" -- a lone one in prose is not a decision."""
    assert extract_rating("我们建议继续持有，等待回调后再买入。") is None
    assert parse_rating("我们建议继续持有。") == "REVIEW"


# ---------------------------------------------------------------------------
# The structured decisions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_portfolio_decision_in_chinese():
    text = render_pm_decision(_decision(PortfolioRating.UNDERWEIGHT, position_size_pct=0.5,
                                        time_horizon="3-6个月"), language=ZH)
    assert text.startswith("**评级**：减持")
    assert "**仓位规模**：组合的 0.5%" in text
    assert "**目标价**：未提供" in text
    assert "**价位警示**：仓位规模有歧义" in text
    _assert_chinese(text)


@pytest.mark.unit
def test_the_trader_proposal_in_chinese():
    proposal = TraderProposal(action="Buy", reasoning="回调后买入。", entry_price=100.0,
                              stop_loss=105.0, target_price=120.0, position_size_pct=5)
    text = render_trader_proposal(proposal, language=ZH)
    assert text.startswith("**操作建议**：买入")
    assert "**入场价**：100.0" in text
    assert "**价位警示**：买入时止损价未低于入场价" in text
    # ystocker lifts this line by its Chinese label.
    assert text.rstrip().endswith("最终交易建议：**买入**")
    _assert_chinese(text)


@pytest.mark.unit
def test_the_reward_risk_ratio_in_chinese():
    proposal = TraderProposal(action="Sell", reasoning="趋势转弱。", entry_price=100.0,
                              stop_loss=105.0, target_price=90.0, position_size_pct=5)
    assert "**收益风险比**：2.0:1" in render_trader_proposal(proposal, language=ZH)


@pytest.mark.unit
def test_every_flag_has_a_chinese_sentence():
    assert set(_FLAG_TEXT_ZH) == set(_FLAG_TEXT)


@pytest.mark.unit
def test_the_research_plan_and_sentiment_in_chinese():
    plan = render_research_plan(ResearchPlan(recommendation="Overweight", rationale="多头胜出。",
                                             strategic_actions="分三批建仓。"), language=ZH)
    assert plan.startswith("**投资建议**：增持")
    _assert_chinese(plan)
    sentiment = render_sentiment_report(SentimentReport(
        overall_band=SentimentBand.MILDLY_BULLISH, overall_score=6.2, confidence="medium",
        narrative="社交平台情绪偏多。"), language=ZH)
    assert sentiment.startswith("**整体情绪：** **温和看涨**（评分：6.2/10）")
    assert "**置信度：** 中" in sentiment
    _assert_chinese(sentiment)


@pytest.mark.unit
def test_the_narratives_in_chinese():
    earnings = render_earnings_narrative(EarningsNarrative(
        guidance_and_commentary="", catalysts=["财报临近。"], confidence="low"), language=ZH)
    assert "## 业绩指引与管理层评论\n暂无" in earnings
    assert "## 风险\n- 暂无" in earnings
    assert earnings.endswith("**叙述置信度：** 低")
    quality = render_quality_narrative(QualityNarrative(
        moat_assessment="规模优势。", confidence="high"), language=ZH)
    valuation = render_valuation_narrative(ValuationNarrative(
        thesis="估值偏贵。", confidence="medium"), language=ZH)
    for text in (earnings, quality, valuation):
        _assert_chinese(text)


@pytest.mark.unit
def test_the_compliance_notice_in_chinese():
    violated = {"status": "violated", "violated": True, "ruling_was_binding": True,
                "reasons": ["size_exceeds_approved"], "final_size_pct": 8.0,
                "approved_size_pct": 5.0}
    text = render_compliance(violated, language=ZH)
    assert "**违反风控闸门裁定**" in text
    assert "最终仓位：组合的 8.0%" in text
    _assert_chinese(text)
    assert set(COMPLIANCE_TEXT_ZH) == set(COMPLIANCE_TEXT)


# ---------------------------------------------------------------------------
# The evidence reports
# ---------------------------------------------------------------------------


def _trend(current, d7=None, d30=None, d90=None, currency="USD", unit="number"):
    def v(x):
        return (Value(x, unit=unit, currency=currency) if x is not None
                else Value.missing("not reported", unit=unit))
    return EstimateTrend(current=v(current), days_ago_7=v(d7), days_ago_30=v(d30),
                         days_ago_60=v(None), days_ago_90=v(d90))


def _count(x):
    return Value(x, unit="count")


def _full_earnings() -> EarningsEvidence:
    """An evidence record that reaches every line of the renderer.

    The breadth is self-contradictory on purpose -- 9 upgrades in 7 days against
    8 in 30, 11 revisions against 5 analysts, the 7- and 30-day trends opposed
    -- so all three retained discrepancies are written.
    """
    period = PeriodEvidence(
        period=FiscalPeriod(key="0y", end_date="2026-12-31"),
        eps=_trend(1.10, d7=1.00, d30=1.20, d90=1.15),
        revenue=_trend(5.0e10, currency="USD", unit="currency_large"),
        breadth=RevisionBreadth(
            up_7d=_count(9), down_7d=_count(1), up_30d=_count(8), down_30d=_count(3),
            up_90d=Value.missing("Alpha Vantage publishes 7- and 30-day revision counts only",
                                 unit="count"),
            down_90d=Value.missing("Alpha Vantage publishes 7- and 30-day revision counts only",
                                   unit="count"),
        ),
        analyst_count=_count(5),
    )
    return finalize_evidence(EarningsEvidence(
        symbol="INTC", as_of="2026-10-05", company_name="英特尔", currency="USD",
        canonical_symbol="INTC.US",
        periods={
            "0y": period,
            "+1y": PeriodEvidence(period=FiscalPeriod(key="+1y"), eps=_trend(1.4)),
            "0q": PeriodEvidence(period=FiscalPeriod(key="0q", end_date="2026-09-30"),
                                 eps=_trend(0.27)),
        },
        calendar=EarningsCalendar(
            next_date="2026-10-22", next_date_range_end="2026-10-27", date_is_estimated=True,
            timing="amc (after market close) — observed pattern, most recently 2026-07-23",
            eps_estimate_avg=Value(0.27, currency="USD"),
            eps_estimate_low=Value(0.20, currency="USD"),
            eps_estimate_high=Value(0.33, currency="USD"),
        ),
        surprises=[
            SurpriseEvent(fiscal_period_end="2026-06-30", announcement_date="2026-07-23",
                          eps_actual=Value(0.30, currency="USD"),
                          eps_estimate=Value(0.10, currency="USD"),
                          eps_difference=Value(0.20, currency="USD"),
                          surprise_pct=Value(2.0, unit="pct_dec")),
            SurpriseEvent(fiscal_period_end="2026-03-31"),
        ],
        drift=[DriftObservation(
            fiscal_period_end="2026-06-30", announcement_date="2026-07-23",
            anchor_session="2026-07-24", sessions=5,
            stock_return=Value(-0.023, unit="pct_dec"),
            benchmark_return=Value(0.011, unit="pct_dec"),
            excess_return=Value(-0.034, unit="pct_dec"),
        )],
        sources=["Alpha Vantage EARNINGS", "Alpha Vantage EARNINGS_ESTIMATES"],
        data_gaps=[
            "Whisper expectations are unavailable: Alpha Vantage publishes sell-side "
            "consensus only, and no free source publishes buy-side whisper numbers.",
            "Consensus margin revisions are unavailable. Reported margins and management "
            "guidance may be discussed but are not consensus margin revisions.",
        ],
        warnings=["as-of mismatch: requested 2026-10-05, evidence carries 2026-10-04"],
    ))


@pytest.mark.unit
def test_every_line_of_the_earnings_report_is_chinese():
    evidence = _full_earnings()
    assert len(evidence.momentum.discrepancies) == 3
    text = render_evidence_report(evidence, language=ZH)
    assert text.startswith("# 盈利与预期修正 — 英特尔（INTC）")
    assert "## 2026 财年（财年截止 2026-12-31） EPS 一致预期" in text
    assert "| **今日** | **1.10美元** |" in text
    assert "- 发布时间：盘后（收盘后发布）——依据历史规律，最近一次为 2026-07-23" in text
    assert "- 营收一致预期：不可用" in text
    assert "下一财年（相对期间 +1y）" in text
    assert "- 近 90 天：不可用（Alpha Vantage 只发布 7 天和 30 天的修正家数）" in text
    _assert_chinese(text)


@pytest.mark.unit
def test_the_intc_report_of_2026_10_05_in_chinese():
    """The report that was reported: no estimates mapped, so nothing scored."""
    evidence = finalize_evidence(EarningsEvidence(
        symbol="INTC", as_of="2026-10-05", company_name="英特尔",
        calendar=EarningsCalendar(next_date="2026-10-22", date_is_estimated=True,
                                  eps_estimate_avg=Value(0.27, currency="USD")),
        sources=["Alpha Vantage EARNINGS", "Alpha Vantage EARNINGS_ESTIMATES"],
        data_gaps=[
            "Alpha Vantage EARNINGS_ESTIMATES returned only horizons this build does not "
            "recognise (fiscal quarter, fiscal year), so no period was mapped. Unrecognised "
            "horizons are skipped rather than guessed.",
        ],
    ))
    text = render_evidence_report(evidence, language=ZH)
    assert "**数据不足**" in text
    assert "- 评分期间：本财年（相对期间 0y）" in text
    assert "动量未评分" in text
    assert "（财季、财年）" in text
    assert "## 业绩惊喜历史\n\n暂无" in text
    _assert_chinese(text)


@pytest.mark.unit
@pytest.mark.parametrize("evidence", [
    EarningsEvidence.unsupported(
        "SPY", "2026-10-05",
        "SPY is a etf, not an operating company, so it has no analyst EPS consensus, "
        "revision history or earnings calendar. Holdings-level earnings would need a "
        "look-through, which this analyst does not perform."),
    EarningsEvidence.pit_unavailable(
        "INTC", "2020-01-02",
        "No estimate snapshot was observed on or before 2020-01-02. Yahoo's revision "
        "lookbacks are relative to the present, so today's figures cannot describe that "
        "date. Snapshots accumulate one per day from the first run onward; a historical "
        "date before the first run has no vintage and this run reports none."),
    EarningsEvidence.no_coverage(
        "XYZ", "2026-10-05",
        "Yahoo recognises XYZ but publishes no analyst EPS estimates, revision history, "
        "or reported-quarter history for it. This is an absence of sell-side coverage, "
        "not a zero: no revision direction can be inferred."),
], ids=["unsupported", "pit_unavailable", "no_coverage"])
def test_the_earnings_refusals_in_chinese(evidence):
    _assert_chinese(render_evidence_report(evidence, language=ZH))


def _quality(**over) -> QualityEvidence:
    fields = dict(
        symbol="INTC", as_of="2026-10-05", company_name="英特尔", currency="USD",
        return_on_equity=Value(-0.1071, unit="pct_dec"),
        operating_margin=Value(0.1219, unit="pct_dec"),
        profit_margin=Value(-0.1979, unit="pct_dec"),
        return_on_assets=Value(0.0141, unit="pct_dec"),
        debt_to_equity=Value(0.49, unit="ratio"),
        current_ratio=Value(1.604, unit="ratio"),
        free_cash_flow=Value(4.87e9, unit="currency_large", currency="USD"),
        total_revenue=Value(5.71e10, unit="currency_large", currency="USD"),
        margin_history=[Value(-0.0004, unit="pct_dec"), Value(-0.0887, unit="pct_dec")],
        margin_history_periods=["2025-12-31", "2024-12-31"],
        sources=["yfinance (Yahoo Finance fundamentals)"],
    )
    fields.update(over)
    return quality_models.finalize_evidence(QualityEvidence(**fields))


@pytest.mark.unit
def test_the_quality_report_in_chinese():
    text = render_quality_report(_quality(), language=ZH)
    assert text.startswith("# 经营质量 — 英特尔（INTC）")
    assert "- 自由现金流：48.70亿美元" in text
    assert "营业利润率历史不足 3 个期间" in text  # two periods: a named gap
    _assert_chinese(text)


@pytest.mark.unit
def test_the_quality_refusals_in_chinese():
    for evidence in (
        QualityEvidence.unsupported(
            "SPY", "2026-10-05",
            "SPY is a etf, not an operating company, so ROE, margins, leverage and "
            "per-share multiples do not describe it the way they describe a company. A "
            "fund's own composition would need a look-through, which this analyst does "
            "not perform."),
        QualityEvidence.no_coverage(
            "XYZ", "2026-10-05",
            "Yahoo recognises XYZ but publishes none of return on equity, operating "
            "margin, debt-to-equity or current ratio for it."),
        quality_models.finalize_evidence(QualityEvidence(symbol="XYZ", as_of="2026-10-05")),
    ):
        _assert_chinese(render_quality_report(evidence, language=ZH))


@pytest.mark.unit
def test_the_valuation_report_in_chinese():
    """INTC's: no trailing P/E, so two named gaps and a partial tier."""
    evidence = valuation_models.finalize_evidence(ValuationEvidence(
        symbol="INTC", as_of="2026-10-05", company_name="英特尔", currency="USD",
        forward_pe=Value(56.33, unit="ratio"), peg_ratio=Value(1.36, unit="ratio"),
        price_to_book=Value(6.693, unit="ratio"), dividend_yield=Value(0.0, unit="pct_dec"),
        market_cap=Value(6.1419e11, unit="currency_large", currency="USD"),
        sources=["yfinance (Yahoo Finance fundamentals)"],
    ))
    text = render_valuation_report(evidence, language=ZH)
    assert "- 历史市盈率：不可用" in text
    assert "- 市值：6,141.90亿美元" in text
    assert "历史市盈率不可用或无定义（常见于亏损公司）" in text
    _assert_chinese(text)


@pytest.mark.unit
def test_the_valuation_refusals_in_chinese():
    for evidence in (
        ValuationEvidence.no_coverage(
            "XYZ", "2026-10-05",
            "Yahoo recognises XYZ but publishes none of trailing P/E, forward P/E, PEG or "
            "price-to-book for it."),
        valuation_models.finalize_evidence(ValuationEvidence(symbol="XYZ", as_of="2026-10-05")),
    ):
        _assert_chinese(render_valuation_report(evidence, language=ZH))


@pytest.mark.unit
def test_a_snapshot_substitution_warning_in_chinese(tmp_path):
    """Produced by the real backfill, so a reworded warning fails here."""
    from tradingagents.dataflows.earnings_snapshot_store import (
        EarningsSnapshotStore,
        backfill_trend_from_snapshots,
    )

    def evidence(as_of, eps):
        return EarningsEvidence(
            symbol="600519", as_of=as_of,
            periods={"0y": PeriodEvidence(period=FiscalPeriod(key="0y", end_date="2026-12-31"),
                                          eps=_trend(eps, currency="CNY"))},
        )

    store = EarningsSnapshotStore(tmp_path / "snapshots.jsonl")
    assert store.append(evidence("2026-09-01", 60.0), observed_date="2026-09-01")
    filled = backfill_trend_from_snapshots(evidence("2026-10-05", 62.0), "600519", store=store)
    assert filled.warnings, "the backfill should have substituted a 30-day lookback"
    text = CHINESE.message(filled.warnings[0])
    assert text.startswith("部分修正期限是用本系统自己带日期的快照重建的")
    assert "本财年（相对期间 0y）EPS的“30 天”回看值取自 2026-09-01 观测的本地快照" in text
    _assert_chinese(text)


# ---------------------------------------------------------------------------
# The catalogue keeps up with the adapters
# ---------------------------------------------------------------------------

_ADAPTERS = [
    "tradingagents/dataflows/alpha_vantage_earnings.py",
    "tradingagents/dataflows/yfinance_earnings.py",
    "tradingagents/dataflows/a_stock_earnings.py",
    "tradingagents/dataflows/fundamentals_evidence.py",
    "tradingagents/dataflows/earnings_models.py",
    "tradingagents/dataflows/quality_models.py",
    "tradingagents/dataflows/valuation_models.py",
]
_REASON_KEYWORDS = {"unavailable_reason", "drift_unavailable_reason"}
_REFUSALS = {"no_coverage", "unsupported", "pit_unavailable"}


def _reader_facing_constants(path: Path) -> list[tuple[int, str]]:
    """Plain-string sentences an adapter puts where a report will print them.

    Data-gap and warning lists, the reasons a value or a section is absent, and
    the detail of a refusal. f-strings are left to the samples above: their
    patterns need real values to match.
    """
    found: list[tuple[int, str]] = []

    def add(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
            found.append((node.lineno, node.value))

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            owner = ast.unparse(fn.value).lower() if isinstance(fn, ast.Attribute) else ""
            if name == "append" and ("gap" in owner or "warn" in owner):
                for arg in node.args:
                    add(arg)
            if name == "missing" and node.args:
                add(node.args[0])
            if name in _REFUSALS and len(node.args) >= 3:
                add(node.args[2])
            for kw in node.keywords:
                if kw.arg in _REASON_KEYWORDS:
                    add(kw.value)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("gap" in n.lower() for n in names):
                values = (node.value.values if isinstance(node.value, ast.Dict)
                          else node.value.elts if isinstance(node.value, ast.List) else [])
                for value in values:
                    add(value)
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            # ``return {}, "the reason no period was mapped"``
            elts = node.value.elts
            if len(elts) == 2 and isinstance(elts[0], ast.Dict) and not elts[0].keys:
                add(elts[1])
    return found


@pytest.mark.unit
@pytest.mark.parametrize("rel", _ADAPTERS)
def test_every_sentence_an_adapter_prints_has_a_chinese_form(rel):
    untranslated = [(line, text) for line, text in _reader_facing_constants(ROOT / rel)
                    if CHINESE.message(text) == text]
    assert not untranslated, (
        f"{rel}: add these to report_language._MESSAGES_ZH (or a pattern): {untranslated}")


@pytest.mark.unit
def test_the_named_gaps_of_all_three_tiers_have_chinese_forms():
    for gaps in (earnings_models._MISSING_SIGNAL_GAPS, quality_models._MISSING_SIGNAL_GAPS,
                 valuation_models._MISSING_SIGNAL_GAPS):
        for key, text in gaps.items():
            assert CHINESE.message(text) != text, key


@pytest.mark.unit
@pytest.mark.parametrize("sample", [
    "Alpha Vantage EARNINGS_ESTIMATES was declined (Thank you for using Alpha Vantage!). "
    "It is premium-gated, and the free tier reports that as a quota notice, so no "
    "consensus revision history is available from this vendor on this key. This is an "
    "entitlement limit, not an absence of analyst coverage.",
    "Alpha Vantage EARNINGS_ESTIMATES was unavailable (timeout).",
    "Alpha Vantage's 3-month earnings calendar lists no upcoming date for INTC at or "
    "after 2026-10-05",
    "Alpha Vantage EARNINGS_CALENDAR declined (quota)",
    "Alpha Vantage EARNINGS_CALENDAR was declined (quota).",
    "Alpha Vantage EARNINGS_CALENDAR unavailable (timeout)",
    "adjusted price history unavailable (timeout)",
    "The point-in-time snapshot store could not be read (disk full). Historical estimate "
    "evidence is unavailable; today's consensus is not a substitute.",
    "同花顺 publishes no consensus-forecast table for 600519, and no other free A-share "
    "source in this project publishes analyst estimates. No EPS consensus, revision "
    "direction, or earnings calendar is available for this symbol.",
    "同花顺's forecast table did not carry the expected 年度 / 均值 columns (saw: 年份, "
    "均价). Its consensus EPS is therefore NOT normalized into a measured figure — a "
    "mis-mapped column would publish a realised result as a forward consensus. Read the "
    "verbatim table under Guidance and do not restate its numbers as 'consensus EPS' "
    "elsewhere.",
    "bmo (before market open) — observed pattern, most recently 2026-07-23",
])
def test_the_sentences_with_a_variable_part(sample):
    translated = CHINESE.message(sample)
    assert translated != sample
    assert not _latin_left(translated) - {"Thank", "you", "for", "using", "timeout",
                                          "quota", "disk", "full"}


@pytest.mark.unit
def test_an_unknown_sentence_stays_english_rather_than_guessed():
    assert CHINESE.message("Something no adapter writes.") == "Something no adapter writes."


@pytest.mark.unit
def test_large_figures_count_in_yi_and_wanyi():
    def large(n, currency="USD"):
        return CHINESE.fmt(Value(n, unit="currency_large", currency=currency))
    assert large(4.87e9) == "48.70亿美元"
    assert large(6.1419e11) == "6,141.90亿美元"
    assert large(1.5e12) == "1.50万亿美元"
    assert large(3.2e9, "CNY") == "32.00亿元"
    assert large(3.2e9, "XYZ") == "32.00亿 XYZ"
    assert CHINESE.fmt(Value(0.27, currency="USD")) == "0.27美元"
    assert CHINESE.fmt(Value.missing("not reported")) == "不可用"
    assert CHINESE.fmt(Value(0.0853, unit="pct_dec")) == "+8.53%"


# ---------------------------------------------------------------------------
# The report around them
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_report_title_follows_the_language_but_the_headings_do_not(tmp_path):
    from tradingagents.reporting import write_report_tree

    state = {"market_report": "市场", "risk_debate_state": {"judge_decision": "**评级**：持有"}}
    with run_config({"output_language": ZH}):
        complete = write_report_tree(state, "INTC", tmp_path).read_text(encoding="utf-8")
    assert complete.startswith("# 交易分析报告：INTC\n\n生成时间：")
    # Consumers split the report on these, so they stay as they are.
    assert "### Market Analyst" in complete
    assert "## V. Portfolio Manager Decision\n\n### Portfolio Manager" in complete


@pytest.mark.unit
def test_the_analyst_is_told_the_band_in_the_report_language():
    from tradingagents.agents.analysts import earnings_analyst

    evidence = finalize_evidence(EarningsEvidence(symbol="INTC", as_of="2026-10-05"))
    with run_config({"output_language": ZH}):
        system = earnings_analyst._synthesis_prompt(
            ticker="INTC", trade_date="2026-10-05", instrument_context="",
            numeric_report="", commentary=None, evidence=evidence,
        )[0].content
    assert "The momentum band is `数据不足` and is final." in system
    assert "Insufficient Data" not in system


@pytest.mark.unit
@pytest.mark.parametrize("step", RATINGS_5_TIER)
def test_each_step_has_one_chinese_word_that_reads_back(step):
    assert parse_rating(f"**评级**：{CHINESE.term(step)}") == step
