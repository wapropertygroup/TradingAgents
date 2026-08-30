# Plan: Earnings & Estimate Revision Agent

## Approved Product Decisions

* The first release adds `earnings` as an **opt-in stock analyst**. Existing API/CLI defaults remain unchanged to avoid silently adding cost and provider requirements.
* Coverage is **US/global-equity first** through yfinance, with Alpha Vantage as an explicitly configured fallback/enrichment and honest partial A-share coverage through the existing Tonghuashun consensus snapshot.
* A missing metric is rendered as unavailable, never as zero and never inferred by the LLM. In particular, whisper expectations, true consensus margin revisions, 90-day breadth, and unavailable A-share history remain explicit gaps.
* The existing full-state logging omission for policy/hot-money/lockup reports will be fixed while adding earnings, because the same audit path must preserve all specialist evidence.

## 1. Change Manifest

| # | File | Action | Summary |
|---|---|---|---|
| 1 | `tradingagents/dataflows/earnings_models.py` | Add | Typed normalized evidence, fiscal-period/value/source metadata, surprise/PEAD models, availability flags, safe numeric coercion, revision calculations, momentum score/band, and deterministic serialization/render inputs. |
| 2 | `tradingagents/dataflows/earnings_snapshot_store.py` | Add | SQLite append-only point-in-time snapshot store under `data_cache_dir`; select only observations at/before `trade_date`; schema versioning and concurrency-safe writes. |
| 3 | `tradingagents/dataflows/yfinance_earnings.py` | Add | Normalize calendar, EPS/revenue estimates, EPS trend/revision breadth, earnings history, and adjusted-price post-earnings drift from yfinance; map provider errors to repository error types. |
| 4 | `tradingagents/dataflows/alpha_vantage_earnings.py` | Add | Add configured fallback for earnings calendar/history/estimates and optional latest-published earnings-call transcript; validate JSON/CSV errors and entitlements. |
| 5 | `tradingagents/dataflows/a_stock_earnings.py` | Add | Wrap the existing Tonghuashun current EPS-consensus table as partial normalized evidence; refuse non-A-share symbols and disclose missing history/calendar/revenue fields. |
| 6 | `tradingagents/dataflows/interface.py` | Modify | Register `earnings_data` and optional `earnings_commentary` categories/methods/providers while preserving explicit vendor-chain semantics and typed fallback. |
| 7 | `tradingagents/default_config.py` | Modify | Add explicit default earnings vendor chains (`a_stock,yfinance`; Alpha Vantage remains opt-in/configurable) and the optional commentary chain. |
| 8 | `tradingagents/agents/utils/earnings_data_tools.py` | Add | Expose one core `get_earnings_evidence` tool and one optional `get_earnings_commentary` tool with stable, machine-readable JSON output. |
| 9 | `tradingagents/agents/utils/agent_utils.py` | Modify | Export the new tools through the shared agent utility surface. |
| 10 | `tradingagents/agents/schemas.py` | Modify | Add bounded `EarningsNarrative` structured-output schema for sourced guidance/commentary, catalysts, risks, confidence, and data gaps; numeric evidence and momentum remain code-owned. |
| 11 | `tradingagents/agents/analysts/earnings_analyst.py` | Add | Implement a deterministic tool-call first pass, structured narrative synthesis second pass, and deterministic final report rendering headed by fiscal consensus, revision breadth, and momentum. |
| 12 | `tradingagents/agents/__init__.py` | Modify | Export `create_earnings_analyst`. |
| 13 | `tradingagents/agents/utils/agent_states.py` | Modify | Add `earnings_report` to `AgentState`. |
| 14 | `tradingagents/graph/propagation.py` | Modify | Initialize `earnings_report` to an empty string for every run. |
| 15 | `tradingagents/graph/analyst_execution.py` | Modify | Register stable wire key/node/tool/report metadata for `earnings`. |
| 16 | `tradingagents/graph/conditional_logic.py` | Modify | Add exact `tools_earnings` / `Msg Clear Earnings` routing. |
| 17 | `tradingagents/graph/setup.py` | Modify | Register the Earnings Analyst factory. |
| 18 | `tradingagents/graph/trading_graph.py` | Modify | Register the Earnings ToolNode, persist `earnings_report` plus the previously omitted specialist reports in full-state logs, and preserve existing defaults/checkpoint behavior. |
| 19 | `tradingagents/agents/researchers/bull_researcher.py` | Modify | Inject earnings momentum/evidence as a distinct high-priority source. |
| 20 | `tradingagents/agents/researchers/bear_researcher.py` | Modify | Inject earnings momentum/evidence and require challenges to coverage/confidence. |
| 21 | `tradingagents/agents/managers/research_manager.py` | Modify | Directly receive Earnings report and reconcile estimate direction, breadth, guidance, surprise, and PEAD rather than depending only on debate summaries. |
| 22 | `tradingagents/agents/trader/trader.py` | Modify | Directly include Earnings report alongside the Research Manager plan so the signal is not lost through summarization. |
| 23 | `tradingagents/agents/risk_mgmt/aggressive_debator.py` | Modify | Add Earnings report to risk evidence. |
| 24 | `tradingagents/agents/risk_mgmt/neutral_debator.py` | Modify | Add Earnings report to risk evidence. |
| 25 | `tradingagents/agents/risk_mgmt/conservative_debator.py` | Modify | Add Earnings report to risk evidence. |
| 26 | `tradingagents/agents/managers/portfolio_manager.py` | Modify | Directly include the Earnings report in the final synthesis, with an instruction not to override unavailable/low-confidence data. |
| 27 | `tradingagents/reporting.py` | Modify | Write `1_analysts/earnings.md` and include `Earnings Analyst` in the consolidated report. |
| 28 | `cli/models.py` | Modify | Add `AnalystType.EARNINGS = "earnings"`. |
| 29 | `cli/utils.py` | Modify | Add the stock-only Earnings Analyst choice; filter it for crypto along with fundamentals. |
| 30 | `cli/main.py` | Modify | Update all analyst/status/report/display registries and final report rendering for `earnings_report`. |
| 31 | `README.md` | Modify | Document the opt-in Earnings Analyst, supported fields, provider configuration, PIT limitation, and unavailable-field policy. |
| 32 | `tests/test_earnings_models.py` | Add | Test normalization, fiscal labels, safe changes, scoring/bands, nullable values, serialization, and deterministic report inputs. |
| 33 | `tests/test_earnings_snapshot_store.py` | Add | Test append-only observations, as-of selection, historical no-look-ahead, schema isolation, duplicate idempotency, and concurrent writes. |
| 34 | `tests/test_yfinance_earnings.py` | Add | Fixture-test every yfinance table shape/error, revision horizons/breadth, surprise filtering, and PEAD session math. |
| 35 | `tests/test_alpha_vantage_earnings.py` | Add | Fixture-test JSON/CSV parsing, API notices/errors/entitlements, estimates, calendar/history, transcript quarter selection, and date filtering. |
| 36 | `tests/test_astock_earnings.py` | Add | Test A-share self-selection, current THS partial evidence, unavailable history, and historical-date refusal. |
| 37 | `tests/test_earnings_analyst.py` | Add | Test deterministic tool calls, structured synthesis, fallback labeling, report completeness, and no fabricated unavailable fields. |
| 38 | `tests/test_analyst_execution.py` | Modify | Cover earnings spec strings, selected order, and wall-time tracking. |
| 39 | `tests/test_astock_analyst_pipeline.py` | Modify | Assert `earnings_report` initial state and Earnings tool contract. |
| 40 | `tests/test_crypto_asset_mode.py` | Modify | Assert Earnings is selectable for stocks and removed for crypto. |
| 41 | `tests/test_seven_report_downstream.py` | Modify | Generalize to eight reports and cover Bull/Bear/all risk debaters plus direct RM/Trader/PM propagation. |
| 42 | `tests/test_reporting.py` | Modify | Assert `earnings.md`, consolidated heading, and output parity. |
| 43 | `tests/test_vendor_routing.py` | Modify | Assert explicit earnings chains, typed no-data/rate-limit behavior, no hidden fallback, and optional commentary degradation. |
| 44 | `tests/test_i18n_coverage.py` | Modify | Include Earnings Analyst in report-agent language coverage. |
| 45 | `tests/test_checkpoint_resume.py` | Modify | Assert changing earnings selection changes graph signature and cannot resume an incompatible graph. |
| 46 | `tests/test_full_state_logging.py` | Add | Verify earnings and all specialist reports survive JSON state logging. |

## 2. Sequencing

- [ ] **Step 1: Define normalized evidence and deterministic scoring** — implement typed models, missing-data semantics, fiscal-period labeling, symmetric change calculations for negative/near-zero EPS, coverage/confidence, and momentum bands. *Depends on: nothing*
- [ ] **Step 2: Implement the point-in-time snapshot store** — create versioned SQLite storage and prove as-of selection/no-look-ahead behavior before any live adapter is connected. *Depends on: Step 1*
- [ ] **Step 3: Implement provider adapters** — add yfinance core coverage, Alpha Vantage fallback/commentary, and partial A-share normalization, with fixture-based parsing and typed errors. *Depends on: Steps 1–2*
- [ ] **Step 4: Register data categories and public tools** — wire provider methods/config and expose stable JSON tools without weakening explicit vendor-chain behavior. *Depends on: Step 3*
- [ ] **Step 5: Implement the Earnings Analyst and report contract** — deterministic tool calls, structured narrative synthesis, deterministic numeric rendering, fallback/coverage flags, and the requested FY consensus/breadth/momentum presentation. *Depends on: Step 4*
- [ ] **Step 6: Wire graph and state** — add the stable analyst spec, factory, ToolNode, conditional route, state initialization, checkpoint-safe selection, and full-state logging. *Depends on: Step 5*
- [ ] **Step 7: Wire every downstream decision consumer** — Bull/Bear, Research Manager, Trader, risk debaters, and Portfolio Manager all receive the report explicitly. *Depends on: Step 6*
- [ ] **Step 8: Wire CLI and report outputs** — stock-only selection, status transitions, streaming/final display, markdown report tree, and documentation. *Depends on: Step 6*
- [ ] **Step 9: Run focused and full verification** — provider/unit/graph/CLI/output tests, then the complete test suite and self-review of the diff for accidental default/cost changes. *Depends on: Steps 7–8*
- [ ] **Step 10: Final audit and handoff** — check every todo item, report test results/data limitations, and propose a commit message. *Depends on: Step 9*

## 3. Impact Analysis

* **Affected consumers**:
  * Programmatic users may opt into `TradingAgentsGraph(selected_analysts=[..., "earnings"])`.
  * CLI users gain an Earnings Analyst checkbox for stocks; existing selections/defaults are unchanged.
  * Research, trading, risk, and portfolio prompts gain a new evidence block only when the report is populated; the initial empty state keeps unselected runs compatible.
  * Report trees and full-state JSON gain an optional `earnings_report`/`earnings.md` section.
  * Data caches gain an `earnings_snapshots.sqlite3` file under the configured cache directory.
* **API surface changes**:
  * New public analyst key: `earnings`.
  * New state field: `earnings_report`.
  * New routed methods/tools: `get_earnings_evidence` and `get_earnings_commentary`.
  * New config categories: `earnings_data` and `earnings_commentary`.
  * These are additive. Existing four-analyst defaults, existing wire keys, and existing explicit vendor chains do not change.
* **Provider/cost impact**:
  * yfinance is already a dependency and supplies the default US-equity baseline.
  * Alpha Vantage is called only when configured in the relevant chain/tool override and a key is present; absent credentials produce a local not-configured result without a network attempt. No hidden premium/API-key dependency is introduced.
  * Earnings commentary is optional and degrades to a sourced data-gap message without failing the core report.
* **Performance impact**:
  * Opting in adds one tool round and one quick-thinking structured synthesis call.
  * PEAD calculation reuses/caches adjusted price history and limits event/horizon counts.
  * SQLite snapshot reads/writes are bounded by symbol and observation date.

## 4. Edge Cases

* Historical `trade_date` with no stored point-in-time snapshot: do not call a live estimates endpoint; return explicit insufficient PIT coverage while allowing already-public historical surprise/PEAD evidence only if its source contract is safe.
* `trade_date` today but provider observation timestamp is later because of timezone differences: compare normalized UTC timestamps and market-local dates explicitly.
* Earnings announced before market, after market, during market, on weekends/holidays, or without timing metadata: choose the first actually tradable session conservatively and record the anchoring assumption.
* Negative EPS, EPS crossing zero, exactly zero EPS, tiny denominators, NaN/inf, or mixed currencies: use symmetric bounded change or mark incomparable; never emit misleading ordinary percentages.
* Fiscal year differs from calendar year or provider exposes only `0y/+1y`: resolve with provider fiscal-end metadata; otherwise render the relative period label rather than invent `FY27`.
* Multiple candidate earnings dates or an outdated/past next date: discard dates before the as-of date and preserve uncertainty/range if supplied.
* No analyst coverage, one analyst, unchanged estimates, conflicting short/long horizons, or breadth counts inconsistent with total analysts: reduce confidence and retain the raw discrepancy.
* 90-day EPS trend exists but 90-day breadth does not: show the trend and render breadth unavailable; do not reuse 30-day counts.
* Revenue revisions unavailable from the active provider: compare local PIT snapshots only when both observations exist; otherwise leave unavailable.
* Margin revisions unavailable: actual margin changes and management guidance may be discussed but must not be labeled consensus margin revisions.
* Whisper expectations unavailable: render unavailable; news/social expectations remain qualitative and cannot become a numeric whisper estimate.
* A-share THS page missing, malformed, rate-limited, or changed: typed no-data/rate-limit result; no silent empty table and no US-provider call for a six-digit A-share unless explicitly configured and valid.
* Alpha Vantage missing key, premium entitlement notice, rate limit, malformed CSV/JSON, empty payload, or transcript not yet published: typed fallback/degradation with the correct source named.
* ETF, index, future, FX, crypto, or non-operating instrument: Earnings is filtered in CLI and returns an unsupported-instrument report for programmatic misuse.
* Structured LLM output fails: deterministic numeric evidence and momentum remain; free-text fallback is labeled unvalidated/low-confidence and cannot alter computed values.
* Provider correction/restatement changes old surprise data: retain source/as-of metadata and do not imply the current payload is a pristine historical vintage.
* Snapshot DB is missing, locked, partially initialized, or from a newer schema: initialize/retry safely or fail with an explicit cache error; never delete or downgrade user data automatically.

## 5. What I'm NOT Changing

* Earnings will not replace or merge with Fundamentals or Sentiment; it remains an independent specialist report.
* Existing default analyst selection remains market/social/news/fundamentals. No unrequested API or LLM cost is added to existing runs.
* No paid whisper-data vendor (Estimize or similar), FactSet/LSEG/Visible Alpha integration, or scraped numeric whisper estimate will be introduced.
* No claim of complete A-share revision history, revenue breadth, margin revision breadth, or 90-day analyst breadth will be made without a validated provider contract.
* No model training, backtested alpha claim, portfolio-sizing rewrite, or RSI/technical-agent removal is part of this feature.
* No external database/service is added; snapshot persistence stays local under the configured cache directory.
* Existing vendor routing will not silently contact unconfigured providers.
* No broad refactor to replace all duplicated analyst registries with a global registry is included; only the necessary Earnings additions and audit-log omission fix are in scope.

## 6. Test Plan

| Test | Type | Validates |
|---|---|---|
| Momentum scoring table tests | Unit | Strong Positive/Positive/Neutral/Negative/Strong Negative boundaries, acceleration, breadth weighting, missing-input reweighting, and confidence floor. |
| Negative/zero EPS change tests | Unit | No division-by-zero, inverted sign, infinite percentage, or misleading standard growth calculation. |
| Fiscal period tests | Unit | Correct FY label when metadata exists and honest relative label when it does not. |
| Evidence serialization tests | Unit | Stable JSON schema, explicit nulls/availability, source/as-of/currency/unit retention. |
| Snapshot append/as-of tests | Unit | Append-only history, idempotency, exact/before-date selection, no later observation leakage, schema-version behavior. |
| Snapshot concurrency/corruption tests | Unit | Safe parallel writes, lock handling, and non-destructive failure. |
| yfinance fixture matrix | Unit/Contract | Calendar, estimate/revenue tables, EPS trend/breadth, missing columns, timezone dates, empty/NaN, typed failures. |
| Surprise/PEAD tests | Unit | Date cutoff, BMO/AMC anchoring, holidays, adjusted stock/benchmark alignment, +1/+5/+20/+60 windows, insufficient history. |
| Alpha Vantage fixture matrix | Unit/Contract | Earnings estimates/history/calendar/transcript, bad key, premium/limit notices, CSV/JSON errors, quarter selection, no future transcript. |
| A-share partial evidence tests | Unit | THS current EPS normalization, symbol self-selection, current-only warning, unavailable fields, historical refusal. |
| Explicit vendor chain tests | Unit | No hidden fallback, configured order, typed no-data/rate-limit behavior, optional commentary degradation. |
| Earnings analyst two-pass tests | Unit | Deterministic initial tool call(s), synthesis only after ToolMessages, code-owned numbers/momentum, structured fallback labeling. |
| Report renderer golden tests | Unit | Requested FY today/30d/breadth/momentum format plus sources, gaps, surprises, drift, guidance sections. |
| Graph compile/routing tests | Integration | Earnings-only and mixed-order graphs compile; exact tool/clear routes; final clear connects to next analyst/Bull. |
| State/checkpoint tests | Integration | Initial empty report, selection-dependent signature, incompatible checkpoint isolation. |
| Downstream sentinel tests | Unit | Bull/Bear/RM/Trader/three risk debaters/PM all receive Earnings evidence. |
| CLI stock/crypto tests | Unit | Stock choice visible, crypto filtered, order/status/report maps consistent. |
| Reporting/logging tests | Unit | `earnings.md`, consolidated report, display state, and full-state JSON preserve all eight specialist reports. |
| Full suite | Regression | Existing analysts, providers, structured agents, CLI, reports, and checkpoint behavior remain green. |

## 7. Rollback Notes

* **Safe revert points**:
  * Steps 1–4 are additive data-layer work and can be reverted without affecting graph behavior.
  * Steps 5–8 are additive behind an opt-in analyst key; reverting the whole feature restores the previous graph and CLI defaults.
* **Persistent artifact**: `earnings_snapshots.sqlite3` is a local cache, not authoritative user data. A code rollback may leave it unused. Do not delete it automatically; users can remove it manually if desired.
* **Points of no return**: None. There is no remote migration, external write, or default-on behavior.
* **Partial-failure rule**: If implementation reveals that a provider payload cannot satisfy its planned contract, stop and revise this plan rather than weakening provenance/PIT guarantees or silently fabricating fields.

## 8. Pseudo-code

### Point-in-time evidence retrieval

```python
def get_earnings_evidence(ticker, trade_date):
    instrument = resolve_company_equity(ticker)
    if not instrument.supports_earnings:
        return EarningsEvidence.unsupported(instrument)

    if trade_date < market_today(instrument.exchange):
        snapshot = store.latest_at_or_before(instrument.symbol, trade_date)
        if snapshot is None:
            return EarningsEvidence.pit_unavailable(
                reason="No estimate snapshot observed on or before trade_date"
            )
        return enrich_with_public_history(snapshot, cutoff=trade_date)

    live = route_to_configured_earnings_vendor(instrument, trade_date)
    normalized = normalize_and_validate(live)
    store.append(normalized, observed_at=utc_now(), schema_version=SCHEMA_VERSION)
    return add_revision_metrics_and_pead(normalized, cutoff=trade_date)
```

### Revision change and momentum

```python
def symmetric_change(today, old):
    if today is None or old is None:
        return None
    denominator = abs(today) + abs(old)
    if denominator < EPSILON:
        return 0.0 if today == old else None
    return 2.0 * (today - old) / denominator

signals = {
    "eps_7d": bounded(symmetric_change(eps.today, eps.days_ago_7), scale=0.02),
    "eps_30d": bounded(symmetric_change(eps.today, eps.days_ago_30), scale=0.04),
    "eps_90d": bounded(symmetric_change(eps.today, eps.days_ago_90), scale=0.08),
    "breadth_30d": safe_ratio(up_30d - down_30d, up_30d + down_30d),
    "revenue_30d": bounded(symmetric_change(revenue.today, revenue.days_ago_30), scale=0.03),
}

# Use only available, internally consistent signals and renormalize weights.
# EPS direction remains primary; surprise/PEAD/guidance are context, not allowed
# to overwrite the revision score.
score = weighted_available_mean(signals, weights={
    "eps_7d": 0.15,
    "eps_30d": 0.35,
    "eps_90d": 0.20,
    "breadth_30d": 0.20,
    "revenue_30d": 0.10,
})

if coverage_is_too_thin(signals):
    momentum = "Insufficient Data"
elif score >= 0.60:
    momentum = "Strong Positive"
elif score >= 0.20:
    momentum = "Positive"
elif score > -0.20:
    momentum = "Neutral"
elif score > -0.60:
    momentum = "Negative"
else:
    momentum = "Strong Negative"
```

The exact scale constants and coverage floor will be locked by table-driven tests during Step 1. Any adjustment that materially changes the product interpretation requires a plan amendment before later steps proceed.

### Deterministic tool round and structured synthesis

```python
def earnings_analyst_node(state):
    tool_messages = tool_results_since_last_ai_message(state["messages"])

    if not tool_messages:
        # Do not rely on the LLM to remember or correctly parameterize the
        # high-priority evidence fetch.
        return {
            "messages": [AIMessage(tool_calls=[
                call("get_earnings_evidence", ticker, trade_date),
                call("get_earnings_commentary", ticker, trade_date),
            ])]
        }

    evidence = parse_validated_evidence(tool_messages)
    narrative = invoke_structured_or_freetext(
        EarningsNarrative,
        prompt=bounded_synthesis_prompt(evidence),
        no_external_tools=True,
    )
    report = render_earnings_report(
        numeric_evidence=evidence,          # source of all displayed numbers
        computed_momentum=evidence.momentum, # cannot be changed by narrative
        narrative=narrative,
    )
    return {"messages": [AIMessage(content=report)], "earnings_report": report}
```
