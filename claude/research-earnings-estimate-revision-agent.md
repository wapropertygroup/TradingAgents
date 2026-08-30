# Research: Earnings & Estimate Revision Agent

## 1. Context & Scope

* **Target repository**: `/Users/yuanxili/workspace/TradingAgents` on `main` (`3fae1c6`). The sibling `TradingAgents-astock` is not the target; the target repository already contains the recent seven-analyst/A-share integration.
* **Requested outcome**: Add an independent Earnings Analyst whose primary signal is the direction, magnitude, and breadth of analyst estimate revisions, with supporting earnings-calendar, guidance, surprise-history, and post-earnings-drift evidence.
* **Primary files investigated in detail**:
  * Graph/state: `tradingagents/graph/analyst_execution.py`, `setup.py`, `conditional_logic.py`, `propagation.py`, `trading_graph.py`, `tradingagents/agents/utils/agent_states.py`.
  * Existing analyst patterns: `tradingagents/agents/analysts/fundamentals_analyst.py`, `sentiment_analyst.py`, `tradingagents/agents/schemas.py`, `tradingagents/agents/utils/structured.py`, `tradingagents/agents/__init__.py`.
  * Data routing/providers: `tradingagents/dataflows/interface.py`, `config.py`, `errors.py`, `y_finance.py`, `a_stock.py`, `alpha_vantage.py`, `alpha_vantage_common.py`, `alpha_vantage_fundamentals.py`, `stockstats_utils.py`, `symbol_utils.py`, `tradingagents/agents/utils/fundamental_data_tools.py`, `signal_data_tools.py`, `agent_utils.py`, `tradingagents/default_config.py`, `pyproject.toml`.
  * Consumers/output: bull/bear researchers, all three risk debaters, Research Manager, Trader, Portfolio Manager, `tradingagents/reporting.py`, `cli/models.py`, `cli/utils.py`, and the analyst/report/status paths in `cli/main.py`.
  * Tests: analyst execution, A-share analyst pipeline/data, downstream report propagation, reporting, vendor routing/errors, structured agents/prompts, crypto asset mode, checkpoint resume, and CLI tests under `tests/`.
* **External capability checks (primary documentation)**:
  * [yfinance Ticker API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html) exposes `calendar`, `earnings_dates`, `earnings_estimate`, `revenue_estimate`, `earnings_history`, `eps_trend`, and `eps_revisions`. Its documented EPS trend includes current/7/30/60/90-day values; revision breadth includes up/down counts for 7 and 30 days.
  * [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/) exposes `EARNINGS_CALENDAR`, `EARNINGS`, `EARNINGS_ESTIMATES` (annual/quarterly EPS and revenue estimates, analyst counts, and revision history), and `EARNINGS_CALL_TRANSCRIPT`.
* **Current as-is flow**:
  1. The CLI or API selects analyst wire keys.
  2. `ANALYST_NODE_SPECS` maps each key to agent, clear, tool, and report nodes.
  3. `GraphSetup` requires a factory, ToolNode, and `should_continue_<key>` router for every selected analyst.
  4. An analyst loops through tool calls, then stores a non-empty report; `create_msg_delete` removes the temporary message/tool history while report state survives.
  5. Researchers and risk debaters receive report fields explicitly. Reporting and CLI display/save paths also enumerate report fields explicitly.
  6. Existing fundamentals provide a live forward-EPS snapshot. A-share fundamentals can scrape a current Tonghuashun EPS-consensus table. Neither path currently provides a trustworthy 7/30/90-day point-in-time revision series.

## 2. Intricacies & Findings

### Data feasibility

| Requested signal | US/global equity MVP | A-share current support | Required treatment |
| --- | --- | --- | --- |
| Next earnings date | yfinance calendar/dates; Alpha Vantage calendar fallback | Not present | Nullable with source/as-of metadata |
| Consensus EPS | yfinance 0q/+1q/0y/+1y; Alpha Vantage fallback | Current THS table only | Map relative periods to actual fiscal year; never hard-code calendar FY |
| Consensus revenue | yfinance and Alpha Vantage | Not present | Nullable by period/currency/unit |
| EPS revision 7/30/90d | yfinance `eps_trend` | Not present | Compute change from provider values; preserve raw values |
| EPS revision breadth | yfinance up/down for 7/30d | Not present | 90d breadth must remain unavailable unless another provider proves it |
| Revenue revisions | Alpha Vantage may cover revision history; yfinance only documents current revenue estimates | Not present | Validate a real payload contract before claiming history; otherwise accumulate local snapshots |
| Margin revisions | No trustworthy current provider contract | Not present | Do not derive/label actual margins as consensus revisions; nullable qualitative guidance only |
| Guidance/commentary | Transcript/news/filing enrichment | News/announcements require a new adapter | Preserve provenance; LLM extraction is qualitative unless provider supplies numeric guidance |
| Surprise history | yfinance earnings history/dates; Alpha Vantage earnings fallback | Not present | Filter strictly by analysis date |
| Post-earnings drift | Deterministic calculation from historical event dates + adjusted OHLCV | Requires event dates first | Calculate +1/+5/+20/+60 trading-day raw and benchmark-adjusted returns |
| Whisper expectations | No reliable installed/free source | None | Explicit `unavailable`; never infer a number from chatter or headlines |

### Hidden dependencies

* A new stable wire key should be `earnings`; the state/report key should be `earnings_report`. Node strings must match exactly across the execution spec, graph setup, conditional router, ToolNode, CLI status registries, and tests.
* `GraphSetup` dynamically calls `ConditionalLogic.should_continue_<key>`. Omitting one registry produces setup-time `KeyError`/`AttributeError`, not graceful degradation.
* The tools bound to the Earnings Analyst must exactly match those executable by `tools_earnings`; otherwise a valid LLM tool call fails at runtime as an unknown tool.
* `create_msg_delete` means only the report field crosses from the analyst stage into research; raw tool messages are intentionally discarded. Important source/as-of facts therefore must be rendered into `earnings_report` itself.
* Analyst order affects actual graph order, CLI status transitions, and wall-time tracking. The CLI has duplicated registries in `cli/utils.py` and `cli/main.py`; both must agree with `ANALYST_NODE_SPECS`.
* `TradingAgentsGraph._run_signature` already includes selected analyst keys, so adding/removing Earnings automatically isolates checkpoint state. The new report field still must be present in the initial state for stable graph shape.
* `TradingAgentsGraph._log_state` currently enumerates only the original four analyst reports and already omits three newer A-share reports. Earnings auditability cannot rely on final state alone; the logger must be updated intentionally.
* Final disk reporting, streaming interim reporting, terminal display, and full-state JSON are separate output paths.
* Downstream agents enumerate source reports rather than consuming a generic collection. Bull, Bear, three risk debaters, Research Manager, Trader, and Portfolio Manager need an explicit policy for the new high-priority signal or it can be lost through repeated summaries.

### State and side effects

* Estimate snapshots are not ordinary TTL data. To support historical analysis without look-ahead, snapshots need append-only point-in-time semantics keyed by canonical symbol, source, observation time, fiscal period, currency/unit, and schema version.
* Live snapshots may have a 6–12 hour refresh window; historical earnings events and transcripts can be cached long term. Atomic write practices can follow the existing temp-file plus `os.replace` pattern.
* Post-earnings drift is derived data. Event timing must distinguish before-market/after-market releases when choosing the first tradable session, align stock and benchmark sessions, use split/dividend-adjusted prices, and report insufficient windows rather than shortening horizons silently.
* Current provider routing has typed `NoMarketDataError`, `VendorRateLimitError`, and `VendorNotConfiguredError`. New adapters must raise these types rather than return strings such as `Error retrieving ...`, because returned error prose is treated as a successful vendor result and blocks fallback.

### Invariants

* Never use a live estimate snapshot to answer a historical `trade_date`. A provider field named `30daysAgo` is relative to the observation timestamp, not automatically relative to an arbitrary historical backtest date.
* Never fill missing values with zero. Zero EPS, zero revisions, no analyst coverage, and unavailable data are different states.
* Every numeric estimate/revision must retain fiscal period, currency/unit, source, and observed-at/as-of metadata.
* Fiscal periods such as `0y`/`+1y` must be resolved against the company's fiscal calendar before rendering labels such as FY27.
* Revision momentum must be deterministic from validated numeric inputs. The LLM may explain the signal but should not invent or override the computed direction/breadth classification.
* Whisper, margin-revision, and guidance fields must allow `unavailable` without failing the entire report.
* Earnings analysis applies to operating-company equities. Crypto must be filtered; ETF/index/future/FX eligibility should use resolved instrument type rather than symbol guessing where possible.
* Existing explicit vendor-chain semantics must remain: the application must not silently call a provider the user did not configure.

## 3. The "Invisible" Assumptions

* The plan will treat the requested first release as an honest, high-quality MVP rather than claim identical coverage across all asset classes and markets.
* `yfinance>=1.4.1` remains the minimum supported version and its documented earnings-analysis properties are available at runtime. Tests must mock provider payloads; implementation cannot depend on live network calls.
* Alpha Vantage's `EARNINGS_ESTIMATES` exact revision payload will be captured and fixture-tested with a real entitled response before revenue-revision breadth/history is advertised.
* A valid Alpha Vantage key may not exist. Therefore yfinance must support the US-equity baseline and missing Alpha Vantage configuration must follow typed fallback semantics.
* The existing A-share THS scraper is a current snapshot, not a point-in-time history provider. A-share reports will expose partial coverage until a historical estimate source or sufficient local snapshots exist.
* “Management commentary” means sourced transcript/filing/announcement content, not an LLM-generated paraphrase without provenance.
* “Post-earnings drift” will be a descriptive historical feature, not a guarantee or a causal alpha estimate.

## 4. Potential Friction Points

* **Point-in-time correctness is the highest risk.** Most free APIs serve today's consensus. Historical TradingAgents runs make accidental look-ahead especially easy.
* **Coverage mismatch.** yfinance provides a strong EPS trend surface but only 7/30-day breadth and no documented revenue trend. Alpha Vantage may improve it, but entitlements, rate limits, and payload contracts need tests.
* **Provider fragility.** Tonghuashun is HTML table scraping; a page-shape change yields silent loss of A-share consensus. It cannot be the sole source of a claimed high-confidence signal.
* **Structured-output tradeoff.** Existing analysts usually use a tool loop and free-text final answer, while strong deterministic rendering uses structured output with no external tools. The clean design is to fetch/normalize evidence deterministically, compute momentum in code, then ask the LLM only for bounded synthesis into a typed report. Trying to mix an unconstrained tool loop with a schema gate would make validation and fallback ambiguous.
* **Context size.** Full earnings-call transcripts are too large for an analyst prompt. Transcript retrieval must select the correct already-published quarter, cap content, and retain provenance.
* **Duplicated registries.** CLI, graph, state, output writer, logger, and downstream prompts all enumerate analysts/report fields separately. Missing one can produce a run that completes but silently hides or drops the Earnings report.
* **Existing logging debt.** The full-state logger already omits policy/hot-money/lockup reports. A focused fix may include those omissions when adding earnings, but that scope must be explicit in planning.
* **No installed local dependencies in the audit shell.** `yfinance` is declared but not installed in the current interpreter, so research used repository contracts and official docs. Implementation verification will need the project environment/dependencies.

## 5. Proposed Next Steps for Planning

1. Define a normalized earnings evidence model and deterministic momentum calculation. Recommended output includes fiscal-period consensus today/7d/30d/90d, EPS breadth 7/30d, revenue evidence when valid, guidance/commentary with provenance, recent surprises, PEAD windows, coverage flags, and a computed momentum band.
2. Add a dedicated `earnings_data` category and one aggregated evidence tool to minimize provider calls. Implement yfinance first, Alpha Vantage as a configured fallback/enrichment, and an explicitly partial A-share adapter based on the existing THS snapshot.
3. Add append-only point-in-time snapshots. Historical runs may use only observations at or before `trade_date`; when absent, render a data-gap report rather than querying live estimates.
4. Implement the Earnings Analyst as fetch/normalize/compute plus structured synthesis and deterministic rendering, keeping missing fields nullable and marking any free-text fallback as unvalidated/low confidence.
5. Wire `earnings`/`earnings_report` through graph, state, CLI, reporting, full-state logging, and all downstream decision stages. Keep the existing default four analysts unchanged initially to avoid an unannounced extra LLM/API cost; make Earnings selectable and documented as recommended for stocks.
6. Add focused unit/contract tests for provider shapes/errors, PIT selection, fiscal-year mapping, momentum math, PEAD session anchoring, graph wiring, CLI filtering/status, downstream propagation, output parity, and checkpoint signature behavior.
7. Do not claim full coverage for 90-day breadth, revenue breadth, margin revisions, or whisper expectations until a provider contract and fixtures prove those fields.

### Open decisions for Planning

* Should Earnings become a default analyst immediately, or remain opt-in for backward compatibility and API-cost control? Research recommendation: opt-in in the first release.
* Is US-equity-first with explicit partial A-share coverage acceptable, or is full A-share revision history a release blocker?
* May the implementation add Alpha Vantage earnings endpoints that require an API key/possibly premium access, or must the first release use yfinance plus locally accumulated snapshots only?
* Is fixing the existing omission of policy/hot-money/lockup fields from `full_states_log` in scope while adding `earnings_report`?
