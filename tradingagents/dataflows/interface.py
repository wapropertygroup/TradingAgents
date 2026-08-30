import logging

from .alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_global_news as get_alpha_vantage_global_news,
    get_income_statement as get_alpha_vantage_income_statement,
    get_indicator as get_alpha_vantage_indicator,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_stock as get_alpha_vantage_stock,
)
from .config import get_config
from .errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from .a_stock import (
    get_balance_sheet as get_astock_balance_sheet,
    get_cashflow as get_astock_cashflow,
    get_fundamentals as get_astock_fundamentals,
    get_global_news as get_astock_global_news,
    get_income_statement as get_astock_income_statement,
    get_indicators as get_astock_indicators,
    get_insider_transactions as get_astock_insider_transactions,
    get_news as get_astock_news,
    get_stock_data as get_astock_stock_data,
    get_concept_blocks as get_astock_concept_blocks,
    get_dragon_tiger_board as get_astock_dragon_tiger_board,
    get_fund_flow as get_astock_fund_flow,
    get_hot_stocks as get_astock_hot_stocks,
    get_industry_comparison as get_astock_industry_comparison,
    get_lockup_expiry as get_astock_lockup_expiry,
    get_northbound_flow as get_astock_northbound_flow,
    get_profit_forecast as get_astock_profit_forecast,
)
from .a_stock_earnings import get_earnings_evidence as get_astock_earnings_evidence
from .alpha_vantage_earnings import (
    get_earnings_commentary as get_alpha_vantage_earnings_commentary,
    get_earnings_evidence as get_alpha_vantage_earnings_evidence,
)
from .fred import get_macro_data as get_fred_macro_data
from .polymarket import get_prediction_markets as get_polymarket_prediction_markets
from .y_finance import (
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_fundamentals as get_yfinance_fundamentals,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
    get_stock_stats_indicators_window,
    get_YFin_data_online,
)
from .fundamentals_evidence import (
    get_quality_evidence as get_yfinance_quality_evidence,
    get_valuation_evidence as get_yfinance_valuation_evidence,
)
from .yfinance_earnings import get_earnings_evidence as get_yfinance_earnings_evidence
from .yfinance_news import get_global_news_yfinance, get_news_yfinance

logger = logging.getLogger(__name__)

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    },
    "macro_data": {
        "description": "Macroeconomic indicators (rates, inflation, labor, growth)",
        "tools": [
            "get_macro_indicators",
        ]
    },
    "prediction_markets": {
        "description": "Market-implied probabilities for forward-looking events",
        "tools": [
            "get_prediction_markets",
        ]
    },
    "signal_data": {
        "description": "A-share forecasts, capital flow, Dragon-Tiger, and lock-up data",
        "tools": [
            "get_profit_forecast", "get_hot_stocks", "get_northbound_flow",
            "get_concept_blocks", "get_fund_flow", "get_dragon_tiger_board",
            "get_lockup_expiry", "get_industry_comparison",
        ],
    },
    "earnings_data": {
        "description": (
            "Analyst EPS/revenue consensus, revision trend and breadth, earnings "
            "calendar, surprise history, and post-earnings drift"
        ),
        "tools": [
            "get_earnings_evidence",
        ],
    },
    "earnings_commentary": {
        "description": "Earnings-call transcript / management commentary",
        "tools": [
            "get_earnings_commentary",
        ],
    },
    "quality_data": {
        "description": (
            "Business-quality fundamentals: ROE, margins, leverage, cash "
            "generation, multi-period margin consistency, and a computed "
            "quality tier"
        ),
        "tools": [
            "get_quality_evidence",
        ],
    },
    "valuation_data": {
        "description": (
            "Valuation multiples: trailing/forward P/E, PEG, price-to-book, "
            "dividend yield, and a computed valuation tier"
        ),
        "tools": [
            "get_valuation_evidence",
        ],
    },
}

VENDOR_LIST = [
    "yfinance",
    "fred",
    "polymarket",
    "alpha_vantage",
    # A-share (沪深京) over 东财/新浪/同花顺. Listed last, and last in each method
    # dict below, so it is only reached once the other vendors have declined —
    # it refuses anything that is not a 6-digit A-share code, so a US ticker
    # never touches it.
    "a_stock",
]

# Optional enrichment categories. These add macro/event context to the news
# analyst but are not core to a decision, so a vendor failure here degrades to a
# sentinel instead of aborting the run (a bad LLM-supplied indicator, a missing
# key, or a network blip should not crash an analysis over flavour data). Core
# categories (prices, fundamentals, news) still raise so a broken primary is loud.
#
# ``signal_data`` belongs here for a sharper reason: it has a single vendor, so
# one 东财 throttle exhausts its whole chain, where the core categories still have
# yfinance behind a_stock. Its eight tools (龙虎榜/资金流/解禁/…) are A-share colour
# for three of seven analysts, and 东财 calls are serialized behind a one-second
# floor, so throttling is expected under a full run. Aborting there discarded the
# other six analysts' completed work and their API spend.
#
# ``earnings_commentary`` is optional for the same shape of reason: its only
# vendor is Alpha Vantage, whose transcript endpoint is premium-gated, so the
# common outcome on a free key is an entitlement notice that exhausts the chain.
# A missing transcript is a data gap the Earnings Analyst states plainly; it must
# not abort a report whose numeric evidence is already in hand.
#
# ``earnings_data`` is deliberately NOT here. It is the Earnings Analyst's core
# payload, so a vendor that breaks outright should be loud, exactly as a broken
# fundamentals or news primary is. The routine "this instrument has no earnings"
# and "no point-in-time vintage exists" outcomes are not failures at all — the
# adapters return them as structured evidence with an explicit status, so they
# never reach this degradation path.
OPTIONAL_CATEGORIES = {
    "macro_data", "prediction_markets", "signal_data", "earnings_commentary",
}

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "a_stock": get_astock_stock_data,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "a_stock": get_astock_indicators,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "a_stock": get_astock_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "a_stock": get_astock_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "a_stock": get_astock_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "a_stock": get_astock_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "a_stock": get_astock_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "a_stock": get_astock_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "a_stock": get_astock_insider_transactions,
    },
    # macro_data
    "get_macro_indicators": {
        "fred": get_fred_macro_data,
    },
    # prediction_markets
    "get_prediction_markets": {
        "polymarket": get_polymarket_prediction_markets,
    },
    "get_profit_forecast": {"a_stock": get_astock_profit_forecast},
    "get_hot_stocks": {"a_stock": get_astock_hot_stocks},
    "get_northbound_flow": {"a_stock": get_astock_northbound_flow},
    "get_concept_blocks": {"a_stock": get_astock_concept_blocks},
    "get_fund_flow": {"a_stock": get_astock_fund_flow},
    "get_dragon_tiger_board": {"a_stock": get_astock_dragon_tiger_board},
    "get_lockup_expiry": {"a_stock": get_astock_lockup_expiry},
    "get_industry_comparison": {"a_stock": get_astock_industry_comparison},
    # earnings_data. yfinance leads: it publishes a real 7/30/60/90-day consensus
    # trend and up/down revision counts for every venue it covers — including
    # Shanghai and Shenzhen listings in Yahoo form, in CNY — which is the signal
    # this analyst exists to read. It self-selects safely here in a way it cannot
    # for prices or news: a bare 6-digit A-share code is refused by string
    # inspection with no network call, and an unknown symbol raises
    # NoMarketDataError, so both fall through rather than succeeding emptily.
    # a_stock backs it up for bare 沪深京 codes with 同花顺's current consensus
    # snapshot, which carries no history and reports momentum as insufficient.
    "get_earnings_evidence": {
        "yfinance": get_yfinance_earnings_evidence,
        "alpha_vantage": get_alpha_vantage_earnings_evidence,
        "a_stock": get_astock_earnings_evidence,
    },
    # quality_evidence / valuation_evidence. yfinance-only for now: unlike
    # earnings, which needed alpha_vantage/a_stock backups because revision
    # history is a thin-coverage category, ROE/margins/P-E/PEG/P-B are core
    # yfinance.info fields with no point-in-time complexity, so a second
    # vendor has not been needed yet. Adding one later is additive -- the
    # vendor-chain shape already supports it.
    "get_quality_evidence": {
        "yfinance": get_yfinance_quality_evidence,
    },
    "get_valuation_evidence": {
        "yfinance": get_yfinance_valuation_evidence,
    },
    # earnings_commentary. Alpha Vantage only: it is the sole source here for an
    # earnings-call transcript. Unconfigured, it raises VendorNotConfiguredError
    # before any network call and the optional-category degradation applies.
    "get_earnings_commentary": {
        "alpha_vantage": get_alpha_vantage_earnings_commentary,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # The configured vendor list IS the chain: we do NOT silently fall back to
    # vendors the user did not choose (#988/#289) — that returned data from an
    # unexpected source and caused cross-vendor inconsistencies. For multi-vendor
    # fallback, list them in order, e.g. data_vendors="yfinance,alpha_vantage".
    # The "default" sentinel (no explicit config) uses all available vendors.
    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(
                f"Configured vendor(s) {explicit} not available for '{method}'. "
                f"Available: {all_available_vendors}."
            )
    else:
        vendor_chain = all_available_vendors

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None
    last_rate_limit: VendorRateLimitError | None = None
    for vendor in vendor_chain:
        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except VendorRateLimitError as e:
            logger.warning("Vendor %r rate-limited for %s; trying next vendor.", vendor, method)
            # Recorded rather than dropped. A multi-vendor chain moves on and this
            # never mattered, but a single-vendor category (signal_data) has
            # nothing to move on to, so the tail below was left with no error at
            # all and reported "No available vendor for 'get_fund_flow'" -- which
            # names a registration bug that did not exist and hides the throttle
            # that did.
            if last_rate_limit is None:
                last_rate_limit = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %r not configured for %s; trying next vendor.", vendor, method)
            if first_error is None:
                first_error = e  # Surface it if no other vendor can serve the call.
            continue
        except NoMarketDataError as e:
            last_no_data = e  # No data here; another configured vendor may have it
            continue
        except Exception as e:
            # Don't let one vendor's failure crash the call when another can
            # serve it, but never swallow silently: a broken primary must be
            # visible in the logs (#989), not hidden behind a fallback's verdict.
            logger.warning("Vendor %r failed for %s: %s", vendor, method, e)
            if first_error is None:
                first_error = e
            continue

    # If any vendor reported "no data", the symbol is genuinely unavailable.
    # Return one explicit, instructive sentinel rather than a vendor-specific
    # empty string, so the agent reports "unavailable" instead of inventing a
    # value. This takes precedence over incidental fallback errors.
    if last_no_data is not None:
        if first_error is not None:
            # A vendor also hit a real error; surface it in logs so the no-data
            # verdict can't hide a broken primary (network/auth/etc.).
            logger.warning(
                "Returning NO_DATA for %s, but a vendor errored earlier: %s",
                method, first_error,
            )
        sym = last_no_data.symbol
        canonical = last_no_data.canonical
        resolved = "" if canonical == sym else f" (resolved to '{canonical}')"
        # Surface the typed error's detail (e.g. "latest row is 2025-06-11 ...
        # stale") so the agent sees the specific reason — invalid symbol, no
        # coverage, or stale data — not just a generic "unavailable".
        reason = f" ({last_no_data.detail})" if last_no_data.detail else ""
        return (
            f"NO_DATA_AVAILABLE: No usable market data for '{sym}'{resolved} from "
            f"any configured vendor{reason}. The symbol may be invalid, delisted, "
            f"not covered, or the vendor returned stale data. Do not estimate or "
            f"fabricate values — report that data is unavailable for this symbol."
        )

    # A throttled vendor is a real failure with a nameable cause, so fold it into
    # first_error here and let the block below apply the same optional-category
    # degradation and reporting that every other failure gets.
    if first_error is None and last_rate_limit is not None:
        first_error = last_rate_limit

    # No vendor returned data and none reported clean "no data" — surface the
    # first real error (e.g. the primary vendor's network failure). Optional
    # enrichment categories degrade to a sentinel instead, so flavour data can't
    # abort the run.
    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
            return (
                f"DATA_UNAVAILABLE: optional {category} could not be retrieved "
                f"({first_error}). Proceed without it; do not fabricate values."
            )
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")
