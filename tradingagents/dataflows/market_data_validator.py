"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows import a_stock
from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _load_ohlcv_for(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV from whichever vendor can actually serve ``symbol``.

    ``load_ohlcv`` is the *yfinance* loader, and Yahoo does not list 沪深京 codes
    under their bare form — ``yf.download("002185")`` returns no rows, so every
    A-share run died here with NoMarketDataError once the web UI stopped
    rejecting the codes upfront. The vendor table in ``interface.py`` routes
    A-shares to ``a_stock``, but this module imports the Yahoo loader directly
    and so bypassed that routing entirely.

    Dispatching on the symbol rather than adding a Yahoo suffix alias is
    deliberate. Yahoo *does* serve ``002185.SZ``, but its rows are
    dividend/split-adjusted, whereas the analyst's own price and indicator tools
    reach the same stock through ``a_stock`` (后复权 from 东财, or unadjusted from
    新浪 when 东财 throttles). This snapshot tells the analyst to treat itself as
    the source of truth and to *flag* any tool output that disagrees with it — so
    a snapshot built on a different adjustment basis than the tools it is meant
    to check would manufacture a discrepancy on every A-share run.
    """
    if a_stock.is_a_share(symbol):
        return a_stock.load_ohlcv(symbol, curr_date)
    return load_ohlcv(symbol, curr_date)


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    The loaders already normalize the Date column and filter out look-ahead
    rows, but we re-apply the cutoff defensively — this is a verification path,
    so it must not trust its input to be pre-filtered.
    """
    data = _load_ohlcv_for(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    # Re-stamp last: copy(), dropna() and the slice above each return a new frame,
    # and .attrs does not survive that on every pandas version. The A-share loader
    # records its vendor and adjustment basis there, which the header renders.
    df.attrs.update(data.attrs)
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
    ]
    # An A-share snapshot must name its adjustment basis. 东财 serves 后复权 and
    # 新浪 serves raw prices, the degrade between them is silent, and this block
    # is what the analyst is told to trust absolutely — an unlabelled price level
    # here is exactly the kind of unfalsifiable number the snapshot exists to
    # prevent.
    if df.attrs.get("source"):
        lines.append(f"- {a_stock.basis_note(df)}")
    lines += [
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
