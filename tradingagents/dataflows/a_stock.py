"""A-share (沪深京) market data over free public HTTP endpoints.

Sources, all direct and keyless:

* **东财 EastMoney** (``push2his`` / ``push2`` / ``datacenter-web`` / ``search``)
  — daily OHLCV with adjustment, live snapshot, valuation fields, company news,
  executive share changes. The primary source for everything it covers.
* **新浪 Sina** (``money.finance.sina.com.cn``) — the three financial statements
  as downloadable TSV, and a K-line fallback when 东财 is throttling.
* **同花顺 THS** — consensus EPS forecast, which the other two do not publish.
* **腾讯 Tencent** (``qt.gtimg.cn``) — a second live-quote source, used only when
  东财's snapshot is unavailable.

Derived from the A-share adapter in TradingAgents-astock (Apache-2.0, see
NOTICE), with two deliberate departures:

**mootdx is optional.** It is attempted first for OHLCV when installed, but its
upstream package metadata pins an old ``httpx`` that conflicts with Gemini.
Deployment therefore installs it without dependency resolution and keeps the
modern Gemini-compatible ``httpx``. Import, connection, or protocol failures
are negatively cached and fall through to 东财 then 新浪; they never abort a run.

**Adjustment is 后复权, never 前复权.** 东财's ``fqt=1`` (前复权) rebases from
today's price, so a long history of a heavily-dividend-paying stock goes
*negative*: 600519 reads -312.68 for 2001. Negative prices make RSI and MACD
meaningless rather than merely wrong, so ``fqt=2`` (后复权) is the default —
adjusted for corporate actions, and monotonically positive.

Market routing is by refusal. This vendor sits after ``yfinance`` in the chain
and raises :class:`NoMarketDataError` for anything that is not an A-share code,
so a US ticker is served by yfinance and never touches these endpoints.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import socket
import threading
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import requests

from .errors import NoMarketDataError, VendorRateLimitError

logger = logging.getLogger(__name__)

# 东财 rejects requests without a browser-ish UA, and drops the connection on
# rapid repeats — measured as RemoteDisconnected after a handful of probes — so
# every call goes through _http() with backoff.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
_EM_HEADERS = {"User-Agent": _UA, "Accept": "*/*",
               "Referer": "https://quote.eastmoney.com/"}

_TIMEOUT = 15
_RETRIES = 3
_EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_EM_SESSION = requests.Session()
_EM_LAST_CALL = 0.0
_EM_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

# 6 digits is the whole A-share code space. The first two decide the exchange,
# which decides 东财's secid prefix (1 = 上海, 0 = 深圳/北京).
_SH_PREFIXES = ("60", "68", "90", "50", "51", "52", "53", "56", "58")   # 主板/科创/B/ETF
_SZ_PREFIXES = ("00", "30", "20", "15", "16", "18", "12", "13")          # 主板/创业/B/基金
_BJ_PREFIXES = ("43", "83", "87", "88", "92")                            # 北交所

_CODE_RE = re.compile(r"^\d{6}$")
# Accepts 600519, sh600519, 600519.SS, 600519.SH — the forms users actually type.
_DECORATED_RE = re.compile(r"^(?:(sh|sz|bj)\.?)?(\d{6})(?:\.(ss|sz|sh|bj))?$", re.I)


def normalize_code(symbol: str) -> Optional[str]:
    """Return the bare 6-digit code for an A-share symbol, else None."""
    s = (symbol or "").strip().lower().replace(" ", "")
    if not s:
        return None
    m = _DECORATED_RE.match(s)
    if not m:
        return None
    code = m.group(2)
    return code if _CODE_RE.match(code) else None


def is_a_share(symbol: str) -> bool:
    code = normalize_code(symbol)
    if not code:
        return False
    return code[:2] in (_SH_PREFIXES + _SZ_PREFIXES + _BJ_PREFIXES)


def _require_a_share(symbol: str) -> str:
    """Bare code, or raise so the router moves on to the next vendor.

    NoMarketDataError rather than a generic Exception on purpose: route_to_vendor
    treats it as "this vendor has nothing for that symbol" and continues quietly,
    which is exactly the semantics wanted when an American ticker reaches here.
    """
    code = normalize_code(symbol)
    if not code or not is_a_share(code):
        raise NoMarketDataError(symbol, None, "not an A-share code (需要 6 位沪深京代码)")
    return code


def _secid(code: str) -> str:
    """东财's market-qualified id: 1.600519 for 上海, 0.000001 for 深圳/北京."""
    if code[:2] in _SH_PREFIXES:
        return f"1.{code}"
    return f"0.{code}"


def _sina_symbol(code: str) -> str:
    return ("sh" if code[:2] in _SH_PREFIXES else "sz") + code


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http(url: str, *, params: dict | None = None, encoding: str | None = None,
          headers: dict | None = None) -> str:
    """GET with backoff, returning text.

    Raises VendorRateLimitError when the host keeps hanging up, so the router can
    try another vendor instead of surfacing a stack trace: 东财 signals overload
    by dropping the connection rather than by returning 429.
    """
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            if "eastmoney.com" in url:
                global _EM_LAST_CALL
                with _EM_LOCK:
                    wait = _EM_MIN_INTERVAL - (time.monotonic() - _EM_LAST_CALL)
                    if wait > 0:
                        time.sleep(wait + random.uniform(0.1, 0.5))
                    resp = _EM_SESSION.get(
                        url, params=params, timeout=_TIMEOUT,
                        headers=headers or _EM_HEADERS,
                    )
                    _EM_LAST_CALL = time.monotonic()
            else:
                resp = requests.get(url, params=params, timeout=_TIMEOUT,
                                    headers=headers or _EM_HEADERS)
            if resp.status_code == 429:
                raise VendorRateLimitError(f"{url} returned 429")
            resp.raise_for_status()
            if encoding:
                resp.encoding = encoding
            return resp.text
        except VendorRateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            if attempt < _RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise VendorRateLimitError(f"{url} unreachable after {_RETRIES} tries: {last}")


def _em_json(url: str, params: dict) -> dict:
    import json

    text = _http(url, params=params)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise VendorRateLimitError(f"non-JSON from {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# 后复权. See the module docstring: fqt=1 yields negative prices on long
# histories, which silently corrupts every indicator computed from them.
_FQT_ADJUSTED = 2


def _fetch_kline_em(code: str, *, fqt: int = _FQT_ADJUSTED) -> pd.DataFrame:
    data = _em_json(_KLINE_URL, {
        "secid": _secid(code),
        "fields1": "f1,f2,f3",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": 101,          # daily
        "fqt": fqt,
        "beg": 0,
        "end": 20500101,
        "lmt": 100000,
    })
    block = data.get("data") or {}
    lines = block.get("klines") or []
    if not lines:
        raise NoMarketDataError(code, _secid(code), "东财 returned no klines")
    rows = []
    for ln in lines:
        p = ln.split(",")
        if len(p) < 6:
            continue
        rows.append({"Date": p[0], "Open": p[1], "Close": p[2], "High": p[3],
                     "Low": p[4], "Volume": p[5]})
    df = pd.DataFrame(rows)
    df.attrs["name"] = block.get("name") or ""
    return df


def _fetch_kline_sina(code: str, datalen: int = 1023) -> pd.DataFrame:
    """新浪 fallback. Caps out around 1023 bars, so it covers the recent window
    an analyst reads but not the full history 东财 gives."""
    import json

    text = _http(
        "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData",
        params={"symbol": _sina_symbol(code), "scale": 240, "datalen": datalen},
        headers={"User-Agent": _UA},
    )
    payload = json.loads(text)
    items = (((payload or {}).get("result") or {}).get("data")) or []
    if not items:
        raise NoMarketDataError(code, _sina_symbol(code), "新浪 returned no klines")
    df = pd.DataFrame([{"Date": it["day"], "Open": it["open"], "Close": it["close"],
                        "High": it["high"], "Low": it["low"], "Volume": it["volume"]}
                       for it in items])
    df.attrs["name"] = ""      # 新浪's kline payload carries no name field
    return df


_MOOTDX_UNAVAILABLE_UNTIL = 0.0
_TDX_SERVERS = (
    ("119.97.185.59", 7709),
    ("124.70.133.119", 7709),
    ("116.205.183.150", 7709),
)


def _fetch_kline_mootdx(code: str) -> pd.DataFrame:
    """Fetch recent daily bars over TDX TCP, failing quickly when unavailable."""
    global _MOOTDX_UNAVAILABLE_UNTIL
    if os.environ.get("TRADINGAGENTS_MOOTDX_ENABLED", "1").lower() in {"0", "false", "off"}:
        raise RuntimeError("mootdx disabled")
    if time.monotonic() < _MOOTDX_UNAVAILABLE_UNTIL:
        raise RuntimeError("mootdx temporarily unavailable")
    try:
        from mootdx.quotes import Quotes

        frame = None
        for server in _TDX_SERVERS:
            try:
                with socket.create_connection(server, timeout=1.5):
                    pass
                client = Quotes.factory(market="std", server=server)
                candidate = client.bars(symbol=code, category=4, offset=800)
                if candidate is not None and not candidate.empty:
                    frame = candidate
                    break
            except Exception:
                continue
        if frame is None or frame.empty:
            raise RuntimeError("mootdx returned no bars from bounded server list")
        frame = frame.drop(
            columns=["datetime", "year", "month", "day", "hour", "minute"],
            errors="ignore",
        ).reset_index()
        frame = frame.rename(columns={
            "datetime": "Date", "open": "Open", "close": "Close",
            "high": "High", "low": "Low", "volume": "Volume",
        })
        frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
        frame.attrs["name"] = ""
        return frame
    except Exception:
        _MOOTDX_UNAVAILABLE_UNTIL = time.monotonic() + 300
        raise


_OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # copy() does not carry .attrs across on every pandas version, so the name
    # the kline payload supplied has to be reattached explicitly.
    out.attrs.update(df.attrs)
    # These endpoints break by changing their field list, not by going down: the
    # row parser then skips every line and hands over a frame with no columns at
    # all. Reporting that as this vendor's own error is what lets the caller fall
    # through to the next source — a bare KeyError escapes the fallback chain and
    # aborts the run, which the module docstring promises never happens.
    missing = [c for c in _OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise NoMarketDataError(
            "", None, f"kline payload missing {', '.join(missing)}")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    # A non-positive close means the adjustment convention went wrong; refusing
    # is better than handing indicators numbers they cannot interpret.
    bad = int((out["Close"] <= 0).sum())
    if bad:
        raise NoMarketDataError("", None,
                                f"{bad} non-positive closes — wrong adjustment mode")
    return out


def load_ohlcv(symbol: str, curr_date: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for an A-share, mootdx → 东财 → 新浪.

    Trimmed at ``curr_date`` so a backtest cannot see past its own date; the
    upstream adapter calls the same thing a look-ahead guard.
    """
    code = _require_a_share(symbol)
    try:
        df = _clean_ohlcv(_fetch_kline_mootdx(code))
        source = "mootdx"
    except Exception as exc:  # optional dependency / TCP source
        logger.info("a_stock: mootdx unavailable for %s (%s); trying 东财", code, exc)
        try:
            df = _clean_ohlcv(_fetch_kline_em(code))
            source = "eastmoney"
        except (VendorRateLimitError, NoMarketDataError) as em_exc:
            logger.warning("a_stock: 东财 kline failed for %s (%s); trying 新浪", code, em_exc)
            df = _clean_ohlcv(_fetch_kline_sina(code))
            source = "sina"
    name = df.attrs.get("name") or ""
    if curr_date:
        cutoff = pd.to_datetime(curr_date, errors="coerce")
        if not pd.isna(cutoff):
            df = df[df["Date"] <= cutoff].reset_index(drop=True)
    if df.empty:
        raise NoMarketDataError(symbol, code, "no rows at or before the requested date")
    # Slicing above also produces a new frame, so re-stamp provenance last.
    df.attrs["source"] = source
    df.attrs["code"] = code
    df.attrs.setdefault("name", name)
    return df


def basis_note(df: pd.DataFrame) -> str:
    """State the adjustment basis of the rows actually returned.

    The two sources do not agree: 东财 here is 后复权, 新浪's K-line is raw. A
    header that says "后复权" over unadjusted rows is worse than no header, and
    the fallback fires silently whenever 东财 throttles.

    Public because the verified-snapshot path renders the same provenance: it
    tells the analyst to treat its numbers as ground truth, so it must say which
    adjustment basis those numbers are on.
    """
    src = df.attrs.get("source")
    if src == "eastmoney":
        return "复权方式: 后复权 (东财 fqt=2) · 数据源: 东方财富"
    if src == "sina":
        return ("复权方式: 不复权 (新浪原始K线) · 数据源: 新浪财经 "
                "— 东财限流时的降级源，除权日附近的跳空未做还原")
    if src == "mootdx":
        return "复权方式: 通达信日线 · 数据源: mootdx/通达信 TCP"
    return f"数据源: {src or 'unknown'}"


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Daily OHLCV as CSV text, matching the shape yfinance's vendor returns."""
    code = _require_a_share(symbol)
    df = load_ohlcv(symbol, end_date)
    start = pd.to_datetime(start_date, errors="coerce")
    if not pd.isna(start):
        df = df[df["Date"] >= start]
    if df.empty:
        raise NoMarketDataError(symbol, code, f"no rows between {start_date} and {end_date}")
    name = df.attrs.get("name") or ""
    title = f"{code} {name}".strip()      # no dangling space when the name is absent
    body = df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")).to_csv(index=False)
    return (f"# {title} 日线 ({start_date} → {end_date})\n"
            f"# {basis_note(df)}\n"
            f"{body}")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def get_indicators(symbol: str, indicator: str, curr_date: str,
                   look_back_days: int = 30) -> str:
    """One indicator's recent values, computed locally from the OHLCV above.

    Computed here rather than fetched, so the indicator and the price series an
    analyst quotes come from the same bars.
    """
    from stockstats import wrap

    code = _require_a_share(symbol)
    df = load_ohlcv(symbol, curr_date)
    frame = df.rename(columns=str.lower)
    stock = wrap(frame)
    try:
        series = stock[indicator]
    except Exception as exc:  # noqa: BLE001 - unknown indicator name
        raise NoMarketDataError(symbol, code, f"unknown indicator {indicator!r}: {exc}")
    # Aligned by position, not by index. stockstats.wrap() reindexes the frame it
    # is given, so pairing df["Date"] with the result by index produced a column
    # of NaN dates -- every value correct, every date lost.
    values = pd.to_numeric(pd.Series(series).reset_index(drop=True), errors="coerce")
    dates = df["Date"].reset_index(drop=True).dt.strftime("%Y-%m-%d")
    tail = pd.DataFrame({"date": dates, indicator: values}).tail(look_back_days)
    lines = [f"{r.date}: {getattr(r, indicator):.4f}"
             for r in tail.itertuples() if pd.notna(getattr(r, indicator))]
    if not lines:
        raise NoMarketDataError(symbol, code, f"{indicator} produced no values")
    return f"## {code} {indicator} (last {len(lines)} sessions)\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Snapshot / fundamentals
# ---------------------------------------------------------------------------

# 东财 field ids. Named here because f-codes are unreadable at the call site.
_EM_FIELDS = {
    "f43": "最新价", "f44": "最高", "f45": "最低", "f46": "今开",
    "f47": "成交量", "f48": "成交额", "f57": "代码", "f58": "名称",
    "f60": "昨收", "f116": "总市值", "f117": "流通市值",
    "f162": "市盈率(动)", "f163": "市盈率(静)", "f164": "市盈率(TTM)",
    "f167": "市净率", "f168": "换手率", "f173": "净资产收益率ROE",
    "f183": "营业收入", "f184": "营收同比", "f185": "净利润同比",
    "f186": "毛利率", "f187": "净利率", "f188": "资产负债率",
}


def _snapshot(code: str) -> dict[str, Any]:
    data = _em_json("https://push2.eastmoney.com/api/qt/stock/get", {
        "secid": _secid(code),
        "fields": ",".join(_EM_FIELDS),
    })
    block = (data.get("data") or {})
    if not block:
        raise NoMarketDataError(code, _secid(code), "东财 snapshot empty")
    return block


def get_fundamentals(symbol: str, curr_date: str | None = None) -> str:
    """Valuation and profitability snapshot, plus consensus EPS when available."""
    code = _require_a_share(symbol)
    block = _snapshot(code)
    name = block.get("f58") or ""

    # 东财 scales several fields; only the ones used here are converted, and the
    # divisor is stated so a reader can check it rather than trust it.
    def num(key: str, scale: float = 1.0, nd: int = 2):
        v = block.get(key)
        if v in (None, "-", ""):
            return None
        try:
            return round(float(v) / scale, nd)
        except (TypeError, ValueError):
            return None

    rows = [
        ("最新价", num("f43", 100)), ("昨收", num("f60", 100)),
        ("总市值(亿元)", num("f116", 1e8)), ("流通市值(亿元)", num("f117", 1e8)),
        ("市盈率TTM", num("f164", 100)), ("市盈率(动)", num("f162", 100)),
        ("市净率", num("f167", 100)), ("换手率(%)", num("f168", 100)),
        ("ROE(%)", num("f173", 100)), ("毛利率(%)", num("f186", 100)),
        ("净利率(%)", num("f187", 100)), ("资产负债率(%)", num("f188", 100)),
        ("营收同比(%)", num("f184", 100)), ("净利润同比(%)", num("f185", 100)),
    ]
    body = "\n".join(f"| {k} | {'—' if v is None else v} |" for k, v in rows)
    out = (f"## {code} {name} 基本面快照\n"
           f"（数据源：东财 push2；as of {curr_date or 'today'}）\n\n"
           f"| 指标 | 数值 |\n| --- | --- |\n{body}\n")
    forecast = _eps_forecast_ths(code)
    if forecast:
        out += f"\n### 同花顺一致预期 EPS\n\n{forecast}\n"
    return out


def _eps_forecast_ths(code: str) -> str:
    """同花顺 consensus EPS. Optional: neither 东财 nor 新浪 publishes it, and a
    missing forecast must not fail the whole fundamentals call."""
    try:
        html = _http(f"https://basic.10jqka.com.cn/{code}/worth.html",
                     encoding="gbk", headers={"User-Agent": _UA})
    except Exception as exc:  # noqa: BLE001
        logger.info("a_stock: 同花顺 EPS forecast unavailable for %s: %s", code, exc)
        return ""
    # The page embeds the forecast table; pull the first table that mentions 预测.
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:  # noqa: BLE001 - page shape changed or no tables
        return ""
    for tbl in tables:
        text = " ".join(str(c) for c in tbl.columns)
        if "预测" in text or "预计" in text:
            return tbl.head(6).to_markdown(index=False)
    return ""


# ---------------------------------------------------------------------------
# Financial statements (新浪)
# ---------------------------------------------------------------------------

_SINA_REPORTS = {
    "balance": ("vDOWN_BalanceSheet", "资产负债表"),
    "cashflow": ("vDOWN_CashFlow", "现金流量表"),
    "income": ("vDOWN_ProfitStatement", "利润表"),
}


def _sina_statement(code: str, kind: str, periods: int = 5) -> str:
    """One statement as a markdown table.

    新浪 serves these as GBK tab-separated text with periods as *columns*, so the
    encoding has to be set explicitly — decoded as UTF-8 it is mojibake, not an
    error, which is the sort of thing that reaches a report unnoticed.
    """
    path, label = _SINA_REPORTS[kind]
    url = (f"http://money.finance.sina.com.cn/corp/go.php/{path}/"
           f"displaytype/4/stockid/{code}/ctrl/all.phtml")
    text = _http(url, encoding="gbk", headers={"User-Agent": _UA})
    rows = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
    if len(rows) < 3:
        raise NoMarketDataError(code, None, f"新浪 {label} empty")
    # Column 0 is the line-item name; keep only the most recent `periods`.
    keep = 1 + periods
    header = rows[0][:keep]
    body = [r[:keep] for r in rows[1:] if len(r) > 1 and r[0].strip()]
    md = ["| " + " | ".join(h.strip() or "项目" for h in header) + " |",
          "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in body[:60]:      # 60 line items is plenty for an analyst summary
        md.append("| " + " | ".join((c or "").strip() for c in r) + " |")
    return (f"## {code} {label}（新浪财经，最近 {periods} 期）\n\n" + "\n".join(md) + "\n")


def get_balance_sheet(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _sina_statement(_require_a_share(symbol), "balance")


def get_cashflow(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _sina_statement(_require_a_share(symbol), "cashflow")


def get_income_statement(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _sina_statement(_require_a_share(symbol), "income")


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def _news_eastmoney(code: str, limit: int = 20) -> list[dict[str, str]]:
    import json

    text = _http("https://search-api-web.eastmoney.com/search/jsonp", params={
        "cb": "cb",
        "param": json.dumps({
            "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default",
                                           "sort": "default", "pageIndex": 1,
                                           "pageSize": limit, "preTag": "",
                                           "postTag": ""}},
        }, ensure_ascii=False),
    })
    body = text[text.index("(") + 1: text.rindex(")")] if "(" in text else text
    payload = json.loads(body)
    items = (((payload.get("result") or {}).get("cmsArticleWebOld")) or [])
    out = []
    for it in items:
        out.append({"date": (it.get("date") or "")[:10],
                    "title": re.sub(r"<[^>]+>", "", it.get("title") or ""),
                    "summary": re.sub(r"<[^>]+>", "", it.get("content") or "")[:300],
                    "source": it.get("mediaName") or "东方财富"})
    return out


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Company news in a date window.

    The argument list mirrors the yfinance vendor exactly (ticker, start, end);
    the router passes them positionally, so a different shape here would silently
    bind a date string to the wrong parameter.
    """
    code = _require_a_share(ticker)
    try:
        items = _news_eastmoney(code)
    except Exception as exc:  # noqa: BLE001
        raise NoMarketDataError(ticker, code, f"东财 news unavailable: {exc}")
    lo, hi = str(start_date or "")[:10], str(end_date or "")[:10]
    items = [i for i in items
             if not i["date"] or ((not lo or i["date"] >= lo) and (not hi or i["date"] <= hi))]
    if not items:
        raise NoMarketDataError(ticker, code, f"no news between {lo} and {hi}")
    symbol = ticker
    parts = [f"## {code} 相关新闻（东方财富，{len(items)} 条）\n"]
    for i in items:
        parts.append(f"### {i['date']} · {i['source']}\n{i['title']}\n\n{i['summary']}\n")
    return "\n".join(parts)


def get_global_news(curr_date: str | None = None, look_back_days: int | None = None,
                    limit: int | None = None) -> str:
    """Market-wide 快讯 from 东财, for the macro slot rather than a single name."""
    import json

    text = _http("https://np-weblist.eastmoney.com/comm/web/getFastNewsList", params={
        "client": "web", "biz": "web_724", "fastColumn": "102",
        "sortEnd": "", "pageSize": 30, "req_trace": str(int(time.time() * 1000)),
    })
    payload = json.loads(text)
    items = ((payload.get("data") or {}).get("fastNewsList")) or []
    if not items:
        raise NoMarketDataError("A-share", None, "东财 快讯 empty")
    cutoff = str(curr_date)[:10] if curr_date else None
    rows, newest, oldest = [], "", ""
    for it in items:
        stamp = (it.get("showTime") or "")[:16]
        newest = newest or stamp
        oldest = stamp or oldest
        if cutoff and stamp[:10] and stamp[:10] > cutoff:
            continue          # never show an analyst news from after its own date
        rows.append(f"- **{stamp}** "
                    f"{re.sub(r'<[^>]+>', '', it.get('title') or it.get('summary') or '')}")
    if not rows:
        # A single page of 快讯 spans only hours, so asking about an earlier date
        # legitimately finds nothing. Say which window was searched instead of
        # implying the feed is broken.
        raise NoMarketDataError(
            "A-share", None,
            f"东财 快讯 only covers {oldest or '?'} → {newest or '?'}, "
            f"nothing at or before {cutoff}")
    return "## A 股市场快讯（东方财富）\n\n" + "\n".join(rows[:(limit or 30)]) + "\n"


# ---------------------------------------------------------------------------
# Insider (高管持股变动)
# ---------------------------------------------------------------------------

def get_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    """Executive shareholding changes — the A-share analogue of insider trades.

    ``curr_date`` is optional and unused by the router today, but kept so a
    look-ahead cutoff can be applied when a caller has one.
    """
    symbol = ticker
    code = _require_a_share(ticker)
    data = _em_json("https://datacenter-web.eastmoney.com/api/data/v1/get", {
        "reportName": "RPT_EXECUTIVE_HOLD_DETAILS",
        "columns": "ALL",
        "pageSize": 30,
        "pageNumber": 1,
        "sortColumns": "CHANGE_DATE",
        "sortTypes": "-1",
        "filter": f'(SECURITY_CODE="{code}")',
        "source": "WEB",
        "client": "WEB",
    })
    rows = ((data.get("result") or {}) or {}).get("data") or []
    if not rows:
        raise NoMarketDataError(symbol, code, "东财 executive-holdings empty")
    keep = [("CHANGE_DATE", "变动日期"), ("PERSON_NAME", "变动人"),
            ("CHANGE_SHARES", "变动股数"), ("AVERAGE_PRICE", "均价"),
            ("CHANGE_REASON", "变动原因"), ("HOLD_NUM_AFTER", "变动后持股")]
    md = ["| " + " | ".join(z for _, z in keep) + " |",
          "| " + " | ".join(["---"] * len(keep)) + " |"]
    for r in rows:
        if curr_date and str(r.get("CHANGE_DATE") or "")[:10] > str(curr_date)[:10]:
            continue
        md.append("| " + " | ".join(str(r.get(k, "—") or "—")[:22] for k, _ in keep) + " |")
    if len(md) == 2:
        raise NoMarketDataError(symbol, code, "no executive changes at or before that date")
    return (f"## {code} 高管持股变动（东方财富）\n\n" + "\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# A-share signal layer (政策/游资/解禁 analysts)
# ---------------------------------------------------------------------------

def _historical_notice(curr_date: str | None, label: str) -> str:
    """Warn when a current-only source is used for a historical analysis."""
    if not curr_date:
        return ""
    try:
        historical = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d").date() < datetime.now().date()
    except ValueError:
        historical = False
    if not historical:
        return ""
    return (
        f"⚠️ 未来函数警告：以下{label}是当前快照，不是 {str(curr_date)[:10]} "
        "当日的历史版本；不得把它描述为分析日已知事实。\n\n"
    )


def _datacenter(report_name: str, *, filter_str: str = "", page_size: int = 50,
                sort_columns: str = "", sort_types: str = "-1") -> list[dict[str, Any]]:
    data = _em_json("https://datacenter-web.eastmoney.com/api/data/v1/get", {
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_str,
        "pageNumber": 1,
        "pageSize": page_size,
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    })
    return (((data.get("result") or {}).get("data")) or [])


def get_profit_forecast(ticker: str, curr_date: str | None = None) -> str:
    """Consensus EPS forecast from Tonghuashun, with explicit snapshot basis."""
    code = _require_a_share(ticker)
    table = _eps_forecast_ths(code)
    if not table:
        return f"NO_DATA_AVAILABLE: 同花顺无 {code} 一致预期覆盖。"
    return (
        _historical_notice(curr_date, "分析师一致预期")
        + f"## {code} 一致预期（同花顺当前快照）\n\n{table}\n"
    )


def get_hot_stocks(curr_date: str = "") -> str:
    """Strong-stock list with Tonghuashun editorial theme attribution."""
    day = curr_date or datetime.now().strftime("%Y-%m-%d")
    text = _http(
        f"http://zx.10jqka.com.cn/event/api/getharden/date/{day}/"
        "orderby/date/orderway/desc/charset/GBK/",
        headers={"User-Agent": _UA},
    )
    payload = json.loads(text)
    rows = payload.get("data") or []
    if not rows:
        return f"NO_DATA_AVAILABLE: 同花顺 {day} 无强势股数据（可能为非交易日）。"
    lines = [f"## {day} 强势股与题材归因（同花顺）", ""]
    for row in rows[:80]:
        lines.append(
            f"- {row.get('code', '')} {row.get('name', '')}: "
            f"涨幅 {row.get('zhangfu', '—')}%，换手 {row.get('huanshou', '—')}%，"
            f"题材：{row.get('reason', '—')}"
        )
    return "\n".join(lines)


def get_northbound_flow(curr_date: str, include_history: bool = False) -> str:
    """Current northbound flow from Tonghuashun; never backdate the snapshot."""
    text = _http(
        "https://data.hexin.cn/market/hsgtApi/method/dayChart/",
        headers={"User-Agent": _UA, "Referer": "https://data.hexin.cn/"},
    )
    payload = json.loads(text)
    times, hgt, sgt = payload.get("time") or [], payload.get("hgt") or [], payload.get("sgt") or []
    if not times:
        return "NO_DATA_AVAILABLE: 同花顺北向资金当前无数据（可能为休市时段）。"
    h_last = float(hgt[-1]) if hgt else 0.0
    s_last = float(sgt[-1]) if sgt else 0.0
    total = h_last + s_last
    rows = [
        _historical_notice(curr_date, "北向资金") + "## 北向资金（同花顺当前快照）",
        f"- 沪股通累计净额：{h_last:.2f} 亿元",
        f"- 深股通累计净额：{s_last:.2f} 亿元",
        f"- 合计：{total:.2f} 亿元",
    ]
    if include_history:
        rows.append("- 历史序列：该免费接口仅保证当前日内快照，未伪造历史值。")
    return "\n".join(rows)


def get_concept_blocks(ticker: str) -> str:
    """Concept and industry membership from Baidu Stock Market."""
    code = _require_a_share(ticker)
    url = "https://finance.pae.baidu.com/api/getrelatedblock"
    stock = json.dumps(
        [{"code": code, "market": "ab", "type": "stock"}],
        ensure_ascii=False,
    )
    text = _http(url, params={"stock": stock, "finClientType": "pc"}, headers={
        "User-Agent": _UA, "Accept": "application/vnd.finance-web.v1+json",
        "Referer": "https://gushitong.baidu.com/",
    })
    payload = json.loads(text)
    groups = (payload.get("Result") or {}).get(code) or []
    if str(payload.get("ResultCode", -1)) != "0" or not groups:
        return f"NO_DATA_AVAILABLE: 百度股市通无 {code} 板块归属。"
    lines = [f"## {code} 所属板块（百度股市通）"]
    for group in groups:
        values = [f"{item.get('name', '')} {item.get('ratio', '')}" for item in group.get("list") or []]
        if values:
            lines.append(f"- {group.get('name', '分类')}：" + "；".join(values))
    return "\n".join(lines)


def get_fund_flow(ticker: str, curr_date: str, include_history: bool = True) -> str:
    """Main/large/medium/small order flow from Eastmoney push2."""
    code = _require_a_share(ticker)
    lines = [f"## {code} 个股资金流（东方财富）"]
    historical = bool(_historical_notice(curr_date, "实时资金流"))
    if historical:
        lines.append("分析日期早于今天，已略去当前日内资金流。")
    else:
        data = _em_json("https://push2.eastmoney.com/api/qt/stock/fflow/kline/get", {
            "secid": _secid(code), "klt": 1,
            "fields1": "f1,f2,f3,f7", "fields2": "f51,f52,f53,f54,f55,f56,f57",
        })
        klines = (data.get("data") or {}).get("klines") or []
        if klines:
            parts = klines[-1].split(",")
            if len(parts) >= 6:
                lines.append(
                    f"- {parts[0]} 主力净流入 {float(parts[1])/1e4:.0f} 万元；"
                    f"大单 {float(parts[4])/1e4:.0f} 万元；超大单 {float(parts[5])/1e4:.0f} 万元"
                )
    if include_history:
        data = _em_json("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get", {
            "secid": _secid(code), "lmt": 60, "klt": 101,
            "fields1": "f1,f2,f3,f7", "fields2": "f51,f52,f53,f54,f55,f56,f57",
        })
        history = (data.get("data") or {}).get("klines") or []
        cutoff = str(curr_date)[:10]
        history = [row for row in history if not cutoff or row.split(",")[0] <= cutoff][-20:]
        lines.extend(f"- 日级：{row}" for row in history)
    if len(lines) == 1:
        return f"NO_DATA_AVAILABLE: 东方财富无 {code} 资金流数据。"
    return "\n".join(lines)


def get_dragon_tiger_board(ticker: str, curr_date: str, look_back_days: int = 30) -> str:
    """Dragon-Tiger appearances and top seats from Eastmoney Datacenter."""
    code = _require_a_share(ticker)
    end = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d")
    start = (end - pd.Timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    rows = _datacenter(
        "RPT_DAILYBILLBOARD_DETAILSNEW",
        filter_str=(f"(TRADE_DATE>='{start}')(TRADE_DATE<='{end:%Y-%m-%d}')"
                    f'(SECURITY_CODE="{code}")'),
        sort_columns="TRADE_DATE",
    )
    if not rows:
        return f"## {code} 龙虎榜（东方财富）\n\n近 {look_back_days} 日未查到上榜记录。"
    lines = [f"## {code} 龙虎榜（东方财富，近 {look_back_days} 日）"]
    for row in rows:
        lines.append(
            f"- {str(row.get('TRADE_DATE', ''))[:10]}：{row.get('EXPLANATION', '—')}；"
            f"净买入 {(row.get('BILLBOARD_NET_AMT') or 0)/1e4:.0f} 万元；"
            f"换手 {row.get('TURNOVERRATE', '—')}%"
        )
    return "\n".join(lines)


def get_lockup_expiry(ticker: str, curr_date: str, forward_days: int = 90) -> str:
    """Restricted-share unlock history and forward calendar."""
    code = _require_a_share(ticker)
    end = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d") + pd.Timedelta(days=forward_days)
    rows = _datacenter(
        "RPT_LIFT_STAGE",
        filter_str=(f'(SECURITY_CODE="{code}")(FREE_DATE>="{str(curr_date)[:10]}")'
                    f'(FREE_DATE<="{end:%Y-%m-%d}")'),
        sort_columns="FREE_DATE", sort_types="1",
    )
    if not rows:
        return f"## {code} 限售解禁（东方财富）\n\n未来 {forward_days} 日未查到待解禁记录。"
    lines = [f"## {code} 未来 {forward_days} 日限售解禁（东方财富）"]
    for row in rows:
        lines.append(
            f"- {str(row.get('FREE_DATE', ''))[:10]}："
            f"{row.get('LIMITED_STOCK_TYPE', '类型未知')}；"
            f"数量 {row.get('FREE_SHARES_NUM', '—')}；占比 {row.get('FREE_RATIO', '—')}"
        )
    return "\n".join(lines)


def get_industry_comparison(ticker: str, curr_date: str) -> str:
    """Current Eastmoney industry ranking with historical-snapshot warning."""
    code = _require_a_share(ticker)
    data = _em_json("https://push2.eastmoney.com/api/qt/clist/get", {
        "pn": 1, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fs": "m:90+t:2", "fields": "f3,f12,f14,f104,f105,f140",
    })
    rows = (data.get("data") or {}).get("diff") or []
    if not rows:
        return "NO_DATA_AVAILABLE: 东方财富当前无行业排名。"
    lines = [_historical_notice(curr_date, "行业排名") + f"## {code} 行业横向比较（东方财富当前快照）"]
    for index, row in enumerate(rows[:30], 1):
        lines.append(
            f"- {index}. {row.get('f14', '—')}：{row.get('f3', '—')}%；"
            f"上涨 {row.get('f104', '—')} / 下跌 {row.get('f105', '—')}；"
            f"领涨 {row.get('f140', '—')}"
        )
    return "\n".join(lines)
