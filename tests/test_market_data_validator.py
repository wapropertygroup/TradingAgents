"""Tests for the deterministic market-data verification snapshot (#830/#881)."""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.market_data_validator as validator


def _sample_ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-05-20")
    closes = [100 + i for i in range(len(dates))]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i for i in range(len(dates))],
    })


@pytest.mark.unit
class TestVerifiedSnapshot:
    def test_excludes_future_rows(self, monkeypatch):
        data = pd.concat([
            _sample_ohlcv(),
            pd.DataFrame({"Date": [pd.Timestamp("2026-06-01")], "Open": [999.0],
                          "High": [999.0], "Low": [999.0], "Close": [999.0], "Volume": [999]}),
        ], ignore_index=True)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)

        snap = validator.build_verified_market_snapshot("COF", "2026-05-13")
        assert "Verified market data snapshot for COF" in snap
        assert "Requested analysis date: 2026-05-13" in snap
        assert "Latest trading row used: 2026-05-13" in snap
        assert "999.00" not in snap          # future row excluded
        assert "boll_lb" in snap             # indicators present

    def test_uses_previous_trading_day_when_date_is_weekend(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        # 2026-05-16 is a Saturday; latest row should be Fri 2026-05-15
        snap = validator.build_verified_market_snapshot("COF", "2026-05-16")
        assert "Latest trading row used: 2026-05-15" in snap
        assert "Recent verified closes" in snap

    def test_raises_when_no_rows_on_or_before_date(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2020-01-01")

    def test_raises_on_empty_data(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2026-05-13")

    def test_look_back_window_capped_at_30(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        snap = validator.build_verified_market_snapshot("COF", "2026-05-20", look_back_days=999)
        # last-N closes table has at most 30 data rows
        close_rows = [ln for ln in snap.splitlines() if ln.startswith("| 2026-")]
        assert 0 < len(close_rows) <= 30


@pytest.mark.unit
class TestTool:
    def test_tool_delegates_to_builder(self, monkeypatch):
        from tradingagents.agents.utils.market_data_validation_tools import (
            get_verified_market_snapshot,
        )
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        out = get_verified_market_snapshot.invoke(
            {"symbol": "COF", "curr_date": "2026-05-20"}
        )
        assert "Verified market data snapshot for COF" in out


def _a_share_frame(source: str = "sina") -> pd.DataFrame:
    """A stub shaped like a real a_stock frame.

    Column *order* differs from the yfinance frame — the A-share loader emits
    Close third (Date, Open, Close, High, Low, Volume), as verified against live
    新浪 data. The snapshot reads fields by name, so order does not matter; the
    stub mirrors it anyway so a future order dependency fails here.
    """
    base = _sample_ohlcv()
    frame = base[["Date", "Open", "Close", "High", "Low", "Volume"]].copy()
    frame.attrs["source"] = source
    frame.attrs["code"] = "002185"
    return frame


def _explode(*_args, **_kwargs):
    raise AssertionError("wrong vendor called for this symbol")


@pytest.mark.unit
class TestAShareRouting:
    """A-shares must reach the a_stock vendor, not Yahoo.

    Yahoo does not list 沪深京 codes in their bare form, so the snapshot used to
    die with NoMarketDataError("Yahoo Finance returned no rows") for every
    A-share — this module imports the yfinance loader directly and so skipped the
    vendor routing in interface.py.
    """

    def test_a_share_does_not_touch_yahoo(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", _explode)
        monkeypatch.setattr(validator.a_stock, "load_ohlcv",
                            lambda s, d: _a_share_frame())

        snap = validator.build_verified_market_snapshot("002185", "2026-05-20")
        assert "Verified market data snapshot for 002185" in snap
        assert "Latest trading row used: 2026-05-20" in snap
        assert "boll_lb" in snap

    @pytest.mark.parametrize("symbol", ["002185", "600519", "SH600519",
                                        "600519.SS", "002185.SZ", "sz002185"])
    def test_decorated_a_share_forms_all_route_to_a_stock(self, monkeypatch, symbol):
        """Every form ystocker's _ASHARE_RE accepts must also route correctly."""
        monkeypatch.setattr(validator, "load_ohlcv", _explode)
        seen = []
        monkeypatch.setattr(
            validator.a_stock, "load_ohlcv",
            lambda s, d: (seen.append(s), _a_share_frame())[1],
        )

        validator.build_verified_market_snapshot(symbol, "2026-05-20")
        assert seen == [symbol]

    def test_us_ticker_still_uses_yahoo(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            validator, "load_ohlcv",
            lambda s, d: (seen.append(s), _sample_ohlcv())[1],
        )
        monkeypatch.setattr(validator.a_stock, "load_ohlcv", _explode)

        validator.build_verified_market_snapshot("COF", "2026-05-20")
        assert seen == ["COF"]

    @pytest.mark.parametrize("source,expected", [
        ("eastmoney", "后复权"),
        ("sina", "不复权"),
    ])
    def test_snapshot_states_the_adjustment_basis(self, monkeypatch, source, expected):
        """The block the analyst is told to trust must say which basis it is on.

        东财 is 后复权, 新浪 is raw, and the degrade between them is silent.
        """
        monkeypatch.setattr(validator, "load_ohlcv", _explode)
        monkeypatch.setattr(validator.a_stock, "load_ohlcv",
                            lambda s, d: _a_share_frame(source))

        snap = validator.build_verified_market_snapshot("002185", "2026-05-20")
        assert expected in snap

    def test_no_basis_line_for_symbols_without_provenance(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        snap = validator.build_verified_market_snapshot("COF", "2026-05-20")
        assert "复权" not in snap
