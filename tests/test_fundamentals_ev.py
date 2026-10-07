"""Enterprise value and its multiples, on one currency basis.

Yahoo's own EV ratios divide the listing's currency by the statements' for an
ADR: on 2026-10-06 ASML read EV/Revenue 1,123 and EV/EBITDA 2,942, and TSM's
EV was neither a dollar nor a TWD figure. ``_enterprise_values`` rebuilds EV
from parts it can convert. The payloads below are Yahoo's, read on the box that
day; ystocker's ``tests/test_data.py`` checks its own copy of this arithmetic
against the same figures, so the report and the site quote one number.

No network: the exchange rate is stubbed.
"""

from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.dataflows import fundamentals_evidence as fe

AS_OF = "2026-10-06"
AAPL = {"currency": "USD", "financialCurrency": "USD", "marketCap": 4869056364544,
        "enterpriseValue": 4880201678848, "totalRevenue": 466822987776,
        "operatingMargins": 0.32623002, "totalDebt": 84343996416, "totalCash": 62399000576,
        "grossMargins": 0.48653, "industry": "Consumer Electronics"}
TSM = {"currency": "USD", "financialCurrency": "TWD", "marketCap": 2501436243968,
       "enterpriseValue": 17749214494720, "totalRevenue": 4440492343296,
       "operatingMargins": 0.60344005, "totalDebt": 1068558516224, "totalCash": 3518010228736,
       "industry": "Semiconductors"}
ASML = {"currency": "USD", "financialCurrency": "EUR", "marketCap": 704477790208,
        "enterpriseValue": 39681748107264, "totalRevenue": 35327500288,
        "operatingMargins": 0.37057, "totalDebt": 1984400000, "totalCash": 7581499904,
        "industry": "Semiconductor Equipment & Materials"}
JPM = {"currency": "USD", "financialCurrency": "USD", "marketCap": 880603955200,
       "enterpriseValue": 721455939584, "totalRevenue": 186328006656,
       "operatingMargins": 0.50394, "grossMargins": 0.0, "industry": "Banks - Diversified"}
SNOW = {"currency": "USD", "financialCurrency": "USD", "marketCap": 118523166720,
        "enterpriseValue": 119885635584, "totalRevenue": 5434647040,
        "operatingMargins": -0.17001, "industry": "Software - Application"}

#: USD per one unit of each, so TWD -> USD is 0.0329.
RATES = {("TWD", "USD"): 0.0329, ("EUR", "USD"): 1.16}


def _ev(info):
    with mock.patch.object(fe, "_fx_rate", lambda a, b: 1.0 if a == b else RATES.get((a, b))):
        return fe._enterprise_values(info, AS_OF)


@pytest.mark.unit
def test_a_us_filer_keeps_yahoos_ev():
    ev, sales, ebit = _ev(AAPL)
    assert ev.value == AAPL["enterpriseValue"] and ev.currency == "USD"
    assert round(sales.value, 2) == 10.45
    assert round(ebit.value, 1) == 32.0


@pytest.mark.unit
def test_an_adr_is_put_on_one_basis():
    ev, sales, ebit = _ev(TSM)
    # 2.501e12 + (1.069e12 - 3.518e12) x 0.0329, in dollars.
    assert round(ev.value / 1e9, 1) == 2420.8 and ev.currency == "USD"
    assert round(sales.value, 2) == 16.57          # Yahoo: 3.997
    assert round(ebit.value, 1) == 27.5


@pytest.mark.unit
def test_asml_no_longer_reads_a_thousand_times_sales():
    _, sales, ebit = _ev(ASML)
    assert round(sales.value, 2) == 17.03          # Yahoo: 1,123
    assert ebit.value < 100


@pytest.mark.unit
def test_without_a_rate_an_adr_has_no_ev_rather_than_a_mixed_one():
    with mock.patch.object(fe, "_fx_rate", lambda a, b: None):
        ev, sales, ebit = fe._enterprise_values(TSM, AS_OF)
    assert not (ev.available or sales.available or ebit.available)
    assert "exchange rate" in sales.unavailable_reason


@pytest.mark.unit
def test_a_bank_has_an_ev_but_no_ev_multiple():
    ev, sales, ebit = _ev(JPM)
    assert ev.available
    assert not sales.available and not ebit.available
    assert "bank or an insurer" in ebit.unavailable_reason


@pytest.mark.unit
def test_an_operating_loss_has_ev_sales_and_no_ev_ebit():
    _, sales, ebit = _ev(SNOW)
    assert round(sales.value, 2) == 22.06
    assert not ebit.available and "operating loss" in ebit.unavailable_reason


@pytest.mark.unit
def test_a_banks_zero_gross_margin_is_not_a_margin():
    assert not fe._gross_margin(JPM["grossMargins"], AS_OF).available
    assert fe._gross_margin(AAPL["grossMargins"], AS_OF).value == pytest.approx(0.48653)


@pytest.mark.unit
def test_the_valuation_evidence_carries_them_and_labels_the_cap_in_its_own_currency():
    fe._shared_snapshot.cache_clear()
    info = dict(TSM, quoteType="EQUITY", shortName="TSMC", trailingPE=30.0, forwardPE=22.0,
                pegRatio=1.1, priceToBook=8.0, dividendYield=1.2)

    class _T:
        def __init__(self, _symbol):
            self.info = info

    with mock.patch.object(fe.yf, "Ticker", _T), \
         mock.patch.object(fe, "_fx_rate", lambda a, b: 1.0 if a == b else RATES.get((a, b))):
        evidence = fe.build_valuation_evidence("TSM", AS_OF)
    fe._shared_snapshot.cache_clear()
    assert evidence.market_cap.currency == "USD"    # was the statements' TWD
    assert round(evidence.ev_to_sales.value, 2) == 16.57
    assert "ev_sales" in evidence.tier.signals and "ev_ebit" in evidence.tier.signals
