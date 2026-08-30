"""Point-in-time snapshot store: no look-ahead, append-only, schema-safe.

The store is the load-bearing piece of the earnings feature, because the failure
it prevents is silent: today's consensus answering a question about a past date
looks exactly like a measurement taken then.
"""

import json
import sqlite3
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tradingagents.dataflows.earnings_models import (
    EarningsEvidence,
    EstimateTrend,
    FiscalPeriod,
    PeriodEvidence,
    RevisionBreadth,
    Value,
    finalize_evidence,
)
from tradingagents.dataflows.earnings_snapshot_store import (
    BACKFILL_MAX_AGE_MULTIPLE,
    DB_SCHEMA_VERSION,
    EarningsSnapshotStore,
    SnapshotStoreError,
    backfill_trend_from_snapshots,
)


def _evidence(symbol="AAPL", as_of="2026-04-01", eps=8.70, revenue=None, key="0y"):
    revenue_trend = EstimateTrend(
        current=(
            Value(revenue, unit="currency_large", currency="USD")
            if revenue is not None
            else Value.missing("absent", unit="currency_large")
        ),
        days_ago_7=Value.missing("absent", unit="currency_large"),
        days_ago_30=Value.missing("absent", unit="currency_large"),
        days_ago_60=Value.missing("absent", unit="currency_large"),
        days_ago_90=Value.missing("absent", unit="currency_large"),
    )
    period = PeriodEvidence(
        period=FiscalPeriod(key=key, end_date="2026-12-31"),
        eps=EstimateTrend(
            current=Value(eps, currency="USD"),
            days_ago_7=Value.missing("absent"),
            days_ago_30=Value.missing("absent"),
            days_ago_60=Value.missing("absent"),
            days_ago_90=Value.missing("absent"),
        ),
        revenue=revenue_trend,
        breadth=RevisionBreadth(),
        analyst_count=Value(37, unit="count"),
    )
    return finalize_evidence(EarningsEvidence(
        symbol=symbol, as_of=as_of, periods={key: period}, sources=["yfinance"],
    ))


class StoreBasicsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "earnings.sqlite3"
        self.store = EarningsSnapshotStore(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_read_back(self):
        self.assertTrue(self.store.append(_evidence(as_of="2026-04-01"),
                                          observed_date="2026-04-01"))
        found = self.store.exact("AAPL", "2026-04-01")
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found.periods["0y"].eps.current.value, 8.70)

    def test_history_accumulates_by_date(self):
        for day, eps in (("2026-02-01", 8.4), ("2026-03-01", 8.5), ("2026-04-01", 8.7)):
            self.store.append(_evidence(as_of=day, eps=eps), observed_date=day)
        self.assertEqual(
            self.store.observed_dates("AAPL"),
            ["2026-02-01", "2026-03-01", "2026-04-01"],
        )

    def test_same_day_writes_converge_rather_than_duplicating(self):
        """Estimates are refetched several times a day under a TTL.

        Writing each as a new row would fabricate intraday history the source
        does not have, so a same-day rewrite replaces.
        """
        self.store.append(_evidence(as_of="2026-04-01", eps=8.70), observed_date="2026-04-01")
        self.store.append(_evidence(as_of="2026-04-01", eps=8.71), observed_date="2026-04-01")
        self.assertEqual(self.store.observed_dates("AAPL"), ["2026-04-01"])
        self.assertAlmostEqual(
            self.store.exact("AAPL", "2026-04-01").periods["0y"].eps.current.value, 8.71
        )

    def test_identical_write_twice_is_idempotent(self):
        payload = _evidence(as_of="2026-04-01")
        self.store.append(payload, observed_date="2026-04-01")
        self.store.append(payload, observed_date="2026-04-01")
        self.assertEqual(len(self.store.observed_dates("AAPL")), 1)

    def test_symbol_case_is_folded_into_one_series(self):
        self.store.append(_evidence(symbol="aapl"), observed_date="2026-04-01")
        self.assertIsNotNone(self.store.latest_at_or_before("AAPL", "2026-04-01"))
        self.assertIsNotNone(self.store.latest_at_or_before("AaPl", "2026-04-01"))

    def test_distinct_symbols_do_not_bleed(self):
        self.store.append(_evidence(symbol="AAPL"), observed_date="2026-04-01")
        self.assertIsNone(self.store.latest_at_or_before("MSFT", "2026-04-01"))


class NoLookAheadTests(unittest.TestCase):
    """The single guarantee the whole feature rests on."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = EarningsSnapshotStore(Path(self._tmp.name) / "e.sqlite3")
        for day, eps in (("2026-02-01", 8.4), ("2026-04-01", 8.7), ("2026-06-01", 9.1)):
            self.store.append(_evidence(as_of=day, eps=eps), observed_date=day)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exact_date_hit(self):
        self.assertAlmostEqual(
            self.store.latest_at_or_before("AAPL", "2026-04-01").periods["0y"].eps.current.value,
            8.7,
        )

    def test_between_dates_returns_the_older_observation(self):
        self.assertAlmostEqual(
            self.store.latest_at_or_before("AAPL", "2026-05-15").periods["0y"].eps.current.value,
            8.7,
        )

    def test_a_later_observation_never_leaks_backwards(self):
        for as_of, expected in (("2026-02-01", 8.4), ("2026-03-31", 8.4), ("2026-05-31", 8.7)):
            with self.subTest(as_of=as_of):
                got = self.store.latest_at_or_before("AAPL", as_of)
                self.assertAlmostEqual(got.periods["0y"].eps.current.value, expected)

    def test_before_the_first_observation_is_none_not_the_earliest(self):
        self.assertIsNone(self.store.latest_at_or_before("AAPL", "2026-01-31"))

    def test_unparseable_date_is_none(self):
        self.assertIsNone(self.store.latest_at_or_before("AAPL", "not-a-date"))

    def test_retrieved_evidence_is_stamped_with_its_observation_date(self):
        """A report built from a vintage must say so, and date it."""
        got = self.store.latest_at_or_before("AAPL", "2026-05-15")
        joined = " ".join(got.sources)
        self.assertIn("point-in-time snapshot observed 2026-04-01", joined)


class WhatIsWorthStoringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = EarningsSnapshotStore(Path(self._tmp.name) / "e.sqlite3")

    def tearDown(self):
        self._tmp.cleanup()

    def test_terminal_statuses_are_not_stored(self):
        """Storing a failure would later satisfy an as-of lookup with it.

        A reader cannot then distinguish "we observed no coverage that day" from
        "we could not reach the vendor that day".
        """
        for factory in (
            EarningsEvidence.unsupported,
            EarningsEvidence.pit_unavailable,
            EarningsEvidence.no_coverage,
        ):
            with self.subTest(factory=factory.__name__):
                self.assertFalse(
                    self.store.append(factory("SPY", "2026-04-01", "reason"),
                                      observed_date="2026-04-01")
                )
        self.assertEqual(self.store.observed_dates("SPY"), [])

    def test_evidence_with_no_periods_is_not_stored(self):
        empty = EarningsEvidence(symbol="AAPL", as_of="2026-04-01")
        self.assertFalse(self.store.append(empty, observed_date="2026-04-01"))

    def test_unresolvable_observation_date_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            self.store.append(_evidence(as_of="garbage"), observed_date="garbage")


class SchemaVersionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "e.sqlite3"

    def tearDown(self):
        self._tmp.cleanup()

    def test_payload_versions_are_isolated_from_each_other(self):
        EarningsSnapshotStore(self.path).append(_evidence(), observed_date="2026-04-01")
        future = EarningsSnapshotStore(self.path, payload_version=999)
        self.assertIsNone(future.latest_at_or_before("AAPL", "2026-12-01"))
        self.assertEqual(future.observed_dates("AAPL"), [])

    def test_an_unreadable_payload_version_is_left_in_place_not_deleted(self):
        EarningsSnapshotStore(self.path).append(_evidence(), observed_date="2026-04-01")
        EarningsSnapshotStore(self.path, payload_version=999).latest_at_or_before(
            "AAPL", "2026-12-01"
        )
        # The v1 row must still be there for a build that understands it.
        self.assertEqual(
            EarningsSnapshotStore(self.path).observed_dates("AAPL"), ["2026-04-01"]
        )

    def test_a_newer_layout_is_refused_rather_than_downgraded(self):
        """The file holds observations that cannot be refetched."""
        EarningsSnapshotStore(self.path).append(_evidence(), observed_date="2026-04-01")
        with sqlite3.connect(self.path) as conn:
            conn.execute(f"PRAGMA user_version={DB_SCHEMA_VERSION + 5}")
        with self.assertRaises(SnapshotStoreError) as ctx:
            EarningsSnapshotStore(self.path).latest_at_or_before("AAPL", "2026-12-01")
        message = str(ctx.exception)
        self.assertIn("newer build", message)
        self.assertIn("do not delete", message.lower())
        self.assertTrue(self.path.exists(), "refusing must not remove the file")

    def test_layout_version_is_stamped_on_creation(self):
        EarningsSnapshotStore(self.path).append(_evidence(), observed_date="2026-04-01")
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(
                int(conn.execute("PRAGMA user_version").fetchone()[0]), DB_SCHEMA_VERSION
            )


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "e.sqlite3"
        self.store = EarningsSnapshotStore(self.path)
        self.store.append(_evidence(as_of="2026-02-01", eps=8.4), observed_date="2026-02-01")
        self.store.append(_evidence(as_of="2026-04-01", eps=8.7), observed_date="2026-04-01")

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_corrupt_row_reads_as_absent_rather_than_failing_every_run(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE earnings_snapshots SET payload='{not json' WHERE observed_date=?",
                ("2026-04-01",),
            )
        fresh = EarningsSnapshotStore(self.path)
        self.assertIsNone(fresh.latest_at_or_before("AAPL", "2026-04-01"))
        # The undamaged earlier vintage is still usable.
        self.assertIsNotNone(fresh.latest_at_or_before("AAPL", "2026-02-01"))

    def test_a_corrupt_row_is_not_deleted(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE earnings_snapshots SET payload='{not json' WHERE observed_date=?",
                ("2026-04-01",),
            )
        EarningsSnapshotStore(self.path).latest_at_or_before("AAPL", "2026-04-01")
        self.assertIn("2026-04-01", EarningsSnapshotStore(self.path).observed_dates("AAPL"))

    def test_a_payload_that_is_valid_json_but_wrong_shape_reads_as_absent(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE earnings_snapshots SET payload=? WHERE observed_date=?",
                (json.dumps(["not", "an", "object"]), "2026-04-01"),
            )
        self.assertIsNone(
            EarningsSnapshotStore(self.path).latest_at_or_before("AAPL", "2026-04-01")
        )


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_writers_all_land(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.sqlite3"
            errors: list[Exception] = []

            def write(day_index: int) -> None:
                try:
                    day = f"2026-04-{day_index + 1:02d}"
                    EarningsSnapshotStore(path).append(
                        _evidence(as_of=day, eps=8.0 + day_index * 0.01),
                        observed_date=day,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded and asserted below
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(i,)) for i in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [], f"concurrent writes raised: {errors}")
            self.assertEqual(len(EarningsSnapshotStore(path).observed_dates("AAPL")), 12)

    def test_parallel_writers_to_the_same_date_converge_without_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.sqlite3"
            errors: list[Exception] = []

            def write(i: int) -> None:
                try:
                    EarningsSnapshotStore(path).append(
                        _evidence(as_of="2026-04-01", eps=8.0 + i * 0.01),
                        observed_date="2026-04-01",
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(EarningsSnapshotStore(path).observed_dates("AAPL"), ["2026-04-01"])


class BackfillTests(unittest.TestCase):
    """Reconstructing a revision history the vendor does not publish."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = EarningsSnapshotStore(Path(self._tmp.name) / "e.sqlite3")

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_thirty_day_old_vintage_fills_the_thirty_day_slot(self):
        self.store.append(_evidence(as_of="2026-04-01", eps=8.40, revenue=4.5e11),
                          observed_date="2026-04-01")
        today = _evidence(as_of="2026-05-01", eps=8.70, revenue=4.7e11)
        filled = backfill_trend_from_snapshots(today, "AAPL", store=self.store)
        eps30 = filled.periods["0y"].eps.days_ago_30
        self.assertTrue(eps30.available)
        self.assertAlmostEqual(eps30.value, 8.40)
        self.assertAlmostEqual(filled.periods["0y"].revenue.days_ago_30.value, 4.5e11)

    def test_the_true_age_travels_with_the_value_and_is_warned_about(self):
        self.store.append(_evidence(as_of="2026-03-18", eps=8.40), observed_date="2026-03-18")
        filled = backfill_trend_from_snapshots(
            _evidence(as_of="2026-05-01", eps=8.70), "AAPL", store=self.store
        )
        eps30 = filled.periods["0y"].eps.days_ago_30
        self.assertEqual(eps30.as_of, "2026-03-18")
        self.assertIn("44 days", eps30.source)
        self.assertTrue(any("actually 44 days earlier" in w for w in filled.warnings))

    def test_a_vendor_published_horizon_is_never_overwritten(self):
        self.store.append(_evidence(as_of="2026-04-01", eps=8.40), observed_date="2026-04-01")
        today = _evidence(as_of="2026-05-01", eps=8.70)
        vendor = Value(8.55, currency="USD", source="yfinance")
        today = replace(today, periods={"0y": replace(
            today.periods["0y"],
            eps=replace(today.periods["0y"].eps, days_ago_30=vendor),
        )})
        filled = backfill_trend_from_snapshots(today, "AAPL", store=self.store)
        self.assertAlmostEqual(filled.periods["0y"].eps.days_ago_30.value, 8.55)
        self.assertEqual(filled.periods["0y"].eps.days_ago_30.source, "yfinance")

    def test_a_vintage_too_old_for_its_slot_is_refused(self):
        """The nearest older snapshot is not automatically a fair proxy."""
        self.assertEqual(BACKFILL_MAX_AGE_MULTIPLE, 2.0)
        # 200 days old, asked to stand in for the 30-day slot (ceiling is 60).
        self.store.append(_evidence(as_of="2025-10-13", eps=7.0), observed_date="2025-10-13")
        filled = backfill_trend_from_snapshots(
            _evidence(as_of="2026-05-01", eps=8.70), "AAPL", store=self.store
        )
        self.assertFalse(filled.periods["0y"].eps.days_ago_30.available)
        self.assertEqual(filled.warnings, [])

    def test_no_vintage_leaves_the_evidence_untouched(self):
        original = _evidence(as_of="2026-05-01", eps=8.70)
        self.assertIs(
            backfill_trend_from_snapshots(original, "AAPL", store=self.store), original
        )

    def test_one_vintage_alone_is_not_enough_to_score(self):
        """A single monthly vintage fills only the 30-day slot: 0.35 weight.

        That is below the 0.50 floor, so the first month of A-share runs still
        reports Insufficient Data. Worth pinning because the opposite assumption —
        "one snapshot unlocks momentum" — would make the feature look broken.
        """
        from tradingagents.dataflows.earnings_models import MOMENTUM_WEIGHTS

        self.store.append(_evidence(symbol="600519", as_of="2026-04-01", eps=64.0),
                          observed_date="2026-04-01")
        today = _evidence(symbol="600519", as_of="2026-05-01", eps=67.85)
        filled = finalize_evidence(
            backfill_trend_from_snapshots(today, "600519", store=self.store)
        )
        self.assertTrue(filled.periods["0y"].eps.days_ago_30.available)
        self.assertAlmostEqual(filled.momentum.available_weight, MOMENTUM_WEIGHTS["eps_30d"])
        self.assertEqual(filled.momentum.band, "Insufficient Data")

    def test_two_vintages_make_a_no_history_vendor_scorable(self):
        """The point of the mechanism: A-share momentum accrues locally.

        The 7-day and 30-day slots together are 0.15 + 0.35 = 0.50, exactly the
        floor, so roughly a month of daily runs is what it takes for a 同花顺-only
        symbol to earn a real momentum band — measured, not guessed.
        """
        from tradingagents.dataflows.earnings_models import compute_momentum

        for day, eps in (("2026-04-01", 64.0), ("2026-04-24", 66.0)):
            self.store.append(
                _evidence(symbol="600519", as_of=day, eps=eps), observed_date=day
            )
        today = _evidence(symbol="600519", as_of="2026-05-01", eps=67.85)
        self.assertEqual(compute_momentum(today.periods["0y"]).band, "Insufficient Data")

        filled = finalize_evidence(
            backfill_trend_from_snapshots(today, "600519", store=self.store)
        )
        self.assertTrue(filled.periods["0y"].eps.days_ago_7.available)
        self.assertTrue(filled.periods["0y"].eps.days_ago_30.available)
        self.assertAlmostEqual(filled.momentum.available_weight, 0.50)
        self.assertNotEqual(filled.momentum.band, "Insufficient Data")
        self.assertGreater(filled.momentum.score, 0)

    def test_a_period_absent_from_the_vintage_is_skipped(self):
        self.store.append(_evidence(as_of="2026-04-01", eps=8.4, key="0y"),
                          observed_date="2026-04-01")
        today = _evidence(as_of="2026-05-01", eps=9.5, key="+1y")
        filled = backfill_trend_from_snapshots(today, "AAPL", store=self.store)
        self.assertFalse(filled.periods["+1y"].eps.days_ago_30.available)


if __name__ == "__main__":
    unittest.main()
