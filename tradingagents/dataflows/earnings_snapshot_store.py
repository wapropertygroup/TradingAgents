"""Point-in-time storage for earnings estimates.

Consensus estimates are an **observed series**, not a cache. Nothing upstream
sells back last month's consensus: Yahoo's ``eps_trend`` gives 7/30/60/90-day
lookbacks for *today*, and asked on a later date it gives lookbacks from that
later date. So a question about a past ``trade_date`` cannot be answered from a
live endpoint, and the failure is silent — today's numbers are shaped exactly
like a measurement taken then.

That makes this store the load-bearing piece of the whole feature. Rules:

**Reads never see the future.** :meth:`latest_at_or_before` selects the newest
observation dated at or before the requested date, full stop. A run for
2026-03-01 executed today reads the March snapshot or nothing.

**A newer schema is never downgraded.** The SQL layout is versioned in
``PRAGMA user_version`` and the payload shape in a ``schema_version`` column.
A database written by a *newer* build raises :class:`SnapshotStoreError`
rather than being migrated backwards or dropped — this file lives in the user's
cache directory, and a series that cannot be refetched must not be destroyed to
make an older binary happy. A payload at a version this build does not
understand is simply invisible to reads and left in place.

**Same-day writes converge; different-day writes accumulate.** Estimates are
re-fetched several times a day under a TTL, and writing each one as a new row
would fabricate intraday history that the source does not have. So the
``(symbol, observed_date, schema_version)`` key upserts, matching the daily
snapshot pattern used elsewhere in this project, and the accumulating axis is
the date.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .earnings_models import SCHEMA_VERSION, EarningsEvidence, safe_date

logger = logging.getLogger(__name__)

#: Layout version of the SQL itself. Bump only for a table/index change.
DB_SCHEMA_VERSION = 1

SNAPSHOT_FILENAME = "earnings_snapshots.sqlite3"

#: SQLite's own lock wait, plus our retry loop on top. Two gunicorn workers and
#: a background warm thread can all reach this file, and a bare ``connect``
#: raises "database is locked" after five seconds by default.
_BUSY_TIMEOUT_S = 10.0
_WRITE_RETRIES = 4
_WRITE_BACKOFF_S = 0.15

_INIT_LOCK = threading.Lock()


class SnapshotStoreError(RuntimeError):
    """The snapshot database is unusable and must not be silently bypassed."""


class EarningsSnapshotStore:
    """Append-only, date-keyed store of normalized earnings evidence."""

    def __init__(self, path: str | os.PathLike[str], *, payload_version: int = SCHEMA_VERSION):
        self.path = Path(path)
        self.payload_version = payload_version
        self._initialized = False

    # -- connection management -------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise SnapshotStoreError(f"cannot open {self.path}: {exc}") from exc
        try:
            conn.row_factory = sqlite3.Row
            # WAL lets a reader proceed while a writer holds the write lock,
            # which is the actual concurrency shape here: many reads on the
            # request path, one background writer.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                # A read-only filesystem or a non-WAL-capable mount. Journaling
                # mode is an optimisation; failing the whole call over it would
                # take out a working store.
                logger.debug("earnings snapshot store: WAL unavailable at %s", self.path)
            conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create the table, or refuse a database from a newer build."""
        if self._initialized:
            return
        with _INIT_LOCK:
            if self._initialized:
                return
            with self._connect() as conn:
                found = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if found > DB_SCHEMA_VERSION:
                    raise SnapshotStoreError(
                        f"{self.path} was written by a newer build "
                        f"(layout v{found} > v{DB_SCHEMA_VERSION}). Refusing to "
                        "downgrade it. Upgrade tradingagents, or move the file "
                        "aside — do not delete it, it holds observations that "
                        "cannot be refetched."
                    )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS earnings_snapshots (
                        symbol         TEXT NOT NULL,
                        observed_date  TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        observed_at    TEXT NOT NULL,
                        source         TEXT,
                        payload        TEXT NOT NULL,
                        PRIMARY KEY (symbol, observed_date, schema_version)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date
                    ON earnings_snapshots (symbol, schema_version, observed_date DESC)
                    """
                )
                if found < DB_SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version={DB_SCHEMA_VERSION}")
                conn.commit()
            self._initialized = True

    # -- writes ----------------------------------------------------------

    def append(
        self,
        evidence: EarningsEvidence,
        *,
        observed_date: str | None = None,
        observed_at: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Record ``evidence`` as the observation for ``observed_date``.

        ``observed_date`` defaults to the evidence's own ``as_of``, which is the
        date the caller asked about — correct because a live fetch only ever
        happens for the current date (see the guard in the adapter layer).

        Returns True when a row was written. Only substantive evidence is
        stored: persisting an ``unsupported``/``pit_unavailable`` result would
        later satisfy an as-of lookup with a record of a *failure*, and the
        reader cannot distinguish "we observed no coverage that day" from "we
        failed to reach the vendor that day".
        """
        if evidence.status in {"unsupported", "pit_unavailable", "no_coverage"}:
            return False
        if not evidence.periods:
            return False

        date_key = safe_date(observed_date or evidence.as_of)
        if date_key is None:
            raise ValueError(
                f"cannot store a snapshot without a resolvable date "
                f"(observed_date={observed_date!r}, as_of={evidence.as_of!r})"
            )

        symbol = _normalize_key(evidence.symbol)
        stamp = observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)

        self._ensure_schema()
        last: sqlite3.Error | None = None
        for attempt in range(_WRITE_RETRIES):
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO earnings_snapshots
                            (symbol, observed_date, schema_version, observed_at, source, payload)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, observed_date, schema_version)
                        DO UPDATE SET
                            observed_at = excluded.observed_at,
                            source      = excluded.source,
                            payload     = excluded.payload
                        """,
                        (
                            symbol,
                            date_key,
                            self.payload_version,
                            stamp,
                            source or ",".join(evidence.sources) or None,
                            payload,
                        ),
                    )
                    conn.commit()
                return True
            except sqlite3.OperationalError as exc:
                # "database is locked" under a concurrent writer. Retry rather
                # than lose the observation, which cannot be re-derived.
                last = exc
                if "locked" not in str(exc).lower() or attempt == _WRITE_RETRIES - 1:
                    break
                time.sleep(_WRITE_BACKOFF_S * (2**attempt))
            except sqlite3.Error as exc:
                last = exc
                break
        raise SnapshotStoreError(
            f"could not write earnings snapshot for {symbol} on {date_key}: {last}"
        ) from last

    # -- reads -----------------------------------------------------------

    def latest_at_or_before(self, symbol: str, as_of: str) -> EarningsEvidence | None:
        """The newest observation dated at or before ``as_of``, or ``None``.

        This is the whole no-look-ahead guarantee. The comparison is on
        ``observed_date`` strings, which are normalized ``YYYY-MM-DD`` and so
        sort lexicographically in date order.
        """
        date_key = safe_date(as_of)
        if date_key is None:
            return None
        row = self._select_one(
            """
            SELECT payload, observed_date, observed_at FROM earnings_snapshots
            WHERE symbol = ? AND schema_version = ? AND observed_date <= ?
            ORDER BY observed_date DESC LIMIT 1
            """,
            (_normalize_key(symbol), self.payload_version, date_key),
        )
        if row is None:
            return None
        return self._decode(row, symbol)

    def exact(self, symbol: str, observed_date: str) -> EarningsEvidence | None:
        """The observation recorded on exactly ``observed_date``, or ``None``."""
        date_key = safe_date(observed_date)
        if date_key is None:
            return None
        row = self._select_one(
            """
            SELECT payload, observed_date, observed_at FROM earnings_snapshots
            WHERE symbol = ? AND schema_version = ? AND observed_date = ?
            """,
            (_normalize_key(symbol), self.payload_version, date_key),
        )
        if row is None:
            return None
        return self._decode(row, symbol)

    def observed_dates(self, symbol: str) -> list[str]:
        """Every date this build has an observation for, oldest first."""
        self._ensure_schema()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT observed_date FROM earnings_snapshots
                    WHERE symbol = ? AND schema_version = ?
                    ORDER BY observed_date ASC
                    """,
                    (_normalize_key(symbol), self.payload_version),
                ).fetchall()
        except sqlite3.Error as exc:
            raise SnapshotStoreError(f"cannot read {self.path}: {exc}") from exc
        return [str(r["observed_date"]) for r in rows]

    def _select_one(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        self._ensure_schema()
        try:
            with self._connect() as conn:
                return conn.execute(sql, params).fetchone()
        except sqlite3.Error as exc:
            raise SnapshotStoreError(f"cannot read {self.path}: {exc}") from exc

    def _decode(self, row: sqlite3.Row, symbol: str) -> EarningsEvidence | None:
        """Rehydrate a stored payload, tolerating one that has gone bad.

        A corrupt row returns ``None`` — which the caller reports as "no
        point-in-time coverage" — rather than raising. The alternative is that
        one unreadable row makes every historical run for that symbol fail,
        and the row is still there to inspect afterwards.
        """
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError) as exc:
            logger.warning(
                "earnings snapshot for %s on %s is not valid JSON (%s); treating as absent",
                symbol, row["observed_date"], exc,
            )
            return None
        try:
            evidence = EarningsEvidence.from_dict(payload)
        except ValueError as exc:
            logger.warning(
                "earnings snapshot for %s on %s does not match the evidence schema (%s)",
                symbol, row["observed_date"], exc,
            )
            return None
        from dataclasses import replace

        # Stamp the retrieval provenance so a report built from a stored
        # observation says so, and dates it. Without this a point-in-time run
        # reads as though it fetched live.
        note = (
            f"point-in-time snapshot observed {row['observed_date']} "
            f"(recorded {row['observed_at']})"
        )
        sources = list(evidence.sources)
        if note not in sources:
            sources.append(note)
        return replace(evidence, sources=sources)


def _normalize_key(symbol: str) -> str:
    """Uppercase, trimmed symbol. Case-folding here so ``aapl`` and ``AAPL``
    are one series rather than two half-length ones."""
    return (symbol or "").strip().upper()


# ---------------------------------------------------------------------------
# Config-bound convenience
# ---------------------------------------------------------------------------

_DEFAULT_STORE: EarningsSnapshotStore | None = None
_DEFAULT_STORE_PATH: Path | None = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> EarningsSnapshotStore:
    """The store under the configured ``data_cache_dir``.

    Rebuilt when the configured directory changes, so a test that repoints
    ``data_cache_dir`` is not served the previous run's database.
    """
    from .config import get_config

    path = Path(get_config()["data_cache_dir"]) / SNAPSHOT_FILENAME
    global _DEFAULT_STORE, _DEFAULT_STORE_PATH
    with _DEFAULT_LOCK:
        if _DEFAULT_STORE is None or path != _DEFAULT_STORE_PATH:
            _DEFAULT_STORE = EarningsSnapshotStore(path)
            _DEFAULT_STORE_PATH = path
        return _DEFAULT_STORE


# ---------------------------------------------------------------------------
# Reconstructing a revision history from local vintages
# ---------------------------------------------------------------------------

#: Lookback horizons, in days, that a stored vintage may stand in for.
BACKFILL_HORIZONS: tuple[int, ...] = (7, 30, 60, 90)

#: A vintage may fill a horizon only if its true age is within this multiple of
#: the horizon. Without a ceiling, the newest snapshot older than 90 days would
#: fill the "90 days ago" slot however old it actually was, and a two-year-old
#: observation would be presented as a quarter's revision. Locked by test.
BACKFILL_MAX_AGE_MULTIPLE = 2.0


def backfill_trend_from_snapshots(
    evidence: EarningsEvidence,
    canonical: str,
    *,
    horizons: tuple[int, ...] = BACKFILL_HORIZONS,
    store: EarningsSnapshotStore | None = None,
) -> EarningsEvidence:
    """Fill missing EPS/revenue lookbacks from this installation's own vintages.

    Some vendors publish a consensus figure with no history behind it — 同花顺 is
    the clear case, and no free source publishes a *revenue* revision series at
    all. But this project records a dated snapshot on every run, so the history
    those vendors lack accrues locally. After a month of daily runs an A-share
    has a real 30-day revision, measured rather than guessed.

    Three rules keep that from becoming a fabrication:

    * **A vendor-supplied horizon is never overwritten.** Only slots that are
      currently unavailable are filled, so Yahoo's own ``30daysAgo`` always wins
      over a local vintage.
    * **The true age travels with the value.** A horizon filled from a 44-day-old
      snapshot says 44 days in its ``source`` and ``as_of``, and the substitution
      is listed in ``warnings``. Labelling it a flat "30 days" would misstate the
      window the change was measured over.
    * **A vintage that is too old for the slot is not used at all**, bounded by
      :data:`BACKFILL_MAX_AGE_MULTIPLE`. The nearest older snapshot is not
      automatically a fair proxy.
    """
    from dataclasses import replace as _replace
    from datetime import date as _date, timedelta as _timedelta

    from .earnings_models import EstimateTrend

    if not evidence.periods:
        return evidence
    try:
        as_of = _date.fromisoformat(evidence.as_of)
    except ValueError:
        return evidence

    active = store or default_store()
    slots = {7: "days_ago_7", 30: "days_ago_30", 60: "days_ago_60", 90: "days_ago_90"}

    periods = dict(evidence.periods)
    substitutions: list[str] = []

    for horizon in sorted(horizons):
        attribute = slots.get(horizon)
        if attribute is None:
            continue
        target = (as_of - _timedelta(days=horizon)).isoformat()
        try:
            vintage = active.latest_at_or_before(canonical, target)
        except SnapshotStoreError as exc:
            logger.info("snapshot backfill unavailable for %s: %s", canonical, exc)
            return evidence
        if vintage is None:
            continue

        observed = _observed_date(vintage)
        if observed is None:
            continue
        try:
            age_days = (as_of - _date.fromisoformat(observed)).days
        except ValueError:
            continue
        if age_days <= 0 or age_days > horizon * BACKFILL_MAX_AGE_MULTIPLE:
            continue

        for key, period in list(periods.items()):
            prior = vintage.periods.get(key)
            if prior is None:
                continue
            updated = period
            for field_name in ("eps", "revenue"):
                trend: EstimateTrend = getattr(updated, field_name)
                existing = getattr(trend, attribute)
                if existing.available:
                    continue  # never overwrite a vendor-published horizon
                source_value = getattr(prior, field_name).current
                if not source_value.available:
                    continue
                updated = _replace(
                    updated,
                    **{
                        field_name: _replace(
                            trend,
                            **{
                                attribute: _replace(
                                    source_value,
                                    source=(
                                        f"local point-in-time snapshot observed "
                                        f"{observed} ({age_days} days before "
                                        f"{evidence.as_of})"
                                    ),
                                    as_of=observed,
                                )
                            },
                        )
                    },
                )
                substitutions.append(
                    f"{key} {field_name} '{horizon}-day' lookback filled from a local "
                    f"snapshot observed {observed}, actually {age_days} days earlier"
                )
            periods[key] = updated

    if not substitutions:
        return evidence

    warnings = list(evidence.warnings)
    note = (
        "Some revision horizons were reconstructed from this installation's own "
        "dated snapshots rather than a vendor-published history, because the vendor "
        "publishes none. Each substitution's true age is stated with the value; the "
        "nominal horizon label is approximate. Substitutions: "
        + "; ".join(substitutions)
    )
    if note not in warnings:
        warnings.append(note)
    return _replace(evidence, periods=periods, warnings=warnings)


def _observed_date(evidence: EarningsEvidence) -> str | None:
    """The date stamped onto a snapshot by :meth:`_decode`."""
    import re

    for source in evidence.sources:
        match = re.search(r"observed (\d{4}-\d{2}-\d{2})", source)
        if match:
            return match.group(1)
    return None
