"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

``extract_rating`` returns ``None`` when no rating can be found, and every
caller turns that into ``REVIEW`` rather than a tradeable position: a decision
nobody can read is not a Hold, and a Hold recorded in its place is quoted back to
the next run as a call that was never made (#1170).
"""

from __future__ import annotations

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# Signal emitted when the model's decision has no recognizable rating. It is not
# a tradeable position: it flags output that needs a human/re-run rather than
# silently degrading to Hold. Callers that map the signal onto the 5-tier enum
# (e.g. ``PortfolioRating(signal)``) should guard with ``is_review`` first.
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating — **X**" — tolerates markdown
# bold wrappers and any dash or colon a model writes as the separator.
_RATING_LABEL_RE = re.compile(r"rating\b[^:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
                              re.IGNORECASE)

# A line presenting the scale rather than a decision ("Rating Scale: Buy, ...").
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

# Standalone 5-tier word anywhere (word boundaries so "Buyer"/"Holding" don't match).
_RATING_WORD_RE = re.compile(
    r"\b(" + "|".join(RATINGS_5_TIER) + r")\b", re.IGNORECASE
)

# The Chinese words for the five steps, each naming exactly one. The first of
# each is what a Chinese report renders (report_language.py); the rest are what a
# model writing Chinese in free text reaches for.
_ZH_RATING_WORDS: tuple[tuple[str, str], ...] = (
    ("买入", "Buy"),
    ("增持", "Overweight"), ("超配", "Overweight"), ("加仓", "Overweight"),
    ("持有", "Hold"), ("中性", "Hold"), ("标配", "Hold"), ("观望", "Hold"),
    ("减持", "Underweight"), ("低配", "Underweight"), ("减仓", "Underweight"),
    ("卖出", "Sell"),
)

# "**评级**：减持", as a Chinese run's Portfolio Manager decision renders, or a
# model's own "最终评级：增持". Only a label that opens its line counts, unlike
# the English one: a mid-sentence "研究经理给出的评级：增持" quotes somebody
# else's call, and Chinese has no capital letter to tell a heading from prose.
_ZH_RATING_LABEL_RE = re.compile(
    r"^[\s>#*\-]*(?:最终|投资|综合)?评级(?:建议|结论|意见)?[\s*]*[:\-\u2010-\u2015][\s*]*(\w+)"
)

# The Chinese for a line presenting the scale ("评级标准：买入、增持……").
_ZH_RATING_SCALE_RE = re.compile(r"评级(?:标准|说明|体系|选项|定义|量表|尺度)")


def _canonical(word: str) -> str | None:
    """The 5-tier rating a labelled word names, in English, or ``None``.

    ``\\w+`` runs on through Chinese, so "减持建议" is read by its first word.
    """
    if word.lower() in _RATING_SET:
        return word.capitalize()
    for zh, rating in _ZH_RATING_WORDS:
        if word.startswith(zh):
            return rating
    return None


def extract_rating(text: str) -> str | None:
    """Extract a 5-tier rating from prose, or ``None`` if none is present.

    Two-pass strategy on the NFKC-normalized text (so fullwidth punctuation like
    ``Rating：Overweight`` is matched the same as ASCII):
    1. An explicit "Rating: X" label (tolerant of markdown bold), or its Chinese
       form "评级：减持". Either label may carry either language's word, and the
       answer is always the English step.
    2. The first standalone 5-tier rating word found anywhere.

    The second pass is English only. The Chinese words are also everyday verbs
    (持有 is "to own", 买入 "to buy into"), so a lone one in prose is not a call.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # The labelled rating, taking the last one written: a decision states its
    # rating after discussing the alternatives. Lines presenting the scale
    # itself are a legend the model echoed, not a call.
    labelled = None
    for line in norm.splitlines():
        if _RATING_SCALE_RE.search(line) or _ZH_RATING_SCALE_RE.search(line):
            continue
        for label_re in (_RATING_LABEL_RE, _ZH_RATING_LABEL_RE):
            m = label_re.search(line)
            rating = _canonical(m.group(1)) if m else None
            if rating:
                labelled = rating
                break
    if labelled:
        return labelled

    # No label. A single rating word in the text is the call; several are an
    # argument, and picking one of them reports a direction nobody decided --
    # prose that rejects a Buy before concluding Underweight read as Buy.
    named = {m.group(1).capitalize() for m in _RATING_WORD_RE.finditer(norm)}
    return named.pop() if len(named) == 1 else None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """Extract a 5-tier rating, or ``REVIEW`` when the decision has none.

    For callers that need a string for every decision, such as the memory log's
    entry tag. The default is the review sentinel, never a tradeable rating.
    """
    rating = extract_rating(text)
    return rating if rating is not None else default


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW sentinel (#1170)."""
    return signal == RATING_REVIEW
