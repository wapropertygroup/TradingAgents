"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision; its free-text fallback is read here)
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
# bold wrappers and any dash or colon a model writes as the separator. "rating"
# must start a word, so "Operating margin: Sell-side" is not a label.
_RATING_LABEL_RE = re.compile(r"(?<![a-z])rating\b[^:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
                              re.IGNORECASE)

# The decision's own rating line, in the shape the Portfolio Manager is asked to
# open with ("- **Rating**: X", "**Final Rating**: X", "## Our rating - X"): an
# optional list marker, emphasis and heading marks, and only words naming the
# decision itself before "rating". "Consensus rating: Buy" or "Trader's rating:
# Buy" is someone else's rating.
_OWN_QUALIFIER = r"(?:(?:final|our|overall|my|recommended|updated|revised|new|current)\s+)*"
_RATING_LINE_RE = re.compile(
    r"\s*(?P<item>(?:[-+*\u2022]|\d+[.)])\s+)?[\s*_#]*" + _OWN_QUALIFIER
    + r"rating[^\w:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(?P<value>\w+)",
    re.IGNORECASE,
)

# A line presenting the scale rather than a decision ("Rating Scale: Buy, ...").
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

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

# The Chinese form of the decision's own rating line: "**评级**：减持", as a
# Chinese run's Portfolio Manager decision renders, or a model's own
# "最终评级：增持". Chinese labels count only in this shape, opening their line:
# a mid-sentence "研究经理给出的评级：增持" quotes somebody else's call, and
# Chinese has no capital letter to tell a heading from prose.
_ZH_RATING_LINE_RE = re.compile(
    r"\s*(?P<item>(?:[-+*\u2022]|\d+[.)])\s+)?[\s*_#>]*(?:最终|投资|综合)?评级(?:建议|结论|意见)?"
    r"[\s*]*[:\-\u2010-\u2015][\s*]*(?P<value>\w+)"
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


def _rating_line(line: str) -> tuple[str, bool] | None:
    """The rating a line in the decision's own rating shape names, in English,
    and whether the line is a list item; ``None`` for any other line."""
    for line_re in (_RATING_LINE_RE, _ZH_RATING_LINE_RE):
        m = line_re.match(line)
        rating = _canonical(m.group("value")) if m else None
        if rating:
            return rating, bool(m.group("item"))
    return None

def extract_rating(text: str) -> str | None:
    """Extract a 5-tier rating from its label, or ``None`` if there is none.

    Reads an explicit "Rating: X" label (tolerant of markdown bold) in the
    NFKC-normalized text, so fullwidth punctuation like ``Rating：Overweight``
    matches as ASCII does: the decision's opening rating line, else its own
    rating lines or, failing those, every label, when they agree.

    A Chinese run's "评级：减持" is a rating line too, read only where it opens
    its line. Either label may carry either language's word, and the answer is
    always the English step.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # The decision is asked to open with its rating, so a first line (after any
    # headings) in that shape is the call. A list item there may also be a quote
    # heading a list of other parties' ratings, so it counts with the decision's
    # other rating lines, which leave out list items; failing those, every label
    # counts. Either way they must agree: ratings that differ, with nothing
    # marking which one is the call, are no call (#1170). Lines presenting the
    # scale itself are a legend the model echoed, not a call.
    lines = [line for line in norm.splitlines()
             if line.strip() and not (_RATING_SCALE_RE.search(line) or _ZH_RATING_SCALE_RE.search(line))]
    first = next((line for line in lines if not line.lstrip().startswith("#")
                  or _RATING_LINE_RE.match(line) or _ZH_RATING_LINE_RE.match(line)), "")
    opening = _rating_line(first)
    if opening and not opening[1]:
        return opening[0]

    own = [opening[0]] if opening else []
    labels = []
    for line in lines:
        line_rating = _rating_line(line)
        if line_rating and not line_rating[1]:
            own.append(line_rating[0])
        labels += [r for r in map(_canonical, _RATING_LABEL_RE.findall(line)) if r]
        # A Chinese label counts wherever it opens its line, a list item's
        # included, as an English one anywhere does just above.
        zh = _ZH_RATING_LINE_RE.match(line)
        rating = _canonical(zh.group("value")) if zh else None
        if rating:
            labels.append(rating)
    # Without a label there is no call to read: a rating word in the prose may be
    # one the text argues against ("not a Sell"), and reading it reports a
    # direction nobody decided.
    found = own or labels
    return found[0] if found and len(set(found)) == 1 else None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """Extract a 5-tier rating, or ``REVIEW`` when the decision has none.

    For callers that need a string for every decision, such as the memory log's
    entry tag. The default is the review sentinel, never a tradeable rating.
    """
    rating = extract_rating(text)
    return rating if rating is not None else default


def run_rating(final_state: dict) -> str:
    """A finished run's rating: the Portfolio Manager's own, else read from its decision.

    The fallback serves a state without ``final_rating``, such as a run an older
    version completed and a checkpoint hands back unchanged.
    """
    return final_state.get("final_rating") or parse_rating(final_state.get("final_trade_decision", ""))


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW sentinel (#1170)."""
    return signal == RATING_REVIEW
