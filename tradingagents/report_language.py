"""Report text that code writes, in the run's output language.

``get_language_instruction()`` asks every agent to write in the configured
output language, but not every line of a report is model prose. The evidence
sections of the Earnings, Quality and Valuation analysts, the headers of the
structured decisions (Research Manager, Trader, Portfolio Manager), the
sentiment band and the risk gate's compliance notice are composed in code, and
until this module they were English whatever the run's language. A Chinese
report read in two languages: its tables, verdicts and data gaps in one, the
analysis around them in the other. The model then quoted the English verdict in
its Chinese prose ("动量评级为Insufficient Data").

Two languages are carried: English, and Simplified Chinese. Any other configured
language keeps the code-written parts in English, as before. A label set
translated into one language and half of another would be worse than a
consistent English one.

What is translated:

- every fixed heading, label, table header and note the renderers write;
- the enumerated verdicts (ratings, sentiment bands, momentum bands, quality and
  valuation tiers, confidence levels), through :meth:`ReportText.term`;
- the data-gap, warning and "unavailable" sentences the evidence adapters emit,
  through :meth:`ReportText.message`: a lookup over the sentences this code base
  actually produces, exact or by pattern. A sentence that is not in it stays
  English rather than being guessed at, because a data gap is a claim about the
  evidence and a loose translation of one is a different claim.

What is not: numbers, tickers, vendor and endpoint names ("Alpha Vantage
EARNINGS_ESTIMATES"), and the team and role headings ``reporting.py`` writes
("### Portfolio Manager"), which consumers split the report on.

A Chinese rating is read back by ``agents/rating.py``, which knows the words
:data:`_TERMS_ZH` gives the five steps. Change one and change the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Which language
# ---------------------------------------------------------------------------

_TRADITIONAL = ("traditional", "繁", "zh-tw", "zh_tw", "zh-hk", "zh_hk",
                "zh-hant", "zh_hant")


def is_chinese(language: str | None) -> bool:
    """Whether ``language`` (an ``output_language`` value) is Simplified Chinese.

    The setting is free text: the CLI offers "Chinese", ystocker sends
    "Simplified Chinese (简体中文)", a user may type "中文" or "zh". Traditional
    Chinese is not carried, and keeps English rather than receiving Simplified
    forms it did not ask for.
    """
    text = str(language or "").strip()
    low = text.casefold()
    if not low or any(mark in low for mark in _TRADITIONAL):
        return False
    return ("chinese" in low or "中文" in text or "简体" in text or "汉语" in text
            or low == "zh" or low.startswith(("zh-", "zh_")))


def configured_language() -> str:
    """The run's ``output_language``, read from the config of the run in progress."""
    from tradingagents.dataflows.config import get_config

    return str(get_config().get("output_language") or "English")


# ---------------------------------------------------------------------------
# Terms: the enumerated verdicts
# ---------------------------------------------------------------------------

_TERMS_ZH: dict[str, str] = {
    # The 5-tier rating and the Trader's three actions. rating.py reads these
    # words back into the English steps.
    "Buy": "买入",
    "Overweight": "增持",
    "Hold": "持有",
    "Underweight": "减持",
    "Sell": "卖出",
    # Sentiment bands.
    "Bullish": "看涨",
    "Mildly Bullish": "温和看涨",
    "Neutral": "中性",
    "Mixed": "分歧",
    "Mildly Bearish": "温和看跌",
    "Bearish": "看跌",
    # Earnings momentum bands ("Neutral" is shared with sentiment).
    "Strong Positive": "强劲正面",
    "Positive": "正面",
    "Negative": "负面",
    "Strong Negative": "强劲负面",
    "Insufficient Data": "数据不足",
    # Quality tiers.
    "High Quality": "高质量",
    "Above Average": "高于平均",
    "Average": "平均",
    "Below Average": "低于平均",
    "Weak": "薄弱",
    # Valuation tiers.
    "Deep Value": "深度价值",
    "Attractive": "有吸引力",
    "Fair": "合理",
    "Expensive": "昂贵",
    "Extreme Premium": "极高溢价",
    # Confidence, which arrives lower-case.
    "low": "低",
    "medium": "中",
    "high": "高",
}

#: The signals in each tier's arithmetic table. English keeps the code's own
#: keys, which name the fields a reader can check against the module.
_SIGNALS_ZH: dict[str, str] = {
    # Earnings momentum.
    "eps_7d": "EPS 一致预期 7 天变化",
    "eps_30d": "EPS 一致预期 30 天变化",
    "eps_90d": "EPS 一致预期 90 天变化",
    "breadth_30d": "30 天修正广度",
    "revenue_30d": "营收一致预期 30 天变化",
    # Quality.
    "roe": "净资产收益率",
    "operating_margin": "营业利润率",
    "debt_to_equity": "债务权益比",
    "current_ratio": "流动比率",
    "fcf_margin": "自由现金流利润率",
    "margin_consistency": "利润率稳定性",
    # Valuation.
    "pe_band": "市盈率区间",
    "peg": "PEG",
    "price_to_book": "市净率",
    "forward_vs_trailing": "远期与历史市盈率对比",
    "dividend_yield": "股息率",
}

#: Source labels the adapters write, where they carry English description.
_SOURCES_ZH: dict[str, str] = {
    "yfinance (Yahoo Finance analyst estimates)": "yfinance（Yahoo Finance 分析师预期）",
    "yfinance (Yahoo Finance fundamentals)": "yfinance（Yahoo Finance 基本面）",
    "同花顺 (10jqka) 一致预期快照": "同花顺（10jqka）一致预期快照",
}

_RELATIVE_PERIODS_ZH: dict[str, str] = {
    "0q": "本季度（相对期间 0q）",
    "+1q": "下季度（相对期间 +1q）",
    "0y": "本财年（相对期间 0y）",
    "+1y": "下一财年（相对期间 +1y）",
}

_CURRENCIES_ZH: dict[str, str] = {
    "USD": "美元", "CNY": "元", "RMB": "元", "HKD": "港元", "JPY": "日元",
    "EUR": "欧元", "GBP": "英镑", "TWD": "新台币", "KRW": "韩元",
    "CAD": "加元", "AUD": "澳元", "CHF": "瑞士法郎", "SGD": "新加坡元",
    "INR": "印度卢比",
}

#: Quote types named in the "not an operating company" sentences.
_QUOTE_TYPES_ZH: dict[str, str] = {
    "etf": "交易所交易基金（ETF）", "mutualfund": "共同基金", "index": "指数",
    "currency": "货币", "cryptocurrency": "加密货币", "future": "期货",
    "option": "期权", "moneymarket": "货币市场基金",
}

# ---------------------------------------------------------------------------
# Messages: the data-gap, warning and "unavailable" sentences
# ---------------------------------------------------------------------------

_MESSAGES_ZH: dict[str, str] = {
    # earnings_models.py
    "7-day EPS consensus trend unavailable": "7 天 EPS 一致预期趋势不可用",
    "30-day EPS consensus trend unavailable": "30 天 EPS 一致预期趋势不可用",
    "90-day EPS consensus trend unavailable": "90 天 EPS 一致预期趋势不可用",
    "30-day analyst revision breadth (up/down counts) unavailable":
        "30 天分析师修正广度（上调/下调家数）不可用",
    "30-day revenue consensus trend unavailable; no free provider publishes a "
    "revenue revision history, so this needs two local point-in-time snapshots":
        "30 天营收一致预期趋势不可用；没有免费数据源发布营收修正历史，"
        "因此需要两份本地时点快照才能计算",
    "not reported": "未报告",
    "not computed": "未计算",
    "malformed serialized calendar": "已序列化的财报日历格式有误",
    "malformed serialized value": "已序列化的数值格式有误",
    "not reported by any consulted provider": "所查询的数据源均未报告",
    "announcement dates were not available, and drift cannot be anchored to a "
    "fiscal quarter end without misdating the market reaction.":
        "没有可用的财报公布日期；若将漂移锚定在财季截止日，会把市场反应的日期弄错。",
    "No detail supplied.": "未提供详情。",
    # alpha_vantage_earnings.py
    "Whisper expectations are unavailable: Alpha Vantage publishes sell-side "
    "consensus only, and no free source publishes buy-side whisper numbers.":
        "耳语预期不可用：Alpha Vantage 只发布卖方一致预期，且没有免费数据源发布买方耳语数字。",
    "Consensus margin revisions are unavailable. Reported margins and management "
    "guidance may be discussed but are not consensus margin revisions.":
        "一致预期利润率修正不可用。可以讨论已报告的利润率和管理层指引，"
        "但它们不是一致预期利润率修正。",
    "not reported by Alpha Vantage": "Alpha Vantage 未报告",
    "Alpha Vantage's earnings calendar carries an EPS estimate only":
        "Alpha Vantage 的财报日历只提供 EPS 预期",
    "Alpha Vantage publishes 7- and 30-day revision counts only":
        "Alpha Vantage 只发布 7 天和 30 天的修正家数",
    "Alpha Vantage EARNINGS_ESTIMATES returned no estimate rows.":
        "Alpha Vantage EARNINGS_ESTIMATES 未返回任何预期数据行。",
    "no announced quarter at or before the analysis date carried a reported date, "
    "so no drift window could be anchored":
        "分析日当日或之前公布的季度均未附公布日期，因此无法锚定漂移窗口",
    "announcement dates were available but the price history did not cover enough "
    "sessions after them to measure any window":
        "有公布日期，但价格历史在其后覆盖的交易日不足，无法测量任何窗口",
    "amc (after market close)": "盘后（收盘后发布）",
    "bmo (before market open)": "盘前（开盘前发布）",
    "amc": "盘后（收盘后发布）",
    "bmo": "盘前（开盘前发布）",
    # yfinance_earnings.py
    "Whisper expectations (buy-side / unofficial estimates) are unavailable: no "
    "free provider publishes them, and news or social sentiment cannot be "
    "converted into a numeric whisper figure.":
        "耳语预期（买方/非官方预期）不可用：没有免费数据源发布此类数据，"
        "新闻或社交情绪也无法换算成数值化的耳语数字。",
    "Consensus margin revisions are unavailable. Reported margin history and "
    "management guidance may be discussed, but neither is a consensus margin "
    "revision and must not be labelled as one.":
        "一致预期利润率修正不可用。可以讨论已报告的利润率历史和管理层指引，"
        "但二者都不是一致预期利润率修正，也不得被标注为此。",
    "Fiscal year end unknown, so annual periods are labelled by Yahoo's relative "
    "keys (0y / +1y) rather than a fiscal year. No FY number is invented from the "
    "calendar year.":
        "财年截止日未知，因此年度期间以 Yahoo 的相对键（0y / +1y）标注，而非具体财年；"
        "不会根据日历年臆造财年编号。",
    "Yahoo returned no upcoming earnings date. It may be unscheduled, or the "
    "issuer may not have confirmed one.":
        "Yahoo 未返回下次财报日期。可能尚未排期，或发行人尚未确认。",
    "Yahoo's earnings-date endpoint returned no usable announcement dates. It is "
    "an HTML scrape rather than a JSON API, so it fails independently of the "
    "estimate tables. Drift is not estimated from fiscal quarter ends: a June "
    "quarter is announced in late July, so that anchor would report weeks of "
    "unrelated trading as the earnings reaction.":
        "Yahoo 的财报日期接口未返回可用的公布日期。该接口是网页抓取而非 JSON API，"
        "因此会独立于预期数据表失败。漂移不以财季截止日估算：6 月季度通常在 7 月下旬"
        "公布，以截止日为锚会把数周无关的交易当作财报反应。",
    "announcement dates were resolved but the price history did not cover enough "
    "sessions after them to measure any window":
        "已解析出公布日期，但价格历史在其后覆盖的交易日不足，无法测量任何窗口",
    "Yahoo publishes no revenue revision history": "Yahoo 不发布营收修正历史",
    "Yahoo publishes 90-day EPS trend but no 90-day revision counts":
        "Yahoo 发布 90 天 EPS 趋势，但不发布 90 天修正家数",
    "benchmark history unavailable": "基准指数历史不可用",
    # fundamentals_evidence.py, quality_models.py, valuation_models.py
    "not reported by Yahoo Finance": "Yahoo Finance 未报告",
    "negative or zero trailing EPS -- no P/E multiple exists to score":
        "历史 EPS 为负或为零——不存在可评分的市盈率",
    "Return on equity unavailable": "净资产收益率不可用",
    "Operating margin unavailable": "营业利润率不可用",
    "Debt-to-equity unavailable": "债务权益比不可用",
    "Current ratio unavailable": "流动比率不可用",
    "Free cash flow or revenue unavailable, so FCF margin could not be computed":
        "自由现金流或营收不可用，因此无法计算自由现金流利润率",
    "Trailing P/E unavailable or undefined (commonly a negative-earnings company)":
        "历史市盈率不可用或无定义（常见于亏损公司）",
    "PEG ratio unavailable": "PEG 比率不可用",
    "Price-to-book unavailable": "市净率不可用",
    "Forward P/E or trailing P/E unavailable, so the comparison could not be "
    "computed":
        "远期或历史市盈率不可用，因此无法比较二者",
    "Dividend yield unavailable": "股息率不可用",
    # a_stock_earnings.py
    "announcement dates are unavailable from this source, and drift anchored to "
    "anything else would misdate the market reaction":
        "该数据源不提供公布日期，而锚定在其他日期会把市场反应的日期弄错",
    "同花顺 publishes a current consensus snapshot with no revision history":
        "同花顺只发布当前一致预期快照，没有修正历史",
    "同花顺 publishes no per-analyst up/down revision counts":
        "同花顺不发布分析师上调/下调家数",
    "covering-institution count not reported": "未报告覆盖机构数",
    "同花顺's consensus table carries EPS only, no revenue estimate":
        "同花顺一致预期表只有 EPS，没有营收预期",
    "同花顺's consensus page carries no scheduled announcement date. 沪深京 issuers "
    "file a 预约披露时间 with their exchange, which this project does not fetch.":
        "同花顺一致预期页面不含预定的披露日期。沪深京发行人会向交易所报送预约披露时间，"
        "本项目未抓取该信息。",
    "同花顺's forecast table carried the expected columns but no row resolved to a "
    "current or future fiscal year with a usable mean estimate.":
        "同花顺预测表包含所需的列，但没有任何一行对应到带可用均值预期的当前或未来财年。",
    "No consensus revision history (7 / 30 / 60 / 90 day) is published for "
    "A-shares by this source. Horizons shown as available were reconstructed "
    "from this installation's own dated snapshots; horizons still unavailable "
    "have no vintage yet.":
        "该数据源不发布 A 股的一致预期修正历史（7 / 30 / 60 / 90 天）。显示为可用的期限"
        "是用本系统自己带日期的快照重建的；仍不可用的期限尚无对应版本。",
    "No analyst up/down revision counts (breadth) are available, so revision "
    "breadth cannot be reported at any window.":
        "没有分析师上调/下调家数（广度）数据，因此任何窗口都无法报告修正广度。",
    "No reported-versus-consensus surprise history is available.":
        "没有实际值与一致预期对比的业绩惊喜历史。",
    "No post-earnings drift can be computed: this source carries no announcement "
    "dates, and drift anchored to anything else would misdate the market "
    "reaction.":
        "无法计算财报后漂移：该数据源不含公布日期，而锚定在其他日期会把市场反应的日期弄错。",
    "Whisper expectations and consensus margin revisions are unavailable.":
        "耳语预期和一致预期利润率修正均不可用。",
}

#: Values inside a pattern's groups that are themselves English: the horizon
#: names Alpha Vantage returns, and the direction in a breadth discrepancy.
_GROUP_WORDS_ZH: dict[str, str] = {
    "fiscal quarter": "财季", "fiscal year": "财年", "none named": "未命名",
    "upgrades": "上调", "downgrades": "下调",
    "eps": "EPS", "revenue": "营收",
}

_P = re.compile

#: Sentences with a variable part, as (pattern, template). A template is
#: ``str.format()``-ed with the match's named groups, each passed through
#: :func:`_group_zh` first.
_PATTERNS_ZH: tuple[tuple[re.Pattern[str], str], ...] = (
    # Earnings timing, with the reporting history it was read from.
    (_P(r"(?P<timing>amc|bmo) \((?:after market close|before market open)\) — "
        r"observed pattern, most recently (?P<date>\S+)"),
     "{timing}——依据历史规律，最近一次为 {date}"),
    # Alpha Vantage.
    (_P(r"Alpha Vantage EARNINGS_ESTIMATES returned only horizons this build does "
        r"not recognise \((?P<horizons>.*)\), so no period was mapped\. "
        r"Unrecognised horizons are skipped rather than guessed\."),
     "Alpha Vantage EARNINGS_ESTIMATES 只返回了本版本无法识别的预测期限（{horizons}），"
     "因此未映射任何期间。无法识别的期限会被跳过，而不是猜测。"),
    (_P(r"Alpha Vantage EARNINGS_ESTIMATES was declined \((?P<exc>.*)\)\. It is "
        r"premium-gated, and the free tier reports that as a quota notice, so no "
        r"consensus revision history is available from this vendor on this key\. "
        r"This is an entitlement limit, not an absence of analyst coverage\."),
     "Alpha Vantage EARNINGS_ESTIMATES 拒绝了请求（{exc}）。该接口需付费订阅，免费套餐会"
     "以配额提示的形式拒绝，因此该密钥无法从此数据源获取一致预期修正历史。这是权限限制，"
     "并非缺少分析师覆盖。"),
    (_P(r"Alpha Vantage EARNINGS_ESTIMATES was unavailable \((?P<exc>.*)\)\."),
     "Alpha Vantage EARNINGS_ESTIMATES 不可用（{exc}）。"),
    (_P(r"Alpha Vantage's 3-month earnings calendar lists no upcoming date for "
        r"(?P<symbol>\S+) at or after (?P<date>\S+)"),
     "Alpha Vantage 的 3 个月财报日历中没有 {symbol} 在 {date} 当日或之后的财报日期"),
    (_P(r"Alpha Vantage EARNINGS_CALENDAR was declined \((?P<exc>.*)\)\."),
     "Alpha Vantage EARNINGS_CALENDAR 拒绝了请求（{exc}）。"),
    (_P(r"Alpha Vantage EARNINGS_CALENDAR declined \((?P<exc>.*)\)"),
     "Alpha Vantage EARNINGS_CALENDAR 拒绝了请求（{exc}）"),
    (_P(r"Alpha Vantage EARNINGS_CALENDAR unavailable \((?P<exc>.*)\)"),
     "Alpha Vantage EARNINGS_CALENDAR 不可用（{exc}）"),
    (_P(r"adjusted price history unavailable \((?P<exc>.*)\)"),
     "复权价格历史不可用（{exc}）"),
    # Yahoo.
    (_P(r"as-of mismatch: requested (?P<requested>\S+), evidence carries "
        r"(?P<carried>\S+)"),
     "截至日不一致：请求为 {requested}，证据为 {carried}"),
    (_P(r"No estimate snapshot was observed on or before (?P<date>\S+)\. Yahoo's "
        r"revision lookbacks are relative to the present, so today's figures cannot "
        r"describe that date\. Snapshots accumulate one per day from the first run "
        r"onward; a historical date before the first run has no vintage and this "
        r"run reports none\."),
     "在 {date} 当日或之前没有观测到任何预期快照。Yahoo 的修正回看以当前日期为基准，"
     "因此今天的数字无法描述那个日期。快照从首次运行起每天累积一份；早于首次运行的"
     "历史日期没有对应版本，本次运行不报告任何数据。"),
    (_P(r"The point-in-time snapshot store could not be read \((?P<exc>.*)\)\. "
        r"Historical estimate evidence is unavailable; today's consensus is not a "
        r"substitute\."),
     "无法读取时点快照存储（{exc}）。历史预期证据不可用；今天的一致预期不能替代。"),
    (_P(r"(?P<symbol>\S+) is a (?P<kind>[a-z]+), not an operating company, so it has "
        r"no analyst EPS consensus, revision history or earnings calendar\. "
        r"Holdings-level earnings would need a look-through, which this analyst does "
        r"not perform\."),
     "{symbol} 是{kind}，并非经营性公司，因此没有分析师 EPS 一致预期、修正历史或财报日历。"
     "持仓层面的盈利需要穿透分析，本分析师不执行此操作。"),
    (_P(r"Yahoo recognises (?P<symbol>.+?) but publishes no analyst EPS estimates, "
        r"revision history, or reported-quarter history for it\. This is an absence "
        r"of sell-side coverage, not a zero: no revision direction can be inferred\."),
     "Yahoo 能识别 {symbol}，但没有发布它的分析师 EPS 预期、修正历史或已报告季度历史。"
     "这是卖方覆盖的缺失，而非零值：无法推断修正方向。"),
    (_P(r"(?P<symbol>\S+) is a (?P<kind>[a-z]+), not an operating company, so ROE, "
        r"margins, leverage and per-share multiples do not describe it the way they "
        r"describe a company\. A fund's own composition would need a look-through, "
        r"which this analyst does not perform\."),
     "{symbol} 是{kind}，并非经营性公司，因此净资产收益率、利润率、杠杆和每股倍数无法像"
     "描述公司那样描述它。基金自身的成分需要穿透分析，本分析师不执行此操作。"),
    (_P(r"Yahoo recognises (?P<symbol>.+?) but publishes none of return on equity, "
        r"operating margin, debt-to-equity or current ratio for it\."),
     "Yahoo 能识别 {symbol}，但没有发布它的净资产收益率、营业利润率、债务权益比或流动比率"
     "中的任何一项。"),
    (_P(r"Yahoo recognises (?P<symbol>.+?) but publishes none of trailing P/E, "
        r"forward P/E, PEG or price-to-book for it\."),
     "Yahoo 能识别 {symbol}，但没有发布它的历史市盈率、远期市盈率、PEG 或市净率中的任何一项。"),
    (_P(r"Fewer than (?P<count>\d+) periods of operating margin history available"),
     "营业利润率历史不足 {count} 个期间"),
    # 同花顺.
    (_P(r"同花顺 publishes no consensus-forecast table for (?P<code>\S+), and no "
        r"other free A-share source in this project publishes analyst estimates\. No "
        r"EPS consensus, revision direction, or earnings calendar is available for "
        r"this symbol\."),
     "同花顺没有发布 {code} 的一致预期表，本项目中也没有其他免费 A 股数据源发布分析师预期。"
     "该代码没有可用的 EPS 一致预期、修正方向或财报日历。"),
    (_P(r"同花顺's forecast table did not carry the expected 年度 / 均值 columns "
        r"\(saw: (?P<columns>.*)\)\. Its consensus EPS is therefore NOT normalized "
        r"into a measured figure — a mis-mapped column would publish a realised "
        r"result as a forward consensus\. Read the verbatim table under Guidance and "
        r"do not restate its numbers as 'consensus EPS' elsewhere\."),
     "同花顺预测表不含所需的“年度 / 均值”列（实际列：{columns}）。因此其一致预期 EPS "
     "未被规范化为实测数值——列映射错误会把已实现的业绩当作前瞻一致预期发布。请阅读"
     "“指引”下的原始表格，不要在别处把其中数字称为“一致预期 EPS”。"),
    # Discrepancies retained by compute_momentum.
    (_P(r"30-day revision count \((?P<total>\d+)\) exceeds reported analyst coverage "
        r"\((?P<analysts>\d+)\); counts are cumulative over the window while coverage "
        r"is a point-in-time figure"),
     "30 天修正家数（{total}）超过报告的分析师覆盖数（{analysts}）；修正家数是窗口内的"
     "累计值，而覆盖数是时点数值"),
    (_P(r"7-day (?P<direction>upgrades|downgrades) \((?P<short>\d+)\) exceed 30-day "
        r"\((?P<long>\d+)\), which is arithmetically impossible; the provider "
        r"refreshes the two windows independently"),
     "7 天{direction}家数（{short}）超过 30 天（{long}），这在算术上不可能；"
     "数据源分别刷新这两个窗口"),
    (_P(r"EPS revision horizons disagree in direction: (?P<signals>.+)"),
     "EPS 修正各期限方向不一致：{signals}"),
    # earnings_snapshot_store.py
    (_P(r"Some revision horizons were reconstructed from this installation's own "
        r"dated snapshots rather than a vendor-published history, because the vendor "
        r"publishes none\. Each substitution's true age is stated with the value; the "
        r"nominal horizon label is approximate\. Substitutions: (?P<substitutions>.+)"),
     "部分修正期限是用本系统自己带日期的快照重建的，而非数据源发布的历史，因为数据源"
     "不发布此类历史。每次替换的真实间隔随数值注明；名义期限标签只是近似。替换明细："
     "{substitutions}"),
)

_SUBSTITUTION_RE = _P(
    r"(?P<period>\S+) (?P<field>eps|revenue) '(?P<days>\d+)-day' lookback filled from "
    r"a local snapshot observed (?P<observed>\S+), actually (?P<age>\d+) days earlier"
)
_SIGNAL_VALUE_RE = _P(r"([a-z0-9_]+)=")


def _group_zh(name: str, value: str | None) -> str:
    """One captured group, in Chinese where it is one of our own words."""
    if value is None:
        return ""
    if name == "kind":
        return _QUOTE_TYPES_ZH.get(value.lower(), value)
    if name == "timing":
        return _MESSAGES_ZH.get(value, value)
    if name == "signals":
        named = _SIGNAL_VALUE_RE.sub(lambda m: _SIGNALS_ZH.get(m.group(1), m.group(1)) + "=",
                                     value)
        return named.replace(", ", "，")
    if name == "substitutions":
        return "；".join(_substitution_zh(part) for part in value.split("; "))
    if name in ("horizons", "direction"):
        return "、".join(_GROUP_WORDS_ZH.get(part, part) for part in value.split(", "))
    return value


def _substitution_zh(text: str) -> str:
    m = _SUBSTITUTION_RE.fullmatch(text.strip())
    if not m:
        return text
    period = _RELATIVE_PERIODS_ZH.get(m["period"], m["period"])
    field = _GROUP_WORDS_ZH[m["field"]]
    return (f"{period}{field}的“{m['days']} 天”回看值取自 {m['observed']} 观测的本地快照，"
            f"实际早 {m['age']} 天")


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportText:
    """Chooses the English or the Chinese form of what a renderer writes.

    Called with both forms, ``L("Data Gaps", "数据缺口")``, so each English string
    sits next to its translation in the renderer rather than behind a key.
    Every lookup falls back to the English it was given: a renderer runs inside
    ``invoke_structured``, where an exception discards the model's whole
    structured answer.
    """

    zh: bool = False

    def __call__(self, en: str, zh: str) -> str:
        return zh if self.zh else en

    def term(self, value: Any) -> str:
        """An enumerated verdict: a rating, band, tier or confidence level."""
        text = str(value or "")
        if not self.zh:
            return text
        return _TERMS_ZH.get(text) or _TERMS_ZH.get(text.lower()) or text

    def signal(self, key: str) -> str:
        return _SIGNALS_ZH.get(key, key) if self.zh else key

    def source(self, label: str) -> str:
        return _SOURCES_ZH.get(label, label) if self.zh else label

    def relative_period(self, key: str) -> str:
        """A relative period key ("0y") with nothing to resolve it to a date."""
        return _RELATIVE_PERIODS_ZH.get(key, key) if self.zh else key

    def message(self, text: str | None) -> str:
        """A data-gap, warning or "unavailable" sentence an adapter wrote."""
        if not text or not self.zh:
            return text or ""
        key = text.strip()
        hit = _MESSAGES_ZH.get(key)
        if hit is not None:
            return hit
        for pattern, template in _PATTERNS_ZH:
            m = pattern.fullmatch(key)
            if m:
                return template.format(**{k: _group_zh(k, v)
                                          for k, v in m.groupdict().items()})
        return text

    def join(self, items) -> str:
        """A run-in list: "a, b" or "a、b"."""
        return ("、" if self.zh else ", ").join(items)

    def fmt(self, value: Any) -> str:
        """A :class:`~tradingagents.dataflows.evidence_values.Value` for a report.

        English is ``_fmt`` exactly. Chinese writes "不可用" for an absent value,
        a currency's name after the figure ("0.27美元"), and large figures in 亿
        and 万亿 ("6,141.90亿美元"), which is how a Chinese reader counts them.
        """
        from tradingagents.dataflows.evidence_values import _fmt

        if not self.zh:
            return _fmt(value)
        if not value.available:
            return "不可用"
        if value.unit == "currency_large":
            return self.large(value.value, value.currency)
        if value.unit in ("pct_dec", "count", "ratio"):
            return _fmt(value)
        return self.money(f"{value.value:,.2f}", value.currency)

    def large(self, number: float, currency: str | None) -> str:
        from tradingagents.dataflows.evidence_values import _fmt_large

        if not self.zh:
            return _fmt_large(number, currency)
        magnitude = abs(number)
        for divisor, suffix in ((1e12, "万亿"), (1e8, "亿"), (1e4, "万")):
            if magnitude >= divisor:
                return self.money(f"{number / divisor:,.2f}{suffix}", currency)
        return self.money(f"{number:,.0f}", currency)

    def money(self, text: str, currency: str | None) -> str:
        if not currency:
            return text
        name = _CURRENCIES_ZH.get(currency.upper()) if self.zh else None
        return f"{text}{name}" if name else f"{text} {currency}"


ENGLISH = ReportText(zh=False)
CHINESE = ReportText(zh=True)


def for_language(language: str | None = None) -> ReportText:
    """The picker for ``language``, or for the run in progress when ``None``."""
    if language is None:
        language = configured_language()
    return CHINESE if is_chinese(language) else ENGLISH
