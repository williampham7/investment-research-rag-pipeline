"""
Pydantic schema + taxonomy for structured extraction (structured_extract.py).

signal_type/asset_class reuse the exact vocabulary already assigned during
corpus curation (arxiv_papers/curation_notes.jsonl) so the two stages stay
joinable and cross-checkable against each other.

Every field is required (no Optional/None) because Gemini's structured-output
mode handles required string/list fields far more reliably than nullable
ones -- "not reported" / an empty list is the sentinel for "not in the paper"
instead of null.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

SIGNAL_TYPE_VALUES = (
    "momentum",
    "reversal",
    "value",
    "size",
    "quality_profitability",
    "low_volatility",
    "volatility_risk_premium",
    "carry",
    "liquidity",
    "seasonality",
    "event_driven",
    "sentiment_text",
    "alternative_data",
    "stat_arb_pairs",
    "ml_composite",
    "other",
)

ASSET_CLASS_VALUES = (
    "equities",
    "fixed_income",
    "fx",
    "commodities",
    "crypto",
    "futures_multi_asset",
    "options_volatility",
    "other",
)

METRIC_UNIT_VALUES = ("ratio", "percent", "bps", "stddev", "count", "currency", "other")
CONFIDENCE_VALUES = ("high", "medium", "low")

SignalType = Literal[SIGNAL_TYPE_VALUES]  # type: ignore[valid-type]
AssetClass = Literal[ASSET_CLASS_VALUES]  # type: ignore[valid-type]
MetricUnit = Literal[METRIC_UNIT_VALUES]  # type: ignore[valid-type]
Confidence = Literal[CONFIDENCE_VALUES]  # type: ignore[valid-type]


class PerformanceMetric(BaseModel):
    metric_name: str = Field(
        description='Name of the reported figure, e.g. "Sharpe ratio", "annualized return", '
        '"CAPM alpha", "information ratio", "max drawdown", "t-statistic", "hit rate".'
    )
    value: float = Field(description="The numeric value exactly as reported (e.g. 1.42, not 142%).")
    unit: MetricUnit = Field(description='"ratio" for Sharpe/IR/t-stat, "percent" for returns/drawdown, "bps" for basis points, "stddev" for volatility, "currency" for a dollar-or-other-currency amount (e.g. "$8,116"), "count" for a plain tally (e.g. number of trades), "other" otherwise.')
    context: str = Field(
        description="What this number describes: which strategy/portfolio/subsample/period it belongs to."
    )
    evidence_quote: str = Field(
        description="A short verbatim quote or table row/cell from the supplied paper text that "
        "contains this exact number. Must be copied from the text, not paraphrased or invented."
    )
    source_location: str = Field(
        description='Where in the paper this came from, e.g. "Table 3", "Section 5.2", "page 12".'
    )


class StructuredExtraction(BaseModel):
    signal_type: List[SignalType] = Field(
        min_length=1,
        description="One or more signal families the paper's strategy is built on. Use 'other' only if "
        "none fit. IMPORTANT: a strategy that trades the mean-reverting SPREAD between two related "
        "instruments (pairs trading, cointegrated spreads, an Ornstein-Uhlenbeck-modeled spread) is "
        "'stat_arb_pairs', not plain 'reversal' -- 'reversal' is for single-asset price reversal "
        "(e.g. short-term reversal of one stock's own past return), not a two-leg spread.",
    )
    asset_class: List[AssetClass] = Field(
        min_length=1,
        description="One or more asset classes the strategy/data is tested on. Use 'other' only if none "
        "fit. IMPORTANT: if the paper trades a diversified basket of futures/CTA-style contracts "
        "spanning many sectors (e.g. equity index, bond, currency, AND commodity futures together, as "
        "is typical of trend-following/managed-futures papers), tag it 'futures_multi_asset' ONLY -- do "
        "not also decompose it into 'equities', 'fixed_income', 'fx', 'commodities' separately. Use "
        "those specific tags only when the paper's focus is genuinely that single asset class.",
    )
    methodology_summary: str = Field(
        description="2-4 sentences describing the empirical method: how the signal is constructed, "
        "portfolio formation/rebalancing, and how it is tested. Not a restatement of the abstract."
    )
    data_period: str = Field(
        description='Sample period covered, e.g. "1963-2016, monthly". "not reported" if absent.'
    )
    universe: str = Field(
        description='The investable universe/dataset, e.g. "NYSE/AMEX/NASDAQ common stocks excluding financials". '
        '"not reported" if absent.'
    )
    performance_metrics: List[PerformanceMetric] = Field(
        description="Every distinct quantitative TRADING/INVESTMENT PERFORMANCE figure explicitly "
        "stated in the text (not computed or inferred by you): returns, Sharpe/Sortino/information "
        "ratios, alpha, drawdown, turnover, hit rate, t-stats ON STRATEGY RETURNS, and similar. Do "
        "NOT include statistical diagnostic/goodness-of-fit tests such as Jarque-Bera, ADF, KPSS, "
        "ARCH-LM, Durbin-Watson, or regression R-squared -- those describe the data/model, not the "
        "strategy's performance. Empty list only if the paper truly reports no performance figures."
    )
    key_findings: List[str] = Field(
        description="2-5 short bullet statements of the paper's main empirical findings/conclusions."
    )
    limitations: List[str] = Field(
        description="Caveats the authors themselves acknowledge (e.g. no transaction costs, in-sample "
        "only, small sample, data snooping risk). Empty list if none are stated."
    )
    extraction_confidence: Confidence = Field(
        description="Your confidence that the above fields are complete and correctly grounded in the text."
    )
    extraction_notes: str = Field(
        description="Anything ambiguous, missing, or uncertain about this extraction. Empty string if none."
    )
