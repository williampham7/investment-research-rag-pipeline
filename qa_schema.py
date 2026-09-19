"""Pydantic schema for the query classifier used by qa.py."""

from typing import List, Literal

from pydantic import BaseModel, Field

from extraction_schema import ASSET_CLASS_VALUES, SIGNAL_TYPE_VALUES

AssetClass = Literal[ASSET_CLASS_VALUES]  # type: ignore[valid-type]
SignalType = Literal[SIGNAL_TYPE_VALUES]  # type: ignore[valid-type]

QueryMode = Literal["structured", "retrieval", "hybrid"]
MetricOperator = Literal[">", ">=", "<", "<=", "none"]


class QueryClassification(BaseModel):
    mode: QueryMode = Field(
        description='"structured": the question asks to filter/list/count/compare papers by tag or a '
        'numeric threshold, answerable exactly from a fact table (e.g. "which papers report Sharpe > 2", '
        '"list FX carry papers", "how many momentum papers are there"). '
        '"retrieval": the question asks to explain/understand/describe methodology, mechanisms, or '
        'findings in prose, needing the paper\'s actual text (e.g. "how is the momentum signal '
        'constructed", "what criticisms of factor investing do these papers raise"). '
        '"hybrid": the question needs BOTH a filtered set of papers AND an explanation of them '
        '(e.g. "what\'s the best-performing FX carry strategy and how does it work").'
    )
    signal_type_filter: List[SignalType] = Field(
        description="Signal type(s) the question mentions or clearly implies. Empty list if none."
    )
    asset_class_filter: List[AssetClass] = Field(
        description="Asset class(es) the question mentions or clearly implies. Empty list if none."
    )
    metric_keyword: str = Field(
        description='A short lowercase keyword for the metric type being asked about, matched by '
        'substring against reported metric names -- e.g. "sharpe", "return", "alpha", "drawdown", '
        '"information ratio", "win rate". Empty string if the question is not about a specific '
        "numeric metric (e.g. a pure tag/count question, or a pure explanation question)."
    )
    metric_operator: MetricOperator = Field(
        description='Comparison implied for metric_keyword, e.g. "Sharpe > 2" -> ">". "none" if there '
        "is no explicit or implied numeric threshold (e.g. \"list the Sharpe ratios of...\" just wants "
        "values reported, not a filter)."
    )
    metric_threshold: float = Field(
        description="The threshold value, e.g. 2.0 for \"Sharpe > 2\". 0 if metric_operator is 'none'."
    )
    reasoning: str = Field(description="One sentence on why this mode/filters were chosen.")
