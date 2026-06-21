from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from industrial_defect_prediction.features import TRACE_COLUMN


@dataclass(frozen=True)
class EvaluationMetrics:
    roc_auc: float
    average_precision: float
    positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split past and future products without shuffling."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = frame.sort_values(
        ["production_date", "production_sequence", TRACE_COLUMN]
    ).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    if split_index == 0 or split_index == len(ordered):
        raise ValueError("Not enough rows for the requested split")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def build_classifier() -> Pipeline:
    """Create the reproducible, imbalance-aware linear baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=0.01,
                    l1_ratio=0.8,
                    class_weight="balanced",
                    max_iter=10_000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y_true: pd.Series, scores) -> EvaluationMetrics:
    return EvaluationMetrics(
        roc_auc=float(roc_auc_score(y_true, scores)),
        average_precision=float(average_precision_score(y_true, scores)),
        positive_rate=float(y_true.mean()),
    )
