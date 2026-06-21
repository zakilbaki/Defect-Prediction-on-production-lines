from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from industrial_defect_prediction.features import (
    TRACE_COLUMN,
    build_feature_frame,
    model_matrix,
)
from industrial_defect_prediction.modeling import (
    build_classifier,
    chronological_split,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the industrial defect baseline.")
    parser.add_argument("--training-inputs", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = pd.read_csv(args.training_inputs)
    output = pd.read_csv(args.training_output)
    merged = inputs.merge(output, on=TRACE_COLUMN, how="inner", validate="one_to_one")

    featured = build_feature_frame(merged)
    train_frame, test_frame = chronological_split(featured, args.test_fraction)
    x_train, y_train = model_matrix(train_frame)
    x_test, y_test = model_matrix(test_frame)

    classifier = build_classifier()
    classifier.fit(x_train, y_train)
    scores = classifier.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, scores)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": classifier, "feature_columns": list(x_train.columns)},
        args.artifacts_dir / "model.joblib",
    )
    report = {
        "evaluation_protocol": "chronological 80/20 holdout by date and sequence",
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "train_positive_rate": float(y_train.mean()),
        "metrics": metrics.to_dict(),
    }
    (args.artifacts_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
