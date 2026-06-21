import numpy as np
import pandas as pd
import pytest

from industrial_defect_prediction.features import (
    TEMPORAL_SENSORS,
    add_history_features,
    add_trace_metadata,
)


def test_add_trace_metadata_parses_date_and_sequence() -> None:
    frame = pd.DataFrame(
        {"PROC_TRACEINFO": ["I-B-XA1207672-190701-00494"]}
    )

    result = add_trace_metadata(frame)

    assert result.loc[0, "product_reference"] == "XA1207672"
    assert result.loc[0, "production_date"] == pd.Timestamp("2019-07-01")
    assert result.loc[0, "production_sequence"] == 494


def test_add_trace_metadata_rejects_invalid_trace() -> None:
    with pytest.raises(ValueError, match="expected trace format"):
        add_trace_metadata(pd.DataFrame({"PROC_TRACEINFO": ["invalid"]}))


def test_history_features_use_only_previous_products() -> None:
    frame = pd.DataFrame(
        {
            "PROC_TRACEINFO": [
                "I-B-XA1207672-190701-00003",
                "I-B-XA1207672-190701-00001",
                "I-B-XA1207672-190701-00002",
            ],
            "production_date": pd.to_datetime(["2019-07-01"] * 3),
            "production_sequence": [3, 1, 2],
        }
    )
    for sensor in TEMPORAL_SENSORS:
        frame[sensor] = [14.0, 10.0, 12.0]

    result = add_history_features(frame, window=2, min_periods=2)
    z_score = result.loc[2, f"{TEMPORAL_SENSORS[0]}_history_z_2"]

    assert result["production_sequence"].tolist() == [1, 2, 3]
    assert np.isnan(result.loc[0, f"{TEMPORAL_SENSORS[0]}_history_z_2"])
    assert np.isclose(z_score, (14.0 - 11.0) / np.std([10.0, 12.0], ddof=1))
