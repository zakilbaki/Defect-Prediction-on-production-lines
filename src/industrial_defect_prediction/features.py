from __future__ import annotations

import numpy as np
import pandas as pd


TRACE_COLUMN = "PROC_TRACEINFO"
TARGET_COLUMN = "Binar OP130_Resultat_Global_v"

TEMPORAL_SENSORS = (
    "OP070_V_1_angle_value",
    "OP070_V_1_torque_value",
    "OP090_SnapRingPeakForce_value",
    "OP110_Vissage_M8_angle_value",
    "OP120_Rodage_I_mesure_value",
)


def add_trace_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the product reference, production date, and sequence from the trace ID."""
    if TRACE_COLUMN not in frame:
        raise KeyError(f"Missing required column: {TRACE_COLUMN}")

    result = frame.copy()
    parts = result[TRACE_COLUMN].str.split("-", expand=True)
    if parts.shape[1] < 5:
        raise ValueError("PROC_TRACEINFO does not match the expected trace format")

    result["product_reference"] = parts[2]
    result["production_date"] = pd.to_datetime(parts[3], format="%y%m%d", errors="coerce")
    result["production_sequence"] = pd.to_numeric(parts[4], errors="coerce")

    if result[["production_date", "production_sequence"]].isna().any().any():
        raise ValueError("Some trace IDs contain an invalid date or production sequence")

    return result


def add_process_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create domain-motivated consistency and ratio features."""
    result = frame.copy()
    result["OP100_missing"] = result["OP100_Capuchon_insertion_mesure"].isna().astype(int)
    result["OP070_angle_diff"] = (
        result["OP070_V_1_angle_value"] - result["OP070_V_2_angle_value"]
    ).abs()
    result["OP070_torque_diff"] = (
        result["OP070_V_1_torque_value"] - result["OP070_V_2_torque_value"]
    ).abs()
    result["OP110_angle_torque_ratio"] = result["OP110_Vissage_M8_angle_value"] / (
        result["OP110_Vissage_M8_torque_value"] + 1e-6
    )
    result["OP090_force_diff"] = (
        result["OP090_SnapRingPeakForce_value"]
        - result["OP090_SnapRingMidPointForce_val"]
    )
    result["OP090_force_ratio"] = result["OP090_SnapRingPeakForce_value"] / (
        result["OP090_SnapRingMidPointForce_val"] + 1e-6
    )
    result["OP120_current_voltage_ratio"] = result["OP120_Rodage_I_mesure_value"] / (
        result["OP120_Rodage_U_mesure_value"] + 1e-6
    )
    return result


def add_history_features(
    frame: pd.DataFrame,
    window: int = 50,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Add rolling z-scores computed strictly from earlier products."""
    if window < 2:
        raise ValueError("window must be at least 2")

    result = frame.sort_values(
        ["production_date", "production_sequence", TRACE_COLUMN]
    ).reset_index(drop=True)

    for column in TEMPORAL_SENSORS:
        previous_values = result[column].shift(1)
        rolling = previous_values.rolling(window=window, min_periods=min_periods)
        previous_mean = rolling.mean()
        previous_std = rolling.std()
        result[f"{column}_history_z_{window}"] = (
            result[column] - previous_mean
        ) / previous_std.replace(0, np.nan)

    return result


def build_feature_frame(frame: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Run the complete deterministic feature pipeline."""
    return add_history_features(add_process_features(add_trace_metadata(frame)), window=window)


def model_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model features and the binary target."""
    if TARGET_COLUMN not in frame:
        raise KeyError(f"Missing required target: {TARGET_COLUMN}")

    excluded = {
        TARGET_COLUMN,
        "production_sequence",
    }
    numeric = frame.select_dtypes(include=["number", "bool"])
    feature_columns = [column for column in numeric.columns if column not in excluded]
    return numeric[feature_columns], frame[TARGET_COLUMN].astype(int)
