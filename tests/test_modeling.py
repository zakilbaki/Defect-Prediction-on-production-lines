import pandas as pd

from industrial_defect_prediction.modeling import chronological_split


def test_chronological_split_keeps_future_rows_in_test() -> None:
    frame = pd.DataFrame(
        {
            "PROC_TRACEINFO": ["later", "earlier", "latest", "middle"],
            "production_date": pd.to_datetime(
                ["2019-07-03", "2019-07-01", "2019-07-04", "2019-07-02"]
            ),
            "production_sequence": [1, 1, 1, 1],
        }
    )

    train, test = chronological_split(frame, test_fraction=0.5)

    assert train["production_date"].max() < test["production_date"].min()
