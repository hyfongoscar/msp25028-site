from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from index import run_hybrid_bulk_model_inference


class DummySelector:
    def transform(self, features_df):
        return features_df.to_numpy(dtype=float)


class DummyScaler:
    def transform(self, features_matrix):
        return features_matrix


class DummyYScaler:
    def inverse_transform(self, values):
        return values


def build_ohlcv_rows(count: int):
    rows = []
    for i in range(count):
        rows.append(
            {
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000.0 + i,
                "adjClose": 100.5 + i,
            }
        )
    return rows


def test_hybrid_bulk_model_inference_preserves_input_length():
    raw_ohlcv_list = build_ohlcv_rows(60)
    predictions = run_hybrid_bulk_model_inference(
        {"weights": np.array([1.0])},
        raw_ohlcv_list,
        DummySelector(),
        DummyScaler(),
        DummyYScaler(),
        ["open", "high", "low", "close", "volume"],
        5,
    )

    assert len(predictions) == len(raw_ohlcv_list)
