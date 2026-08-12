"""Unit tests for the LSTM forecaster and preprocessing utilities."""
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import LSTMForecaster, anomaly_score  # noqa: E402
from data_preprocessing import build_rolling_windows, train_val_test_split  # noqa: E402


def test_lstm_forward_shape():
    model = LSTMForecaster(input_size=3, hidden_size=16, num_layers=1, output_size=3)
    x = torch.randn(8, 20, 3)
    out = model(x)
    assert out.shape == (8, 3)


def test_anomaly_score_is_nonnegative():
    y_true = torch.randn(10, 3)
    y_pred = torch.randn(10, 3)
    score = anomaly_score(y_true, y_pred)
    assert (score >= 0).all()


def test_build_rolling_windows_shapes():
    values = np.arange(100).reshape(-1, 1).astype("float32")
    X, y = build_rolling_windows(values, window_size=10, horizon=1)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90, 1, 1)


def test_train_val_test_split_sizes():
    values = np.arange(100).reshape(-1, 1).astype("float32")
    X, y = build_rolling_windows(values, window_size=10, horizon=1)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(X_train) > len(X_val)
