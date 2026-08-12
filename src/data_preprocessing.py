"""Data loading and preprocessing utilities for LSTM time-series forecasting.

Builds rolling windows from a financial time-series CSV, scales the
features, and returns train/validation/test splits along with the
fitted scaler (needed later for inverse-transforming predictions).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_series(csv_path, timestamp_col="timestamp", value_cols=None):
    """Load a CSV file and return a sorted DataFrame indexed by timestamp."""
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    if value_cols is not None:
        df = df[[timestamp_col] + value_cols]
    return df


def build_rolling_windows(values, window_size=90, horizon=1):
    """Convert a 2D array (time, features) into overlapping rolling windows.

    Returns X of shape (n_samples, window_size, n_features) and y of shape
    (n_samples, horizon, n_features).
    """
    X, y = [], []
    n = len(values)
    for start in range(n - window_size - horizon + 1):
        end = start + window_size
        X.append(values[start:end])
        y.append(values[end:end + horizon])
    return np.array(X), np.array(y)


def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15):
    """Chronological split so validation/test always come after training data."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def fit_scaler(train_values):
    scaler = StandardScaler()
    scaler.fit(train_values)
    return scaler


def prepare_dataset(csv_path, timestamp_col="timestamp", value_cols=None, window_size=90, horizon=1):
    """End-to-end preprocessing pipeline used by train.py and evaluate.py."""
    df = load_series(csv_path, timestamp_col, value_cols)
    feature_cols = value_cols or [c for c in df.columns if c != timestamp_col]
    raw_values = df[feature_cols].values.astype("float32")

    n_train = int(len(raw_values) * 0.7)
    scaler = fit_scaler(raw_values[:n_train])
    scaled_values = scaler.transform(raw_values)

    X, y = build_rolling_windows(scaled_values, window_size=window_size, horizon=horizon)
    splits = train_val_test_split(X, y)
    return splits, scaler, feature_cols
