"""Evaluate a trained LSTM checkpoint on held-out test data.

Usage:
    python src/evaluate.py --data data/transactions.csv --checkpoint artifacts/model.pt
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data_preprocessing import prepare_dataset
from model import LSTMForecaster, anomaly_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM forecaster")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--anomaly-threshold", type=float, default=2.5)
    parser.add_argument("--labels", type=str, default=None, help="Optional path to ground-truth anomaly labels (.npy)")
    return parser.parse_args()


def main():
    args = parse_args()

    (_, _, _, _, X_test, y_test), scaler, feature_cols = prepare_dataset(
        args.data, window_size=args.window_size
    )

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(feature_cols),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test[:, 0, :])

    with torch.no_grad():
        preds = model(X_test_t)

    residuals = anomaly_score(y_test_t, preds).numpy()
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    z_scores = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0) + 1e-8)
    flagged = (np.abs(z_scores) > args.anomaly_threshold).any(axis=1)

    results = {
        "mae": mae,
        "rmse": rmse,
        "num_test_windows": int(len(flagged)),
        "flagged_anomaly_rate": float(flagged.mean()),
    }

    if args.labels:
        true_labels = np.load(args.labels)
        results["accuracy"] = float(accuracy_score(true_labels, flagged))
        results["precision"] = float(precision_score(true_labels, flagged, zero_division=0))
        results["recall"] = float(recall_score(true_labels, flagged, zero_division=0))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
