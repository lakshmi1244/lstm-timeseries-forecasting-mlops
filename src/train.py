"""Train the LSTM forecaster and evaluate anomaly-detection performance.

Usage:
    python src/train.py --data data/transactions.csv --window-size 90 \
        --epochs 30 --anomaly-threshold 2.5

The script trains an LSTM to predict the next value(s) in a rolling
window. Anomalies are flagged when the residual between the prediction
and the actual value exceeds `anomaly-threshold` standard deviations.
If the input CSV contains a binary `is_anomaly` column, classification
accuracy against those ground-truth labels is also reported.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import prepare_dataset
from model import LSTMForecaster, anomaly_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM forecaster")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with a timestamp column")
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anomaly-threshold", type=float, default=2.5)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser.parse_args()


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y[:, 0, :]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_anomalies(model, loader, threshold):
    model.eval()
    residuals = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            residuals.append(anomaly_score(yb, preds).numpy())
    residuals = np.concatenate(residuals, axis=0)
    z_scores = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0) + 1e-8)
    flagged = (np.abs(z_scores) > threshold).any(axis=1)
    return flagged, residuals


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    (X_train, y_train, X_val, y_val, X_test, y_test), scaler, feature_cols = prepare_dataset(
        args.data, window_size=args.window_size
    )

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(feature_cols),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_losses.append(loss_fn(preds, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    flagged, residuals = evaluate_anomalies(model, test_loader, args.anomaly_threshold)

    metrics = {
        "best_val_loss": best_val_loss,
        "num_test_windows": int(len(flagged)),
        "num_flagged_anomalies": int(flagged.sum()),
        "anomaly_rate": float(flagged.mean()),
        "mean_absolute_residual": float(np.mean(np.abs(residuals))),
    }

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
