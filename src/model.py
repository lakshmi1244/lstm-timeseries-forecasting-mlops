"""LSTM forecasting model definition."""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM network for multivariate time-series forecasting.

    The model consumes a rolling window of historical features and
    predicts the value(s) for the next time step. It is also used as
    the backbone for residual-based anomaly detection: large deviations
    between predicted and actual values are flagged as anomalies.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.norm(last_step)
        return self.head(last_step)


def anomaly_score(y_true, y_pred):
    """Compute an absolute residual anomaly score between prediction and truth."""
    return (y_true - y_pred).abs()
