# LSTM-Based Time-Series Forecasting & MLOps Deployment

A PyTorch LSTM model for multivariate time-series forecasting and residual-based anomaly detection, with reference deployment scripts for AWS SageMaker, GCP Vertex AI, and Azure ML, plus a CloudWatch-based drift monitoring and retraining trigger.

## Project structure

```
src/
  data_preprocessing.py   # Rolling-window dataset construction and scaling
  model.py                 # LSTMForecaster model + anomaly scoring
  train.py                  # Training loop, checkpointing, anomaly evaluation
  evaluate.py               # Evaluate a saved checkpoint on test data
deployment/
  sagemaker_deploy.py       # Deploy to AWS SageMaker + CloudWatch drift alarm
  vertex_ai_deploy.py       # Deploy to GCP Vertex AI
  azure_ml_deploy.py        # Deploy to Azure ML managed online endpoint
monitoring/
  drift_detection.py        # PSI-based drift detection + retraining trigger
scripts/
  generate_sample_data.py   # Synthetic financial time series generator
tests/
  test_model.py              # Unit tests
```

## Getting started

```bash
pip install -r requirements.txt

# 1. Generate a synthetic dataset (or point to your own CSV with a
#    timestamp column and one or more numeric feature columns)
python scripts/generate_sample_data.py --output data/transactions.csv --days 1000

# 2. Train the model
python src/train.py --data data/transactions.csv --window-size 90 --epochs 30

# 3. Evaluate on held-out test data
python src/evaluate.py --data data/transactions.csv --checkpoint artifacts/model.pt
```

## Deployment

Each script in `deployment/` packages the trained model and deploys it to a real-time inference endpoint on the respective cloud platform:

- `sagemaker_deploy.py` also configures a CloudWatch alarm on a custom drift metric.
- `vertex_ai_deploy.py` deploys to a Vertex AI endpoint with autoscaling.
- `azure_ml_deploy.py` registers the model and deploys an Azure ML managed online endpoint.

`monitoring/drift_detection.py` computes the Population Stability Index (PSI) between a reference and current residual distribution, publishes it as a CloudWatch metric, and can trigger a SageMaker retraining job when drift exceeds a configurable threshold.

## Note on results

This repository is a runnable reference implementation and demo/portfolio project. Exact accuracy, latency, and deployment-time metrics depend on your dataset, hyperparameters, and cloud configuration — run `train.py`/`evaluate.py` yourself to reproduce metrics for your own data rather than treating any numbers elsewhere as guaranteed results of this exact codebase.

## License

MIT
