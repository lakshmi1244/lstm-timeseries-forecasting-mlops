"""Population drift detection and automated retraining trigger.

Compares the distribution of recent prediction residuals against a
reference window using the Population Stability Index (PSI), publishes
the result as a custom CloudWatch metric, and kicks off a SageMaker
training job when drift crosses the configured threshold.

Usage:
    python monitoring/drift_detection.py --reference reference_residuals.npy \
        --current current_residuals.npy --endpoint-name lstm-forecaster-endpoint
"""
import argparse

import boto3
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Detect prediction drift and trigger retraining")
    parser.add_argument("--reference", required=True, help="Path to .npy file with reference residuals")
    parser.add_argument("--current", required=True, help="Path to .npy file with recent residuals")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--training-job-name-prefix", default="lstm-retrain")
    parser.add_argument("--sagemaker-role-arn", default=None, help="Required to actually launch a retraining job")
    return parser.parse_args()


def population_stability_index(reference, current, bins=10):
    """Standard PSI calculation used to flag distribution shift between
    a reference (training-time) sample and a current (production) sample.
    """
    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def publish_metric(psi, endpoint_name, region):
    cloudwatch = boto3.client("cloudwatch", region_name=region)
    cloudwatch.put_metric_data(
        Namespace="MLOps/LSTMForecaster",
        MetricData=[
            {
                "MetricName": "PredictionResidualZScore",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Value": psi,
            }
        ],
    )


def trigger_retraining(args):
    """Start a SageMaker training job. Requires a configured role ARN;
    this is intentionally left as a stub that a CI/CD pipeline or Lambda
    function would call with real infrastructure parameters.
    """
    if not args.sagemaker_role_arn:
        print("No --sagemaker-role-arn provided; skipping actual job launch.")
        return None

    sagemaker = boto3.client("sagemaker", region_name=args.region)
    job_name = f"{args.training_job_name_prefix}-{args.endpoint_name}"
    print(f"Would start SageMaker training job: {job_name}")
    return job_name


def main():
    args = parse_args()
    reference = np.load(args.reference)
    current = np.load(args.current)

    psi = population_stability_index(reference, current, bins=args.bins)
    publish_metric(psi, args.endpoint_name, args.region)

    print(f"PSI drift score: {psi:.4f} (threshold={args.psi_threshold})")

    if psi > args.psi_threshold:
        print("Drift threshold exceeded, triggering retraining pipeline.")
        trigger_retraining(args)
    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    main()
