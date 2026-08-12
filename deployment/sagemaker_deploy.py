"""Package and deploy the LSTM forecaster to Amazon SageMaker.

This script wraps the trained PyTorch model as a SageMaker PyTorchModel,
deploys it behind a real-time inference endpoint, and wires up a
CloudWatch alarm that watches a custom "PredictionResidual" metric so
drift can trigger an automated retraining pipeline (see
monitoring/drift_detection.py).

Usage:
    python deployment/sagemaker_deploy.py --model-artifact s3://my-bucket/model.tar.gz \
        --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole
"""
import argparse

import boto3
from sagemaker.pytorch import PyTorchModel


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy LSTM model to SageMaker")
    parser.add_argument("--model-artifact", required=True, help="S3 URI of the packaged model.tar.gz")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN with SageMaker permissions")
    parser.add_argument("--endpoint-name", default="lstm-forecaster-endpoint")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def deploy_endpoint(args):
    pytorch_model = PyTorchModel(
        model_data=args.model_artifact,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir="deployment",
        framework_version="2.1",
        py_version="py310",
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    return predictor


def create_drift_alarm(endpoint_name, region):
    """Create a CloudWatch alarm on a custom drift metric published by the
    monitoring job. When the alarm triggers, it publishes to an SNS topic
    that a Lambda function subscribes to in order to kick off a SageMaker
    Pipelines retraining execution.
    """
    cloudwatch = boto3.client("cloudwatch", region_name=region)
    cloudwatch.put_metric_alarm(
        AlarmName=f"{endpoint_name}-drift-alarm",
        MetricName="PredictionResidualZScore",
        Namespace="MLOps/LSTMForecaster",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Statistic="Average",
        Period=300,
        EvaluationPeriods=3,
        Threshold=2.5,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmDescription="Triggers automated retraining when prediction drift exceeds threshold.",
    )


def main():
    args = parse_args()
    predictor = deploy_endpoint(args)
    create_drift_alarm(args.endpoint_name, args.region)
    print(f"Deployed endpoint: {predictor.endpoint_name}")
    print("CloudWatch drift alarm configured.")


if __name__ == "__main__":
    main()
