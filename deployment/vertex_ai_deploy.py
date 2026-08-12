"""Package and deploy the LSTM forecaster to GCP Vertex AI.

Mirrors the SageMaker deployment flow so the same trained model artifact
can be served consistently across clouds, reducing cross-cloud MLOps
abstraction overhead.

Usage:
    python deployment/vertex_ai_deploy.py --model-artifact-uri gs://my-bucket/model \
        --project my-gcp-project --region us-central1
"""
import argparse

from google.cloud import aiplatform


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy LSTM model to Vertex AI")
    parser.add_argument("--model-artifact-uri", required=True, help="GCS URI of the model directory")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--model-display-name", default="lstm-forecaster")
    parser.add_argument("--endpoint-display-name", default="lstm-forecaster-endpoint")
    parser.add_argument("--machine-type", default="n1-standard-4")
    parser.add_argument(
        "--serving-container-image-uri",
        default="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
    )
    return parser.parse_args()


def upload_model(args):
    return aiplatform.Model.upload(
        display_name=args.model_display_name,
        artifact_uri=args.model_artifact_uri,
        serving_container_image_uri=args.serving_container_image_uri,
    )


def deploy_model(model, args):
    endpoint = aiplatform.Endpoint.create(display_name=args.endpoint_display_name)
    model.deploy(
        endpoint=endpoint,
        machine_type=args.machine_type,
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100,
    )
    return endpoint


def main():
    args = parse_args()
    aiplatform.init(project=args.project, location=args.region)

    model = upload_model(args)
    endpoint = deploy_model(model, args)

    print(f"Model uploaded: {model.resource_name}")
    print(f"Endpoint deployed: {endpoint.resource_name}")


if __name__ == "__main__":
    main()
