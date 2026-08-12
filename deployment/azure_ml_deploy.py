"""Package and deploy the LSTM forecaster to Azure Machine Learning.

Completes the multi-cloud MLOps abstraction layer alongside the
SageMaker and Vertex AI deployment scripts, so the same trained
artifact can be served consistently on Azure ML managed online
endpoints.

Usage:
    python deployment/azure_ml_deploy.py --model-path artifacts/model.pt \
        --subscription-id <sub-id> --resource-group <rg> --workspace-name <ws>
"""
import argparse

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy LSTM model to Azure ML")
    parser.add_argument("--model-path", required=True, help="Local or blob path to the model artifact")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--endpoint-name", default="lstm-forecaster-endpoint")
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    return parser.parse_args()


def get_client(args):
    credential = DefaultAzureCredential()
    return MLClient(
        credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def register_model(client, args):
    model = Model(path=args.model_path, name="lstm-forecaster", description="LSTM time-series forecaster")
    return client.models.create_or_update(model)


def deploy_model(client, model, args):
    endpoint = ManagedOnlineEndpoint(name=args.endpoint_name, auth_mode="key")
    client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model.id,
        instance_type=args.instance_type,
        instance_count=1,
    )
    client.online_deployments.begin_create_or_update(deployment).result()

    endpoint.traffic = {"blue": 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()


def main():
    args = parse_args()
    client = get_client(args)
    model = register_model(client, args)
    deploy_model(client, model, args)
    print(f"Deployed model '{model.name}' (v{model.version}) to endpoint '{args.endpoint_name}'.")


if __name__ == "__main__":
    main()
