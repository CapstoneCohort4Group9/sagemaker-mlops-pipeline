import boto3, os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.session import Session as SageMakerSession
from pipeline.steps.preprocessing import get_preprocessing_step
from pipeline.steps.training import get_training_step, get_register_step
from pipeline.steps.evaluation import get_evaluation_step


os.environ["AWS_SAGEMAKER_DISABLE_STUDIO_AUTOCONFIG"] = "true"

# --- AWS Config ---
region = "us-east-1"
profile = "Developer"
assume_role_arn  = "arn:aws:iam::109038807292:role/service-role/AmazonSageMaker-ExecutionRole-20250622T130599"

sts_client = boto3.Session(profile_name=profile).client("sts", region_name=region)
assume_role_response = sts_client.assume_role(
    RoleArn=assume_role_arn,
    RoleSessionName="SageMakerPipelineSession"
)

# Extract credentials
creds = assume_role_response["Credentials"]

# Create a new boto3 session using the assumed role credentials
assumed_session = boto3.Session(
    aws_access_key_id=creds["AccessKeyId"],
    aws_secret_access_key=creds["SecretAccessKey"],
    aws_session_token=creds["SessionToken"],
    region_name=region
)

# Use this session to create the SageMaker session and pipeline session
sagemaker_session = PipelineSession(
    boto_session=assumed_session
)

# Use the same `role` for upsert
role = assume_role_arn  # Use the same role ARN


# Pipeline Parameters ---
train_input_s3 = ParameterString(name="TrainInputS3Uri", default_value="s3://sagemaker-studio-7oabgjsj0b7/datasets/raw/train.jsonl")
test_input_s3 = ParameterString(name="TestInputS3Uri", default_value="s3://sagemaker-studio-7oabgjsj0b7/datasets/raw/test.jsonl")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
mlflow_tracking_uri = ParameterString(name="MLflowTrackingURI", default_value="arn:aws:sagemaker:us-east-1:109038807292:mlflow-tracking-server/tracking-server-4gnc5g87j3w4vb-3rt7rqb89qmt5z-dev")

# Pipeline Steps ---
preprocess_step, train_output, eval_output = get_preprocessing_step(
    sagemaker_session, role, train_input_s3, test_input_s3
)

train_step, estimator = get_training_step(
    sagemaker_session=sagemaker_session,
    role=role,
    preprocess_step=preprocess_step,
    mlflow_tracking_uri=mlflow_tracking_uri,
    entry_point="train_lora.py"
)

eval_step = get_evaluation_step(
    sagemaker_session, role, train_step
)

eval_step.add_depends_on([train_step])

register_step = get_register_step(
    sagemaker_session, role, estimator, train_step, model_approval_status, eval_step
)

# Update pipeline steps – inference_step removed
pipeline = Pipeline(
    name="Hermes2Pro-Mistral-Finetuning-Pipeline",
    parameters=[train_input_s3, test_input_s3, model_approval_status, mlflow_tracking_uri],
    steps=[preprocess_step, train_step, eval_step, register_step],
    sagemaker_session=sagemaker_session
)

print(pipeline.definition())
try:
   pipeline.create(role_arn=role)
   print("SageMaker pipeline successfully created and updated.")
except Exception as e:
    print("Failed to create pipeline:", str(e))
