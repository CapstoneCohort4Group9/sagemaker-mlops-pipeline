from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model


def get_training_step(sagemaker_session, role, preprocess_step, mlflow_tracking_uri, entry_point):
    estimator = HuggingFace(
        entry_point=entry_point,
        source_dir="src",
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=role,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04",
        transformers_version="4.49.0",
        pytorch_version="2.5.1",
        py_version="py311",
        checkpoint_s3_uri="s3://sagemaker-studio-7oabgjsj0b7/hermespro/checkpoints/",
        hyperparameters={
            "HF_MODEL_NAME": "NousResearch/Hermes-2-Pro-Mistral-7B",
            "DATA_PATH": "/opt/ml/input/data/train/train.jsonl",
            "EVAL_PATH": "/opt/ml/input/data/eval/test.jsonl",
            "OUTPUT_DIR": "/opt/ml/model"
        },
        environment={
            "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        },
        sagemaker_session=sagemaker_session,
        base_job_name="hermes2pro-mistral-lora",
        use_spot_instances=False,
        max_run=32000,
        #max_wait=32000
    )

    training_step = TrainingStep(
        name="TrainHermesProModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train_output"].S3Output.S3Uri,
                content_type="application/json"
            ),
            "eval": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["test_output"].S3Output.S3Uri,
                content_type="application/json"
            )
        },
        cache_config=CacheConfig(enable_caching=True, expire_after="30d")
    )

    return training_step, estimator

def get_register_step(sagemaker_session, role, estimator, train_step, model_approval_status, eval_step):

    # Reference model artifacts from training step
    model = Model(
        image_uri=estimator.image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role
    )
     
    # Define model metrics if evaluation step is provided
    model_metrics = None
    if eval_step is not None:
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="s3://sagemaker-studio-7oabgjsj0b7/metrics/evaluation.json",
                content_type="application/json"
            )
        )

    # Use ModelStep to ensure train card shows up correctly
    register_step = ModelStep(
        name="HopJetAir",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.g5.2xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="HopJetAirHermesProMistral",
            approval_status=model_approval_status,
            model_metrics=model_metrics
        ),
        depends_on=[eval_step.name] 
    )

    return register_step

