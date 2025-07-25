from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

def get_evaluation_step(sagemaker_session, role, train_step):
    processor = ScriptProcessor(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04",
        command=["python3"],
        role=role,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        base_job_name="hermes-eval",
        sagemaker_session=sagemaker_session
    )

    property_file = PropertyFile(
        name="EvaluationMetrics",
        output_name="evaluation",
        path="evaluation.json"
    )

    step = ProcessingStep(
        name="EvaluateHopJetAirModel",
        processor=processor,
        code="src/run_evaluation.py",  
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source="s3://sagemaker-studio-7oabgjsj0b7/datasets/evaluation/test_prompts.jsonl",
                destination="/opt/ml/processing/input"
            ),
            ProcessingInput( 
                source="src/requirements.txt",
                destination="/opt/ml/processing/code"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                output_name="evaluation",
                destination="s3://sagemaker-studio-7oabgjsj0b7/metrics/"
            )
        ],
        job_arguments=[
            "--model_dir", "/opt/ml/processing/model",
            "--test_input", "/opt/ml/processing/input/test_prompts.jsonl",
            "--output_file", "/opt/ml/processing/evaluation/evaluation.json"
        ],
        property_files=[property_file]
    )

    return step