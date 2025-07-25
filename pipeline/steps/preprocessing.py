from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


def get_preprocessing_step(sagemaker_session, role, train_input_s3, test_input_s3):
    processor = ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        base_job_name="hermes-preprocess",
        sagemaker_session=sagemaker_session
    )

    step = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        code="src/preprocess.py",
        inputs=[
            ProcessingInput(
                source=train_input_s3,
                destination="/opt/ml/processing/train_input",
                input_name="train_input"
            ),
            ProcessingInput(
                source=test_input_s3,
                destination="/opt/ml/processing/test_input",
                input_name="test_input"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/train_output",
                destination="s3://sagemaker-studio-7oabgjsj0b7/datasets/train/",
                output_name="train_output"
            ),
            ProcessingOutput(
                source="/opt/ml/processing/test_output",
                destination="s3://sagemaker-studio-7oabgjsj0b7/datasets/eval/",
                output_name="test_output"
            )
        ],
        job_arguments=[
            "--train_input", "/opt/ml/processing/train_input/train.jsonl",
            "--test_input", "/opt/ml/processing/test_input/test.jsonl",
            "--train_output", "/opt/ml/processing/train_output/train.jsonl",
            "--test_output", "/opt/ml/processing/test_output/test.jsonl"
        ]
    )

    return step, "s3://sagemaker-studio-7oabgjsj0b7/datasets/train/train.jsonl", "s3://sagemaker-studio-7oabgjsj0b7/datasets/eval/test.jsonl"
