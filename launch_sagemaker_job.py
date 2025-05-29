import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# AWS and SageMaker Session
sess = sagemaker.Session()

# Explicitly specify the SageMaker execution role ARN (replace with your actual role ARN)
sagemaker_role = "arn:aws:iam::400621361756:role/service-role/AmazonSageMaker-ExecutionRole-20250528T085874"
print(f"Using SageMaker Execution Role: {sagemaker_role}")

region = sess.boto_region_name
s3_bucket_name = "qwen-finetune-data"  # Your specified bucket

# Training script and source directory
source_dir_s3_path = (
    f"s3://{s3_bucket_name}/qwen3-medical-finetune-mix/source/sourcedir.tar.gz"
)
entry_point = "train.py"

# S3 paths for input data and output
s3_input_train = f"s3://{s3_bucket_name}/prepared_medical_mix_data/train.jsonl"
s3_input_eval = f"s3://{s3_bucket_name}/prepared_medical_mix_data/eval.jsonl"
s3_output_path = f"s3://{s3_bucket_name}/qwen3-medical-finetune/output"

# Hyperparameters for training
hyperparameters = {
    "train_data_path": "/opt/ml/input/data/train/train.jsonl",
    "eval_data_path": "/opt/ml/input/data/eval/eval.jsonl",
    "output_dir": "/opt/ml/model",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "optim": "paged_adamw_32bit",
    "save_steps": 100,
    "logging_steps": 10,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "max_steps": 200,  # Small number for testing
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "eval_steps": 50,
    "max_seq_length": 512,
}

# --- Create HuggingFace Estimator ---
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=None,
    code_location=source_dir_s3_path,
    role=sagemaker_role,
    instance_type="ml.g5.xlarge",
    instance_count=1,
    pytorch_version="2.5.1",
    transformers_version="4.49.0",
    py_version="py311",
    hyperparameters=hyperparameters,
    output_path=s3_output_path,
    max_run=7200,
    keep_alive_period_in_seconds=0,
    volume_size=30,
)

# Data channels
train_input = sagemaker.inputs.TrainingInput(
    s3_input_train, content_type="application/jsonlines"
)
eval_input = sagemaker.inputs.TrainingInput(
    s3_input_eval, content_type="application/jsonlines"
)

# --- Start Training ---
try:
    huggingface_estimator.fit({"train": train_input, "eval": eval_input})
    print("Training job started successfully!")
    print(f"Job name: {huggingface_estimator.latest_training_job.name}")
except Exception as e:
    print(f"Error starting training job: {e}")
    raise
