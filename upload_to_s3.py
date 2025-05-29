import boto3
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def upload_file_to_s3(local_file, bucket, s3_key):
    """Upload a file to S3"""
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_file, bucket, s3_key)
        print(f"Successfully uploaded {local_file} to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error uploading {local_file}: {e}")
        return False


# Upload the prepared data files
bucket = "qwen-finetune-data"
upload_file_to_s3(
    "./prepared_medical_mix_data/train.jsonl",
    bucket,
    "prepared_medical_mix_data/train.jsonl",
)
upload_file_to_s3(
    "./prepared_medical_mix_data/eval.jsonl",
    bucket,
    "prepared_medical_mix_data/eval.jsonl",
)
