import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os
import boto3  # Added for S3 download
import argparse  # Added for command-line arguments
from dotenv import load_dotenv  # Added for .env file loading

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# These will now be defaults, can be overridden by command-line arguments
DEFAULT_S3_BUCKET = "qwen-finetune-data"
DEFAULT_S3_KEY = (
    "medical_o1_sft_mix.json"  # Assuming this is the key for your file in the bucket
)
DEFAULT_LOCAL_DOWNLOAD_PATH = (
    "./downloaded_source.json"  # Temporary local path for the downloaded file
)

# Keys for instruction/question and response/answer in your JSON objects
# CRITICAL: Verify these keys against the actual structure of medical_o1_sft_mix.json
QUESTION_KEY = "Question"
ANSWER_KEY = "Response"

DEFAULT_OUTPUT_DIR = "./prepared_medical_mix_data"  # Changed output directory name
DEFAULT_TRAIN_FILE_JSONL = "train.jsonl"
DEFAULT_EVAL_FILE_JSONL = "eval.jsonl"
TEST_SPLIT_SIZE = 0.1  # 10% for evaluation
RANDOM_STATE = 42


# --- Helper Function to Download from S3 ---
def download_from_s3(bucket, key, download_path):
    """Downloads a file from S3."""
    s3 = boto3.client("s3")
    print(f"Attempting to download s3://{bucket}/{key} to {download_path}...")
    try:
        s3.download_file(bucket, key, download_path)
        print(f"Successfully downloaded to {download_path}")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False


# --- Helper Function to Format Data ---
def format_example(item):
    """
    Formats a single data item into the required text string for SFT.
    Example format: "<s>[INST] Question Text [/INST] Answer Text </s>"
    Adjust if Qwen3 has a very specific chat template you want to adhere to.
    For base models, this common instruction format often works well.
    """
    question = item.get(QUESTION_KEY)  # Use .get() for safer access
    answer = item.get(ANSWER_KEY)

    if question is None or answer is None:
        print(f"Warning: Missing '{QUESTION_KEY}' or '{ANSWER_KEY}' in item: {item}")
        # Decide how to handle missing keys: skip item, raise error, or use placeholder
        # For now, let's return None to filter it out later
        return None

    return {"text": f"<s>[INST] {question} [/INST] {answer} </s>"}


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare medical data for SFT.")
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=DEFAULT_S3_BUCKET,
        help="S3 bucket containing the source JSON file.",
    )
    parser.add_argument(
        "--s3_key",
        type=str,
        default=DEFAULT_S3_KEY,
        help="S3 key for the source JSON file.",
    )
    parser.add_argument(
        "--local_download_path",
        type=str,
        default=DEFAULT_LOCAL_DOWNLOAD_PATH,
        help="Local temporary path to download the S3 file.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default=QUESTION_KEY,
        help="JSON key for the question/instruction field.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default=ANSWER_KEY,
        help="JSON key for the answer/response field.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save prepared train.jsonl and eval.jsonl.",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default=DEFAULT_TRAIN_FILE_JSONL,
        help="Filename for the output training data.",
    )
    parser.add_argument(
        "--eval_filename",
        type=str,
        default=DEFAULT_EVAL_FILE_JSONL,
        help="Filename for the output evaluation data.",
    )

    args = parser.parse_args()

    # 0. Download data from S3
    if not download_from_s3(args.s3_bucket, args.s3_key, args.local_download_path):
        print("Exiting due to S3 download failure.")
        exit(1)

    # 1. Load Raw Data from the downloaded file
    print(f"Loading raw data from local file: {args.local_download_path}")
    try:
        with open(args.local_download_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print(f"Successfully loaded {len(raw_data)} records.")
    except Exception as e:
        print(f"Error loading JSON file from {args.local_download_path}: {e}")
        exit(1)

    # Verify that the assumed keys exist in the first item (optional, good for debugging)
    if raw_data and (
        args.question_key not in raw_data[0] or args.answer_key not in raw_data[0]
    ):
        print(
            f"Warning: Assumed keys '{args.question_key}' or '{args.answer_key}' not found in the first data item."
        )
        print(f"First item keys: {raw_data[0].keys()}")
        print(
            "Please verify QUESTION_KEY and ANSWER_KEY or pass them as arguments if they are different."
        )
        # Decide if this should be a fatal error or just a warning. For now, continue.

    # 2. Split Data (before formatting to save processing time on eval set if it's large)
    print(
        f"Splitting data into train and evaluation sets (eval size: {TEST_SPLIT_SIZE*100}%)..."
    )
    train_data, eval_data = train_test_split(
        raw_data, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training set size: {len(train_data)}")
    print(f"Evaluation set size: {len(eval_data)}")

    # 3. Apply Formatting
    print("Formatting datasets...")
    # Use args.question_key and args.answer_key in format_example if you modify it to accept them
    # For simplicity, format_example currently uses global-like QUESTION_KEY, ANSWER_KEY
    # which are set from args at the start of the script if not overridden.
    # To be more robust, pass args.question_key and args.answer_key to format_example
    formatted_train_data = [
        fmt_item
        for item in train_data
        if (fmt_item := format_example(item)) is not None
    ]
    formatted_eval_data = [
        fmt_item for item in eval_data if (fmt_item := format_example(item)) is not None
    ]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Save to JSONL
    train_output_path = os.path.join(args.output_dir, args.train_filename)
    eval_output_path = os.path.join(args.output_dir, args.eval_filename)

    print(f"Saving formatted training data to: {train_output_path}")
    with open(train_output_path, "w", encoding="utf-8") as f:
        for item in formatted_train_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saving formatted evaluation data to: {eval_output_path}")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        for item in formatted_eval_data:
            f.write(json.dumps(item) + "\n")

    # Clean up downloaded file
    if os.path.exists(args.local_download_path):
        print(f"Cleaning up downloaded file: {args.local_download_path}")
        os.remove(args.local_download_path)

    print("\nData preparation complete.")
    print(f"Prepared data saved in: {args.output_dir}")
    print(f"Next steps: Upload '{train_output_path}' and '{eval_output_path}' to S3.")
    print(
        f"Suggested S3 path for prepared data: s3://{args.s3_bucket}/prepared_medical_mix_data/"
    )
