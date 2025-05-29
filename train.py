import argparse
import os
import logging
import sys

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ConstantLR,
    LinearLR,
    CosineAnnealingLR,
)
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOGLEVEL", "INFO")),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Add for debugging
logger.info("=" * 50)
logger.info("STARTING TRAIN.PY SCRIPT")
logger.info("=" * 50)

# --- Add these lines for version debugging ---
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"LRScheduler import successful: {LRScheduler is not None}")
try:
    import transformers

    logger.info(f"Transformers version: {transformers.__version__}")
except ImportError:
    logger.error("Transformers library not found.")
try:
    import bitsandbytes

    logger.info(f"BitsandBytes version: {bitsandbytes.__version__}")
except ImportError:
    logger.error("BitsandBytes library not found.")
try:
    import peft

    logger.info(f"PEFT version: {peft.__version__}")
except ImportError:
    logger.error("PEFT library not found.")
try:
    import accelerate

    logger.info(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    logger.error("Accelerate library not found.")
try:
    import trl

    logger.info(f"TRL version: {trl.__version__}")
except ImportError:
    logger.error("TRL library not found.")

# Check for the problematic attribute directly
try:
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        if hasattr(torch.cuda, "is_bf16_supported"):
            logger.info(f"torch.cuda.is_bf16_supported available: True")
            logger.info(
                f"torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}"
            )
        else:
            logger.info(f"torch.cuda.is_bf16_supported available: False")
        # This check is for the misspelled attribute, it should ideally be False or not present
        logger.info(
            f"torch.cuda.is_bf1oter_available (misspelled) available: {hasattr(torch.cuda, 'is_bf1oter_available')}"
        )
except Exception as e:
    logger.error(f"Error during CUDA attribute checks: {e}")
# --- End of added lines ---


def main(args):
    logger.info("Starting main() function")

    # 1. Load Tokenizer and Model
    # ---
    # Using Qwen2.5 with transformers>=4.51.0 from requirements.txt
    model_id = (
        "Qwen/Qwen2.5-1.5B"  # Back to Qwen2.5 now that we have newer transformers
    )

    # QLoRA Configuration (4-bit quantization)
    logger.info("Creating BitsAndBytesConfig...")
    # BitsAndBytesConfig is used to tell the Hugging Face transformers library how to load the pre-trained model with quantization enabled.
    # configure the 4-bit quantization parameters for the base Qwen2.5-1.5B model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Recommended: "nf4" or "fp4"
        bnb_4bit_compute_dtype=torch.bfloat16,  # For Ampere GPUs and newer
        # bnb_4bit_compute_dtype=torch.float16, # For older GPUs
        bnb_4bit_use_double_quant=True,
    )
    logger.info("BitsAndBytesConfig created successfully")

    logger.info(f"Loading base model: {model_id} with BitsAndBytesConfig.")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},  # Load model on GPU 0
            # device_map="auto", # For multi-GPU, but for 1.5B on a single GPU, {"":0} is fine
            trust_remote_code=True,  # Qwen models may require this
            # torch_dtype=torch.bfloat16, # Already handled by bnb_config compute_dtype
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    model.config.use_cache = False  # Recommended for fine-tuning
    logger.info("Model configuration updated")

    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Common practice
        logger.info("Padding token set to EOS token")

    logger.info("Model and tokenizer loaded.")

    # Prepare model for k-bit training (important for QLoRA)
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration
    # Find target_modules: print(model) and look for linear layers like q_proj, v_proj, etc.
    # Common for Qwen-like models:
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # LoraConfig is used to define the properties of the Low-Rank Adaptation layers that will be added to the quantized base model.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    logger.info("PEFT model created.")
    model.print_trainable_parameters()

    # 2. Load and Preprocess Dataset
    # ---
    # The dataset should be on the SageMaker instance, typically copied from S3.
    # args.train_data_dir is provided by SageMaker.
    logger.info(f"Loading training dataset from: {args.train_data_path}")
    try:
        # Correctly load the JSON Lines file for training
        train_dataset = load_dataset(
            "json", data_files=args.train_data_path, split="train"
        )
        logger.info(
            f"Training dataset loaded successfully. Features: {train_dataset.features}"
        )
        logger.info(f"Number of training examples: {len(train_dataset)}")
    except Exception as e:
        logger.error(f"Error loading training dataset from {args.train_data_path}: {e}")
        # Log fallback attempts if any, or just raise
        raise

    logger.info(f"Loading evaluation dataset from: {args.eval_data_path}")
    try:
        # Correctly load the JSON Lines file for evaluation
        eval_dataset = load_dataset(
            "json", data_files=args.eval_data_path, split="train"
        )  # Often, eval files also just have a 'train' split name
        logger.info(
            f"Evaluation dataset loaded successfully. Features: {eval_dataset.features}"
        )
        logger.info(f"Number of evaluation examples: {len(eval_dataset)}")
    except Exception as e:
        logger.error(
            f"Error loading evaluation dataset from {args.eval_data_path}: {e}"
        )
        # Log fallback attempts if any, or just raise
        raise

    # Tokenization will be handled by SFTTrainer by providing the 'dataset_text_field'
    # or by providing a pre-tokenized dataset. SFTTrainer prefers a text field.

    # 3. Training
    # ---
    output_dir = args.output_dir  # SageMaker's output directory /opt/ml/model

    # Determine if evaluation should be performed
    do_eval = eval_dataset is not None and args.eval_steps > 0

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,  # "paged_adamw_32bit" is good for QLoRA
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,  # 0.3 is a common value for QLoRA
        max_steps=args.max_steps if args.max_steps > 0 else None,
        num_train_epochs=(
            args.num_train_epochs if args.max_steps <= 0 else 0
        ),  # Trainer prefers max_steps if both specified
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,  # e.g., "cosine" or "linear"
        report_to="tensorboard",  # or "wandb" if configured
        fp16=False,  # Not used when using BitsAndBytes 4-bit
        bf16=args.bf16,  # Use bf16 hyperparameter passed from launch script
        eval_steps=args.eval_steps if do_eval else None,
        # load_best_model_at_end=True if do_eval else False, # Optional: requires save_strategy to match evaluation_strategy
        # group_by_length=True, # Can speed up training by batching similar length sequences
    )

    # Data collator - SFTTrainer handles this internally if dataset_text_field is specified.
    # If you pre-tokenizee, you might need a DataCollatorForLanguageModeling.
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Initialize SFTTrainer
    # ---
    # Ensure your dataset has the expected column names for SFTTrainer
    # (e.g., 'text', or 'prompt'/'response' if using a formatting_func_map)
    # For simplicity, if your JSONL has a 'text' field that's already formatted for instruction tuning:
    logger.info("Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
            # formatting_func=formatting_func, # Or provide a formatting function if your data needs structuring
            # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), # Or a more specific collator if needed
        )
        logger.info("SFTTrainer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SFTTrainer: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Training complete. Saving model to {output_dir}")
    # SFTTrainer automatically saves the LoRA adapter.
    # To save the full model (if you wanted to merge later, not typical for SageMaker deployment of LoRA):
    # trainer.save_model(output_dir) # This saves adapter & tokenizer
    # For just the adapter:
    model.save_pretrained(output_dir)  # Saves adapter_model.bin
    tokenizer.save_pretrained(output_dir)  # Save tokenizer

    logger.info(f"LoRA adapters and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Model Fine-Tuning")

    # Data and Model Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/ml/model",
        help="Directory to save the model and outputs.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/opt/ml/input/data/train/train.jsonl",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="/opt/ml/input/data/eval/eval.jsonl",
        help="Path to the evaluation data file.",
    )

    # Model and LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)  # LoRA rank
    parser.add_argument("--lora_alpha", type=int, default=32)  # LoRA alpha
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU for training.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument(
        "--max_steps", type=int, default=-1
    )  # -1 means use num_train_epochs
    parser.add_argument(
        "--num_train_epochs", type=int, default=1
    )  # Alternative to max_steps
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument(
        "--max_seq_length", type=int, default=2048
    )  # Adjust based on your data & VRAM
    parser.add_argument(
        "--packing", type=bool, default=False
    )  # Set to True for SFTTrainer to pack sequences
    parser.add_argument(
        "--eval_steps", type=int, default=0
    )  # Add eval_steps argument, default to 0 (no eval) if not provided
    parser.add_argument(
        "--bf16", type=bool, default=True, help="Whether to use bf16 precision."
    )

    args = parser.parse_args()

    # If max_steps is set, it overrides num_train_epochs in TrainingArguments
    # SFTTrainer/Trainer handles the logic of max_steps vs num_train_epochs.
    # We adjust num_train_epochs to 0 if max_steps is given, as some Trainer versions
    # might prioritize num_train_epochs if it's a positive integer.
    if args.max_steps > 0:
        logger.info(
            f"max_steps is set to {args.max_steps}, num_train_epochs will be effectively managed by Trainer."
        )
        # No need to explicitly set args.num_train_epochs to None or 0 here,
        # TrainingArguments constructor will handle it.
        # Just ensure max_steps is passed correctly.
    elif args.num_train_epochs <= 0 and args.max_steps <= 0:
        logger.warning(
            "Neither max_steps nor num_train_epochs is set to a positive value. Training might not run as expected. Defaulting num_train_epochs to 1 for TrainingArguments if not otherwise specified."
        )
        # TrainingArguments defaults num_train_epochs to 3.0 if not set, which is fine.

    main(args)
