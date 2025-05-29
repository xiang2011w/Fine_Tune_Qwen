from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# --- Configuration ---
base_model_id = "Qwen/Qwen3-1.7B-Base"
# Path to your downloaded/extracted LoRA adapters (e.g., from S3 output)
adapter_path = "/path/to/your/downloaded_adapters_from_sagemaker_output/"  # This directory should contain adapter_config.json, adapter_model.bin, etc.

# --- Load Base Model with QLoRA config (if you trained with it) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map={"": 0},  # Or "auto"
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load LoRA Adapters ---
print(f"Loading LoRA adapters from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.eval()  # Set to evaluation mode

print("Model ready for inference.")


# --- Inference Example ---
def generate_answer(question):
    # Format the prompt similar to your training data
    # Check Qwen3's specific chat template if available and use tokenizer.apply_chat_template
    # For this example, using the same INST format as in training
    prompt = f"<s>[INST] {question} [/INST]"

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(model.device)

    print(f"\nGenerating answer for: {question}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Adjust as needed
            temperature=0.7,  # Adjust for creativity
            top_p=0.9,
            top_k=50,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,  # Important if batching or using padding
        )

    response_ids = outputs[0][
        inputs["input_ids"].shape[1] :
    ]  # Get only the generated tokens
    answer = tokenizer.decode(response_ids, skip_special_tokens=True)
    return answer


# Test
medical_question = "What are the common symptoms of influenza?"
generated_answer = generate_answer(medical_question)
print(f"Question: {medical_question}")
print(f"Generated Answer: {generated_answer}")

medical_question_2 = "How is type 2 diabetes managed?"
generated_answer_2 = generate_answer(medical_question_2)
print(f"Question: {medical_question_2}")
print(f"Generated Answer: {generated_answer_2}")
