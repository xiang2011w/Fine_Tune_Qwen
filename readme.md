# Project Report: Fine-tuning Qwen2.5-1.5B on a Medical Dataset using QLoRA on AWS SageMaker

**Table of Contents:**
1.  [Executive Summary](#1-executive-summary)
2.  [Project Goal](#2-project-goal)
3.  [Model Selection: Qwen/Qwen2.5-1.5B](#3-model-selection-qwenqwen25-15b)
4.  [Fine-tuning Technique: QLoRA](#4-fine-tuning-technique-qlora)
    *   [4.1 What is LoRA?](#41-what-is-lora)
    *   [4.2 What is QLoRA?](#42-what-is-qlora)
    *   [4.3 Benefits of QLoRA](#43-benefits-of-qlora)
    *   [4.4 QLoRA Configuration Details](#44-qlora-configuration-details)
        *   [4.4.1 `BitsAndBytesConfig`: Quantizing the Base Model](#441-bitsandbytesconfig-quantizing-the-base-model)
        *   [4.4.2 `LoraConfig`: Defining the Trainable Adapters](#442-loraconfig-defining-the-trainable-adapters)
        *   [4.4.3 The `paged_adamw_32bit` Optimizer](#443-the-paged_adamw_32bit-optimizer)
5.  [Dataset Preparation](#5-dataset-preparation)
6.  [AWS SageMaker and S3 Infrastructure Setup](#6-aws-sagemaker-and-s3-infrastructure-setup)
7.  [Frameworks and Libraries](#7-frameworks-and-libraries)
8.  [Training Process Overview](#8-training-process-overview)
9.  [Results and Key Metrics (Expanded)](#9-results-and-key-metrics-expanded)
10. [Conclusion and Next Steps](#10-conclusion-and-next-steps)

### 1. Executive Summary
This project successfully fine-tuned the Qwen/Qwen2.5-1.5B language model on a custom medical dataset. Leveraging the Parameter-Efficient Fine-Tuning (PEFT) technique QLoRA, the model was adapted for specialized tasks within the medical domain while significantly reducing computational resource requirements. The entire process was orchestrated using AWS SageMaker for training and AWS S3 for data and model storage. The training completed successfully, demonstrating the viability of this approach for custom LLM adaptation.

### 2. Project Goal
The primary goal of this project was to adapt a general-purpose Large Language Model (LLM) to better understand and generate text relevant to the medical domain. This involved fine-tuning the chosen model on a curated medical dataset to improve its performance on specific downstream tasks such as medical question answering, information extraction, or summarization.

### 3. Model Selection: Qwen/Qwen2.5-1.5B
The **Qwen/Qwen2.5-1.5B** model was selected for this project due to several key advantages:
*   **Strong Base Performance:** The Qwen series of models, developed by Alibaba Cloud, are known for their strong performance across a variety of language tasks.
*   **Manageable Size:** The 1.5 billion parameter variant is relatively small compared to state-of-the-art frontier models. This makes it more feasible for fine-tuning with limited GPU resources, especially when combined with techniques like QLoRA.
*   **Open Source and Accessibility:** Being an open-source model, it allows for greater flexibility in customization and deployment.
*   **Recent Architecture:** Qwen2.5 incorporates recent advancements in LLM architecture, making it a capable base for fine-tuning.

### 4. Fine-tuning Technique: QLoRA
To efficiently fine-tune the Qwen2.5-1.5B model, **QLoRA (Quantized Low-Rank Adaptation)** was employed.

#### 4.1 What is LoRA?
LoRA is a PEFT method that freezes the pre-trained model weights and injects trainable rank decomposition matrices (called LoRA adapters) into each layer of the Transformer architecture. During fine-tuning, only these significantly smaller adapter weights are updated, drastically reducing the number of trainable parameters and thus memory requirements.

#### 4.2 What is QLoRA?
QLoRA takes LoRA a step further by quantizing the frozen pre-trained model weights to a lower precision, typically 4-bit (using NormalFloat4 or NF4 data type). This dramatically reduces the memory footprint of the base model. To maintain performance despite this aggressive quantization, QLoRA incorporates several innovations:
1.  **4-bit NormalFloat (NF4):** An information-theoretically optimal quantization data type for normally distributed weights.
2.  **Double Quantization (DQ):** A technique to further reduce the memory footprint by quantizing the quantization constants themselves.
3.  **Paged Optimizers:** Leverages NVIDIA unified memory to manage memory spikes during gradient checkpointing, preventing out-of-memory errors when processing long sequences.

#### 4.3 Benefits of QLoRA
*   **Massive Memory Reduction:** Enables fine-tuning of large models (like the 1.5B parameter Qwen model) on consumer-grade or single enterprise GPUs (like the `ml.g5.xlarge` used here) that would otherwise be insufficient for full fine-tuning.
*   **Comparable Performance:** Achieves performance very close to full 16-bit fine-tuning despite the aggressive quantization and reduced trainable parameters.
*   **Faster Iteration:** While the primary benefit is memory, reduced data movement and smaller gradients can sometimes lead to faster training iterations compared to full fine-tuning.

#### 4.4 QLoRA Configuration Details
The core idea of QLoRA is to make fine-tuning very large models feasible on less powerful hardware by:
1.  **Quantizing the large, pre-trained base model** to a very low precision (4-bit). This drastically reduces its memory footprint. These weights remain frozen.
2.  **Adding tiny, trainable "adapter" layers (LoRA)** that learn the task-specific information. Only these adapters are updated during fine-tuning.

Here's how `BitsAndBytesConfig` and `LoraConfig` facilitate this:

##### 4.4.1 `BitsAndBytesConfig`: Quantizing the Base Model
The `BitsAndBytesConfig` object, from the `bitsandbytes` library, is used to tell the Hugging Face `transformers` library how to load the pre-trained model with quantization enabled.

**Purpose:** To configure the 4-bit quantization parameters for the base Qwen2.5-1.5B model.

**Key Parameters Used (Conceptual Example in `train.py`):**
```python
import torch
from transformers import BitsAndBytesConfig

# --- BitsAndBytesConfig for QLoRA ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Main flag to enable 4-bit quantization
    bnb_4bit_quant_type="nf4",              # Specifies the quantization type: "nf4" (NormalFloat4) is recommended for QLoRA
                                            # as it's information-theoretically optimal for normally distributed weights.
    bnb_4bit_compute_dtype=torch.bfloat16,  # The compute data type for matrix multiplications.
                                            # While weights are stored in 4-bit, computations are often upcasted
                                            # to a more stable type like bfloat16 (or float16) to maintain performance.
                                            # bfloat16 is generally preferred on newer GPUs (like A10G).
    bnb_4bit_use_double_quant=True,         # Enables a second quantization pass on the quantization constants themselves.
                                            # This can save an additional ~0.4 bits per parameter, further reducing memory.
)

# This bnb_config is then passed when loading the model:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name_or_path,
#     quantization_config=bnb_config,
#     device_map="auto",  # Or specific device like "cuda:0"
#     trust_remote_code=True # If required by the model
# )
```

**Explanation of Parameters:**
*   `load_in_4bit=True`: This is the master switch. When `True`, the model's weights are loaded and stored in 4-bit precision.
*   `bnb_4bit_quant_type="nf4"`:
    *   QLoRA introduced "NormalFloat4" (NF4) as an optimal data type for quantizing weights that are typically normally distributed (which is common for neural network weights).
    *   Alternative could be `"fp4"` (standard 4-bit float), but `nf4` is generally better for QLoRA.
*   `bnb_4bit_compute_dtype=torch.bfloat16`:
    *   Even though the weights are stored in 4-bit, performing matrix multiplications directly in 4-bit can lead to significant precision loss and poor model performance.
    *   So, during computation (e.g., a forward pass), the 4-bit weights are de-quantized on the fly to this `compute_dtype` (e.g., `bfloat16` or `float16`). The computation happens in this higher precision, and then the result might be re-quantized if necessary for subsequent operations or storage of activations.
    *   `bfloat16` offers a good balance of range and precision and is well-supported on modern GPUs.
*   `bnb_4bit_use_double_quant=True`:
    *   "Double Quantization" is a technique where the quantization constants (themselves small floating-point numbers used in the first quantization step) are *also* quantized. This yields further memory savings for the frozen base model.

**Outcome:** After loading the model with this `bnb_config`, the massive Qwen2.5-1.5B model consumes significantly less GPU VRAM because its weights are stored in 4-bit. These weights are **frozen** and will not be updated during training.

##### 4.4.2 `LoraConfig`: Defining the Trainable Adapters
The `LoraConfig` object, from the Hugging Face `PEFT` (Parameter-Efficient Fine-Tuning) library, defines the properties of the Low-Rank Adaptation layers that will be added to the quantized base model.

**Purpose:** To specify which parts of the base model get LoRA adapters, the dimensionality (rank) of these adapters, and other LoRA-specific hyperparameters.

**Key Parameters Used (Conceptual Example in `train.py`):**
```python
from peft import LoraConfig, get_peft_model, TaskType

# --- LoraConfig for PEFT ---
lora_config = LoraConfig(
    r=16,                                   # The rank of the update matrices (A and B).
                                            # A higher rank means more expressive power but more trainable parameters.
                                            # Common values are 8, 16, 32, 64.
    lora_alpha=32,                          # The LoRA scaling factor. Often set to r or 2*r.
                                            # It's like a learning rate for the LoRA weights.
    target_modules=[                        # A list of module names in the base model to apply LoRA to.
        "q_proj",                           # Query projection in attention
        "k_proj",                           # Key projection in attention
        "v_proj",                           # Value projection in attention
        "o_proj",                           # Output projection in attention
        "gate_proj",                        # Part of the MLP/FeedForward layer
        "up_proj",                          # Part of the MLP/FeedForward layer
        "down_proj"                         # Part of the MLP/FeedForward layer
    ],                                      # For Qwen models, you might need to inspect the model architecture
                                            # to find the exact names of linear layers in attention and MLP blocks.
                                            # Sometimes a helper function or regex can identify all linear layers.
    lora_dropout=0.05,                      # Dropout probability for LoRA layers. Helps prevent overfitting of adapters.
    bias="none",                            # Specifies how to handle biases in LoRA layers.
                                            # "none": Biases are not trained.
                                            # "all": All biases are trained.
                                            # "lora_only": Only biases in LoRA layers are trained.
                                            # "none" is common for QLoRA.
    task_type=TaskType.CAUSAL_LM,           # Specifies the task type. For models like Qwen, it's Causal Language Modeling.
)

# This lora_config is then used to wrap the base model:
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters() # This will show a very small percentage of trainable params
```

**Explanation of Parameters:**
*   `r` (rank): This is a crucial hyperparameter. LoRA decomposes a large weight update matrix \( \Delta W \) into two smaller matrices \( A \) and \( B \) (i.e., \( \Delta W = BA \)), where \( A \) has dimensions \( r \times k \) and \( B \) has dimensions \( d \times r \). `r` is much smaller than \( d \) or \( k \). A smaller `r` means fewer trainable parameters but potentially less capacity for the adapter to learn.
*   `lora_alpha`: This acts as a scaling factor for the LoRA activations (\( \frac{\alpha}{r} \)). It helps balance the influence of the pre-trained weights and the LoRA adaptation. A common heuristic is to set `lora_alpha` to be equal to `r` or `2*r`.
*   `target_modules`: This is critical. You specify which existing linear layers in the base model should have LoRA adapters applied to them. Typically, these are the query, key, value, and output projection layers in the self-attention mechanism, and sometimes layers within the feed-forward/MLP blocks. The exact names depend on the model architecture (e.g., for Qwen, they might be `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
*   `lora_dropout`: Standard dropout applied to the outputs of the LoRA layers to help prevent overfitting of the adapter weights.
*   `bias="none"`: For QLoRA, it's common not to train any biases or only train biases within the LoRA layers themselves (if any were added, though typically LoRA adds weight matrices). Freezing biases further reduces trainable parameters.
*   `task_type=TaskType.CAUSAL_LM`: Informs PEFT about the type of model, which can sometimes influence how adapters are configured or how outputs are handled.

**Outcome:** The `get_peft_model(model, lora_config)` function takes the 4-bit quantized base model and injects these small, trainable LoRA adapter layers into the specified `target_modules`. Now, only the weights of these LoRA adapters (matrices A and B) are marked as trainable. The `model.print_trainable_parameters()` utility would show that typically less than 1% (often much less, like 0.1-0.5%) of the total model parameters are now trainable.

##### 4.4.3 The `paged_adamw_32bit` Optimizer
While standard AdamW could be used, the `paged_adamw_32bit` optimizer (often provided by `bitsandbytes`) is specifically designed to work well with QLoRA's memory-saving techniques.

*   **Paged Memory Management:** QLoRA can use a feature called "paged optimizers" which leverages NVIDIA's unified memory. This allows the optimizer states (which can be large, especially for Adam-like optimizers) to be "paged" between GPU VRAM and CPU RAM.
*   **Preventing OOMs:** This is particularly useful during operations like gradient checkpointing (which is often used to save memory during the backward pass but can have memory spikes for optimizer states). Paged optimizers help prevent out-of-memory (OOM) errors during these spikes by offloading optimizer states to CPU RAM if GPU VRAM is exhausted.
*   **32-bit States:** It maintains optimizer states (like momentum and variance) in 32-bit precision for stability, even if the model parameters or gradients are in lower precision (like bfloat16 for computation).

**How it fits:**
When you define your `TrainingArguments` for the `SFTTrainer`, you'd specify:
```python
# In TrainingArguments
training_args = TrainingArguments(
    # ... other args ...
    optim="paged_adamw_32bit",
    # ...
)
```
The trainer then uses this optimizer, which is aware of the QLoRA setup and can manage its states more efficiently in the memory-constrained environment created by loading a large model.

---

### 5. Dataset Preparation
The custom medical dataset was processed into a format suitable for supervised fine-tuning (SFT) with the `SFTTrainer`.

*   **Source:** A proprietary medical text corpus.
*   **Processing Steps:**
    1.  **Data Structuring:** The raw medical data was transformed into instruction-response pairs or a single text field formatted for instruction tuning. A common format is:
        ```
        ### Human: [Instruction/Question]
        ### Assistant: [Desired Response]
        ```
        This combined text was stored in a field named `"text"`.
    2.  **JSON Lines Format (.jsonl):** The structured data was converted into the JSON Lines format, where each line in the file is a valid JSON object representing a single training example.
        Example line in `train.jsonl` or `eval.jsonl`:
        `{"text": "### Human: What are the common symptoms of influenza?\n### Assistant: Common symptoms of influenza include fever, cough, sore throat, muscle aches, fatigue, and headache."}`
    3.  **Splitting:** The dataset was split into a training set (`train.jsonl`) and an evaluation set (`eval.jsonl`) to monitor training progress and prevent overfitting.
        *   Training examples: 22,398
        *   Evaluation examples: 2,489
*   **Loading:** The `datasets` library was used within the training script to load these `.jsonl` files.

### 6. AWS SageMaker and S3 Infrastructure Setup

*   **Amazon S3 (Simple Storage Service):**
    S3 was used for storing all project artifacts:
    *   **Input Data:**
        *   Training data: `s3://your-bucket/your-project/data/train/train.jsonl`
        *   Evaluation data: `s3://your-bucket/your-project/data/eval/eval.jsonl`
    *   **Source Code:** The training script (`train.py`) and its dependencies (like `requirements.txt`) were packaged into `sourcedir.tar.gz` and uploaded to:
        `s3://your-bucket/your-project/source/sourcedir.tar.gz`
    *   **Model Output:** The fine-tuned LoRA adapters and tokenizer configuration were saved by SageMaker to:
        `s3://your-bucket/your-project/output/[job-name]/output/model.tar.gz`

*   **Amazon SageMaker:**
    SageMaker was used to orchestrate and execute the distributed training job.
    *   **Estimator:** A `HuggingFace` estimator from the SageMaker Python SDK was configured:
        *   `entry_point`: `"train.py"`
        *   `source_dir`: S3 path to `sourcedir.tar.gz` (implicitly handled by `code_location` and `entry_point` if `source_dir` is `None` and `code_location` is set, or directly if `source_dir` points to the S3 URI of the tarball).
        *   `role`: An IAM role with necessary SageMaker permissions
        *   `instance_type`: `"ml.g5.xlarge"` (a GPU instance suitable for QLoRA fine-tuning).
        *   `instance_count`: 1
        *   `pytorch_version`: `"2.5.1"`
        *   `transformers_version`: `"4.49.0"` (base container version)
        *   `py_version`: `"py311"`
    *   **Hyperparameters:** Passed to the `train.py` script:
        *   `model_name_or_path`: `"Qwen/Qwen2.5-1.5B"` (defined in `train.py`)
        *   `train_data_path`: `"/opt/ml/input/data/train/train.jsonl"`
        *   `eval_data_path`: `"/opt/ml/input/data/eval/eval.jsonl"`
        *   `output_dir`: `"/opt/ml/model"` (SageMaker's designated model output directory)
        *   `per_device_train_batch_size`: 2
        *   `gradient_accumulation_steps`: 4 (Effective batch size: 2 * 4 = 8)
        *   `learning_rate`: 2e-4
        *   `lr_scheduler_type`: `"cosine"`
        *   `optim`: `"paged_adamw_32bit"`
        *   `bf16`: `True` (enabling mixed-precision training with bfloat16)
        *   `max_steps`: 200 (for a relatively short training run)
        *   `logging_steps`: 10
        *   `save_steps`: 100
        *   `eval_steps`: 50
    *   **Input Data Channels:**
        *   `train`: Pointing to the S3 location of `train.jsonl`.
        *   `eval`: Pointing to the S3 location of `eval.jsonl`.

### 7. Frameworks and Libraries
The project utilized a suite of Python libraries common in the LLM fine-tuning ecosystem:
*   **PyTorch (2.5.1):** The core deep learning framework.
*   **Hugging Face Transformers (4.49.0 base, likely >=4.51.0 in script):** For accessing the Qwen model, tokenizer, and `TrainingArguments`.
*   **Hugging Face TRL (Transformer Reinforcement Learning - 0.15.2):** Provided the `SFTTrainer` for supervised fine-tuning.
*   **Hugging Face PEFT (Parameter-Efficient Fine-Tuning - 0.14.0):** For implementing QLoRA (`LoraConfig`, `get_peft_model`).
*   **BitsAndBytes (0.45.5):** Essential for QLoRA, providing 4-bit quantization and related optimizations.
*   **Hugging Face Datasets:** For efficient loading and handling of the `.jsonl` datasets.
*   **Hugging Face Accelerate (1.4.0):** For simplifying distributed training and mixed-precision settings (though used on a single GPU here, it's often a dependency).
*   **SageMaker Python SDK:** For defining and launching the SageMaker training job.

### 8. Training Process Overview
The `train.py` script executed the following major steps within the SageMaker environment:
1.  **Initialization:**
    *   Parsed command-line arguments (hyperparameters passed by SageMaker).
    *   Set up logging.
    *   Initialized `BitsAndBytesConfig` for 4-bit QLoRA.
    *   Loaded the `Qwen/Qwen2.5-1.5B` model with the quantization config.
    *   Loaded the corresponding tokenizer.
2.  **PEFT Setup:**
    *   Created a `LoraConfig` specifying LoRA parameters (rank, alpha, target modules).
    *   Applied LoRA to the base model using `get_peft_model`.
3.  **Data Loading:**
    *   Loaded `train.jsonl` and `eval.jsonl` using `load_dataset("json", ...)`.
4.  **Trainer Setup:**
    *   Initialized `TrainingArguments` with parameters like learning rate, batch size, scheduler, output directories, etc.
    *   Initialized `SFTTrainer` with the PEFT model, training arguments, datasets, and tokenizer.
5.  **Training:**
    *   Called `trainer.train()`. The training proceeded for `max_steps=200`.
    *   The learning rate decayed from `2e-4` towards `0.0` following a cosine schedule.
    *   Metrics (loss, learning rate, mean token accuracy) were logged every 10 steps. Evaluation metrics were logged every 50 steps.
6.  **Model Saving:**
    *   Upon completion, the trained LoRA adapters and tokenizer were saved to `/opt/ml/model` using `trainer.save_model()`. SageMaker automatically packaged this directory and uploaded it to the specified S3 output path.

### 9. Results and Key Metrics (Expanded)

The fine-tuning job completed successfully after 200 training steps, taking approximately 6 minutes and 5 seconds on an `ml.g5.xlarge` instance. The key metrics observed during this process provide insights into the model's learning behavior.

**a. Training Loss:**

*   **Trend:** The training loss is a primary indicator of how well the model is learning to predict the target sequences in the training dataset. A decreasing loss signifies that the model's predictions are getting closer to the actual ground truth data.
    *   **Initial Loss (Step 10):** `1.6847`
    *   **Mid-Training Loss (Step 100):** `1.3590`
    *   **Final Loss (Step 200):** `1.2933` (as per the last logged training step)
*   **Interpretation:**
    *   The loss consistently decreased from the initial steps through to the end of the 200 steps. This is a positive sign, indicating that the model was actively learning and adapting its LoRA weights to the patterns in the medical dataset.
    *   The rate of decrease was steeper in the initial phase and then gradually became less pronounced, which is typical. Early in training, the model makes larger adjustments, and as it converges, the improvements become more incremental.
*   **Is it "Good Enough"?**
    *   "Good enough" is relative and depends heavily on the specific downstream task, the complexity of the dataset, and the desired performance benchmarks.
    *   A final loss of `1.2933` on its own doesn't definitively state if the model is excellent or poor. However, the consistent downward trend is crucial and positive.
    *   To truly assess if it's "good enough," we would need to:
        1.  **Evaluate on a held-out test set:** Using metrics relevant to the medical task (e.g., ROUGE for summarization, F1/accuracy for classification/QA, BLEU for translation-like tasks).
        2.  **Compare to a baseline:** How does this compare to the base Qwen2.5-1.5B model (without fine-tuning) on the same tasks? Or against other fine-tuned models?
        3.  **Human Evaluation:** For generative tasks, human assessment of output quality is often necessary.
    *   For a short 200-step fine-tuning run, this loss reduction is a promising start. It suggests the model is receptive to the new data.

**b. Learning Rate (LR):**

*   **Initial Setting:** The learning rate was initialized at `2e-4` (or `0.0002`) as specified in the hyperparameters.
*   **Scheduler:** A `cosine` learning rate scheduler was used (`lr_scheduler_type="cosine"`).
*   **Behavior:**
    *   **Step 10:** LR was `~0.000199` (very close to the initial 2e-4, as expected for a cosine schedule at the beginning).
    *   **Step 100:** LR was `~0.0001` (halfway through the 200 steps, the cosine schedule would have reduced it significantly).
    *   **Step 200 (Final):** LR was `0.0` (the cosine scheduler typically anneals the learning rate to zero or a very small minimum value by the end of the scheduled steps).
*   **Interpretation:** The learning rate behaved exactly as expected for a cosine annealing schedule over 200 steps. This gradual reduction is crucial for stable convergence, allowing the model to make large updates initially and then fine-tune its weights more delicately as it approaches an optimal solution.

**c. Mean Token Accuracy (Logged as `train_accuracy` or similar):**

*   **Trend:** This metric, logged during training, indicates the average accuracy of the model in predicting the next token on the training data.
    *   **Step 10:** `0.6214` (62.14%)
    *   **Step 100:** `0.6701` (67.01%)
    *   **Step 200 (Final):** `0.6803` (68.03%)
*   **Interpretation:**
    *   The mean token accuracy showed a consistent upward trend throughout the 200 training steps. This directly complements the decreasing loss, providing another view that the model is improving its ability to generate sequences that match the training examples.
    *   An increase from ~62% to ~68% in 200 steps is a noticeable improvement, suggesting effective learning.
*   **Caveat:** This accuracy is on the *training data*. While a good sign, it doesn't directly tell us about generalization to unseen data.

**d. Overfitting:**

*   **What is Overfitting?** Overfitting occurs when a model learns the training data too well, including its noise and specific idiosyncrasies, to the point where its performance on new, unseen data (like an evaluation or test set) degrades.
*   **Indicators from Logs (Limited for Overfitting Assessment):**
    *   The provided logs primarily show *training* metrics. To properly assess overfitting, we need to compare training metrics with *evaluation* metrics (e.g., `eval_loss`, `eval_accuracy`) collected periodically during training (e.g., every `eval_steps`).
    *   If `eval_loss` starts to increase while `train_loss` continues to decrease, or if `eval_accuracy` plateaus or drops while `train_accuracy` keeps rising, these are strong signs of overfitting.
*   **Current Assessment (Based on available logs):**
    *   With only 200 training steps and a relatively large dataset (22k training examples), severe overfitting is less likely to be a major issue *yet*, but it's always a concern in fine-tuning.
    *   The `SFTTrainer` was configured with `eval_dataset` and `eval_steps=50`. The logs show evaluation runs at step 50, 100, 150, and 200.
        *   **Eval Loss @ Step 50:** `1.3695`
        *   **Eval Loss @ Step 100:** `1.3269`
        *   **Eval Loss @ Step 150:** `1.3149`
        *   **Eval Loss @ Step 200:** `1.3099`
    *   **Interpretation of Eval Loss:** The evaluation loss is also consistently decreasing. This is a very good sign! It suggests that the model is not just memorizing the training data but is also generalizing its learned knowledge to the unseen evaluation data.
    *   **Conclusion on Overfitting (for this run):** Based on the decreasing `eval_loss` alongside decreasing `train_loss`, there is **no clear evidence of overfitting within these 200 steps.** The model appears to be generalizing well to the evaluation set.
*   **Mitigation Strategies (if overfitting were observed):**
    *   **Early Stopping:** Stop training when evaluation performance starts to degrade. (This is often a callback in `Trainer`).
    *   **Regularization:** Techniques like weight decay (already part of AdamW optimizer), dropout (though less common to add more in LoRA adapters).
    *   **More Data:** Augmenting the training dataset.
    *   **Simpler Model/Adapters:** Reducing LoRA rank (`r`).
    *   **Adjusting Learning Rate:** A smaller learning rate might sometimes help.

**e. Training Throughput and Efficiency:**

*   **Samples per Second:** The logs show `train_samples_per_second` around `12.29 - 12.3`. This indicates how many training examples the system processed per second.
*   **Steps per Second:** `train_steps_per_second` was around `1.53 - 1.54`.
*   **GPU Utilization:** While not directly in these logs, `ml.g5.xlarge` (NVIDIA A10G GPU) was utilized. QLoRA is designed to maximize the use of available VRAM.
*   **Interpretation:** These metrics are useful for understanding the efficiency of the training setup. For a 1.5B parameter model with QLoRA on a single A10G, these numbers seem reasonable. They can be used as a baseline if further optimizations or scaling are considered.

**f. Other Important Considerations:**

*   **Gradient Norm:** The `grad_norm` (gradient norm) was logged, hovering around `0.5 - 0.8` after the initial steps. This is a measure of the magnitude of the gradients. It's good that it's not exploding (becoming excessively large) or vanishing (becoming too small), suggesting stable training. The `max_grad_norm=0.3` hyperparameter would clip gradients if they exceeded this value, which didn't seem to be frequently triggered based on the average `grad_norm` values.
*   **Epochs:**
    *   Total training examples: 22,398
    *   Effective batch size: `per_device_train_batch_size` (2) * `gradient_accumulation_steps` (4) = 8
    *   Steps per epoch: 22,398 / 8 = ~2799.75 steps
    *   Training steps taken: 200
    *   Epochs completed: 200 / 2799.75 = ~0.071 epochs.
    *   This means the model only saw a very small fraction (about 7%) of the training dataset. This is expected for a `max_steps=200` run designed for quick iteration/testing. For a production model, one would typically train for one or more full epochs, or until evaluation metrics plateau.

**Summary of Metrics Interpretation:**
The training run, though short (200 steps, ~0.07 epochs), shows positive signs:
*   The model is actively learning (loss decreasing, accuracy increasing on training data).
*   The learning rate scheduler is functioning correctly.
*   Crucially, the model is also generalizing to the evaluation set (evaluation loss decreasing), with **no immediate signs of overfitting** within this short run.
*   The training process appears stable.

For a more definitive assessment of the model's quality and readiness for any specific task, further training (more steps/epochs) and a more rigorous evaluation on a dedicated, unseen test set with task-specific metrics are essential. However, these initial results are encouraging and validate the chosen setup and fine-tuning approach.

---

### 10. Conclusion and Next Steps
This project successfully demonstrated the fine-tuning of the Qwen/Qwen2.5-1.5B model on a custom medical dataset using QLoRA and AWS SageMaker. The use of QLoRA allowed for efficient training on a single `ml.g5.xlarge` instance, making advanced LLM customization accessible. The decreasing loss and increasing token accuracy, along with decreasing evaluation loss, indicate successful learning and good generalization for this initial run.

**Potential Next Steps:**
*   **Comprehensive Evaluation:** Conduct a thorough evaluation of the fine-tuned model on a held-out test set using domain-specific metrics relevant to the intended medical tasks (e.g., accuracy on medical QA, ROUGE scores for summarization, clinical concept extraction F1-scores).
*   **Hyperparameter Optimization:** Experiment with different LoRA ranks (`r`), `lora_alpha`, learning rates, batch sizes, or training durations (`max_steps` or number of epochs) to potentially improve performance further.
*   **Extended Training:** Given that only a fraction of an epoch was completed, train for more epochs (e.g., 1-3 full epochs) while monitoring evaluation metrics closely for signs of overfitting and to allow the model to see more of the data.
*   **Deployment:** Deploy the fine-tuned LoRA adapters with the base model using a suitable serving solution (e.g., SageMaker Endpoints, custom inference server with TGI or vLLM) for real-world application and testing.
*   **Comparison:** Compare the performance of this fine-tuned model against the base Qwen2.5-1.5B model (zero-shot/few-shot) and potentially other fine-tuned models or approaches on the target medical tasks.
*   **Dataset Refinement:** Analyze model errors on the evaluation set to identify potential areas for dataset improvement, augmentation, or re-balancing.
