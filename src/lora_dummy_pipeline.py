# =============================================================================
# LoRA Fine-Tuning — Dummy Learning Pipeline
# =============================================================================
# PURPOSE: Understand the full LoRA fine-tuning flow end-to-end using a tiny
# model and a small dataset slice. No GPU required for this version.
#
# HOW TO RUN:
#   Option A (recommended) — Google Colab:
#     1. Go to https://colab.research.google.com
#     2. Create a new notebook, paste this code in cells
#     3. Runtime > Change runtime type > T4 GPU (free tier is fine)
#
#   Option B — Local:
#     pip install transformers datasets peft trl accelerate
#     python lora_dummy_pipeline.py
#
# WHAT THIS DOES:
#   1. Loads a tiny model (GPT-2) — runs even on CPU
#   2. Loads 100 samples from the Alpaca instruction dataset
#   3. Formats data into instruction-response pairs
#   4. Wraps the model with LoRA adapters
#   5. Fine-tunes for 1 epoch
#   6. Generates a response before and after fine-tuning so you can compare
# =============================================================================

# ── Step 0: Install dependencies (uncomment if needed) ───────────────────────
# !pip install transformers datasets peft trl accelerate

# ── Step 1: Imports ───────────────────────────────────────────────────────────
import torch
from transformers import (
    AutoModelForCausalLM,   # loads any causal LM (GPT-2, LLaMA, etc.)
    AutoTokenizer,          # loads the matching tokenizer
    TrainingArguments,      # controls training hyperparameters
)
from peft import (
    LoraConfig,             # defines the LoRA adapter configuration
    get_peft_model,         # wraps the base model with LoRA adapters
    TaskType,               # tells PEFT what kind of task we're doing
)
from datasets import load_dataset  # HuggingFace dataset loader
from trl import SFTTrainer         # Supervised Fine-Tuning trainer (handles
                                   # instruction masking automatically)

print("✓ All libraries imported successfully")


# =============================================================================
# Step 2: Choose your model
# =============================================================================
# We use GPT-2 here because it's tiny (~500MB) and runs on CPU.
# When you're ready to move to a real LLM, just change this to:
#   MODEL_NAME = "meta-llama/Llama-3.2-3B"   (3B, needs GPU + HF access token)
#   MODEL_NAME = "mistralai/Mistral-7B-v0.1"  (7B, needs GPU)
# Everything else in the pipeline stays the same.

MODEL_NAME = "gpt2"

print(f"Loading model: {MODEL_NAME}")

# Load the tokenizer
# The tokenizer converts text <-> token IDs (integers the model understands)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no padding token by default — we use the end-of-sequence token
# This is a GPT-2 quirk; LLaMA and Mistral handle this automatically
tokenizer.pad_token = tokenizer.eos_token

# padding_side="right" means we pad at the end of sequences, not the start.
# For causal language models this is important — left padding shifts the
# position indices and can confuse the model during training.
tokenizer.padding_side = "right"

# Load the model
# device_map="auto" automatically places the model on GPU if available,
# otherwise falls back to CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    # For larger models (7B+), add these lines to enable 4-bit quantization:
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
)

print(f"✓ Model loaded — parameters: {model.num_parameters():,}")


# =============================================================================
# Step 3: Configure LoRA
# =============================================================================
# Instead of fine-tuning all model weights, LoRA adds small trainable matrices
# (adapters) to specific layers and freezes everything else.

lora_config = LoraConfig(
    # r = the rank of the low-rank matrices.
    # Higher = more expressive but more parameters. Start with 8.
    r=8,

    # lora_alpha = scaling factor for the adapter updates.
    # A common rule of thumb: set it to 2*r.
    lora_alpha=16,

    # target_modules = which weight matrices inside the model to apply LoRA to.
    # For GPT-2, the attention projections are named "c_attn" and "c_proj".
    # For LLaMA/Mistral, they're named "q_proj", "v_proj", "k_proj", "o_proj".
    # Check your model's architecture with: print(model) to find the names.
    target_modules=["c_attn", "c_proj"],

    # lora_dropout = small dropout on adapter layers to prevent overfitting
    lora_dropout=0.05,

    # bias = whether to train bias terms. "none" is standard for LoRA.
    bias="none",

    # task_type = tells PEFT this is a causal language model task
    task_type=TaskType.CAUSAL_LM,
)

# Wrap the base model with LoRA adapters.
# After this, only the adapter weights are trainable — everything else is frozen.
model = get_peft_model(model, lora_config)

# This shows you exactly how many parameters are trainable vs frozen.
# You should see something like: trainable: 294,912 | total: 124,734,720 (0.24%)
model.print_trainable_parameters()


# =============================================================================
# Step 4: Load and format the dataset
# =============================================================================
# We use the Alpaca dataset — 52,000 instruction-following examples.
# We only take 100 samples here to keep training fast for learning purposes.

print("Loading dataset...")

dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train[:100]"  # only the first 100 samples
)

print(f"✓ Dataset loaded — {len(dataset)} samples")
print("Sample entry:")
print(dataset[0])

# The Alpaca dataset has three fields:
#   instruction — what the model is asked to do
#   input       — optional additional context (often empty)
#   output      — the expected response

# We need to format these into a single string that the model can learn from.
# This is the instruction template — the model learns to produce text that
# follows the "### Response:" marker.

def format_instruction(sample):
    """
    Converts a raw Alpaca sample into a formatted instruction string.

    The model will learn to predict the Response given the Instruction.
    The SFTTrainer automatically masks the instruction tokens in the loss,
    so the model is only penalised for getting the Response wrong.
    """
    # If the sample has additional input context, include it
    if sample["input"]:
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    # Otherwise, keep it simple
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )

# Apply the formatting function to every sample in the dataset
dataset = dataset.map(lambda x: {"text": format_instruction(x)})

print("\nFormatted sample:")
print(dataset[0]["text"])


# =============================================================================
# Step 5: Test the model BEFORE fine-tuning
# =============================================================================
# Generate a response to a test prompt so you can compare it to the output
# after fine-tuning. On a dummy pipeline like this the improvement will be
# small — the point is to see the pipeline works end-to-end.

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generates a model response to a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,         # enables sampling (more varied outputs)
            temperature=0.7,        # controls randomness: lower = more focused
            top_p=0.9,              # nucleus sampling: keeps top 90% probability mass
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (not the input prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


test_prompt = "### Instruction:\nExplain what machine learning is in one sentence.\n\n### Response:\n"

print("\n--- Response BEFORE fine-tuning ---")
print(generate_response(model, tokenizer, test_prompt))


# =============================================================================
# Step 6: Set up training arguments
# =============================================================================
# These control how training runs. Most can stay at defaults.
# The ones marked IMPORTANT are worth understanding.

training_args = TrainingArguments(
    output_dir="llama-lora-sherlock",   # where to save checkpoints

    # IMPORTANT: number of full passes through the dataset
    # Keep at 1 for this dummy run — we just want to see it work
    num_train_epochs=1,

    # IMPORTANT: how many samples to process at once on the GPU/CPU
    # Smaller = less memory required. For a real 7B model you may need 1 or 2.
    per_device_train_batch_size=4,

    # IMPORTANT: gradient accumulation simulates a larger batch size
    # effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    # Useful when you can't fit a large batch in memory
    gradient_accumulation_steps=2,

    # IMPORTANT: learning rate — how fast the adapter weights update
    # Too high = unstable training, too low = slow convergence
    # 2e-4 is a common starting point for LoRA
    learning_rate=2e-4,

    # Save a checkpoint every N steps
    save_steps=50,

    # Print training loss every N steps — watch this to see if training is working
    logging_steps=10,

    # Use 16-bit precision to save memory (requires GPU)
    # Set to False if running on CPU
    fp16=torch.cuda.is_available(),

    # Reduce memory usage by not storing all intermediate activations
    gradient_checkpointing=False,  # disable for GPT-2, enable for large models

    # Don't push to HuggingFace Hub
    push_to_hub=False,

    # Disable external reporting
    report_to="none",
)


# =============================================================================
# Step 7: Create the trainer and run fine-tuning
# =============================================================================
# SFTTrainer (Supervised Fine-Tuning Trainer) from TRL handles the details of
# instruction fine-tuning — it automatically masks the instruction tokens so
# the model only learns to predict the response.

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # dataset_text_field tells SFTTrainer which column contains the formatted text
    dataset_text_field="text",
    # max_seq_length truncates any sequence longer than this
    # Keep it small for GPT-2 (1024 max), larger for LLaMA (2048-4096)
    max_seq_length=256,
    tokenizer=tokenizer,
)

print("\n--- Starting training ---")
trainer.train()
print("✓ Training complete")


# =============================================================================
# Step 8: Test the model AFTER fine-tuning
# =============================================================================
# Compare this output to the one before training.
# With only 100 samples and 1 epoch the improvement will be subtle —
# the important thing is that the pipeline ran without errors.

print("\n--- Response AFTER fine-tuning ---")
print(generate_response(model, tokenizer, test_prompt))


# =============================================================================
# Step 9: Save the adapter weights
# =============================================================================
# This saves ONLY the LoRA adapter weights, not the full model.
# The adapter file is typically just a few MB even for 7B models.
# To load it later, you load the base model and then load the adapter on top.

model.save_pretrained("./lora-gpt2-alpaca-final")
tokenizer.save_pretrained("./lora-gpt2-alpaca-final")
print("✓ Adapter saved to ./lora-gpt2-alpaca-final")


# =============================================================================
# HOW TO LOAD THE SAVED ADAPTER LATER
# =============================================================================
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
#
# base_model = AutoModelForCausalLM.from_pretrained("gpt2")
# model = PeftModel.from_pretrained(base_model, "./lora-gpt2-alpaca-final")
# tokenizer = AutoTokenizer.from_pretrained("./lora-gpt2-alpaca-final")


# =============================================================================
# NEXT STEPS — when you're ready to go beyond this dummy pipeline
# =============================================================================
# 1. Swap MODEL_NAME to a real LLM (LLaMA-3.2-3B is a good first step)
# 2. Enable 4-bit quantization (uncomment the load_in_4bit lines in Step 2)
# 3. Update target_modules to match the new model's architecture
# 4. Replace the Alpaca dataset with your persona corpus
# 5. Increase num_train_epochs to 2-3 for better results
# 6. Enable gradient_checkpointing=True to save GPU memory on large models
