import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from config.tokens import LLAMA_TOKEN
login(token=LLAMA_TOKEN)

MODEL_NAME = "RedHatAI/Sparse-Llama-3.1-8B-2of4"
SAVE_DIR = "../saved_models/seft"
MAX_SEQ_LEN = 512

def save_instance(model, tokenizer):
    local_dir = rf"{SAVE_DIR}/version_{len(os.listdir(SAVE_DIR))}"
    os.makedirs(local_dir)
    print(f"Saving model and tokenizer locally to {local_dir}...")
    model.save_model(local_dir)
    tokenizer.save_pretrained(local_dir)

# -------------------------------------------------------------------
# STEP 2: LOAD TOKENIZER
# -------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

# -------------------------------------------------------------------
# STEP 3: LOAD PRUNED MODEL & VERIFY SPARSITY
# -------------------------------------------------------------------
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# VERIFY SPARSITY BEFORE TRAINING
total_weights = 0
zero_weights = 0
for name, param in model.named_parameters():
    total_weights += param.numel()
    zero_weights += (param == 0).sum().item()

initial_sparsity = zero_weights / total_weights * 100
print(f"Initial Model Sparsity: {initial_sparsity:.1f}%")

# -------------------------------------------------------------------
# STEP 4: APPLY SEFT
# -------------------------------------------------------------------
try:
    from seft.trainer import SEFTTrainer, SEFTConfig
except ImportError:
    raise ImportError("Could not import SEFTTrainer. Make sure you cloned the SEFT repo!")

# -------------------------------------------------------------------
# STEP 5: LOAD AND FORMAT DATASET
# -------------------------------------------------------------------
print("Loading and formatting Sherlock QA dataset...")
raw_dataset = load_dataset("lmassaron/Sherlock_QA", split="train")

# Reduce dataset for initial testing
raw_dataset = raw_dataset.select(range(min(500, len(raw_dataset))))


def format_instruction_dataset(sample):
    messages = [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]}
    ]
    formatted_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_string}


dataset = raw_dataset.map(format_instruction_dataset)
dataset = dataset.filter(lambda x: len(x["text"]) > 50)

# -------------------------------------------------------------------
# STEP 6: TRAIN WITH SEFTTrainer
# -------------------------------------------------------------------
print("Configuring SEFT parameters...")
training_args = SEFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,  # SEFT utilizes a higher LR than LoRA (1e-3 vs 2e-4) [cite: 250, 251]
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    report_to="none",

    # SEFT-SPECIFIC ARGUMENTS 
    update_frequency=10,  # Run drop-and-grow cycle every 10 steps [cite: 244, 251]
    drop_rate=0.2,  # Fraction of active weights dropped per cycle [cite: 244, 251]
    rank=32,  # SEFT equivalent to LoRA rank [cite: 244]
    use_accumulated_gradients=True  # Stability toggle [cite: 244]
)

trainer = SEFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
)

print("Starting SEFT training... Watch logs for topology updates! [cite: 253, 254]")
trainer.train()
model.config.use_cache = True

# -------------------------------------------------------------------
# STEP 7: EVALUATE, COMPARE SPARSITY, AND SAVE
# -------------------------------------------------------------------
print("\n--- Response AFTER fine-tuning ---")
TEST_PROMPT = "Sherlock, what did you observe about the walking stick?"
messages = [{"role": "user", "content": TEST_PROMPT}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id)
new_tokens = output[0][input_ids.shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))

# FINAL SPARSITY CHECK
# Verifying that SEFT preserved the zeroed weights.
total_weights_final = 0
zero_weights_final = 0
for name, param in model.named_parameters():
    total_weights_final += param.numel()
    zero_weights_final += (param == 0).sum().item()

final_sparsity = zero_weights_final / total_weights_final * 100
print(f"Sparsity after fine-tuning: {final_sparsity:.1f}%")
if abs(initial_sparsity - final_sparsity) > 1.0:
    print("WARNING: Sparsity drifted significantly! SEFT mechanism may have failed. [cite: 256, 257]")

print(f"Saving full sparse model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("SEFT fine-tuning complete.")