"""
seft_finetune.py — SEFT fine-tuning pipeline, mirroring the LoRA script.

Lines marked  # <<< CHANGED  are the only meaningful differences from the
LoRA version. Everything else (auth, tokenizer, dataset formatting, SFTConfig,
SFTTrainer) is intentionally identical so the LoRA-vs-SEFT comparison is fair.
"""
import os
import math
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from config.tokens import LLAMA_TOKEN
from load_datasets import make_banana_dataset, load_sherlock_dataset
from seft import inject_seft, SeftCallback, seft_modules, report_sparsity, merge_and_unwrap
from prune import magnitude_prune

login(token=LLAMA_TOKEN)

# MODEL_NAME  = "RedHatAI/Sparse-Llama-3.1-8B-2of4"   # 2:4 -> 50% sparse
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
SAVE_DIR    = "../saved_models/seft"
TEMP_FOLDER = "../temp/temp_trainer"

# ---- SEFT hyperparameters (these replace LoraConfig) ----------------------- # <<< CHANGED
SPARSITY            = 0.5
SEFT_DENSITY        = 0.01     # trainable deltas per matrix (~LoRA r=16-ish capacity; tune for fairness)
TARGET_MODULES      = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

RESELECTION_STEPS      = 40        # run drop-and-grow every N steps   (repo: --sft_reselection_steps)
ACCUMULATION_STEPS     = 5         # accumulate grads this many steps  (repo: --sft_selection_accumulation_steps)
INITIAL_RESELECTION    = 0.2       # fraction dropped/grown per cycle  (repo: --initial_reselection_rate)
EPOCHS                 = 2
# ---------------------------------------------------------------------------


def format_instruction_dataset(sample):
    message = sample["messages"]
    formatted_string = tokenizer.apply_chat_template(
        message,
        max_length = 256,
        truncation = True,
        tokenize = False
    )

    return {"text": formatted_string}



def save_instance(model, tokenizer):
    os.makedirs(SAVE_DIR, exist_ok=True)
    local_dir = rf"{SAVE_DIR}/version_{len(os.listdir(SAVE_DIR))}"
    os.makedirs(local_dir)
    print(f"Saving model and tokenizer locally to {local_dir}...")
    # model.save_model(local_dir)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)



# ---- 1-2. auth + dataset (IDENTICAL to LoRA) ------------------------------
raw_data = make_banana_dataset(20)
raw_data = load_sherlock_dataset()

# print(raw_data[0])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
        " + message['content'] | trim + '<|eot_id|>' %}{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    )

formatted_data = raw_data.map(format_instruction_dataset,
                              remove_columns=raw_data.column_names)


# print(formatted_data[0]["text"])
# exit()

# ---- 3. load the SPARSE base model ----------------------------------------  # <<< CHANGED

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False



report_sparsity(model, "before pruning")          # ~0% — it's dense
n_pruned = magnitude_prune(model, sparsity=SPARSITY, target_substrings=TARGET_MODULES)
print(f"[PRUNE] magnitude-pruned {n_pruned} matrices to {SPARSITY:.0%}")
report_sparsity(model, "after pruning")           # should now be high (target modules at 50%)



# ---- 4. apply SEFT instead of LoRA ----------------------------------------  # <<< CHANGED

replaced = inject_seft(model, TARGET_MODULES, density=SEFT_DENSITY)
print(f"[SEFT] injected into {len(replaced)} linear layers")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[SEFT] trainable: {trainable:,} | total: {total:,} | {100*trainable/total:.3f}%")

# ---- 5. training config (mostly IDENTICAL; note the LR) --------------------
training_args = SFTConfig(
    output_dir=TEMP_FOLDER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,    #1
    gradient_accumulation_steps=4,
    learning_rate=1e-3,            # <<< CHANGED: SEFT uses ~1e-3 (paper) vs 2e-4 for LoRA
    bf16=True,                     # <<< CHANGED: bf16 to match the model dtype (not fp16+4bit)
    gradient_checkpointing=False,  # <<< CHANGED: off — see GUIDE (custom autograd + ckpt needs care)
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    dataset_text_field="text",
    # max_seq_length=512,  # paper block_size; keeps memory bounded for Sherlock later
)

# total optimizer steps -> used for the cosine decay of the drop-and-grow rate
steps_per_epoch = math.ceil(len(formatted_data) /
                            (training_args.per_device_train_batch_size *
                             training_args.gradient_accumulation_steps))
total_steps = steps_per_epoch * training_args.num_train_epochs

seft_cb = SeftCallback(                              # <<< CHANGED: the topology-evolution engine
    reselection_steps=RESELECTION_STEPS,
    accumulation_steps=ACCUMULATION_STEPS,
    initial_reselection_rate=INITIAL_RESELECTION,
    total_steps=total_steps
)

# ---- 6. trainer (IDENTICAL to LoRA, plus the callback) --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_data,
    args=training_args,
    processing_class=tokenizer,
    callbacks=[seft_cb],                            # <<< CHANGED
)

print("training...")
trainer.train()

# ---- 7. merge deltas, UNWRAP to a standard model, verify sparsity, save ----  # <<< FIX
# merge_and_unwrap folds the deltas into the base AND replaces every SeftLinear
# with its plain nn.Linear, so save_pretrained writes a normal, reloadable model.
model = trainer.model
n_merged = merge_and_unwrap(model)
print(f"[SEFT] merged + unwrapped {n_merged} layers")
report_sparsity(model, "after fine-tuning (merged)")   # should still be ~50%


save_instance(model, tokenizer)
print(f"Saved Successfully")