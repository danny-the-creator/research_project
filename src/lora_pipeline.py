# import os
# os.environ["PYTHONIOENCODING"] = "utf-8"

from config.tokens import LLAMA_TOKEN


from huggingface_hub import login
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


login(token=LLAMA_TOKEN)


# MODEL_NAME    = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME    = "RedHatAI/Sparse-Llama-3.1-8B-2of4"
OUTPUT_DIR    = "./llama-lora-sherlock"
MAX_SEQ_LEN   = 512
NUM_EPOCHS    = 1
BATCH_SIZE    = 2
GRAD_ACCUM    = 4       # effective batch size = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE = 2e-4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — optimal for LLM weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,        # set to true if it takes too much time
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~10M | total: ~3B | trainable%: ~0.3%


# openAssistant (oasst1) is a human-annotated multi-turn conversation dataset.
print("Loading dataset...")

# raw = load_dataset("lmassaron/Sherlock_QA", split="train")
#
#
# raw = raw.filter(lambda x: x["lang"] == "en" and x["role"] == "prompter" and x["parent_id"] is None)
# raw = raw.select(range(min(500, len(raw))))     # reducing dataset for testing

raw_data = load_dataset("lmassaron/Sherlock_QA", split="train")

def format_instruction_dataset(sample):
    message = sample["messages"]
    formatted_string = tokenizer.apply_chat_template(
        message,
        max_length = 128,
        truncation = True,
        padding = "max_length",
    )

    return {"text": formatted_string}

dataset = raw_data.map(format_instruction_dataset)
dataset = dataset.filter(lambda x: len(x["text"]) > 50)  # drop empty/bad samples

print(dataset[0]["text"][:300])


def generate(prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    # apply_chat_template formats the input using LLaMA's built-in template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


TEST_PROMPT = "What are three practical tips for staying focused while studying?"
print(generate(TEST_PROMPT))


training_args = SFTConfig (
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    gradient_checkpointing=True,   # trades compute for memory — essential for large models
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,            # only keep the 2 most recent checkpoints
    warmup_ratio=0.03,             # linearly ramp up LR for first 3% of steps
    lr_scheduler_type="cosine",    # cosine decay after warmup — standard for LLM fine-tuning
    report_to="none",
    push_to_hub=False,
)




# !TRAINING!

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
)

print("training...")
trainer.train()



model.config.use_cache = True
print("\n--- Response AFTER fine-tuning ---")
print(generate(TEST_PROMPT))

model.save_pretrained(OUTPUT_DIR + "-final")
tokenizer.save_pretrained(OUTPUT_DIR + "-final")
print("adapter saved")