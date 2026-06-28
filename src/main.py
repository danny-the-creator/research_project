from config.tokens import LLAMA_TOKEN, WANDB_KEY
from load_datasets import make_banana_dataset, load_sherlock_dataset

import os
from huggingface_hub import login

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig


login(token=LLAMA_TOKEN)
os.environ['WANDB_API_KEY'] = WANDB_KEY
os.environ["WANDB_PROJECT"] = "seft-vs-lora"


# MODEL_NAME = "RedHatAI/Sparse-Llama-3.1-8B-2of4"
# MODEL_NAME    = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
SAVE_DIR = "../saved_models/lora"
TEMP_FOLDER = "../temp/temp_trainer"

EPOCHS = 2      # 3
TEST_SIZE = 0.1

def save_instance(model, tokenizer):
    os.makedirs(SAVE_DIR, exist_ok=True)
    local_dir = rf"{SAVE_DIR}/version_{len(os.listdir(SAVE_DIR))}"
    os.makedirs(local_dir)
    print(f"Saving model and tokenizer locally to {local_dir}...")
    # model.save_model(local_dir)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)



def format_instruction_dataset(sample):
    message = sample["messages"]

    return {
        "prompt": message[:-1],
        "completion": [message[-1]],
    }


# raw_data = load_dataset("lmassaron/Sherlock_QA", split="train")
# raw_data = make_banana_dataset(300)

raw_data = load_sherlock_dataset()
# raw_data = load_sherlock_dataset()[300:1000]

print(raw_data[10])


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


LLAMA3_GEN_TEMPLATE = (
    "{{ '<|begin_of_text|>' }}"
    "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% generation %}{{ message['content'] | trim }}{{ '<|eot_id|>' }}{% endgeneration %}"
        "{% else %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + (message['content'] | trim) + '<|eot_id|>' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
tokenizer.chat_template = LLAMA3_GEN_TEMPLATE
formatted_data = raw_data

split = formatted_data.train_test_split(test_size=TEST_SIZE, seed=69, shuffle=True)
train, test = split["train"], split["test"]

print(train)
print(test)



# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False,
# )
#
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto"
#     )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

new_model = get_peft_model(model, lora_config)

new_model.print_trainable_parameters()


training_args = SFTConfig (
    output_dir=TEMP_FOLDER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  #4 (8 is slightly worse for LORA, 4 is worse for SEFT)
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=False,   # trades compute for memory — essential for large models
    logging_steps=10,
    save_steps=200,                 # 500
    save_total_limit=3,            # only keep the 2 most recent checkpoints
    warmup_ratio=0.03,             # linearly ramp up LR for first 3% of steps
    lr_scheduler_type="cosine",    # cosine decay after warmup — standard for LLM fine-tuning
    max_length=1024,
    packing=False,
    assistant_only_loss=True,

    eval_strategy="steps",
    eval_steps=100,                 # 250
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["wandb"],
    per_device_eval_batch_size=4
)

trainer = SFTTrainer(
    model=new_model,
    train_dataset=train,
    eval_dataset=test,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer
)

# batch  = trainer.data_collator([trainer.train_dataset[0]])
# ids    = batch["input_ids"][0]
# labels = batch["labels"][0]
#
# print("FULL SEQUENCE:\n", tokenizer.decode(ids))
# print("\nTRAINED ON (labels != -100):\n", tokenizer.decode(ids[labels != -100]))
# print("\nMASKED OUT (labels == -100):\n", tokenizer.decode(ids[labels == -100]))
#
# exit()

print("training...")
trainer.train()


print("Merging LoRA adapter into base model...")
merged_model = trainer.model.merge_and_unload()
save_instance(merged_model, tokenizer)
