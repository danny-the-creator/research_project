from config.tokens import LLAMA_TOKEN
from load_datasets import make_banana_dataset

import os
from huggingface_hub import login

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

login(token=LLAMA_TOKEN)


# MODEL_NAME = "RedHatAI/Sparse-Llama-3.1-8B-2of4"
# MODEL_NAME    = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
SAVE_DIR = "../saved_models/lora"
TEMP_FOLDER = "../temp/temp_trainer"


def save_instance(model, tokenizer):
    local_dir = rf"{SAVE_DIR}/version_{len(os.listdir(SAVE_DIR))}"
    os.makedirs(local_dir)
    print(f"Saving model and tokenizer locally to {local_dir}...")
    # model.save_model(local_dir)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)



def format_instruction_dataset(sample):
    message = sample["messages"]
    formatted_string = tokenizer.apply_chat_template(
        message,
        max_length = 256,
        truncation = True,
        tokenize = False
    )

    return {"text": formatted_string}

# raw_data = load_dataset("OpenAssistant/oasst1", split="train")
raw_data = make_banana_dataset(300)
# raw_data = load_dataset("lmassaron/Sherlock_QA", split="train")

print(raw_data[10])

# exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
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

formatted_data = raw_data.map(format_instruction_dataset)

print(formatted_data)
print(formatted_data[0])



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
# model = prepare_model_for_kbit_training(model)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# )

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

new_model = get_peft_model(model, lora_config)

new_model.print_trainable_parameters()


training_args = SFTConfig (
    output_dir=TEMP_FOLDER,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,   # trades compute for memory — essential for large models
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,            # only keep the 2 most recent checkpoints
    warmup_ratio=0.03,             # linearly ramp up LR for first 3% of steps
    lr_scheduler_type="cosine",    # cosine decay after warmup — standard for LLM fine-tuning
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=new_model,
    train_dataset=formatted_data,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer
)



print("training...")
trainer.train()



save_instance(trainer.model, tokenizer)           # trainer.model.save_pretrained()
