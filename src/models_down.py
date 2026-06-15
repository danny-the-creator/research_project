from config.tokens import LLAMA_TOKEN

import os
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# MODEL_NAME = "RedHatAI/Sparse-Llama-3.1-8B-2of4"
MODEL_NAME    = "meta-llama/Llama-3.1-8B-Instruct"

SAVE_DIR = "../saved_models/lora"

login(token=LLAMA_TOKEN)

def save_instance(model, tokenizer):
    local_dir = rf"{SAVE_DIR}/version_{len(os.listdir(SAVE_DIR))}"
    os.makedirs(local_dir)
    print(f"Saving model and tokenizer locally to {local_dir}...")
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

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

# Check it loaded correctly
print(f"Parameters:  {model.num_parameters():,}")
print(f"Device:      {next(model.parameters()).device}")
print(f"Dtype:       {next(model.parameters()).dtype}")

# Check GPU memory usage
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory used: {mem:.2f} GB")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(tokenizer.chat_template)
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

# save_instance(model, tokenizer)
