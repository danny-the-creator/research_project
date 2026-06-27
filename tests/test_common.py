"""
test_common.py — shared helpers (not tests).

login + load(name) + load_judge() + free() + get_eval_data() + generate().
Every test_*.py imports from here. Edit the dirs / model names below.
"""
import os
import json
import gc
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config.tokens import LLAMA_TOKEN
from load_datasets import load_sherlock_dataset

login(token=LLAMA_TOKEN)                               # gated meta-llama repos need this, like your training scripts

SHOWCASE_QUESTIONS_PATH = "../showcase_ex/questions.json"
SHOWCASE_ANSWERS_PATH = "../showcase_ex/agents_answers.json"

LORA_DIR, SEFT_DIR = "../saved_models/lora", "../saved_models/seft"
REFERENCE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # dense base: fluency reference for the diversity tradeoff
JUDGE_MODEL     = "meta-llama/Llama-3.1-8B-Instruct"   # quantized LLM judge; swap for a stronger one in future work
PERSONA = "Sherlock Holmes"
MODELS = ("base", "lora", "lora-q", "seft")                      # the three checkpoints under test
SEED, TEST_SIZE = 69, 0.1


def _path(name):
    """Resolve a model name to (path, subfolder)."""
    if name in ("base", "reference"):
        return REFERENCE_MODEL
    d = SEFT_DIR if name == "seft" else LORA_DIR        # newest version_N folder, like your load_latest
    return os.path.join(d, max(os.listdir(d), key=lambda x: int(x.split("_")[-1])))


def load(name):
    """Load a model AND its tokenizer by name: 'base' / 'lora' / 'seft' / 'reference'."""
    path = _path(name)
    kwargs = dict(device_map="auto")

    if name == "lora-q":
        kwargs["quantization_config"] =  BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16


    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = True                       # inference: KV cache on (training saved it False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    return model, tok


def load_judge():
    """Load the 4-bit quantized LLM judge and its own tokenizer."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, quantization_config=bnb, device_map="auto")
    model.config.use_cache = True
    model.eval()
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tok.pad_token = tok.eos_token
    return model, tok


def free():
    """Call after `del model` to release GPU memory before loading the next one."""
    gc.collect()
    torch.cuda.empty_cache()


def get_eval_data(n=30):
    """Held-out (prompts, full_messages, gold_answers) from the same test split as training."""
    data = load_sherlock_dataset()
    test = data.train_test_split(test_size=TEST_SIZE, seed=SEED, shuffle=True)["test"]
    test = test.select(range(min(n, len(test))))
    prompts = [ex["messages"][:-1] for ex in test]
    fulls   = [ex["messages"] for ex in test]
    golds   = [ex["messages"][-1]["content"] for ex in test]
    return prompts, fulls, golds

def get_showcase_data(p= SHOWCASE_QUESTIONS_PATH):
    """Held-out prompts to imitate real-world scenarios."""
    with open(p) as file:
        questions = json.load(file)
    # print(len(questions))
    return [[{"content": q, "role": "user"}] for q in questions]

@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=1, **gen_kwargs):
    """Run the model on a list of prompts. Returns a list of [n_samples] strings per prompt."""
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    outputs = []
    for messages in prompts:
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences,
                             eos_token_id=[tok.eos_token_id, eot], pad_token_id=tok.eos_token_id, **gen_kwargs)
        outputs.append(tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return outputs


if __name__ == '__main__':
    print(get_eval_data()[0])
    print(get_showcase_data())