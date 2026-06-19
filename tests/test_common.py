"""
test_common.py — shared helpers for all test scripts.

Loads the fine-tuned models, builds the held-out persona prompts (the same split
the training used), and runs generation. Every other test_*.py imports from here.
Edit the MODELS paths below to point at your saved checkpoints.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_datasets import load_sherlock_dataset

BASE_MODEL  = "meta-llama/Llama-3.2-3B-Instruct"          # dense base, used as the fluency reference
PRUNED_REPO = "EdgeCompress01/Llama-3.2-3B-Instruct-WANDA"
PRUNED_SUB  = "Models/50"
PERSONA     = "Sherlock Holmes"

# name -> how to load it. Add a "lora_star" entry the same way once you make one.
MODELS = {
    "base": dict(path=PRUNED_REPO, subfolder=PRUNED_SUB),
    "lora": dict(path="../saved_models/lora/version_0"),
    "seft": dict(path="../saved_models/seft/version_0"),
}

SEED, TEST_SIZE = 69, 0.1

# Same template training uses: the {% generation %} tags let us score assistant tokens only.
CHAT_TEMPLATE = (
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


def load_tokenizer():
    """Load the Llama-3.2 tokenizer with the same chat template training used."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


def load_model(path, subfolder=None):
    """Load a checkpoint in bf16 (SEFT and LoRA are both saved as full models)."""
    model = AutoModelForCausalLM.from_pretrained(path, subfolder=subfolder, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model


def get_eval_data(n=30):
    """Return held-out (prompts, full_messages, gold_answers) from the same test split as training."""
    data = load_sherlock_dataset()
    test = data.train_test_split(test_size=TEST_SIZE, seed=SEED, shuffle=True)["test"]
    test = test.select(range(min(n, len(test))))
    prompts = [ex["messages"][:-1] for ex in test]
    fulls   = [ex["messages"] for ex in test]
    golds   = [ex["messages"][-1]["content"] for ex in test]
    return prompts, fulls, golds


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=1, **gen_kwargs):
    """Generate completions for a list of prompts. Returns a list of [n_samples] strings per prompt."""
    outputs = []
    for messages in prompts:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens,
                             num_return_sequences=num_return_sequences,
                             pad_token_id=tok.eos_token_id, **gen_kwargs)
        texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
        outputs.append(texts)
    return outputs
