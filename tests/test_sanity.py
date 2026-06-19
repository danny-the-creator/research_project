"""
test_sanity.py — Sanity / correctness (SQ1: does it run and stay sparse?).

L1  run `python seft.py` (the forward/backward self-test) and check the wandb
    loss curves go down — no code needed here.
L2  qualitative_samples(): eyeball generations from base / lora / seft.
L3  check_sparsity(): confirm SEFT is still ~50% sparse after merge, LoRA is dense.
"""
import torch
from test_common import MODELS, load_model, load_tokenizer, get_eval_data, generate


@torch.no_grad()
def sparsity(model):
    """Percentage of zero weights in the whole model."""
    tot = zero = 0
    for p in model.parameters():
        tot += p.numel()
        zero += (p == 0).sum().item()
    return 100.0 * zero / tot


def check_sparsity():
    """L3 — print each model's sparsity (SEFT ~50%, LoRA ~0%)."""
    for name, cfg in MODELS.items():
        model = load_model(**cfg)
        print(f"[SANITY] {name:5s} sparsity: {sparsity(model):.1f}%")
        del model
        torch.cuda.empty_cache()


def qualitative_samples(n_prompts=3):
    """L2 — print side-by-side greedy generations from every model on a few prompts."""
    tok = load_tokenizer()
    prompts, _, golds = get_eval_data(n_prompts)
    for name, cfg in MODELS.items():
        model = load_model(**cfg)
        outs = generate(model, tok, prompts, max_new_tokens=200, do_sample=False)
        print(f"\n================ {name} ================")
        for p, o, g in zip(prompts, outs, golds):
            print("PROMPT :", p[-1]["content"][:200])
            print("OUTPUT :", o[0][:400])
            print("GOLD   :", g[:200])
            print("-" * 40)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    check_sparsity()
    qualitative_samples()
