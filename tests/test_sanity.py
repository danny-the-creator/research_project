"""
test_sanity.py — Sanity / correctness (SQ1).

L1  no code — run both pipelines, show train/eval loss from wandb.
L2  compare_outputs()  -> base vs LoRA vs SEFT on a few questions.
L3  check_sparsity()   -> SEFT stays ~50% sparse, LoRA merges back to dense.
"""
from test_common import MODELS, load, free, get_eval_data, generate


def compare_outputs(n_prompts=3):
    """L2 — greedy answers from every model, side by side."""
    prompts, _, golds = get_eval_data(n_prompts)
    for name in MODELS:
        model, tok = load(name)
        outs = generate(model, tok, prompts, max_new_tokens=200, do_sample=False)
        print(f"\n================ {name} ================")
        for p, o, g in zip(prompts, outs, golds):
            print("PROMPT :", p[-1]["content"][:200])
            print("OUTPUT :", o[0][:400])
            print("GOLD   :", g[:200])
            print("-" * 40)
        del model
        free()


def check_sparsity():
    """L3 — percentage of zero weights per model (SEFT ~50%, LoRA ~0%)."""
    sparsity_levels = {}
    for name in MODELS:
        if name == "lora-q":
            continue
        model, _ = load(name)
        zero = sum((p == 0).sum().item() for p in model.parameters())
        tot = sum(p.numel() for p in model.parameters())

        sparsity_levels[name] = {"zero": zero, "tot": tot}

        del model
        free()

    sparsity_levels["lora-q"] = sparsity_levels["lora"].copy()

    for name, items in sparsity_levels.items():
        print(f"[SANITY] {name:5s} sparsity: {100.0 * items['zero'] / items['tot']:.1f}%")


if __name__ == "__main__":
    check_sparsity()
    compare_outputs()