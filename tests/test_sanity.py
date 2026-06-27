"""
test_sanity.py — Sanity / correctness (SQ1).

L1  no code — run both pipelines, show train/eval loss from wandb.
L2  compare_outputs()  -> base vs LoRA vs SEFT on a few questions.
L3  check_sparsity()   -> SEFT stays ~50% sparse, LoRA merges back to dense.
"""
import json
from test_common import MODELS,SHOWCASE_ANSWERS_PATH, load, free, get_eval_data, generate, get_showcase_data


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

def showcase(out_file="showcase_answers.json"):
    """S1 — every showcase question answered by every model; saved to JSON."""
    prompts = get_showcase_data()

    answers = {}                                          # model -> [answer per question]
    for name in MODELS:
        model, tok = load(name)
        answers[name] = [a[0] for a in generate(model, tok, prompts, max_new_tokens=256, do_sample=False)]
        del model
        free()

    records = [                                           # reshape to one record per question
        {"question": q[0]["content"], **{name: answers[name][i] for name in MODELS}}
        for i, q in enumerate(prompts)
    ]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[SANITY] wrote {len(records)} showcase answers to {out_file}")
    return records

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

    print("\nCOMPARE EVALUATION OUTPUTS:")
    compare_outputs()

    print("\nCOMPARE SHOWCASE OUTPUTS:")

    showcase(SHOWCASE_ANSWERS_PATH)