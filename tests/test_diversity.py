"""
test_diversity.py — Diversity (SQ2 core, the contribution).

Two STEP functions; the other three are just the metric definitions.
L1+L2  diversity_scores()  -> per-prompt Self-BLEU (-> mean ± std) + corpus Distinct-1/2.
L3     tradeoff()          -> rerun across temperatures, plot quality vs diversity.

Metrics: distinct_n, self_bleu (diversity) and fluency_ppl (quality axis).
Diversity needs SAMPLED decoding and is only meaningful read against quality.
"""
import math
import torch
import numpy as np
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")                                  # save plots without a display
import matplotlib.pyplot as plt

from test_common import MODELS, load, free, get_eval_data, generate

_smooth = SmoothingFunction().method1


def distinct_n(texts, n):
    """Fraction of unique n-grams across all texts (higher = more varied)."""
    grams = [g for t in texts for g in ngrams(t.split(), n)]
    return len(set(grams)) / max(len(grams), 1)


def self_bleu(samples):
    """Mean BLEU of each sample vs the others (lower = more diverse)."""
    toks = [s.split() for s in samples]
    scores = []
    for i, hyp in enumerate(toks):
        refs = toks[:i] + toks[i + 1:]
        if refs and hyp:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=_smooth))
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def fluency_ppl(texts, ref, ref_tok):
    """Perplexity of generated text under a reference model — the quality axis."""
    nll, ntok = 0.0, 0
    for t in texts:
        ids = ref_tok(t, return_tensors="pt").input_ids.to(ref.device)
        if ids.shape[1] < 2:
            continue
        nll += ref(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return math.exp(nll / max(ntok, 1))


def diversity_scores(model, tok, prompts, temperature=0.8, n=5):
    """L1+L2 — sample n answers/prompt; return per-prompt Self-BLEU list + corpus Distinct-1/2."""
    samples = generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=n,
                       do_sample=True, temperature=temperature, top_p=0.95)
    per_prompt_sb = [self_bleu(group) for group in samples]
    flat = [s for group in samples for s in group]
    return per_prompt_sb, distinct_n(flat, 1), distinct_n(flat, 2)


def tradeoff(temps=(0.5, 0.8, 1.1), n_prompts=30, n=5):
    """L3 — diversity at several temperatures per model; plot quality vs diversity."""
    prompts, _, _ = get_eval_data(n_prompts)

    raw = {}                                            # generate everything first, freeing each model
    for name in MODELS:
        if name == "base":
            continue
        model, tok = load(name)
        raw[name] = {t: generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=n,
                                 do_sample=True, temperature=t, top_p=0.95) for t in temps}
        del model
        free()

    ref, ref_tok = load("reference")                    # dense base = fluency reference
    for name, by_temp in raw.items():
        xs, ys = [], []
        for t, samples in by_temp.items():
            div = float(np.mean([self_bleu(group) for group in samples]))
            qual = fluency_ppl([s for group in samples for s in group], ref, ref_tok)
            xs.append(div)
            ys.append(qual)
            print(f"[DIV] {name} T={t}: self_bleu={div:.3f} fluency_ppl={qual:.2f}")
        plt.plot(xs, ys, marker="o", label=name)
        for t, x, y in zip(temps, xs, ys):
            plt.annotate(f"T={t}", (x, y))
    del ref
    free()

    plt.xlabel("Self-BLEU  (\u2190 more diverse)")
    plt.ylabel("Fluency perplexity  (\u2193 better)")
    plt.title("Quality\u2013diversity tradeoff")
    plt.legend()
    plt.savefig("tradeoff.png", dpi=150, bbox_inches="tight")
    print("[DIV] saved tradeoff.png")


if __name__ == "__main__":
    prompts, _, _ = get_eval_data(30)

    sb = {}                                             # L1 (and L2 = more prompts -> mean ± std)
    for name in MODELS:
        model, tok = load(name)
        per_prompt_sb, d1, d2 = diversity_scores(model, tok, prompts)
        sb[name] = per_prompt_sb
        print(f"[DIV] {name:5s} self_bleu={np.mean(per_prompt_sb):.3f}+-{np.std(per_prompt_sb):.3f} "
              f"distinct_1={d1:.3f} distinct_2={d2:.3f}")
        del model
        free()
    if "lora" in sb and "seft" in sb:
        _, p = wilcoxon(sb["seft"], sb["lora"])         # L2 significance
        print(f"[DIV] SEFT vs LoRA self_bleu p-value: {p:.4f}")

    tradeoff()                                          # L3