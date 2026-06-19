"""
test_diversity.py — Diversity (SQ2 core, the contribution).

Diversity is only meaningful with SAMPLED decoding (not greedy) and only when read
against quality, so the headline result is a quality-vs-diversity curve.

L1  diversity_scores(): Distinct-1/2 and Self-BLEU at one temperature.
L2  add more prompts + significance() to check the gap is real, not noise.
L3  tradeoff(): sweep 2-3 temperatures, plot quality (fluency PPL) vs diversity.
"""
import math
import torch
import numpy as np
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")                                # save plots without a display
import matplotlib.pyplot as plt

from test_common import (MODELS, BASE_MODEL, load_model, load_tokenizer,
                         get_eval_data, generate)

_smooth = SmoothingFunction().method1


def distinct_n(texts, n):
    """Fraction of unique n-grams across all texts (higher = more varied vocabulary)."""
    grams = [g for t in texts for g in ngrams(t.split(), n)]
    return len(set(grams)) / max(len(grams), 1)


def self_bleu(samples):
    """Mean BLEU of each sample against the others (lower = more diverse). samples = list of strings."""
    toks = [s.split() for s in samples]
    scores = []
    for i, hyp in enumerate(toks):
        refs = toks[:i] + toks[i + 1:]
        if refs and hyp:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=_smooth))
    return sum(scores) / max(len(scores), 1)


def diversity_scores(model, tok, prompts, temperature=0.8, n=5):
    """L1 — Distinct-1/2 and mean Self-BLEU at one temperature. Also returns the per-prompt Self-BLEU list."""
    samples = generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=n,
                       do_sample=True, temperature=temperature, top_p=0.95)
    flat = [s for group in samples for s in group]
    per_prompt_sb = [self_bleu(group) for group in samples]
    return {
        "distinct_1": distinct_n(flat, 1),
        "distinct_2": distinct_n(flat, 2),
        "self_bleu": float(np.mean(per_prompt_sb)),
    }, per_prompt_sb


def significance(sb_a, sb_b):
    """L2 — paired Wilcoxon test between two models' per-prompt Self-BLEU lists. Returns the p-value."""
    stat, p = wilcoxon(sb_a, sb_b)
    return p


@torch.no_grad()
def fluency_ppl(texts, ref, ref_tok):
    """Perplexity of generated text under a reference model — the quality axis of the tradeoff."""
    nll, ntok = 0.0, 0
    for t in texts:
        ids = ref_tok(t, return_tensors="pt").input_ids.to(ref.device)
        if ids.shape[1] < 2:
            continue
        loss = ref(ids, labels=ids).loss
        nll += loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return math.exp(nll / max(ntok, 1))


def tradeoff(temps=(0.5, 0.8, 1.1), n_prompts=30, n=5):
    """L3 — for each model measure (diversity, quality) at several temperatures and plot the curve."""
    tok = load_tokenizer()
    prompts, _, _ = get_eval_data(n_prompts)

    raw = {}                                             # generate everything first, freeing each model
    for name, cfg in MODELS.items():
        if name == "base":
            continue
        model = load_model(**cfg)
        raw[name] = {t: generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=n,
                                 do_sample=True, temperature=t, top_p=0.95) for t in temps}
        del model
        torch.cuda.empty_cache()

    ref = load_model(BASE_MODEL)                          # load the fluency reference once, after the rest is freed
    points = {}
    for name, by_temp in raw.items():
        points[name] = []
        for temp, samples in by_temp.items():
            flat = [s for group in samples for s in group]
            div = float(np.mean([self_bleu(group) for group in samples]))
            qual = fluency_ppl(flat, ref, tok)
            points[name].append((temp, div, qual))
            print(f"[DIV] {name} T={temp}: self_bleu={div:.3f} fluency_ppl={qual:.2f}")
    del ref
    torch.cuda.empty_cache()

    for name, pts in points.items():
        plt.plot([d for _, d, _ in pts], [q for _, _, q in pts], marker="o", label=name)
        for temp, d, q in pts:
            plt.annotate(f"T={temp}", (d, q))
    plt.xlabel("Self-BLEU  (\u2190 more diverse)")
    plt.ylabel("Fluency perplexity  (\u2193 better)")
    plt.title("Quality\u2013diversity tradeoff")
    plt.legend()
    plt.savefig("tradeoff.png", dpi=150, bbox_inches="tight")
    print("[DIV] saved tradeoff.png")
    return points


if __name__ == "__main__":
    tok = load_tokenizer()
    prompts, _, _ = get_eval_data(30)

    results, sb = {}, {}                                 # L1 + L2
    for name, cfg in MODELS.items():
        model = load_model(**cfg)
        results[name], sb[name] = diversity_scores(model, tok, prompts)
        print(f"[DIV] {name}: {results[name]}")
        del model
        torch.cuda.empty_cache()
    if "seft" in sb and "lora" in sb:
        print(f"[DIV] SEFT vs LoRA Self-BLEU p-value: {significance(sb['seft'], sb['lora']):.4f}")

    tradeoff()                                           # L3
