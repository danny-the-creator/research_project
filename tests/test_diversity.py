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
from transformers import set_seed
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")                                  # save plots without a display
import matplotlib.pyplot as plt

from test_common import MODELS, load, free, get_eval_data, generate

_smooth = SmoothingFunction().method1

SEEDS = [0, 1, 2]


def repetition_rate(text, n=3):
    grams = list(ngrams(text.split(), n))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


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
    avg_rep = np.mean([repetition_rate(s) for g in samples for s in g])
    return per_prompt_sb, distinct_n(flat, 1), distinct_n(flat, 2), avg_rep


def tradeoff(temps=(0.3, 0.5, 0.7, 0.8, 0.9, 1.1, 1.3), n_prompts=30, n=5, seeds=SEEDS):
    """L3 — diversity at several temperatures per model; plot quality vs diversity."""
    prompts, _, _ = get_eval_data(n_prompts)

    raw = {}                                            # generate everything first, freeing each model
    for name in MODELS:
        if name == "base":
            continue
        model, tok = load(name)
        raw[name] = {}
        for t in temps:
            per_seed = []
            for s in seeds:  # NEW: one generation per seed
                set_seed(s)
                per_seed.append(generate(model, tok, prompts, max_new_tokens=200, num_return_sequences=n,
                                         do_sample=True, temperature=t, top_p=0.95))
            raw[name][t] = per_seed
        del model
        free()

    ref, ref_tok = load("reference")
    for name, by_temp in raw.items():
        xs, ys, xerr, yerr = [], [], [], []  # NEW: means + stds per temperature
        for t, per_seed in by_temp.items():
            divs = [float(np.mean([self_bleu(g) for g in samples])) for samples in per_seed]
            quals = [fluency_ppl([s for g in samples for s in g], ref, ref_tok) for samples in per_seed]
            xs.append(np.mean(divs))
            ys.append(np.mean(quals))
            xerr.append(np.std(divs))  # NEW: spread across seeds
            yerr.append(np.std(quals))
            print(f"[DIV] {name} T={t}: self_bleu {np.mean(divs):.3f}±{np.std(divs):.3f} "
                  f"fluency_ppl {np.mean(quals):.2f}±{np.std(quals):.2f}")
        plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker="o", capsize=3, label=name)
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

    sb_seed0 = {}                                      # NEW: per-prompt Self-BLEU at the first seed, for the paired test
    for name in MODELS:
        model, tok = load(name)
        sb_means, d1s, d2s, reps = [], [], [], []               # NEW: collect one value per seed
        for i, s in enumerate(SEEDS):
            print(f"Seed: {s}")
            set_seed(s)                                # NEW: fix randomness for this pass
            per_prompt_sb, d1, d2, rep = diversity_scores(model, tok, prompts)   # unchanged call
            sb_means.append(np.mean(per_prompt_sb))
            d1s.append(d1)
            d2s.append(d2)
            reps.append(rep)
            if i == 0:
                sb_seed0[name] = per_prompt_sb         # keep first-seed list for Wilcoxon
        print(f"[DIV] {name:6s} self_bleu {np.mean(sb_means):.3f} ± {np.std(sb_means):.3f} "    # NEW: mean ± std
              f"distinct_1 {np.mean(d1s):.3f} distinct_2 {np.mean(d2s):.3f}  "
              f"avg_repeat={np.mean(reps):.5f} ({len(SEEDS)} seeds)")
        del model
        free()

    if "lora" in sb_seed0 and "seft" in sb_seed0:
        _, p = wilcoxon(sb_seed0["seft"], sb_seed0["lora"])                  # CHANGED: now on the first-seed lists
        print(f"[DIV] SEFT vs LoRA self_bleu p-value (seed {SEEDS[0]}): {p:.4f}")

    tradeoff()