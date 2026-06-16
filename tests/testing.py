"""
evaluate.py — shared evaluation pipeline for the LoRA vs SEFT comparison.

Runs IDENTICALLY on both outputs: your SEFT checkpoint (a standard model after
merge_and_unwrap) and your LoRA output (adapter auto-merged onto its base).

Metrics, mapped to your proposal:
  Quality    : perplexity on held-out persona text ; LLM-as-judge persona consistency
  Diversity  : Self-BLEU (lower = more diverse) ; Distinct-1 / Distinct-2
  Efficiency : inference latency (tokens/sec) ; peak GPU memory (GB)

Usage:
  python evaluate.py --model ./saved_models/seft/version_0 --persona "Sherlock Holmes"
  # or import evaluate_model(...) / compare([...]) from your own script.
"""
import os, time, json, argparse, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------- diversity (no model)
def distinct_n(texts, n, tokenizer=None):
    """Unique n-grams / total n-grams across `texts`. Higher = more lexical variety."""
    total, seen = 0, set()
    for t in texts:
        toks = tokenizer.tokenize(t) if tokenizer else t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams); seen.update(grams)
    return len(seen) / max(total, 1)


def self_bleu(texts, max_n=4):
    """Mean BLEU of each text against the others as references. Lower = more diverse.
    Standard Texygen-style Self-BLEU. Computed within a group of responses."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    tok = [t.split() for t in texts if t.strip()]
    if len(tok) < 2:
        return float("nan")
    w = tuple([1.0 / max_n] * max_n)
    scores = []
    for i in range(len(tok)):
        refs = tok[:i] + tok[i + 1:]
        scores.append(sentence_bleu(refs, tok[i], weights=w, smoothing_function=smooth))
    return sum(scores) / len(scores)


def diversity_metrics(per_prompt_generations):
    """per_prompt_generations: list (one per prompt) of lists of response strings."""
    pooled = [g for group in per_prompt_generations for g in group]
    sb = [self_bleu(group) for group in per_prompt_generations if len(group) > 1]
    sb = [s for s in sb if not math.isnan(s)]
    return {
        "self_bleu": (sum(sb) / len(sb)) if sb else float("nan"),  # avg over prompts; lower=better
        "distinct_1": distinct_n(pooled, 1),
        "distinct_2": distinct_n(pooled, 2),
    }


# ----------------------------------------------------------------------------- model loading
def load_model(path, base_model=None, dtype=torch.bfloat16, device="cuda"):
    """Load a full checkpoint, OR a LoRA adapter (auto-detected, merged onto base)."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_lora_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    if is_lora_adapter:
        from peft import PeftModel
        if base_model is None:
            cfg = json.load(open(os.path.join(path, "adapter_config.json")))
            base_model = cfg["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, path).merge_and_unload()  # dense merge for eval
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return model.to(device).eval(), tokenizer


# ----------------------------------------------------------------------------- quality: perplexity
@torch.no_grad()
def perplexity(model, tokenizer, texts, max_len=512, device="cuda"):
    """Token-weighted corpus perplexity on held-out text. Lower = better fit to the style."""
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1                 # number of predicted tokens
        total_nll += out.loss.item() * n     # loss is mean CE over the n predictions
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


# ----------------------------------------------------------------------------- generation + efficiency
@torch.no_grad()
def generate(model, tokenizer, prompts, n_per_prompt=5, max_new_tokens=128,
             temperature=0.8, top_p=0.95, device="cuda"):
    """Sample n responses per prompt. Returns (per_prompt_generations, efficiency dict)."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    per_prompt, total_new, t0 = [], 0, time.time()
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt").to(device)
        outs = model.generate(
            ids, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, num_return_sequences=n_per_prompt,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = outs[:, ids.shape[1]:]                      # strip the prompt
        total_new += int((gen != tokenizer.pad_token_id).sum())
        per_prompt.append([tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen])
    dt = time.time() - t0
    eff = {"tokens_per_sec": total_new / max(dt, 1e-9),
           "gen_seconds": dt,
           "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None}
    return per_prompt, eff


# ----------------------------------------------------------------------------- quality: LLM-as-judge
JUDGE_PROMPT = (
    "You are evaluating whether a response stays in character as {persona}.\n"
    "Rate ONLY persona consistency (voice, style, vocabulary, knowledge) on a 1-5 scale:\n"
    "1 = clearly out of character, 5 = indistinguishable from {persona}.\n\n"
    "User prompt: {prompt}\nResponse: {response}\n\n"
    "Reply with a single integer 1-5 and nothing else."
)

def judge_persona(per_prompt_generations, prompts, persona, judge_fn=None):
    """judge_fn(text) -> str/int score. Pluggable: wire it to any LLM you have access to
    (see make_*_judge below). Returns mean score, or None if no judge is configured."""
    if judge_fn is None:
        print("[judge] no judge_fn provided -> skipping persona-consistency "
              "(wire make_anthropic_judge / make_openai_judge to enable).")
        return None
    scores = []
    for prompt, group in zip(prompts, per_prompt_generations):
        for resp in group:
            raw = judge_fn(JUDGE_PROMPT.format(persona=persona, prompt=prompt, response=resp))
            try:
                scores.append(int(str(raw).strip()[0]))
            except (ValueError, IndexError):
                continue
    return (sum(scores) / len(scores)) if scores else None

# --- example judge factories (uncomment + install the client you use) -------------------
# def make_anthropic_judge(model="claude-haiku-4-5-20251001"):
#     from anthropic import Anthropic
#     client = Anthropic()
#     def fn(prompt):
#         m = client.messages.create(model=model, max_tokens=4,
#                                     messages=[{"role": "user", "content": prompt}])
#         return m.content[0].text
#     return fn
#
# def make_openai_judge(model="gpt-4o-mini"):
#     from openai import OpenAI
#     client = OpenAI()
#     def fn(prompt):
#         r = client.chat.completions.create(model=model, max_tokens=4,
#                                             messages=[{"role": "user", "content": prompt}])
#         return r.choices[0].message.content
#     return fn


# ----------------------------------------------------------------------------- orchestration
def evaluate_model(model_path, prompts, held_out_texts, persona="the target persona",
                   base_model=None, n_per_prompt=5, judge_fn=None, device="cuda"):
    model, tokenizer = load_model(model_path, base_model=base_model, device=device)
    ppl = perplexity(model, tokenizer, held_out_texts, device=device)
    gens, eff = generate(model, tokenizer, prompts, n_per_prompt=n_per_prompt, device=device)
    div = diversity_metrics(gens)
    persona_score = judge_persona(gens, prompts, persona, judge_fn=judge_fn)
    results = {
        "model": model_path,
        "quality":   {"perplexity": ppl, "persona_consistency": persona_score},
        "diversity": div,
        "efficiency": eff,
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results, gens


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else str(x)

def compare(model_paths, prompts, held_out_texts, **kw):
    """Run several models and print a side-by-side table."""
    rows = [evaluate_model(p, prompts, held_out_texts, **kw)[0] for p in model_paths]
    keys = [("quality", "perplexity"), ("quality", "persona_consistency"),
            ("diversity", "self_bleu"), ("diversity", "distinct_1"),
            ("diversity", "distinct_2"), ("efficiency", "tokens_per_sec"),
            ("efficiency", "peak_gpu_gb")]
    head = ["metric"] + [os.path.basename(r["model"].rstrip("/")) for r in rows]
    print(" | ".join(f"{h:>22}" for h in head))
    for a, b in keys:
        line = [f"{a}.{b}"] + [_fmt(r[a][b]) for r in rows]
        print(" | ".join(f"{c:>22}" for c in line))
    return rows


# ----------------------------------------------------------------------------- helpers for TRAINING-time efficiency
# (drop these around trainer.train() in your training scripts to capture RQ3's training metrics)
def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_mem_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_model", default=None, help="base id for a LoRA adapter")
    ap.add_argument("--persona", default="the target persona")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    held_out = ["It is a capital mistake to theorize before one has data.",
                "You see, but you do not observe. The distinction is clear."]
    eval_prompts = ["What do you make of the muddy footprints by the door?",
                    "How should one approach an unsolvable problem?"]

    res, gens = evaluate_model(args.model, eval_prompts, held_out,
                               persona=args.persona, base_model=args.base_model,
                               n_per_prompt=args.n)
    print(json.dumps(res, indent=2))
    print("\nsample generation:\n", gens[0][0][:300])