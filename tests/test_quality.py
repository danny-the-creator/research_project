"""
test_quality.py — Quality (SQ2).

L1  perplexity = exp(eval_loss), already in your training logs — perplexity() below
    recomputes it cleanly so base/LoRA/SEFT are measured the same way (so L1 folds into L2).
L2  perplexity()   -> held-out perplexity on the assistant tokens.
L3  judge_score()  -> average LLM-as-judge persona rating.
"""
import re
import json
import math
import torch
from test_common import (MODELS, PERSONA, PERSONA_RUBRIC, HELP_RUBRIC, SHOWCASE_ANSWERS_PATH,
                         load, load_judge, free, get_eval_data, generate)



def _ask(judge, judge_tok, content):
    inputs = judge_tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(judge.device)
    out = judge.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=judge_tok.eos_token_id)
    m = re.search(r"[1-5]", judge_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return int(m.group()) if m else None

@torch.no_grad()
def perplexity(n_prompts=500):
    """L1+L2 — held-out perplexity per model, scoring assistant tokens only.
    Skips base: its stock tokenizer has no {% generation %} markers."""
    _, fulls, _ = get_eval_data(n_prompts)
    for name in MODELS:
        if name == "base":
            continue
        model, tok = load(name)
        nll, ntok = 0.0, 0
        for messages in fulls:
            enc = tok.apply_chat_template(messages, return_dict=True, return_tensors="pt",
                                          return_assistant_tokens_mask=True)
            ids = enc["input_ids"].to(model.device)
            mask = torch.as_tensor(enc["assistant_masks"]).bool().reshape(ids.shape).to(model.device)
            labels = ids.clone()
            labels[~mask] = -100
            n = mask.sum().item()
            nll += model(ids, labels=labels).loss.item() * n
            ntok += n
        print(f"[QUALITY] {name:5s} perplexity: {math.exp(nll / ntok):.2f}")
        del model
        free()


@torch.no_grad()
def persona_score(n_prompts=300):
    """Q4 (persona axis) — average 1-5 persona rating on held-out prompts, per model."""
    prompts, _, _ = get_eval_data(n_prompts)

    answers = {}                                        # generate first, free each model, then judge
    for name in MODELS:
        model, tok = load(name)
        answers[name] = [a[0] for a in generate(model, tok, prompts, max_new_tokens=200, do_sample=False)]
        del model
        free()

    judge, judge_tok = load_judge()
    for name, texts in answers.items():
        scores = [s for s in (_ask(judge, judge_tok, PERSONA_RUBRIC.format(persona=PERSONA, text=t))
                              for t in texts) if s is not None]
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((x - mean) ** 2 for x in scores) / n) ** 0.5
        print(f"[QUALITY] {name:6s} persona {mean:.2f} ± {std:.2f}  (n={n})")
    del judge
    free()


def helpfulness_score(answers_file="showcase_answers.json"):
    """Q4 (helpfulness axis) — judge how completely each model answers the showcase questions.
    Reads the records S1's showcase() already wrote, so no regeneration."""
    with open(answers_file, encoding="utf-8") as f:
        records = json.load(f)                            # [{"question":..., "base":..., "lora":..., ...}, ...]

    judge, judge_tok = load_judge()
    for name in MODELS:
        scores = []
        for rec in records:
            s = _ask(judge, judge_tok, HELP_RUBRIC.format(question=rec["question"], text=rec[name]))
            if s is not None:
                scores.append(s)
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((x - mean) ** 2 for x in scores) / n) ** 0.5
        print(f"[QUALITY] {name:6s} helpfulness {mean:.2f} ± {std:.2f}  (n={n})")
    del judge
    free()

if __name__ == "__main__":
    perplexity()
    persona_score()
    helpfulness_score(SHOWCASE_ANSWERS_PATH)