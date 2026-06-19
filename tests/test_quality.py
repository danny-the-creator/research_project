"""
test_quality.py — Quality (SQ2).

L1  perplexity = exp(eval_loss), already in your training logs — perplexity() below
    recomputes it cleanly so base/LoRA/SEFT are measured the same way (so L1 folds into L2).
L2  perplexity()   -> held-out perplexity on the assistant tokens.
L3  judge_score()  -> average LLM-as-judge persona rating.
"""
import re
import math
import torch
from test_common import MODELS, PERSONA, load, load_judge, free, get_eval_data, generate

RUBRIC = ("Rate from 1 to 5 how strongly the passage below reads like {persona} "
          "(1 = no resemblance, 5 = unmistakably {persona}). Reply with only the number.\n\n"
          "Passage:\n{text}")


@torch.no_grad()
def perplexity(n_prompts=50):
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
def judge_score(n_prompts=25):
    """L3 — average 1-5 persona rating from the LLM judge, per model."""
    prompts, _, _ = get_eval_data(n_prompts)

    answers = {}                                        # generate first, free each model, then judge
    for name in MODELS:
        model, tok = load(name)
        answers[name] = [a[0] for a in generate(model, tok, prompts, max_new_tokens=200, do_sample=False)]
        del model
        free()

    judge, judge_tok = load_judge()
    for name, texts in answers.items():
        scores = []
        for text in texts:
            prompt = RUBRIC.format(persona=PERSONA, text=text)
            inputs = judge_tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True).to(judge.device)
            out = judge.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=judge_tok.eos_token_id)
            m = re.search(r"[1-5]", judge_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            if m:
                scores.append(int(m.group()))
        print(f"[QUALITY] {name:5s} persona score: {sum(scores) / len(scores):.2f}  (n={len(scores)})")
    del judge
    free()


if __name__ == "__main__":
    perplexity()      # L1 + L2
    judge_score()     # L3