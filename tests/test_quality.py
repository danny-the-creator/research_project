"""
test_quality.py — Quality (SQ2: coherent and in-persona?).

L1  perplexity = exp(eval_loss) straight from training — read eval_loss off wandb,
    or pass it to ppl_from_loss() below. No generation needed.
L2  eval_perplexity(): clean held-out perplexity on the persona (assistant) tokens.
L3  persona_scores(): an LLM judge rates how strongly each output sounds like the persona.
"""
import re
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from test_common import MODELS, PERSONA, load_model, load_tokenizer, get_eval_data, generate

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # quantized for the draft; swap for a stronger judge later
RUBRIC = ("Rate from 1 to 5 how strongly the passage below reads like {persona} "
          "(1 = no resemblance, 5 = unmistakably {persona}). Reply with only the number.\n\n"
          "Passage:\n{text}")


def ppl_from_loss(eval_loss):
    """L1 — convert a logged eval_loss into perplexity."""
    return math.exp(eval_loss)


@torch.no_grad()
def eval_perplexity(model, tok, full_messages):
    """L2 — perplexity over held-out persona text, scoring assistant tokens only."""
    nll, ntok = 0.0, 0
    for messages in full_messages:
        enc = tok.apply_chat_template(messages, return_dict=True, return_tensors="pt",
                                      return_assistant_tokens_mask=True)
        ids = enc["input_ids"].to(model.device)
        mask = torch.as_tensor(enc["assistant_masks"]).bool().reshape(ids.shape).to(model.device)
        labels = ids.clone()
        labels[~mask] = -100                         # score only the assistant answer
        n = mask.sum().item()
        nll += model(ids, labels=labels).loss.item() * n
        ntok += n
    return math.exp(nll / ntok)


def load_judge():
    """Load the 4-bit quantized judge model and its own tokenizer."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, quantization_config=bnb, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def judge_one(text, judge, judge_tok):
    """Ask the judge for a single 1-5 persona score for one passage."""
    prompt = RUBRIC.format(persona=PERSONA, text=text)
    ids = judge_tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        add_generation_prompt=True, return_tensors="pt").to(judge.device)
    out = judge.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge_tok.eos_token_id)
    resp = judge_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"[1-5]", resp)
    return int(m.group()) if m else None


def persona_scores(n_prompts=25):
    """L3 — average LLM-as-judge persona score for each model."""
    tok = load_tokenizer()
    prompts, _, _ = get_eval_data(n_prompts)

    generations = {}                                 # generate first, free each model, then judge
    for name, cfg in MODELS.items():
        model = load_model(**cfg)
        generations[name] = [o[0] for o in generate(model, tok, prompts, max_new_tokens=200, do_sample=False)]
        del model
        torch.cuda.empty_cache()

    judge, judge_tok = load_judge()
    for name, texts in generations.items():
        scores = [s for s in (judge_one(t, judge, judge_tok) for t in texts) if s is not None]
        print(f"[QUALITY] {name:5s} persona score: {sum(scores) / len(scores):.2f}  (n={len(scores)})")
    del judge
    torch.cuda.empty_cache()


if __name__ == "__main__":
    tok = load_tokenizer()
    _, fulls, _ = get_eval_data(50)
    for name, cfg in MODELS.items():                 # L2
        model = load_model(**cfg)
        print(f"[QUALITY] {name:5s} perplexity: {eval_perplexity(model, tok, fulls):.2f}")
        del model
        torch.cuda.empty_cache()
    persona_scores()                                 # L3
