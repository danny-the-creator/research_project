"""
test_efficiency.py — Efficiency (SQ3).

TRAINING (measured during training, not here): add MemTimeCallback to your
SFTTrainer callbacks in main.py / seft_pipeline.py. memory_components() is an
optional finer breakdown you can call from inside the training script.

INFERENCE (run on saved checkpoints):
L2  throughput(): decoding speed in tokens/sec (expect SEFT == LoRA on a GPU).
L3  flops_analysis(): the FLOP reduction sparsity would give with a sparse engine.
"""
import time
import torch
from transformers import TrainerCallback
from test_common import MODELS, load_model, load_tokenizer, get_eval_data

TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MemTimeCallback(TrainerCallback):
    """E-train-L2 — log peak GPU memory and wall-clock time per epoch."""
    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()

    def on_epoch_begin(self, args, state, control, **kw):
        self._t = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kw):
        print(f"[EFF] epoch time: {time.perf_counter() - self._t:.1f}s")

    def on_train_end(self, args, state, control, **kw):
        print(f"[EFF] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


@torch.no_grad()
def memory_components(model, optimizer):
    """E-train-L3 (optional) — rough static memory split in GB (activations not included)."""
    GB = 1e9
    params = sum(t.numel() * t.element_size() for t in model.parameters()) / GB
    grads = sum(t.grad.numel() * t.grad.element_size() for t in model.parameters() if t.grad is not None) / GB
    optim = sum(s.numel() * s.element_size()
                for st in optimizer.state.values() for s in st.values() if torch.is_tensor(s)) / GB
    print(f"[EFF] params={params:.2f}GB grads={grads:.2f}GB optimizer={optim:.2f}GB")
    return dict(params=params, grads=grads, optimizer=optim)


@torch.no_grad()
def throughput(model, tok, prompt, max_new_tokens=256):
    """E-inf-L2 — decoding speed (tokens/sec) for one prompt."""
    ids = tok.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)
    model.generate(ids, max_new_tokens=8, pad_token_id=tok.eos_token_id)        # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return (out.shape[1] - ids.shape[1]) / (time.perf_counter() - t0)


@torch.no_grad()
def flops_analysis(model):
    """E-inf-L3 — theoretical FLOP reduction from sparsity in the target linear layers."""
    nz = tot = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and any(t in name for t in TARGET):
            nz += (m.weight != 0).sum().item()
            tot += m.weight.numel()
    frac = nz / max(tot, 1)
    return dict(nonzero_frac=frac, theoretical_reduction=1 - frac, ideal_speedup=1 / max(frac, 1e-9))


if __name__ == "__main__":
    tok = load_tokenizer()
    prompts, _, _ = get_eval_data(1)
    for name, cfg in MODELS.items():
        model = load_model(**cfg)
        tps = throughput(model, tok, prompts[0])
        fa = flops_analysis(model)
        print(f"[EFF] {name:5s} {tps:5.1f} tok/s | nonzero={fa['nonzero_frac']:.2f} "
              f"reduction={fa['theoretical_reduction']:.2f} ideal_speedup={fa['ideal_speedup']:.2f}x")
        del model
        torch.cuda.empty_cache()
