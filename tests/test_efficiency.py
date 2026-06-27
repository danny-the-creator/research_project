"""
test_efficiency.py — Efficiency (SQ3).

Training:
  L1  no code — peak memory + time/epoch already from your runs (align the two configs first!).
  L2  MemTimeCallback    -> add to your SFTTrainer callbacks in main.py / seft_pipeline.py.
  L3  ignored for now.
Inference (on saved checkpoints):
  L2     throughput()      -> decoding speed in tokens/sec (expect SEFT == LoRA on a GPU).
  L1+L3  flops_analysis()  -> nonzero fraction + theoretical FLOP reduction (for the write-up).
"""
import time
import torch
from transformers import TrainerCallback
from test_common import MODELS, load, free, get_eval_data

TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MemTimeCallback(TrainerCallback):
    """Training L2 — log peak GPU memory and wall-clock time per epoch."""
    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()

    def on_epoch_begin(self, args, state, control, **kw):
        self._t = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kw):
        print(f"[EFF] epoch time: {time.perf_counter() - self._t:.1f}s")

    def on_train_end(self, args, state, control, **kw):
        print(f"[EFF] peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


@torch.no_grad()
def throughput(model, tok, prompt, max_new_tokens=256):
    """Inference L2 — decoding speed (tokens/sec) for one prompt."""
    inputs = tok.apply_chat_template(prompt, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    model.generate(**inputs, max_new_tokens=8, pad_token_id=tok.eos_token_id)    # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return (out.shape[1] - inputs["input_ids"].shape[1]) / (time.perf_counter() - t0)


@torch.no_grad()
def flops_analysis(model):
    """Inference L1+L3 — nonzero fraction + theoretical FLOP reduction in the target linears."""
    nz = tot = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and any(t in name for t in TARGET):
            nz += (m.weight != 0).sum().item()
            tot += m.weight.numel()
    frac = nz / max(tot, 1)
    return dict(nonzero_frac=frac, theoretical_reduction=1 - frac, ideal_speedup=1 / max(frac, 1e-9))

@torch.no_grad()
def model_size_gb(model):
    """Stored size of the weights in GB (handles bf16 and 4-bit packed params)."""
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_ / 1e9

if __name__ == "__main__":
    prompts, _, _ = get_eval_data(1)
    for name in MODELS:
        model, tok = load(name)

        size = model_size_gb(model)

        torch.cuda.reset_peak_memory_stats()
        tps = throughput(model, tok, prompts[0])
        peak = torch.cuda.max_memory_allocated() / 1e9

        if name != "lora-q":
            fa = flops_analysis(model)
            print(f"[EFF] {name:6s} {tps:5.1f} tok/s | size={size:.2f}GB peak={peak:.2f}GB | "  
                  f"nonzero={fa['nonzero_frac']:.2f} reduction={fa['theoretical_reduction']:.2f} "
                  f"ideal_speedup={fa['ideal_speedup']:.2f}x")
        else:
            print(f"[EFF] {name:6s} {tps:5.1f} tok/s | size={size:.2f}GB peak={peak:.2f}GB | " 
                  f"flops n/a (4-bit)")

        del model
        free()