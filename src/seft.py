"""
seft.py — Sparsity Evolution Fine-Tuning (SEFT), self-contained.

Faithful (but simplified) reimplementation of the core mechanism from
"Leave it to the Specialist: Repair Sparse LLMs with Sparse Fine-Tuning via
Sparsity Evolution" (Xiao et al., 2025, arXiv:2505.24037), designed to drop
into a modern transformers/trl stack and mirror a LoRA pipeline.

What it does, per target nn.Linear with a (sparse) frozen weight W:
  * adds k trainable scalar DELTAS at chosen positions inside W
  * trains only those deltas (W itself stays frozen and sparse)
  * every `reselection_steps`, runs RigL-style DROP-AND-GROW:
      - drop the lowest-|value| active deltas
      - grow new deltas at the highest accumulated-|gradient| positions
      - growth is restricted to currently-nonzero base positions, so the
        model's sparsity pattern is preserved throughout

Differences from the official repo: the official code uses a custom CUDA
kernel (linear-sd) and an SftAdamW optimizer; this uses pure PyTorch + a
TrainerCallback so it runs anywhere. For paper-exact 8B results use the repo.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


class _SeftLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, values, indices, bias, mod):
        eff = weight.clone()
        eff.view(-1).scatter_add_(0, indices, values.to(eff.dtype))
        ctx.save_for_backward(x, eff, values, indices)
        ctx.mod = mod
        ctx.has_bias = bias is not None
        ctx.in_f = weight.shape[1]
        return F.linear(x, eff, bias)

    @staticmethod
    def backward(ctx, g):
        x, eff, values, indices = ctx.saved_tensors
        mod, in_f = ctx.mod, ctx.in_f
        gx = g @ eff
        g2 = g.reshape(-1, g.shape[-1]).float()
        x2 = x.reshape(-1, x.shape[-1]).float()
        # cheap per-delta grad: only the active positions (no dense [out,in] matmul)
        rows = indices // in_f
        cols = indices % in_f
        gvalues = (g2[:, rows] * x2[:, cols]).sum(0).to(values.dtype)
        # expensive dense grad ONLY during the accumulation window (for growth)
        if mod is not None and mod.accumulate:
            gW = g2.transpose(0, 1) @ x2          # [out, in]
            mod.grad_accum += gW.abs().reshape(-1)
        gb = g2.sum(0).to(eff.dtype) if ctx.has_bias else None
        return gx, None, gvalues, None, gb, None


class SeftLinear(nn.Module):
    """Wraps a frozen (sparse) nn.Linear and adds k trainable weight deltas."""
    def __init__(self, base: nn.Linear, k: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.n = out_f * in_f
        self.k = int(k)
        dev = base.weight.device
        self.values = nn.Parameter(torch.zeros(self.k, device=dev, dtype=torch.float32))
        nz_idx = (base.weight.reshape(-1) != 0).nonzero(as_tuple=True)[0]
        if nz_idx.numel() < self.k:                # very sparse layer guard
            self.k = max(1, nz_idx.numel())
            self.values = nn.Parameter(torch.zeros(self.k, device=dev, dtype=torch.float32))
        perm = nz_idx[torch.randperm(nz_idx.numel(), device=dev)[:self.k]]
        self.register_buffer("indices", perm.long())
        self.register_buffer("grad_accum", torch.zeros(self.n, device=dev))
        self.accumulate = False

    def forward(self, x):
        return _SeftLinearFn.apply(x, self.base.weight, self.values,
                                   self.indices, self.base.bias, self)

    @torch.no_grad()
    def reselect(self, drop_frac):
        """RigL drop-and-grow. Returns the local delta-slots that changed."""
        n_drop = max(1, int(self.k * drop_frac))
        drop_local = torch.topk(self.values.abs(), n_drop, largest=False).indices
        scores = self.grad_accum.clone()
        scores[self.indices] = -1.0                         # exclude active
        scores[self.base.weight.reshape(-1) == 0] = -1.0    # preserve base sparsity
        grow = torch.topk(scores, n_drop, sorted=False).indices
        self.indices[drop_local] = grow
        self.values.data[drop_local] = 0.0                  # new deltas start at 0
        self.grad_accum.zero_()
        return drop_local

    @torch.no_grad()
    def merge_into_base(self):
        self.base.weight.view(-1).scatter_add_(
            0, self.indices, self.values.data.to(self.base.weight.dtype))


def inject_seft(model, target_substrings, density):
    """Replace matching nn.Linear layers with SeftLinear; freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if not any(t in name for t in target_substrings):
            continue
        if not isinstance(module, nn.Linear):
            continue
        k = max(1, int(density * module.weight.numel()))
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        setattr(parent, child, SeftLinear(module, k))
        replaced.append(name)
    return replaced


def seft_modules(model):
    return [m for m in model.modules() if isinstance(m, SeftLinear)]


class SeftCallback(TrainerCallback):
    """Drives the topology evolution during training (the heart of SEFT)."""
    def __init__(self, reselection_steps=60, accumulation_steps=5,
                 initial_reselection_rate=0.2, total_steps=None):
        self.R = reselection_steps
        self.A = accumulation_steps
        self.rate0 = initial_reselection_rate
        self.total = total_steps

    def _rate(self, step):                     # cosine decay of the drop fraction (RigL)
        if not self.total:
            return self.rate0
        return self.rate0 / 2 * (1 + math.cos(math.pi * step / self.total))

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kw):
        step = state.global_step
        mods = seft_modules(model)
        phase = step % self.R
        # turn dense-grad accumulation ON only in the A steps before a boundary
        acc = (self.R - phase) <= self.A
        for m in mods:
            m.accumulate = acc
        # at a boundary: drop-and-grow, then reset optimizer state for changed slots
        if step > 0 and phase == 0:
            rate = self._rate(step)
            for m in mods:
                changed = m.reselect(rate)
                m.accumulate = False
                if optimizer is not None:
                    st = optimizer.state.get(m.values, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][changed] = 0.0
            if state.is_world_process_zero:
                print(f"[SEFT] step {step}: topology update, drop/grow rate={rate:.3f}")


@torch.no_grad()
def report_sparsity(model, tag=""):
    tot = zero = 0
    for _, p in model.named_parameters():
        tot += p.numel(); zero += (p == 0).sum().item()
    for m in seft_modules(model):              # count frozen base weights too
        w = m.base.weight
        tot += w.numel(); zero += (w == 0).sum().item()
    pct = 100.0 * zero / max(tot, 1)
    print(f"[SEFT] sparsity {tag}: {pct:.1f}%  ({zero:,}/{tot:,} zeros)")
    return pct


# --------------------------- self test ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    lin = nn.Linear(8, 6)
    with torch.no_grad():
        lin.weight[torch.rand_like(lin.weight) < 0.5] = 0.0
    sl = SeftLinear(lin, k=5)
    sl.values.data = torch.randn(5)
    x = torch.randn(3, 8, requires_grad=True)

    # dense reference
    vr = sl.values.data.clone().requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    effr = lin.weight.detach().clone()
    effr.view(-1).scatter_add_(0, sl.indices, vr)
    ref = F.linear(xr, effr, lin.bias.detach())
    out = sl(x)
    out.sum().backward(); ref.sum().backward()
    print("forward match :", torch.allclose(out, ref, atol=1e-5))
    print("grad x  match :", torch.allclose(x.grad, xr.grad, atol=1e-5))
    print("grad dv match :", torch.allclose(sl.values.grad, vr.grad, atol=1e-5))

    # accumulation + reselect preserves budget & base sparsity
    base_nz = int((lin.weight != 0).sum())
    sl.accumulate = True
    sl.zero_grad(); sl(torch.randn(4, 8)).sum().backward()
    changed = sl.reselect(0.4)
    print("budget constant     :", sl.indices.numel() == 5)
    print("grow on nonzero only:", bool((lin.weight.reshape(-1)[sl.indices] != 0).all()))
    print("base sparsity kept  :", int((lin.weight != 0).sum()) == base_nz)
    sl.merge_into_base()
    print("merge keeps sparsity:", int((lin.weight != 0).sum()) == base_nz)