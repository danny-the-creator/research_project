"""
seft.py — Sparsity Evolution Fine-Tuning (SEFT), self-contained.

Faithful (but simplified) reimplementation of the core mechanism from
"Leave it to the Specialist: Repair Sparse LLMs with Sparse Fine-Tuning via
Sparsity Evolution" (Xiao et al., 2025, arXiv:2505.24037).

Per target nn.Linear with a (sparse) frozen weight W:
  * adds k trainable scalar DELTAS at chosen positions inside W
  * trains only those deltas (W stays frozen and sparse)
  * every `reselection_steps`, runs RigL-style DROP-AND-GROW:
      - drop the lowest-|value| active deltas
      - grow new deltas at the highest accumulated-|gradient| positions
      - growth is restricted to currently-nonzero base positions, so the
        model's sparsity pattern is preserved throughout
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


class _SeftLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, values, indices, bias, mod):
        eff = weight.clone()                                   # transient: built, used, freed
        eff.view(-1).scatter_add_(0, indices, values.to(eff.dtype))
        out = F.linear(x, eff, bias)
        # <<< FIX (memory cliff): save a REFERENCE to the frozen weight, NOT the clone.
        # Saving `eff` held a full weight copy per layer until backward (~5.6GB across a 3B).
        ctx.save_for_backward(x, weight, values, indices)
        ctx.mod = mod
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, g):
        x, weight, values, indices = ctx.saved_tensors
        in_f = weight.shape[1]
        # recompute eff transiently for grad-input (one matrix at a time, then freed)
        eff = weight.clone()
        eff.view(-1).scatter_add_(0, indices, values.to(eff.dtype))
        gx = g @ eff
        del eff
        g2 = g.reshape(-1, g.shape[-1]).float()
        x2 = x.reshape(-1, x.shape[-1]).float()
        out_f = g2.shape[1]; n = out_f * in_f; N = g2.shape[0]; k = indices.numel()
        mod = ctx.mod
        if mod is not None and mod.accumulate:
            gW = g2.transpose(0, 1) @ x2                        # [out,in]; reuse for grad + growth
            gvalues = gW.reshape(-1)[indices].to(values.dtype)
            mod._accumulate_candidates(gW.abs().reshape(-1))
            del gW
        elif N * k < n:
            # <<< FIX (speed): gather is only cheap when N*k < out*in (small k / short seq)
            rows = indices // in_f; cols = indices % in_f
            gvalues = (g2[:, rows] * x2[:, cols]).sum(0).to(values.dtype)
        else:
            # large delta budget (e.g. MLP): matmul+index avoids the huge [N,k] gather (~10x faster)
            gvalues = (g2.transpose(0, 1) @ x2).reshape(-1)[indices].to(values.dtype)
        gb = g2.sum(0).to(weight.dtype) if ctx.has_bias else None
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
        self.n_nonzero = int(nz_idx.numel())
        if self.n_nonzero < self.k:                      # very sparse layer guard
            self.k = max(1, self.n_nonzero)
            self.values = nn.Parameter(torch.zeros(self.k, device=dev, dtype=torch.float32))
        perm = nz_idx[torch.randperm(self.n_nonzero, device=dev)[:self.k]]
        self.register_buffer("indices", perm.long())
        # <<< FIX (memory): bounded candidate "leaderboard" instead of a full [n] buffer.
        # Permanent footprint ~ O(cand_M) floats (KB), not O(out*in) (GB).
        self.cand_M = min(self.n_nonzero, max(int(0.5 * self.k), 2048))  # <<< FIX: bound top-k cost
        self.cand_idx = None     # plain attrs (training-only state; not saved, not in state_dict)
        self.cand_val = None
        self.accumulate = False

    def forward(self, x):
        return _SeftLinearFn.apply(x, self.base.weight, self.values,
                                   self.indices, self.base.bias, self)

    @torch.no_grad()
    def _accumulate_candidates(self, gabs_flat):
        """Merge this step's top-M grad positions into the running leaderboard."""
        gabs_flat[self.indices] = 0.0                              # exclude active deltas
        gabs_flat[self.base.weight.reshape(-1) == 0] = 0.0         # preserve base sparsity
        take = min(self.cand_M, gabs_flat.numel())
        v, i = torch.topk(gabs_flat, take)
        if self.cand_idx is None:
            self.cand_idx, self.cand_val = i.clone(), v.clone()
        else:
            cat_i = torch.cat([self.cand_idx, i])
            cat_v = torch.cat([self.cand_val, v])
            uniq, inv = torch.unique(cat_i, return_inverse=True)
            summed = torch.zeros(uniq.numel(), device=uniq.device, dtype=torch.float32)
            summed.index_add_(0, inv, cat_v)
            if uniq.numel() > self.cand_M:
                vv, ii = torch.topk(summed, self.cand_M)
                self.cand_idx, self.cand_val = uniq[ii], vv
            else:
                self.cand_idx, self.cand_val = uniq, summed

    @torch.no_grad()
    def _reset_candidates(self):
        self.cand_idx = None
        self.cand_val = None

    @torch.no_grad()
    def reselect(self, drop_frac):
        """RigL drop-and-grow. Returns the local delta-slots that changed."""
        n_drop = max(1, int(self.k * drop_frac))
        drop_local = torch.topk(self.values.abs(), n_drop, largest=False).indices
        if self.cand_idx is None or self.cand_idx.numel() == 0:
            self._reset_candidates()                 # no grad signal this cycle -> keep topology
            return drop_local[:0]
        take = min(n_drop, self.cand_idx.numel())
        top = torch.topk(self.cand_val, take).indices
        grow = self.cand_idx[top]
        # safety: only ever grow into nonzero-base positions (preserve sparsity exactly)
        valid = self.base.weight.reshape(-1)[grow] != 0
        grow = grow[valid]
        take = grow.numel()
        drop_local = drop_local[:take]
        self.indices[drop_local] = grow
        self.values.data[drop_local] = 0.0           # new deltas start at 0
        self._reset_candidates()
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


def merge_and_unwrap(model):
    """<<< FIX (saving): merge deltas into the base, then replace each SeftLinear with
    its plain nn.Linear, so the model is a STANDARD architecture again and
    model.save_pretrained() produces a normal, reloadable checkpoint."""
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, SeftLinear)]
    for _, m in targets:
        m.merge_into_base()
    for name, m in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        setattr(parent, child, m.base)
    return len(targets)


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
        acc = (self.R - phase) <= self.A           # accumulate in the A steps before a boundary
        for m in mods:
            m.accumulate = acc
        if step > 0 and phase == 0:                # boundary: drop-and-grow
            rate = self._rate(step)
            for m in mods:
                changed = m.reselect(rate)
                m.accumulate = False
                if optimizer is not None and changed.numel() > 0:
                    st = optimizer.state.get(m.values, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][changed] = 0.0     # fresh deltas: no stale Adam momentum
            if state.is_world_process_zero:
                print(f"[SEFT] step {step}: topology update, drop/grow rate={rate:.3f}")


@torch.no_grad()
def report_sparsity(model, tag=""):
    """<<< FIX (reporting): count each weight ONCE and skip SEFT delta params.
    (The old version added SeftLinear base weights twice -> inflated totals.)"""
    tot = zero = 0
    for name, p in model.named_parameters():
        if name.endswith(".values"):           # SEFT deltas: not 'weights'; folded into base on merge
            continue
        tot += p.numel()
        zero += (p == 0).sum().item()
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

    base_nz = int((lin.weight != 0).sum())
    sl.accumulate = True
    sl.zero_grad(); sl(torch.randn(4, 8)).sum().backward()
    changed = sl.reselect(0.4)
    print("budget constant     :", sl.indices.numel() == 5)
    print("grow on nonzero only:", bool((lin.weight.reshape(-1)[sl.indices] != 0).all()))
    print("base sparsity kept  :", int((lin.weight != 0).sum()) == base_nz)
    sl.merge_into_base()
    print("merge keeps sparsity:", int((lin.weight != 0).sum()) == base_nz)
    print("no full-n buffer    :", not hasattr(sl, "grad_accum"),
          "| cand_M =", sl.cand_M, "(<<", sl.n, ")")