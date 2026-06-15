"""
prune.py — quick magnitude pruning to make a DENSE model sparse for SEFT.

meta-llama/Llama-3.2-3B-Instruct ships dense, but SEFT only does something
meaningful on an already-sparse model (it evolves deltas inside the existing
nonzeros). So we prune first.

For pipeline testing, unstructured per-matrix magnitude pruning is enough:
zero the smallest-|w| `sparsity` fraction of each target weight matrix.
For the *real* quality numbers later, prefer a calibrated method (Wanda /
SparseGPT) — magnitude is the fast, dependency-free way to get the pipeline
running. For a fair LoRA-vs-SEFT comparison, prune ONCE and reuse the same
sparse base for both pipelines.
"""
import torch
import torch.nn as nn

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def magnitude_prune(model, sparsity=0.5, target_substrings=None):
    """Zero the smallest-|w| `sparsity` fraction of each matching nn.Linear weight.
    Returns the number of matrices pruned."""
    if target_substrings is None:
        target_substrings = DEFAULT_TARGETS
    pruned = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_substrings):
            continue
        w = module.weight.data
        k = int(sparsity * w.numel())
        if k <= 0:
            continue
        # k-th smallest |w| -> threshold; zero everything at or below it
        thresh = torch.kthvalue(w.abs().reshape(-1).float(), k).values.to(w.dtype)
        w[w.abs() <= thresh] = 0.0
        pruned += 1
    return pruned
