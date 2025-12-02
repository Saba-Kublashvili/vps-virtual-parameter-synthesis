import torch
import torch.nn as nn

@torch.no_grad()
def householder_chain(Phi, q=4):
    """Build orthogonal rotation via Householder reflectors"""
    d = Phi.shape[0]
    R = torch.eye(d, device=Phi.device, dtype=Phi.dtype)
    M = Phi.clone()

    for _ in range(q):
        vals, vecs = torch.linalg.eigh(M)
        v = vecs[:, -1]  # dominant eigenvector
        v = v / (v.norm() + 1e-8)
        H = torch.eye(d, device=Phi.device) - 2.0 * torch.outer(v, v)
        R = H @ R
        M = H @ M @ H.t()

    return R

@torch.no_grad()
def accumulate_coactivation_stats(model, dataloader, tokenizer, limit_batches=100):
    """Collect co-activation statistics for rotation optimization"""
    stats = []
    layers = []

    for m in model.modules():
        if m.__class__.__name__ == "VPSLinear":
            layers.append(m)
            stats.append(torch.zeros((m.base.weight.shape[0], m.base.weight.shape[0]),
                                     device=m.base.weight.device, dtype=torch.float32))

    for i, batch in enumerate(dataloader):
        if i >= limit_batches: break
        texts = batch.get("text", batch.get("question", batch.get("input", [])))
        if not texts: continue

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
            for j, m in enumerate(layers):
                if getattr(m, "last_h", None) is not None:
                    h = m.last_h.float()
                    stats[j] += h.T @ h

    # Compute rotations via Householder chains
    rotations = []
    for S in stats:
        S = S / (S.trace() + 1e-6)
        R = householder_chain(S + 1e-6*torch.eye(S.shape[0], device=S.device))
        rotations.append(R)

    return rotations

@torch.no_grad()
def apply_mlp_rotations(model, rotations):
    """Apply orthogonal rotations to MLP layers"""
    i = 0
    for name, mod in model.named_modules():
        # Find MLP blocks (various naming conventions)
        if hasattr(mod, "up_proj") and hasattr(mod, "down_proj"):
            if i < len(rotations):
                R = rotations[i]
                i += 1

                up = mod.up_proj
                down = mod.down_proj

                # Handle both wrapped and unwrapped layers
                if hasattr(up, "base"):
                    up_linear = up.base
                    down_linear = down.base
                else:
                    up_linear = up
                    down_linear = down

                if isinstance(up_linear, nn.Linear) and isinstance(down_linear, nn.Linear):
                    # Check dimension compatibility
                    if up_linear.weight.shape[1] == R.shape[0]:
                        up_linear.weight.data = up_linear.weight @ R
                    if down_linear.weight.shape[0] == R.shape[1]:
                        down_linear.weight.data = R.T @ down_linear.weight

    return model


