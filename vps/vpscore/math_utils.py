from __future__ import annotations
import torch

# ---------------------------
# Entropy utilities
# ---------------------------

@torch.no_grad()
def compute_token_entropy(
    logits: torch.Tensor,
    dim: int = -1,
    normalize: bool = False
) -> torch.Tensor:
    """
    Per-token entropy from logits.

    Args:
        logits: Tensor of shape (..., V) where V = vocab size.
        dim:    Dimension of the vocabulary (usually -1).
        normalize: If True, divides by log(V) to get [0,1] range.

    Returns:
        Tensor of shape logits.shape without the vocab dimension (i.e., reduced along `dim`).
    """
    # Use float32 for numerical stability on T4 (half/bfloat16 can trip SVD/softmax edges)
    if logits.dtype in (torch.float16, torch.bfloat16):
        l = logits.float()
    else:
        l = logits

    log_probs = torch.log_softmax(l, dim=dim)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=dim)

    if normalize:
        V = logits.size(dim)
        denom = torch.log(torch.tensor(float(V), device=logits.device, dtype=ent.dtype)).clamp_min(1e-8)
        ent = ent / denom
    return ent

# ---------------------------
# Spectral / clipping helpers
# ---------------------------

@torch.no_grad()
def spectral_clip_cols(M: torch.Tensor, clamp: float) -> torch.Tensor:
    # column-wise ℓ2 clip
    if clamp <= 0:
        return M
    col_norms = torch.linalg.vector_norm(M, dim=0) + 1e-8
    scale = torch.clamp(col_norms / clamp, min=1.0)
    return M / scale

@torch.no_grad()
def spectral_norm_clip(A: torch.Tensor, B: torch.Tensor, tau: float):
    """
    Clamp each rank-1 component A[:,i] B[:,i]^T so that ||A_i||*||B_i|| <= tau.
    This keeps AB^T stable without expensive SVD on half/bfloat.
    """
    if tau <= 0:
        return A, B
    an = torch.linalg.vector_norm(A, dim=0) + 1e-8
    bn = torch.linalg.vector_norm(B, dim=0) + 1e-8
    scale = torch.clamp((an * bn) / tau, min=1.0)
    s = torch.sqrt(scale)
    return A / s, B / s

# ---------------------------
# Feature selection helpers
# ---------------------------

@torch.no_grad()
def topk_indices_by_activation(X: torch.Tensor, k: int):
    """
    X: (tokens, features)
    returns top-k feature indices by mean |activation|
    """
    feats = X.abs().mean(dim=0)
    k = min(k, feats.numel())
    return torch.topk(feats, k).indices

# ---------------------------
# Kronecker / Sylvester (safe on T4)
# ---------------------------

@torch.no_grad()
def batched_kronecker(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Safe Kronecker for small r×r blocks, avoiding view/stride issues on CUDA.
    A,B: (r,r)
    """
    r = A.shape[-1]
    return (A.reshape(r,1,r,1) * B.reshape(1,r,1,r)).reshape(r*r, r*r)

@torch.no_grad()
def solve_sylvester_via_kron(Gx: torch.Tensor, Gb: torch.Tensor, rhs_vec: torch.Tensor, alpha: float=1e-1):
    """
    Solve (Gb ⊗ Gx + α I) vec(S) = rhs_vec  for small r (<= 64 typically).
    Falls back to float32 CPU if needed (half/bfloat SVD not implemented on some GPUs).
    """
    r = Gx.shape[0]
    device = Gx.device
    dtype = torch.float32  # robust on T4

    K = batched_kronecker(Gx.to(dtype), Gb.t().to(dtype))
    M = K + alpha * torch.eye(r*r, dtype=dtype)
    sol = torch.linalg.solve(M.cpu(), rhs_vec.reshape(-1).to(dtype).cpu()).to(device)
    return sol.reshape(r, r).to(Gx.dtype)


