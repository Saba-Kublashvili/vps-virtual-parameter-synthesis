from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn

# ---- Small config facade so builders read only what they need ----
@dataclass
class _CfgView:
    rank: int = 2
    topk: int = 32
    builder: str = "hybrid"
    alpha: float = 1e-3  # ridge in SC

def _safe_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = int(min(max(1, k), scores.numel()))
    _, idx = torch.topk(scores, k, largest=True, sorted=False)
    return idx

def _scores_in_out(x2d: torch.Tensor, W: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    # x2d: (N, in_features)
    in_scores = x2d.abs().mean(dim=0)                           # (in,)
    with torch.no_grad():
        y2d = x2d @ W.weight.t()                                # (N, out)
    out_scores = y2d.abs().mean(dim=0)                          # (out,)
    return in_scores, out_scores

class SKBuilder:
    """
    Sparse selector builder.
    Returns U (in_features x r) and V (out_features x r), one-hot columns at top-k indices.

      A = W.t() @ V   -> (in, r)
      B = W     @ U   -> (out, r)
    """
    def __init__(self, cfg):
        self.cfg = _CfgView(
            rank=getattr(cfg, "rank", 2),
            topk=getattr(cfg, "topk", 32),
            builder=getattr(cfg, "builder", "sk"),
            alpha=getattr(cfg, "alpha", 1e-3),
        )

    @torch.no_grad()
    def __call__(
        self,
        x2d: torch.Tensor,                 # (N, in_features)
        h2d: torch.Tensor,                 # (N, hidden)  (unused here)
        W: nn.Linear,                      # base linear (out_features x in_features)
        grad_h: Optional[torch.Tensor],    # (N, hidden)  (unused here)
        target_step: Optional[int],        # unused hook
    ):
        device = W.weight.device
        dtype  = W.weight.dtype
        out_f, in_f = W.weight.shape      # (out, in)  <- PyTorch Linear layout

        in_scores, out_scores = _scores_in_out(x2d, W)
        in_sel  = _safe_topk(in_scores, min(self.cfg.topk, in_f))
        out_sel = _safe_topk(out_scores, min(self.cfg.topk, out_f))

        r = int(min(self.cfg.rank, in_sel.numel(), out_sel.numel()))
        r = max(1, r)

        U = torch.zeros(in_f,  r, device=device, dtype=dtype)   # (in, r)
        V = torch.zeros(out_f, r, device=device, dtype=dtype)   # (out, r)
        cols = torch.arange(r, device=device)
        U[in_sel[:r],  cols] = 1.0
        V[out_sel[:r], cols] = 1.0

        return U, V, in_f, out_f

class SCBuilder(SKBuilder):
    """
    Sylvester-style coupling refinement on top of SK (robust to fp16).
    Solves a small ridge least-squares in float32 and folds it into V (column mixing).
    Falls back gracefully to pure SK if numerics are unhappy.
    """
    @torch.no_grad()
    def __call__(
        self,
        x2d: torch.Tensor,
        h2d: torch.Tensor,
        W: nn.Linear,
        grad_h: Optional[torch.Tensor],
        target_step: Optional[int],
    ):
        # start from valid SK selectors
        U, V, in_f, out_f = super().__call__(x2d, h2d, W, grad_h, target_step)

        try:
            r = U.shape[1]
            if r == 0:
                return U, V, in_f, out_f

            # recover selected indices from one-hot columns
            in_mask  = U.abs().sum(dim=1) > 0                      # (in,)
            out_mask = V.abs().sum(dim=1) > 0                      # (out,)
            in_idx   = in_mask.nonzero(as_tuple=False).flatten()[:r]
            out_idx  = out_mask.nonzero(as_tuple=False).flatten()[:r]

            # compact views (float32 for stable linear algebra)
            XA = x2d[:, in_idx].to(torch.float32)                  # (N, r)
            Y  = (x2d @ W.weight.t())[:, out_idx].to(torch.float32)  # (N, r)

            # ridge: (X^T X + alpha I) T = X^T Y
            Gx  = XA.T @ XA                                        # (r, r)
            RHS = XA.T @ Y                                         # (r, r)
            I   = torch.eye(r, device=Gx.device, dtype=Gx.dtype)
            alpha = float(getattr(self.cfg, "alpha", 1e-3))
            T   = torch.linalg.solve(Gx + alpha * I, RHS)          # (r, r)

            # mix columns of V by T^T (stays (out, r)), then column-normalize
            V_mix = (V @ T.to(V.dtype).t())                        # (out, r)
            col_norm = V_mix.norm(dim=0, keepdim=True).clamp_min(1e-6)
            V = (V_mix / col_norm).to(V.dtype)
        except Exception:
            # keep SK result if anything goes wrong
            pass

        return U, V, in_f, out_f

class HybridBuilder:
    """Use SC when gradient information is available; otherwise SK."""
    def __init__(self, cfg):
        self.sc = SCBuilder(cfg)
        self.sk = SKBuilder(cfg)

    @torch.no_grad()
    def __call__(
        self,
        x2d: torch.Tensor,
        h2d: torch.Tensor,
        W: nn.Linear,
        grad_h: Optional[torch.Tensor],
        target_step: Optional[int],
    ):
        if grad_h is not None:
            return self.sc(x2d, h2d, W, grad_h, target_step)
        return self.sk(x2d, h2d, W, grad_h, target_step)

def make_builder(name: str, cfg):
    """
    Factory: returns a callable builder instance with signature
        (x2d, h2d, W, grad_h, target_step) -> (U, V, in_features, out_features)
    """
    key = (name or "").lower()
    if key in {"sk", "sks", "sparse", "selector"}:
        return SKBuilder(cfg)
    if key in {"sc", "sylvester", "coupled"}:
        return SCBuilder(cfg)
    return HybridBuilder(cfg)  # default

# Back-compat alias for older imports
HybridBuilderOut = HybridBuilder

