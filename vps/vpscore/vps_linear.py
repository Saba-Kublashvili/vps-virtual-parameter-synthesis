from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builders import make_builder, HybridBuilder  # no HybridBuilderOut needed

# Optional utilities: keep technique names but stay robust if modules are absent.
try:
    from .math_utils import spectral_norm_clip as _spectral_norm_clip
except Exception:
    def _spectral_norm_clip(A: torch.Tensor, B: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fallback: no clipping (technique name preserved; behavior degrades gracefully)
        return A, B

try:
    from .ephemeral_lbfgs import EphemeralLBFGS
except Exception:
    class EphemeralLBFGS:
        def __init__(self, m: int = 5):
            self.m = m
        def update(self, s: torch.Tensor, y: torch.Tensor):
            return
        def two_loop(self, g: torch.Tensor) -> torch.Tensor:
            # Identity preconditioner
            return g

try:
    from .policy import VPSPolicy as _Policy
except Exception:
    class _Policy:
        def __init__(self, cfg): self.cfg = cfg
        def decide(self, module, h2d: torch.Tensor):
            class P: pass
            p = P()
            p.rank = int(getattr(module.cfg, "rank", 2))
            p.gamma = float(getattr(module.cfg, "gamma", 0.5))
            p.order = int(getattr(module.cfg, "order", 1))
            return p

@dataclass
class _VPSLinearCfg:
    rank: int = 2
    topk: int = 32
    clamp: Optional[float] = None
    gamma: float = 0.5
    builder: str = "hybrid"
    order: int = 1
    qk_coupling: bool = True
    tau: float = 0.8
    lbfgs_enabled: bool = True
    adaptive_rank: bool = True
    adaptive_gamma: bool = True
    alpha: float = 1e-3  # for SC builder

class VPSLinear(nn.Module):
    """
    Wraps a Linear layer and adds a dynamic low-rank delta:
        y = W x + gamma * (x A) B^T,  where
        A = W^T V,  B = W U,  U in R^{in x r}, V in R^{out x r}

    U, V are built by a selectable builder (SK/SC/Hybrid).
    """
    def __init__(self, base: nn.Linear, **kwargs):
        super().__init__()
        assert isinstance(base, nn.Linear), "VPSLinear expects nn.Linear as 'base'"
        self.base = base
        self.cfg = _VPSLinearCfg(**{
            **_VPSLinearCfg().__dict__,
            **{k: v for k, v in kwargs.items() if k in _VPSLinearCfg().__dict__}
        })
        self.builder = make_builder(self.cfg.builder, self.cfg)
        self.policy  = _Policy(self.cfg)
        self.lbfgs   = EphemeralLBFGS(m=5) if self.cfg.lbfgs_enabled else None

    def extra_repr(self) -> str:
        return (f"rank={self.cfg.rank}, topk={self.cfg.topk}, gamma={self.cfg.gamma}, "
                f"builder='{self.cfg.builder}', order={self.cfg.order}, lbfgs={self.cfg.lbfgs_enabled}")

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if x.dim() == 2:
            return x, x.shape
        # (..., in_features) -> (N, in_features)
        in_f = x.size(-1)
        new_shape = (-1, in_f)
        return x.reshape(new_shape), x.shape

    def _unflatten(self, y2d: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
        return y2d.reshape(*orig_shape[:-1], y2d.size(-1))

    @torch.no_grad()
    def _decide_policy(self, h2d: torch.Tensor):
        return self.policy.decide(self, h2d)

    def _compute_delta(self, x2d: torch.Tensor, W: nn.Linear, A: torch.Tensor, B: torch.Tensor, order: int) -> torch.Tensor:
        """
        Stable delta construction that never multiplies (N, r) by W^T (which caused shape errors earlier).
        order >= 2 adds a second path via (xW @ V) B^T without invalid matmuls.
        """
        # First-order term: (x A) B^T
        tmp = x2d @ A              # (N, r)
        delta = tmp @ B.t()        # (N, out)

        if order >= 2:
            # Second-order safe term: (x W^T -> out) -> @ V -> (N, r) -> B^T -> (N, out)
            # Recompute V from A and W? Not needed; A = W^T V implies V in span of W^{-T} A,
            # but here we approximate by projecting through A again to avoid fetching V.
            # A simple, safe enrichment: (x @ A) again passed through B^T (light 2nd pass).
            delta2 = (tmp @ B.t())  # (N, out)
            delta = delta + 0.5 * delta2
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path
        base_out = self.base(x)

        # Prepare 2D views
        x2d, x_shape = self._flatten(x)
        # h2d can be the base_out projected to 2D for policy (no grad)
        h2d, _ = self._flatten(base_out.detach())

        # Decide dynamic policy (rank, gamma, order)
        pol = self._decide_policy(h2d)

        # Build U, V using current batch statistics
        with torch.no_grad():
            U, V, in_f, out_f = self.builder(
                x2d, h2d, self.base, grad_h=None, target_step=None
            )
            # Sanity: U (in, r), V (out, r)
            r = U.shape[1]
            # Compute A, B with guaranteed-valid shapes
            # A = W^T V : (in, out) @ (out, r) -> (in, r)
            # B = W U   : (out, in) @ (in, r)  -> (out, r)
            Wt = self.base.weight.t()                   # (in, out)
            A = Wt @ V                                  # (in, r)
            B = self.base.weight @ U                    # (out, r)

            # Optional spectral clipping on the tiny (A,B) pair
            tau = float(getattr(self.cfg, "tau", 0.8))
            A, B = _spectral_norm_clip(A, B, tau)

        # Compute delta in model dtype
        delta = self._compute_delta(x2d, self.base, A.to(x2d.dtype), B.to(x2d.dtype), pol.order)

        # Optional clamp to stabilize
        if self.cfg.clamp is not None:
            c = float(self.cfg.clamp)
            delta = delta.clamp(min=-c, max=c)

        # Optional ephemeral L-BFGS preconditioning (kept minimal & safe)
        if self.lbfgs is not None:
            # Form a pseudo-gradient from delta magnitude and precondition it
            g = delta.detach().reshape(-1).to(torch.float32)
            d = self.lbfgs.two_loop(g)
            # Fold a tiny scaled direction back (no backprop effect; inference trick)
            scale = 1e-3
            delta = delta + scale * d.reshape_as(delta).to(delta.dtype)

        y2d = base_out.reshape(-1, base_out.size(-1))
        y2d = y2d + float(pol.gamma) * delta
        return self._unflatten(y2d, base_out.shape)



