import torch
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class LayerPolicy:
    use: bool
    rank: int
    topk: int
    gamma: float
    order: int
    use_fisher: bool = False
    budget_tight: bool = False

class VPSPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_improved = False
        self.improvement_history = []
        self.token_entropy = 0.0

    def update_outcome(self, improved: bool, loss_delta: float = 0.0):
        self.last_improved = improved
        self.improvement_history.append((improved, loss_delta))
        if len(self.improvement_history) > 10:
            self.improvement_history.pop(0)

    def set_token_entropy(self, entropy: float):
        self.token_entropy = entropy

    def decide(self, layer_module, h2d, token_entropy: Optional[float] = None) -> LayerPolicy:
        cfg = self.cfg

        # ---- robust energy (no NaN/Inf) ----
        energy_t = (h2d * h2d).mean()
        energy_t = torch.nan_to_num(energy_t, nan=0.0, posinf=1e6, neginf=0.0)
        energy = float(energy_t.item())

        # Base decision
        use = (not cfg.enable_policy) or (energy >= cfg.min_energy_threshold)

        # ---- scale in [0,1], numerically safe ----
        e_clamped = max(0.0, min(1e4, energy))
        try:
            scale = 1.0 - math.exp(-e_clamped)
        except OverflowError:
            scale = 1.0
        if not math.isfinite(scale):
            scale = 0.5  # safe fallback, only if upstream produced NaNs

        # Token entropy influence
        if token_entropy is not None:
            self.token_entropy = token_entropy
        if math.isfinite(self.token_entropy) and self.token_entropy > 0:
            scale = max(scale, min(1.0, self.token_entropy / 3.0))

        # Improvement history influence
        recent_improvements = sum(1 for imp, _ in self.improvement_history[-5:] if imp)
        if recent_improvements >= 3:
            scale = min(1.0, scale * 1.3)
        elif recent_improvements == 0 and len(self.improvement_history) >= 5:
            scale *= 0.7

        # ---- Adaptive parameters ----
        r_lo, r_hi = cfg.rank_bounds
        t_lo, t_hi = cfg.topk_bounds
        g_lo, g_hi = cfg.gamma_bounds
        o_lo, o_hi = cfg.order_bounds

        rank = int(r_lo + (r_hi - r_lo) * scale) if cfg.adaptive_rank else cfg.rank
        topk = int(t_lo + (t_hi - t_lo) * scale)
        gamma = g_lo + (g_hi - g_lo) * scale if cfg.adaptive_gamma else cfg.gamma
        order = int(o_lo + (o_hi - o_lo) * scale) if cfg.adaptive_order else cfg.order

        # Clamp to bounds
        rank = max(1, min(rank, r_hi))
        topk = max(t_lo, min(topk, t_hi))
        gamma = max(g_lo, min(gamma, g_hi))
        order = max(o_lo, min(order, o_hi))

        # Builder selection heuristics
        use_fisher = self.last_improved and scale > 0.7
        budget_tight = scale < 0.3 or energy < cfg.min_energy_threshold * 2

        return LayerPolicy(
            use=use, rank=rank, topk=topk, gamma=gamma, order=order,
            use_fisher=use_fisher, budget_tight=budget_tight
        )


import os, sys
sys.path.append("/content/vps")
os.environ["PYTHONPATH"] = "/content/vps:" + os.environ.get("PYTHONPATH","")

