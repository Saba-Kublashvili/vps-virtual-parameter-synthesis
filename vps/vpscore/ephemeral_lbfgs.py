from __future__ import annotations
import torch
from collections import deque

class EphemeralLBFGS:
    """
    Tiny L-BFGS memory; we keep curvature info but ensure 1D vectors.
    """
    def __init__(self, m: int = 5):
        self.m = m
        self.S = deque(maxlen=m)
        self.Y = deque(maxlen=m)

    @torch.no_grad()
    def update(self, s: torch.Tensor, y: torch.Tensor):
        # flatten to 1D
        s = s.reshape(-1).detach()
        y = y.reshape(-1).detach()
        if s.numel() == 0 or y.numel() == 0:
            return
        # basic curvature check
        if torch.dot(y, s) > 1e-8:
            self.S.append(s)
            self.Y.append(y)

