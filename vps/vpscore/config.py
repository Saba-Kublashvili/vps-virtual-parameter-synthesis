from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class VPSConfig:
    # ---- HF model I/O ----
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device_map: str = "auto"

    # Either set one (string like "float16" / "bfloat16") or leave None to rely on torch_dtype_str.
    dtype: Optional[str] = None

    # Back-compat with your infer script:
    # Used like: getattr(torch, cfg.torch_dtype_str, torch.float16)
    torch_dtype_str: str = "float16"   # Good default on Colab T4

    # ---- generation ----
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 0

    # ---- patch selection ----
    apply_to: List[str] = field(default_factory=lambda: [
        "q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"
    ])

    # ---- VPS hyperparams (forwarded to VPSLinear) ----
    rank: int = 2
    topk: int = 32
    clamp: Optional[float] = None
    gamma: float = 0.5
    builder: str = "hybrid"       # "sk" | "sc" | "hybrid"
    order: int = 1
    qk_coupling: bool = True
    tau: float = 0.8
    lbfgs_enabled: bool = True
    adaptive_rank: bool = True
    adaptive_gamma: bool = True
    alpha: float = 1e-3           # for SC builder

    # Convenience helper if you want it elsewhere
    def resolve_dtype_name(self) -> str:
        """
        Returns the dtype name string your code should map to a torch dtype.
        Prefers `dtype` if set, otherwise falls back to `torch_dtype_str`.
        """
        return (self.dtype or self.torch_dtype_str)


