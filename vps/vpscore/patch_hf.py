from __future__ import annotations
import torch.nn as nn
from typing import List, Any
from .vps_linear import VPSLinear

NAMES = {
    "attn_q": ["q_proj","wq","query","q"],
    "attn_k": ["k_proj","wk","key","k"],
    "attn_v": ["v_proj","wv","value","v"],
    "attn_o": ["o_proj","wo","out_proj","o"],
    "mlp_up": ["up_proj","w1","fc_in","dense_h_to_4h","gate_proj","w3","gate"],
    "mlp_down":["down_proj","w2","fc_out","dense_4h_to_h"],
}

# Global registry for Q/K pairing across modules (keyed by parent module id)
PAIR_REGISTRY = {}

def _match_bucket(name:str, apply_to:List[str]):
    lname = name.lower()
    for key, aliases in NAMES.items():
        if key in apply_to and any(a in lname for a in aliases):
            return key
    return None

def _wrap(module: nn.Module, apply_to: List[str], vps_kwargs: dict):
    for name, child in list(module.named_children()):
        _wrap(child, apply_to, vps_kwargs)
        if isinstance(child, nn.Linear):
            bucket = _match_bucket(name, apply_to)
            if bucket is not None:
                vps = VPSLinear(child, **vps_kwargs)
                setattr(module, name, vps)
                # Register possible q/k pairs under the parent block
                if bucket in ["attn_q", "attn_k"]:
                    key = id(module)
                    PAIR_REGISTRY.setdefault(key, {})[bucket] = vps

def patch_model_with_vps(model: nn.Module, apply_to: List[str], *args: Any, **kwargs: Any):
    """
    Flexible wrapper: accepts either
      A) legacy long-form:
         (model, apply_to, rank, topk, clamp, gamma, builder, [softgrad_builder=False], policy_cfg=None)
         -- or extended long-form some users used:
         (model, apply_to, rank, topk, clamp, gamma, builder, order, qk_coupling, tau,
          lbfgs_enabled, adaptive_rank, adaptive_gamma, [softgrad_builder=False], policy_cfg=None)

      B) cfg-form:
         (model, apply_to, cfg)   where cfg is VPSConfig
    """
    # Defaults
    softgrad_builder = kwargs.pop("softgrad_builder", False)
    policy_cfg = kwargs.pop("policy_cfg", None)
    qk_coupling_flag = False  # will set from cfg or long-form if provided

    # ---- Parse inputs ----
    if len(args) == 1 and not isinstance(args[0], (int, float, str, bool)):
        # cfg-form
        cfg = args[0]
        rank   = getattr(cfg, "rank", 2)
        topk   = getattr(cfg, "topk", 32)
        clamp  = getattr(cfg, "clamp", 0.2)
        gamma  = getattr(cfg, "gamma", 0.5)
        builder= getattr(cfg, "builder", "hybrid")
        softgrad_builder = getattr(cfg, "softgrad_builder", softgrad_builder)
        policy_cfg = cfg
        qk_coupling_flag = bool(getattr(cfg, "qk_coupling", False))
    else:
        # long-form(s)
        if len(args) >= 6:
            rank, topk, clamp, gamma, builder = args[0:5+1]
            # extended long-form might pass order, qk_coupling, tau, lbfgs_enabled, adaptive_rank, adaptive_gamma next
            if len(args) >= 12:
                # args[6] is order (unused here), args[7] is qk_coupling
                qk_coupling_flag = bool(args[7])
            # optional trailing softgrad_builder, policy_cfg via kwargs or positional
            if len(args) >= 13:
                # softgrad might be at position 12 (after 12 required)
                if isinstance(args[12], bool):
                    softgrad_builder = args[12]
                if len(args) >= 14:
                    policy_cfg = args[13]
        else:
            raise TypeError("patch_model_with_vps: unsupported argument pattern. "
                            "Pass VPSConfig as third arg OR full long-form params.")

    vps_kwargs = dict(
        rank=rank,
        topk=topk,
        clamp=clamp,
        gamma=gamma,
        builder=builder,
        softgrad_builder=softgrad_builder,
        policy_cfg=policy_cfg,
    )

    # Actually swap in VPSLinear for selected Linear layers
    _wrap(model, apply_to, vps_kwargs)

    # Finalize Q/K pairing if enabled
    if qk_coupling_flag:
        for pair in PAIR_REGISTRY.values():
            q = pair.get("attn_q")
            k = pair.get("attn_k")
            if q is not None and k is not None:
                q.is_Q = True; k.is_K = True
                q._peer = k;   k._peer = q

    return model

