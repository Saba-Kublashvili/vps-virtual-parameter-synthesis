# scripts/run_ablations.py
import argparse
import json
import torch
from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from vpscore.config import VPSConfig
from vpscore.patch_hf import patch_model_with_vps
from vpscore.hooks import HookManager
from vpscore.builders import make_builder
from vpscore.patch_hf import PAIR_REGISTRY
from scripts.eval_gsm8k import evaluate_gsm8k

@dataclass
class AblationConfig:
    name: str
    vps_enabled: bool = True
    gamma: float = 0.5
    rank: int = 2
    topk: int = 32
    order: int = 1
    builder: str = "hybrid"
    qk_coupling: bool = True
    lbfgs_enabled: bool = True
    tau: float = 0.8
    adaptive_rank: bool = True
    adaptive_gamma: bool = True

def run_ablation(model, tok, hooks, test_path, ablation_cfg: AblationConfig,
                 base_cfg: VPSConfig, limit=None):
    """Run a single ablation configuration"""
    # Update config with ablation settings
    cfg = VPSConfig()
    cfg.gamma = ablation_cfg.gamma
    cfg.rank = ablation_cfg.rank
    cfg.topk = ablation_cfg.topk
    cfg.order = ablation_cfg.order
    cfg.builder = ablation_cfg.builder
    cfg.qk_coupling = ablation_cfg.qk_coupling
    cfg.lbfgs_enabled = ablation_cfg.lbfgs_enabled
    cfg.tau = ablation_cfg.tau
    cfg.adaptive_rank = ablation_cfg.adaptive_rank
    cfg.adaptive_gamma = ablation_cfg.adaptive_gamma

    # Apply configuration to model
    for m in model.modules():
        if m.__class__.__name__ == "VPSLinear":
            if m.builder_name != cfg.builder:
                m.builder = make_builder(cfg.builder, cfg.rank, cfg.topk, m.clamp)
                m.builder_name = cfg.builder
            m.default_gamma = cfg.gamma
            m.default_rank = cfg.rank
            m.default_topk = cfg.topk
            m.order = cfg.order
            m.tau = cfg.tau
            if m.policy:
                m.policy.cfg = cfg

    # Re-bind Q/K if policy toggled
    for pair in PAIR_REGISTRY.values():
        q = pair.get("attn_q"); k = pair.get("attn_k")
        if not (q and k): continue
        if cfg.qk_coupling:
            q.is_Q = True; k.is_K = True; q._peer = k; k._peer = q
        else:
            q.is_Q = k.is_K = False; q._peer = k._peer = None

    # Run evaluation
    result = evaluate_gsm8k(model, tok, hooks, test_path, cfg,
                           use_vps=ablation_cfg.vps_enabled, limit=limit)

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", type=str, required=True)
    ap.add_argument("--limit", type=int, default=100, help="Limit examples for quick testing")
    ap.add_argument("--output", type=str, default="ablation_results.json")
    args = ap.parse_args()

    # Define ablation configurations
    ablations = [
        AblationConfig("baseline", vps_enabled=False),
        AblationConfig("vps_default"),
        AblationConfig("no_qk_coupling", qk_coupling=False),
        AblationConfig("no_lbfgs", lbfgs_enabled=False),
        AblationConfig("gamma_0.3", gamma=0.3),
        AblationConfig("gamma_0.7", gamma=0.7),
        AblationConfig("rank_1", rank=1),
        AblationConfig("rank_4", rank=4),
        AblationConfig("order_1", order=1),
        AblationConfig("order_2", order=2),
        AblationConfig("order_3", order=3),
        AblationConfig("builder_gn", builder="gn"),
        AblationConfig("builder_fisher", builder="fisher"),
        AblationConfig("builder_secant", builder="secant"),
        AblationConfig("builder_skew", builder="skew"),
        AblationConfig("builder_sparse", builder="sparse"),
        AblationConfig("tau_0.5", tau=0.5),
        AblationConfig("tau_1.0", tau=1.0),
        AblationConfig("fixed_rank", adaptive_rank=False),
        AblationConfig("fixed_gamma", adaptive_gamma=False),
        AblationConfig("topk_16", topk=16),
        AblationConfig("topk_64", topk=64),
    ]

    # Load model once
    cfg = VPSConfig()
    torch.manual_seed(cfg.seed)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[cfg.dtype]

    print(f"Loading model: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype,
                                                 device_map=cfg.device_map)
    model = patch_model_with_vps(model, cfg.apply_to, cfg.rank, cfg.topk,
                                 cfg.clamp, cfg.gamma, cfg.builder,
                                 cfg.softgrad_builder, policy_cfg=cfg)
    hooks = HookManager()
    hooks.attach(model)

    # Run ablations
    results = {}
    for ablation in ablations:
        print(f"\n=== Running ablation: {ablation.name} ===")
        result = run_ablation(model, tok, hooks, args.test_path, ablation, cfg, args.limit)
        results[ablation.name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "config": {
                "gamma": ablation.gamma,
                "rank": ablation.rank,
                "topk": ablation.topk,
                "order": ablation.order,
                "builder": ablation.builder,
                "qk_coupling": ablation.qk_coupling,
                "lbfgs": ablation.lbfgs_enabled,
                "tau": ablation.tau
            }
        }
        print(f"{ablation.name}: {result['accuracy']:.2f}%")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== ABLATION SUMMARY ===")
    baseline = results.get("baseline", {}).get("accuracy", 0)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        diff = res["accuracy"] - baseline
        print(f"{name:20s}: {res['accuracy']:6.2f}% ({diff:+.2f}%)")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()


