import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vpscore.config import VPSConfig
from vpscore.patch_hf import patch_model_with_vps
from vpscore.verifiers.composite_verifier import CompositeVerifier
from vpscore.utils.generation import generate
from vpscore.hooks import HookManager
from vpscore.math_utils import compute_token_entropy

DEFAULT_VERIFIER_WEIGHTS = {
    "exact_match": 0.8,
    "coherence": 0.2,
    "length_penalty": 0.0,
}

def build(cfg: VPSConfig):
    # ---- robust fallbacks so missing config fields don't crash ----
    seed = getattr(cfg, "seed", 1234)
    torch.manual_seed(seed)

    dtype_name = getattr(cfg, "dtype", "bf16")
    _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = _dtype_map.get(dtype_name, torch.bfloat16)

    model_name = getattr(cfg, "model_name", None)
    if not model_name:
        raise ValueError("cfg.model_name is not set in VPSConfig")

    device_map = getattr(cfg, "device_map", "auto")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Use dtype= (not torch_dtype=) to avoid the deprecation warning.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)

    # Your patcher: the variant we standardized earlier takes (model, apply_to, cfg)
    apply_to = getattr(cfg, "apply_to", None)
    model = patch_model_with_vps(model, apply_to, cfg)

    hooks = HookManager()
    hooks.attach(model)
    return tok, model, hooks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--gold", type=str, default=None)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    cfg = VPSConfig()

    tok, model, hooks = build(cfg)

    # Fall back if cfg.verifier_weights missing
    verifier_weights = getattr(cfg, "verifier_weights", DEFAULT_VERIFIER_WEIGHTS)
    verifier = CompositeVerifier(verifier_weights)

    # Also fallbacks for gen params
    max_new_tokens = getattr(cfg, "max_new_tokens", 128)
    temperature = getattr(cfg, "temperature", 0.7)
    top_p = getattr(cfg, "top_p", 0.9)

    # ----------------- Iteration 0 (plain decode) -----------------
    text = generate(model, tok, args.prompt, max_new_tokens, temperature, top_p)
    print(f"=== Iteration 0 ===\n{text}\n")

    if args.gold is None or args.iters < 2:
        return

    prev_loss = float("inf")

    for it in range(1, args.iters):
        hooks.clear_buffers()

        # Clear any L-BFGS memories (keeps your technique intact)
        for m in model.modules():
            if hasattr(m, "clear_lbfgs"):
                m.clear_lbfgs()

        # Forward pass to compute token entropy for the adaptive policy
        inputs = tok(args.prompt, return_tensors="pt").to(model.device)
        with torch.enable_grad():
            out = model(**inputs)
            logits = out.logits
            entropy = compute_token_entropy(logits[0, -1, :])

        # Feed entropy to policies (again, preserving your mechanism)
        for m in model.modules():
            if hasattr(m, "policy") and m.policy is not None:
                m.policy.set_token_entropy(entropy)

        # Deterministic decode for verifier evaluation
        out_ids = model.generate(
            **inputs,
            max_new_tokens=min(64, max_new_tokens),
            do_sample=False,
            return_dict_in_generate=True,
        )
        pred_text = tok.decode(out_ids.sequences[0], skip_special_tokens=True)

        # Multi-objective verifier (unchanged logic)
        losses = verifier.compute_loss(pred_text, args.gold, model, tok, args.prompt)
        vloss = losses["total"]

        # Feedback loop to policy
        improved = vloss < prev_loss
        for m in model.modules():
            if hasattr(m, "policy") and m.policy is not None:
                m.policy.update_outcome(improved, prev_loss - vloss)
        prev_loss = vloss

        # Short CE surrogate to populate grads → VPS uses grad_h
        gold_ids = tok(args.gold, return_tensors="pt").to(model.device)["input_ids"][0]
        T = min(8, gold_ids.shape[0])
        target = gold_ids[-T:].unsqueeze(0)  # [1, T]

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out2 = model(**inputs)
            logits2 = out2.logits[:, -T:, :]  # [1, T, V]
            ce = torch.nn.functional.cross_entropy(
                logits2.reshape(-1, logits2.size(-1)),
                target.reshape(-1)
            )
            ce.backward()

        # VPS-enhanced generation
        text2 = generate(model, tok, args.prompt, max_new_tokens, temperature, top_p)
        print(f"=== Iteration {it} (VPS enhanced) ===")
        print(f"Losses: {losses}")
        print(f"Output:\n{text2}\n")

if __name__ == "__main__":
    main()

