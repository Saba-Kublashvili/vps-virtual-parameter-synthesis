from typing import Optional, List, Dict, Any
import torch

def _as_model_inputs(x):
    """Normalize tokenizer outputs into a dict with input_ids and attention_mask."""
    if isinstance(x, torch.Tensor):
        return {"input_ids": x, "attention_mask": torch.ones_like(x)}
    elif isinstance(x, dict):
        if "attention_mask" not in x and "input_ids" in x:
            x = dict(x)
            x["attention_mask"] = torch.ones_like(x["input_ids"])
        return x
    else:
        t = torch.as_tensor(x, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

def _chat_inputs(tokenizer, prompt: str, add_generation_prompt: bool = True):
    """
    Build input_ids using the tokenizer's chat_template when available.
    Returns a dict of tensors on CPU.
    """
    if getattr(tokenizer, "apply_chat_template", None) and getattr(tokenizer, "chat_template", None):
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return _as_model_inputs(out)
    # Fallback: plain prompt
    return _as_model_inputs(tokenizer(prompt, return_tensors="pt"))

def generate(model, tok, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.7, top_p: float = 0.95,
             do_sample: Optional[bool] = None) -> str:
    if do_sample is None:
        do_sample = temperature > 0.0

    inputs = _chat_inputs(tok, prompt, add_generation_prompt=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        return_dict_in_generate=True,
    )
    seq = out.sequences  # [bsz, prompt_len + new_len]

    # Decode ONLY the newly generated tokens (no role echoes)
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = seq[:, prompt_len:]
    return tok.decode(new_ids[0], skip_special_tokens=True).strip()



