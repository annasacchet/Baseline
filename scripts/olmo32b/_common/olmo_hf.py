"""HuggingFace transformers backend for the OLMo-3.1-32B pipeline.

Drop-in replacement for olmo_vllm.py exposing the SAME three entry points
(load_vllm / render_chat / generate_batch_vllm) so the rewriting + Answer F1
scripts can switch backends with a single --backend flag, no other changes.

Why this exists
---------------
On a CUDA 12.7 driver, vLLM 0.10.x + OLMo-3.1 keeps trying to JIT-build
FlashInfer / custom kernels with the system nvcc (<12) and fails. The HF path
loads the bf16 checkpoint in NF4 4-bit via bitsandbytes (~18-20 GB, fits on a
single 48 GB A6000) and generates with model.generate — slower than vLLM but
robust, no compilation, and identical to what perplexity/OpenFActScore already
use on this node.

Generation is done in mini-batches with left-padding so a whole batch of
prompts can be decoded in one forward loop.
"""

from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def hf_login_if_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("HF login OK", flush=True)


# generate_batch_vllm only receives the model (vLLM signature), so we stash
# the tokenizer module-globally at load time.
_TOKENIZER = None


def load_vllm(
    model_id: str,
    *,
    tensor_parallel_size: int | None = None,   # ignored (HF uses device_map)
    max_model_len: int = 8192,                  # ignored here
    gpu_memory_utilization: float = 0.90,       # ignored here
    quantization: str | None = None,            # "bitsandbytes"/"4bit"/None
    dtype: str = "bfloat16",
    enforce_eager: bool = False,                # ignored here
):
    """Load OLMo via HF transformers. Signature mirrors olmo_vllm.load_vllm so
    callers don't need to change. Returns (model, tokenizer)."""
    global _TOKENIZER
    use_4bit = quantization in ("bitsandbytes", "4bit", "nf4")
    print(f"Loading {model_id} with HF transformers "
          f"(4bit={use_4bit}, dtype={dtype}) ...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left padding so generated tokens are contiguous at the right edge.
    tokenizer.padding_side = "left"

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    _TOKENIZER = tokenizer
    print(f"  loaded in {time.time() - t0:.1f}s · "
          f"device map: {getattr(model, 'hf_device_map', 'n/a')}", flush=True)
    return model, tokenizer


def render_chat(tokenizer, system_prompt: str | None, user_prompt: str) -> str:
    """Apply the tokenizer chat template (same as olmo_vllm.render_chat)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return (system_prompt + "\n\n" if system_prompt else "") + user_prompt


# Mini-batch size for HF generation. Smaller than vLLM's continuous batching;
# tune via OLMO_HF_BATCH_SIZE if you hit OOM during decode.
_HF_BATCH_SIZE = int(os.environ.get("OLMO_HF_BATCH_SIZE", "8"))


@torch.no_grad()
def generate_batch_vllm(
    model, prompts: list[str], *,
    temperature: float, max_new_tokens: int, top_p: float = 0.95,
) -> list[str]:
    """Generate one continuation per prompt via HF model.generate.

    Name kept as generate_batch_vllm for drop-in compatibility. The first arg
    is an HF model (not a vLLM engine); the tokenizer is taken from the model
    config cache set up at load time.
    """
    if not prompts:
        return []

    tokenizer = _TOKENIZER
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))

    n_batches = (len(prompts) + _HF_BATCH_SIZE - 1) // _HF_BATCH_SIZE
    print(f"  [hf] generate_batch_vllm: {len(prompts)} prompts, "
          f"batch_size={_HF_BATCH_SIZE}, {n_batches} batches", flush=True)
    outputs: list[str] = []
    for i in range(0, len(prompts), _HF_BATCH_SIZE):
        batch_idx = i // _HF_BATCH_SIZE + 1
        t0 = time.time()
        batch = prompts[i:i + _HF_BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        print(f"  [hf] batch {batch_idx}/{n_batches}: tokenized "
              f"(input_ids {tuple(enc['input_ids'].shape)}), calling generate ...",
              flush=True)
        out = model.generate(**enc, **gen_kwargs)
        # strip the prompt: with left padding, new tokens start at input width.
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(t.strip() for t in decoded)
        print(f"    [hf] batch {batch_idx}/{n_batches} done in {time.time()-t0:.1f}s", flush=True)
    return outputs
