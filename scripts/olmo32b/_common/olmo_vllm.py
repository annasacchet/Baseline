"""vLLM helper for the OLMo-3.1-32B-Instruct pipeline (bf16).

vLLM gives us tensor parallelism (both GPUs work on every forward, not
pipeline), paged attention, and prefix caching — together this is ~10-15x
faster than HF transformers on a 32B model on 2x consumer GPUs.

What we get from this module:
  - load_vllm(model_id, ...) returns a vllm.LLM
  - generate_batch_vllm(llm, prompts, ...) -> list[str]
  - render_chat(tokenizer, system_prompt, user_prompt) -> str

Notes
-----
* vLLM owns its own tokenizer internally, but we still load AutoTokenizer
  separately so we can apply chat templates with system + user roles
  identically to how the HF path does it.
* OLMo-3.1-32B is loaded full-precision in bf16; on a 2x 48 GB Homer node,
  tensor_parallel_size=2 splits each layer's weights across both GPUs.
  Combined ~64 GB of weights fits comfortably with the default KV cache.
"""

from __future__ import annotations

import os
import time

from transformers import AutoTokenizer


def hf_login_if_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("HF login OK", flush=True)


def load_vllm(
    model_id: str,
    *,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    quantization: str | None = None,
    dtype: str = "bfloat16",
    enforce_eager: bool = False,
):
    """Load a vLLM engine for the given model id.

    Parameters
    ----------
    tensor_parallel_size : how many GPUs vLLM splits the model across.
        Defaults to the number of visible CUDA devices.
    max_model_len : max total tokens (prompt + completion) per request.
        8k is enough for MuSiQue chains and keeps the KV-cache memory bounded.
    quantization : None for full-precision bf16 (default for OLMo-32B).
        Can be set to 'awq_marlin' / 'gptq' if a pre-quantized repo is used,
        or 'bitsandbytes' for on-the-fly NF4 4-bit quantization of the bf16
        checkpoint (load_format is auto-set to 'bitsandbytes' in that case).
    """
    from vllm import LLM

    if tensor_parallel_size is None:
        import torch
        tensor_parallel_size = max(1, torch.cuda.device_count())

    print(f"Loading {model_id} with vLLM (tp={tensor_parallel_size}, "
          f"quant={quantization}, dtype={dtype}, max_len={max_model_len}) ...",
          flush=True)
    t0 = time.time()
    kwargs = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
    )
    if quantization:
        kwargs["quantization"] = quantization
        if quantization == "bitsandbytes":
            kwargs["load_format"] = "bitsandbytes"
    llm = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return llm, tokenizer


def render_chat(tokenizer, system_prompt: str | None, user_prompt: str) -> str:
    """Apply the tokenizer chat template to a system+user conversation."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return (system_prompt + "\n\n" if system_prompt else "") + user_prompt


def generate_batch_vllm(
    llm, prompts: list[str], *,
    temperature: float, max_new_tokens: int, top_p: float = 0.95,
) -> list[str]:
    """Run vLLM.generate on a batch of pre-rendered prompts.

    Returns one decoded continuation per prompt, in the same order.
    """
    from vllm import SamplingParams

    if not prompts:
        return []
    sampling = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p) if temperature > 0 else 1.0,
        max_tokens=int(max_new_tokens),
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]
