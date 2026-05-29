"""Quick isolation test: can vLLM load OLMo-3.1-32B (bnb 4-bit) and generate
one completion WITHOUT JIT-building any CUDA kernel (no nvcc needed)?

Run on Homer (single GPU) with FlashInfer disabled:

  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  CUDA_VISIBLE_DEVICES=1 \
  python scripts/olmo32b/_common/test_vllm_olmo.py

If FLASH_ATTN also tries to JIT-compile, retry with XFORMERS:
  VLLM_ATTENTION_BACKEND=XFORMERS ...

Success = it prints a non-empty REWRITE and "OK". Then we flip the launcher
back to --backend vllm.
"""
from __future__ import annotations

import os
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "allenai/OLMo-3.1-32B-Instruct"

print("env VLLM_ATTENTION_BACKEND =", os.environ.get("VLLM_ATTENTION_BACKEND"), flush=True)
print("env CUDA_VISIBLE_DEVICES   =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

t0 = time.time()
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    dtype="bfloat16",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enforce_eager=True,            # no CUDA graph capture / torch.compile
    trust_remote_code=True,
)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompt = tok.apply_chat_template(
    [
        {"role": "system", "content": "You are a careful text rewriting assistant."},
        {"role": "user", "content": "Rephrase this to be more formal:\n\nHey, the cat sat on the mat."},
    ],
    tokenize=False, add_generation_prompt=True,
)

t1 = time.time()
out = llm.generate([prompt], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128), use_tqdm=False)
print(f"generated in {time.time()-t1:.1f}s", flush=True)
print("REWRITE:", repr(out[0].outputs[0].text.strip()), flush=True)
print("OK" if out[0].outputs[0].text.strip() else "EMPTY OUTPUT", flush=True)
