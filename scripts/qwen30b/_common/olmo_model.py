"""Shared model-loading helpers for the OLMo-3.1-32B pipeline (HF transformers).

`allenai/OLMo-3.1-32B-Instruct` is open-weights (no gating). We default to
bf16 across `device_map="auto"`; on a 2x 48GB Homer node the ~64 GB of
weights split across both GPUs with headroom for activations.

Optional 4-bit NF4 (`use_4bit=True`) is exposed for memory-constrained
runs (e.g. a single 48 GB GPU) but is slower and not the default.
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
    else:
        print("HF_TOKEN not set — proceeding without login "
              "(OLMo-3.1-32B-Instruct is open-weights, this is usually fine)",
              flush=True)


def load_olmo(model_id: str, *, use_4bit: bool = False, padding_side: str = "left"):
    """Load OLMo-3.1-32B (or any HF causal LM) in bf16 by default."""
    print(f"Loading {model_id} (4-bit={use_4bit}) ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side

    kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    print(
        f"  loaded in {time.time() - t0:.1f}s · device map: "
        f"{getattr(model, 'hf_device_map', 'n/a')}",
        flush=True,
    )
    return tok, model
