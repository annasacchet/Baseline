"""Shared model-loading helpers for the Llama-3.1-70B 4-bit pipeline.

`meta-llama/Llama-3.1-70B-Instruct` is a gated repo — set HF_TOKEN before
loading. 4-bit NF4 brings the weights to ~40 GB, which fits across two
RTX A6000 48GB via `device_map="auto"`.
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
              "(Llama-3.1-70B is gated, this will likely fail)", flush=True)


def load_llama(model_id: str, *, use_4bit: bool = True, padding_side: str = "left"):
    """Load Llama-3.1-70B (or any HF causal LM) with 4-bit NF4 by default."""
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
