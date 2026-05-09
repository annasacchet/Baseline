"""
Minimal OpenAI Chat client used by the smoke-test pipeline.

Goal: drop-in replacement for the rewriter / QA / AFG / AFV models in the
existing pipeline, so we can sanity-check the whole chain end-to-end without
loading any 32B HF model.

The class exposes two interfaces that match what the rest of the codebase
already calls:

  - `complete(user_prompt)`           used by rewriting + QA (single-message)
  - `generate(system, user, max_new)` matches HFChatModel.generate signature
                                      used by openfactscore_eval (AFG / AFV)

Behaviour
---------
- Reads OPENAI_API_KEY from the environment.
- Retries transient errors (rate limit / connection) with exponential backoff.
- Supports any chat model (default: gpt-4o-mini).
- temperature=0 by default → deterministic, repeatable smoke tests.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)


class OpenAIChat:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        role_label: str = "openai",
        temperature: float = 0.0,
        max_retries: int = 5,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Run with: "
                "OPENAI_API_KEY=sk-... python script.py"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.role_label = role_label
        self.temperature = temperature
        self.max_retries = max_retries
        print(f"[{role_label}] using OpenAI model: {model} (temp={temperature})", flush=True)

    def _call(self, messages: list[dict], max_tokens: int) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except (AuthenticationError, PermissionDeniedError, BadRequestError, NotFoundError) as err:
                # Permanent errors — retrying won't help. Fail loudly and immediately.
                raise RuntimeError(
                    f"OpenAI {type(err).__name__}: {err}\n"
                    "  → check OPENAI_API_KEY, model name, and project permissions."
                ) from err
            except (RateLimitError, APIConnectionError, APIError) as err:
                last_err = err
                wait = 2 ** attempt
                print(
                    f"  [{self.role_label}] {type(err).__name__} on attempt {attempt+1} "
                    f"({err}); retrying in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")

    def complete(self, user_prompt: str, max_tokens: int = 4096) -> str:
        """Single-user-message completion (rewriter / QA usage)."""
        return self._call(
            [{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )

    def generate(self, system_prompt: str | None, user_prompt: str, max_new_tokens: int) -> str:
        """HFChatModel.generate-compatible signature (used by AFG / AFV)."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._call(messages, max_tokens=max_new_tokens)
